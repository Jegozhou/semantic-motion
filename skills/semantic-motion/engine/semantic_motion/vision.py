"""Visual scene understanding for layout: people, faces, hands/gestures, motion, shots, existing text.

One decode pass at analysis resolution and full frame rate (for frame-accurate cut detection);
heavier models run on a sub-sampled stream (default 4 fps). Everything is reduced to a coarse
grid (80 cells on the long side) which is what the layout solver consumes.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .media import FrameReader, MediaInfo, even
from .paths import model_path
from .util import log, warn

GRID_LONG = 80


@dataclass
class VisionData:
    W: int
    H: int
    gw: int
    gh: int
    t: np.ndarray                 # sample times [N]
    sample_shot: np.ndarray       # shot index per sample [N]
    person: np.ndarray            # [N,gh,gw] person probability
    head: np.ndarray              # [N,gh,gw] face-skin + hair probability
    motion: np.ndarray            # [N,gh,gw] normalized motion energy
    faces: list                   # per sample: list of [x,y,w,h,score] normalized
    hands: list                   # per sample: list of [x,y,vis] normalized (wrists+index tips)
    shots: list                   # [{id,start,end,f0,f1,...}]
    shot_clutter: np.ndarray      # [S,gh,gw]
    shot_lum: np.ndarray          # [S,gh,gw]
    shot_rgb: np.ndarray          # [S,gh,gw,3]
    shot_text: np.ndarray         # [S,gh,gw] fraction of samples with detected text
    overlay_text: np.ndarray      # [gh,gw] text persisting across shots / time (burned-in subs, logos)
    meta: dict = field(default_factory=dict)

    def save(self, path: str) -> None:
        np.savez_compressed(path, W=self.W, H=self.H, gw=self.gw, gh=self.gh, t=self.t, sample_shot=self.sample_shot,
                            person=self.person.astype(np.float16), head=self.head.astype(np.float16),
                            motion=self.motion.astype(np.float16), shot_clutter=self.shot_clutter.astype(np.float16),
                            shot_lum=self.shot_lum.astype(np.float16), shot_rgb=self.shot_rgb.astype(np.float16),
                            shot_text=self.shot_text.astype(np.float16), overlay_text=self.overlay_text.astype(np.float16),
                            faces=json.dumps(self.faces), hands=json.dumps(self.hands), shots=json.dumps(self.shots),
                            meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path: str) -> "VisionData":
        z = np.load(path, allow_pickle=False)
        f32 = lambda k: z[k].astype(np.float32)  # noqa: E731
        return cls(W=int(z["W"]), H=int(z["H"]), gw=int(z["gw"]), gh=int(z["gh"]), t=z["t"], sample_shot=z["sample_shot"],
                   person=f32("person"), head=f32("head"), motion=f32("motion"), faces=json.loads(str(z["faces"])),
                   hands=json.loads(str(z["hands"])), shots=json.loads(str(z["shots"])), shot_clutter=f32("shot_clutter"),
                   shot_lum=f32("shot_lum"), shot_rgb=f32("shot_rgb"), shot_text=f32("shot_text"),
                   overlay_text=f32("overlay_text"), meta=json.loads(str(z["meta"])))

    # ------------------------------------------------------------------ queries
    def samples_between(self, t0: float, t1: float) -> np.ndarray:
        idx = np.where((self.t >= t0 - 1e-6) & (self.t <= t1 + 1e-6))[0]
        if len(idx) == 0:
            idx = np.array([int(np.argmin(np.abs(self.t - (t0 + t1) / 2)))])
        return idx

    def shot_at(self, t: float) -> int:
        for i, s in enumerate(self.shots):
            if s["start"] - 1e-6 <= t < s["end"]:
                return i
        return len(self.shots) - 1

    def face_boxes_between(self, t0: float, t1: float) -> list:
        out = []
        for i in self.samples_between(t0, t1):
            out.extend(self.faces[int(i)])
        return out


def _grid_dims(W: int, H: int) -> tuple[int, int]:
    if W >= H:
        gw = GRID_LONG
        gh = max(8, int(round(GRID_LONG * H / W)))
    else:
        gh = GRID_LONG
        gw = max(8, int(round(GRID_LONG * W / H)))
    return gw, gh


def _pool(m: np.ndarray, gw: int, gh: int) -> np.ndarray:
    return cv2.resize(m.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)


class _Models:
    def __init__(self, use_pose: bool = True, use_text: bool = True):
        import mediapipe as mp  # type: ignore
        from mediapipe.tasks import python as mpt  # type: ignore
        from mediapipe.tasks.python import vision as mpv  # type: ignore

        self.mp = mp
        VIDEO = mpv.RunningMode.VIDEO
        self.seg = mpv.ImageSegmenter.create_from_options(mpv.ImageSegmenterOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(model_path("selfie_multiclass_256x256.tflite"))),
            running_mode=VIDEO, output_confidence_masks=True, output_category_mask=False))
        self.face = mpv.FaceDetector.create_from_options(mpv.FaceDetectorOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(model_path("blaze_face_short_range.tflite"))),
            running_mode=VIDEO, min_detection_confidence=0.55))
        self.pose = None
        if use_pose:
            try:
                self.pose = mpv.PoseLandmarker.create_from_options(mpv.PoseLandmarkerOptions(
                    base_options=mpt.BaseOptions(model_asset_path=str(model_path("pose_landmarker_lite.task"))),
                    running_mode=VIDEO, num_poses=2, min_pose_detection_confidence=0.5))
            except Exception as e:  # pragma: no cover
                warn(f"pose model unavailable: {e}")
        self.text = None
        if use_text:
            try:
                tm = cv2.dnn.TextDetectionModel_DB(str(model_path("text_detection_cn_ppocrv3_2023may.onnx")))
                tm.setBinaryThreshold(0.3).setPolygonThreshold(0.5).setMaxCandidates(200).setUnclipRatio(2.0)
                self.text = tm
            except Exception as e:  # pragma: no cover
                warn(f"text detector unavailable: {e}")

    def segment(self, rgb: np.ndarray, ts_ms: int):
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.seg.segment_for_video(img, ts_ms)
        cms = [m.numpy_view() for m in res.confidence_masks]
        # classes: 0 background, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 others
        person = 1.0 - cms[0]
        head = np.clip(cms[1] + cms[3], 0, 1)
        fres = self.face.detect_for_video(img, ts_ms)
        h, w = rgb.shape[:2]
        faces = []
        for d in fres.detections:
            bb = d.bounding_box
            sc = d.categories[0].score if d.categories else 1.0
            faces.append([bb.origin_x / w, bb.origin_y / h, bb.width / w, bb.height / h, float(sc)])
        hands = []
        if self.pose is not None:
            pres = self.pose.detect_for_video(img, ts_ms)
            for lm in pres.pose_landmarks:
                for k in (15, 16, 19, 20):  # wrists, index fingers
                    p = lm[k]
                    vis = getattr(p, "visibility", 1.0) or 0.0
                    if vis > 0.5 and 0 <= p.x <= 1 and 0 <= p.y <= 1:
                        hands.append([float(p.x), float(p.y), float(vis)])
        return person, head, faces, hands

    def close(self) -> None:
        for m in (self.seg, self.face, self.pose):
            try:
                if m is not None:
                    m.close()
            except Exception:
                pass

    def detect_text(self, rgb: np.ndarray) -> list:
        if self.text is None:
            return []
        h, w = rgb.shape[:2]
        iw, ih = (640, 352) if w >= h else (352, 640)
        self.text.setInputParams(1.0 / 255.0, (iw, ih), (122.67891434, 116.66876762, 104.00698793))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        try:
            polys, conf = self.text.detect(bgr)
        except Exception:
            return []
        boxes = []
        for p in polys:
            p = np.array(p, np.float32)
            x0, y0 = p[:, 0].min() / w, p[:, 1].min() / h
            x1, y1 = p[:, 0].max() / w, p[:, 1].max() / h
            if (x1 - x0) * (y1 - y0) > 0.0004:
                boxes.append([float(x0), float(y0), float(x1 - x0), float(y1 - y0)])
        return boxes


def detect_cuts(scores: np.ndarray, fps: float, min_shot: float = 0.4) -> list[int]:
    cuts = []
    n = len(scores)
    win = max(3, int(fps))
    last = 0  # frame 0 starts a shot: a "cut" in the first min_shot seconds is a fade-in/encoder artefact
    for i in range(1, n):
        lo = max(1, i - win)
        ctx = scores[lo:i]
        base = float(np.median(ctx)) if len(ctx) else 0.0
        thr = max(0.16, base * 4 + 0.05)
        if scores[i] > thr and (i - last) >= min_shot * fps:
            cuts.append(i)
            last = i
    if cuts and n - cuts[-1] < min_shot * fps:
        cuts.pop()  # same at the very end
    return cuts


def analyze(info: MediaInfo, sample_fps: float = 4.0, use_pose: bool = True, use_text: bool = True,
            text_every: float = 1.0) -> VisionData:
    W, H = info.width, info.height
    gw, gh = _grid_dims(W, H)
    aw = 640 if W >= H else 360
    ah = even(aw * H / W)
    log(f"vision: analysis {aw}x{ah}, grid {gw}x{gh}, sample {sample_fps} fps")
    models = _Models(use_pose, use_text)
    fps = info.fps
    step = max(1, int(round(fps / sample_fps)))
    text_step = max(1, int(round(fps * text_every)))
    reader = FrameReader(info, aw, ah, fast=True)
    cut_scores = []
    prev_thumb = None
    prev_hist = None
    grays: list = []
    S_t, S_person, S_head, S_motion, S_faces, S_hands, S_frame = [], [], [], [], [], [], []
    clutter_acc, lum_acc, rgb_acc, text_acc, frame_samples = [], [], [], [], []
    for fi, frame in enumerate(reader):
        rgb = frame[:, :, :3]
        small = cv2.resize(rgb, (48, 27) if W >= H else (27, 48), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        if prev_thumb is None:
            cut_scores.append(0.0)
        else:
            d = float(np.mean(np.abs(g - prev_thumb))) / 255.0
            hc = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
            cut_scores.append(0.55 * min(1.0, d * 4) + 0.45 * max(0.0, 1 - hc))
        prev_thumb, prev_hist = g, hist
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        grays.append(gray)
        if len(grays) > 8:
            grays.pop(0)
        if fi % step == 0:
            ts = fi / fps
            person, head, faces, hands = models.segment(rgb, int(round(ts * 1000)))
            ref = grays[0]
            mot = cv2.GaussianBlur(cv2.absdiff(gray, ref).astype(np.float32), (5, 5), 0) / 255.0
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx * gx + gy * gy) / 255.0
            S_t.append(ts)
            S_frame.append(fi)
            S_person.append(_pool(person, gw, gh))
            S_head.append(_pool(head, gw, gh))
            S_motion.append(_pool(mot, gw, gh))
            S_faces.append(faces)
            S_hands.append(hands)
            clutter_acc.append(_pool(mag, gw, gh))
            lum = (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]) / 255.0
            lum_acc.append(_pool(lum, gw, gh))
            rgb_acc.append(cv2.resize(rgb.astype(np.float32) / 255.0, (gw, gh), interpolation=cv2.INTER_AREA))
        if fi % text_step == 0:
            boxes = models.detect_text(rgb)
            m = np.zeros((gh, gw), np.float32)
            for x, y, w, h in boxes:
                x0, y0 = int(math.floor(x * gw)), int(math.floor(y * gh))
                x1, y1 = int(math.ceil((x + w) * gw)), int(math.ceil((y + h) * gh))
                m[max(0, y0):min(gh, y1), max(0, x0):min(gw, x1)] = 1.0
            text_acc.append(m)
            frame_samples.append(fi)
    models.close()
    n_frames = len(cut_scores)
    if n_frames == 0:
        raise RuntimeError("no frames decoded")
    cuts = detect_cuts(np.array(cut_scores), fps)
    bounds = [0] + cuts + [n_frames]
    shots = []
    for k in range(len(bounds) - 1):
        f0, f1 = bounds[k], bounds[k + 1]
        shots.append({"id": k + 1, "f0": f0, "f1": f1, "start": round(f0 / fps, 4), "end": round(f1 / fps, 4)})
    S_frame_a = np.array(S_frame)
    sample_shot = np.array([next(i for i, s in enumerate(shots) if s["f0"] <= f < s["f1"]) for f in S_frame], np.int32)
    text_frames = np.array(frame_samples)
    S = len(shots)
    shot_clutter = np.zeros((S, gh, gw), np.float32)
    shot_lum = np.zeros((S, gh, gw), np.float32)
    shot_rgb = np.zeros((S, gh, gw, 3), np.float32)
    shot_text = np.zeros((S, gh, gw), np.float32)
    person = np.array(S_person)
    head = np.array(S_head)
    motion = np.array(S_motion)
    # normalize motion by a robust global scale, ignoring camera-wide changes
    mscale = float(np.percentile(motion, 99)) if motion.size else 1.0
    motion = np.clip(motion / max(mscale, 1e-3), 0, 1)
    for si, s in enumerate(shots):
        idx = np.where(sample_shot == si)[0]
        if len(idx) == 0:
            idx = np.array([int(np.argmin(np.abs(S_frame_a - s["f0"])))])
        shot_clutter[si] = np.mean(np.array(clutter_acc)[idx], axis=0)
        shot_lum[si] = np.mean(np.array(lum_acc)[idx], axis=0)
        shot_rgb[si] = np.mean(np.array(rgb_acc)[idx], axis=0)
        tidx = np.where((text_frames >= s["f0"]) & (text_frames < s["f1"]))[0]
        if len(tidx):
            shot_text[si] = np.mean(np.array(text_acc)[tidx], axis=0)
        # summarize speaker presence
        faces_in = [S_faces[i] for i in idx]
        s["face_ratio"] = round(float(np.mean([1.0 if f else 0.0 for f in faces_in])), 3)
        s["person_cov"] = round(float(np.mean(person[idx] > 0.5)), 3)
        s["speaker"] = bool(s["face_ratio"] > 0.5 and s["person_cov"] > 0.03)
        # temporal motion of the camera (global) -> 'static' / 'moving'
        s["camera"] = "moving" if float(np.mean(motion[idx][:, person[idx].mean(axis=0) < 0.2])) > 0.18 else "static"
    # overlay text: present in >=60% of all text samples, or in the same place across >=2 shots
    if text_acc:
        ta = np.array(text_acc)
        freq = ta.mean(axis=0)
        multi = (shot_text > 0.25).sum(axis=0) >= max(2, int(math.ceil(0.5 * S))) if S >= 2 else np.zeros_like(freq, bool)
        overlay = np.clip(np.maximum(freq * 1.6, multi.astype(np.float32)), 0, 1)
        # subtitles change content but sit in a band: dilate horizontally within rows with frequent text
        overlay = cv2.dilate(overlay, np.ones((1, 3), np.uint8))
    else:
        overlay = np.zeros((gh, gw), np.float32)
    meta = {"analysis_size": [aw, ah], "sample_fps": sample_fps, "fps": fps, "frames": n_frames,
            "cut_frames": cuts, "cut_scores_p95": float(np.percentile(cut_scores, 95))}
    log(f"vision: {len(S_t)} samples, {len(shots)} shots, cuts at {[round(c / fps, 2) for c in cuts][:20]}")
    return VisionData(W=W, H=H, gw=gw, gh=gh, t=np.array(S_t, np.float32), sample_shot=sample_shot, person=person, head=head,
                      motion=motion, faces=S_faces, hands=S_hands, shots=shots, shot_clutter=shot_clutter, shot_lum=shot_lum,
                      shot_rgb=shot_rgb, shot_text=shot_text, overlay_text=overlay, meta=meta)


# ---------------------------------------------------------------------------------- summaries
def _box_from_mask(m: np.ndarray, thr: float = 0.5):
    ys, xs = np.where(m > thr)
    if len(xs) == 0:
        return None
    gh, gw = m.shape
    return [xs.min() / gw, ys.min() / gh, (xs.max() + 1) / gw, (ys.max() + 1) / gh]


def largest_free_rect(occ: np.ndarray) -> tuple[int, int, int, int, int]:
    """Maximal all-free rectangle (histogram method). occ: bool [gh,gw] True = occupied.
    Returns (area, x0, y0, x1, y1) in grid cells (exclusive end)."""
    gh, gw = occ.shape
    heights = np.zeros(gw, int)
    best = (0, 0, 0, 0, 0)
    for y in range(gh):
        heights = np.where(occ[y], 0, heights + 1)
        stack: list = []
        for x in range(gw + 1):
            h = heights[x] if x < gw else 0
            start = x
            while stack and stack[-1][1] >= h:
                sx, sh = stack.pop()
                area = sh * (x - sx)
                if area > best[0]:
                    best = (area, sx, y - sh + 1, x, y + 1)
                start = sx
            stack.append((start, h))
    return best


def shot_composition(vd: VisionData, si: int, safe: Optional[tuple] = None) -> dict:
    idx = np.where(vd.sample_shot == si)[0]
    if len(idx) == 0:
        return {}
    pmax = vd.person[idx].max(axis=0)
    hmax = vd.head[idx].max(axis=0)
    person_box = _box_from_mask(pmax, 0.5)
    faces = [f for i in idx for f in vd.faces[int(i)]]
    face_box = None
    if faces:
        fa = np.array(faces)
        face_box = [float(np.median(fa[:, 0])), float(np.median(fa[:, 1])), float(np.median(fa[:, 2])), float(np.median(fa[:, 3]))]
    hands = [h for i in idx for h in vd.hands[int(i)]]
    mot = vd.motion[idx].mean(axis=0)
    occ = (pmax > 0.35) | (vd.overlay_text > 0.3) | (vd.shot_text[si] > 0.4)
    if safe:
        x0, y0, x1, y1 = safe
        m = np.ones_like(occ)
        m[int(y0 * vd.gh):int(math.ceil(y1 * vd.gh)), int(x0 * vd.gw):int(math.ceil(x1 * vd.gw))] = False
        occ = occ | m
    free = []
    occ2 = occ.copy()
    for _ in range(3):
        area, fx0, fy0, fx1, fy1 = largest_free_rect(occ2)
        if area < 0.04 * vd.gw * vd.gh:
            break
        free.append([round(fx0 / vd.gw, 3), round(fy0 / vd.gh, 3), round(fx1 / vd.gw, 3), round(fy1 / vd.gh, 3)])
        occ2[fy0:fy1, fx0:fx1] = True
    cx = (person_box[0] + person_box[2]) / 2 if person_box else 0.5
    left_free = float((~occ[:, : vd.gw // 2]).mean())
    right_free = float((~occ[:, vd.gw // 2:]).mean())
    open_side = "right" if right_free > left_free + 0.05 else ("left" if left_free > right_free + 0.05 else "balanced")
    return {
        "person_box": [round(v, 3) for v in person_box] if person_box else None,
        "face_box": [round(v, 3) for v in face_box] if face_box else None,
        "speaker_x": round(cx, 3),
        "open_side": open_side,
        "free_regions": free,
        "hands_active": bool(len(hands) > 0.5 * len(idx)),
        "hand_zone": _box_from_mask(_hand_heat(hands, vd.gw, vd.gh), 0.2) if hands else None,
        "motion_mean": round(float(mot.mean()), 3),
        "clutter_mean": round(float(vd.shot_clutter[si].mean()), 3),
        "lum_mean": round(float(vd.shot_lum[si].mean()), 3),
        "has_text": bool((vd.shot_text[si] > 0.3).any()),
    }


def _hand_heat(hands: list, gw: int, gh: int) -> np.ndarray:
    m = np.zeros((gh, gw), np.float32)
    for x, y, v in hands:
        cx, cy = int(x * gw), int(y * gh)
        cv2.circle(m, (cx, cy), max(2, gw // 16), 1.0, -1)
    if len(hands):
        m = m / max(1.0, m.max())
    return m
