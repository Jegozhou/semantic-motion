"""Semantic Motion — vision layer (faces, shots, safe zones).

Contract: ``semotion.vision/1`` (docs/INTERFACES.md §3.2).

Design notes
------------
* **One decode for every signal.** The clip is decoded once at a bounded
  detection resolution (longest side ``det_max_side`` = 640 px) at
  ``fps_scan`` Hz; the same RGB frames feed face detection (scaled back to
  native pixels) and shot detection (downscaled grey thumbnails + HSV
  histograms). No second pass over the file.
* **Face detection** uses OpenCV YuNet (``face_detection_yunet_2023mar.onnx``),
  score threshold 0.6. Detections are grouped into tracks by greedy
  IoU / centre-distance association across sampled frames. Boxes are stored in
  **native pixels**; interpolation to any frame is nearest-neighbour (see
  :func:`box_at_frame`), which is what the contract asks for and what
  ``verify`` needs.
* **No fabrication.** A clip with no detectable face (e.g. designed artwork)
  yields ``faces == []`` / ``face_tracks == []`` and downstream code must fall
  back to a static safe layout. Nothing is guessed.
* **Shot detection (documented, measured).** For consecutive sampled frames we
  compute, on 64×36 thumbnails:
    ``rgb`` = mean |Δ| over R,G,B (0..1) — "how much of the frame changed";
    ``bh``  = Bhattacharyya distance of L1-normalised HSV histograms
              (H30/S10/V10) — "was the colour distribution replaced".
  A cut is declared when ``bh >= 0.55`` **or** ``rgb >= 0.12`` **or**
  (``rgb >= 0.035`` and ``bh >= 0.35``).

  Why these numbers (measured on the 4 fixtures, see
  ``provenance.shot_detection.transitions``):

  | set | rgb | bh |
  |---|---|---|
  | real footage, no cut (fx_pd a/b, 238 transitions) | p90 0.009, max 0.011 | p90 0.151, max 0.276 |
  | synthetic static / slow_pan / slow_zoom, no cut | p90 0.001 | p90 0.047, max 0.073 |
  | cuts (fx_c at 8.0 s / 16.0 s) | 0.055 / 0.679 | 1.0 / 1.0 |

  The two cut types need *different* signals: 16 s is a brightness cut
  (``rgb`` 0.68, caught by the single-signal rule), while 8 s switches between
  two palettes that render to almost the same luminance (``rgb`` 0.055,
  invisible to any grey-diff method, but ``bh`` = 1.0). So neither a
  "grey diff only" rule nor a "both signals" rule is sufficient; the
  single-signal hist rule plus a combined middle band covers both, and the
  non-cut maxima stay 2–5× below the thresholds. A hard cut between two
  *visually near-identical* shots is undetectable from pixels (and mostly
  invisible to a viewer); that is a documented limitation, not a silent one.
  Grey diff is still recorded, as a diagnostic only.
* **Determinism.** All outputs are a pure function of the decoded pixels and
  the parameters below; ``provenance.created_utc`` / ``command`` are the only
  run-varying fields.
* **Cache**: ``artifacts/cache/vision/<source_sha256>.json``; a hit is only
  accepted when the stored parameters (``scanned_fps``, detector model) match
  the request, so a different ``fps_scan`` can never return stale geometry.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from . import media

__all__ = [
    "SCHEMA",
    "DEFAULT_MODEL",
    "CUT",
    "analyze",
    "load",
    "validate",
    "face_tracks",
    "box_at_frame",
    "face_box_at",
    "face_union_box",
    "iou",
    "safe_area_boxes",
]

SCHEMA = "semotion.vision/1"
DEFAULT_MODEL = "assets/models/face_detection_yunet_2023mar.onnx"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Shot-detection thresholds — see the module docstring for the measured basis.
CUT = {
    "hist_alone": 0.55,        # colour distribution replaced outright
    "rgb_alone": 0.12,         # most of the frame's pixels replaced
    "rgb_with_hist": 0.035,    # combined band: area change ...
    "hist_with_rgb": 0.35,     # ... together with a colour-distribution change
}

THUMB_W, THUMB_H = 64, 36
SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 5000
DET_MAX_SIDE = 640

# Fixed safe zones per canvas aspect: (title, caption, card candidates).
# Coordinates are normalised [x, y, w, h]; the caption band keeps the bottom
# of the frame clear and the card never overlaps title/caption.
_ZONES: dict[str, dict[str, Any]] = {
    "9:16": {
        "title": [0.08, 0.06, 0.84, 0.12],
        "caption": [0.08, 0.80, 0.84, 0.11],
        "cards": [[0.06, 0.30, 0.88, 0.42], [0.06, 0.22, 0.88, 0.30],
                  [0.06, 0.44, 0.88, 0.30]],
    },
    "16:9": {
        "title": [0.06, 0.06, 0.88, 0.14],
        "caption": [0.10, 0.80, 0.80, 0.12],
        "cards": [[0.20, 0.26, 0.60, 0.44], [0.05, 0.26, 0.40, 0.46],
                  [0.55, 0.26, 0.40, 0.46]],
    },
    "1:1": {
        "title": [0.08, 0.06, 0.84, 0.13],
        "caption": [0.08, 0.79, 0.84, 0.12],
        "cards": [[0.10, 0.25, 0.80, 0.44], [0.10, 0.22, 0.80, 0.28]],
    },
    "4:5": {
        "title": [0.08, 0.06, 0.84, 0.12],
        "caption": [0.08, 0.80, 0.84, 0.11],
        "cards": [[0.08, 0.28, 0.84, 0.42], [0.08, 0.22, 0.84, 0.30]],
    },
}


# --------------------------------------------------------------------------- paths

def cache_root() -> Path:
    return Path(os.environ.get("SEMOTION_CACHE_ROOT") or (PROJECT_ROOT / "artifacts" / "cache"))


def cache_path(source_sha256: str) -> Path:
    return cache_root() / "vision" / f"{source_sha256}.json"


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# -------------------------------------------------------------------------- decode

def _det_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    scale = min(1.0, float(max_side) / float(max(width, height)))
    dw = max(2, int(round(width * scale / 2.0)) * 2)
    dh = max(2, int(round(height * scale / 2.0)) * 2)
    return dw, dh


def _sample_step(fps: float, fps_scan: float) -> int:
    """Frames between detections: ``ceil(fps / fps_scan)`` (>= 1)."""
    if fps_scan <= 0:
        raise media.MediaError("E_FFMPEG", f"fps_scan must be positive, got {fps_scan}")
    return max(1, int(math.ceil(float(fps) / float(fps_scan) - 1e-9)))


def _decode_sampled(video: Path, *, step: int, det_w: int, det_h: int) -> np.ndarray:
    """Decode every ``step``-th frame at detection size as uint8 RGB frames.

    Returns an ``(n, det_h, det_w, 3)`` array. Uses ``media.run`` only — the
    package never shells out to ffmpeg directly.
    """
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tf:
        raw_path = Path(tf.name)
    try:
        vf = (f"select='not(mod(n\\,{step}))',"
              f"scale={det_w}:{det_h}:flags=bilinear")
        media.run([media.FFMPEG, "-v", "error", "-y", "-i", str(video),
                   "-vf", vf, "-fps_mode", "passthrough",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", str(raw_path)])
        data = np.fromfile(str(raw_path), dtype=np.uint8)
    finally:
        raw_path.unlink(missing_ok=True)
    frame_bytes = det_w * det_h * 3
    if frame_bytes <= 0 or data.size == 0:
        raise media.MediaError("E_FFMPEG", f"no frames decoded from {video}")
    usable = (data.size // frame_bytes) * frame_bytes
    return data[:usable].reshape(-1, det_h, det_w, 3)


# --------------------------------------------------------------------------- faces

def _create_detector(model_path: Path, det_w: int, det_h: int):
    import cv2

    if not model_path.exists():
        raise media.MediaError("E_FFMPEG", f"face detector model missing: {model_path}")
    det = cv2.FaceDetectorYN.create(str(model_path), "", (det_w, det_h),
                                    SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K)
    # OpenCV 5.x needs an explicit input size before the first detect().
    det.setInputSize((det_w, det_h))
    return det


def _detect_faces(detector, rgb: np.ndarray) -> list[dict[str, Any]]:
    import cv2

    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    _retval, faces = detector.detect(bgr)
    out: list[dict[str, Any]] = []
    if faces is None:
        return out
    for row in faces:
        x, y, bw, bh = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
        score = float(row[-1])
        if bw <= 1.0 or bh <= 1.0 or score < SCORE_THRESHOLD:
            continue
        out.append({"box_det": [x, y, bw, bh], "score": score})
    return out


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two ``[x, y, w, h]`` boxes."""
    ax0, ay0, aw, ah = (float(v) for v in a[:4])
    bx0, by0, bw, bh = (float(v) for v in b[:4])
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _center_norm_dist(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay = float(a[0]) + float(a[2]) / 2.0, float(a[1]) + float(a[3]) / 2.0
    bx, by = float(b[0]) + float(b[2]) / 2.0, float(b[1]) + float(b[3]) / 2.0
    norm = max(float(a[2]), float(a[3]), float(b[2]), float(b[3]), 1.0)
    return math.hypot(ax - bx, ay - by) / norm


def _associate_tracks(per_frame: dict[int, list[dict[str, Any]]], *,
                      iou_thr: float = 0.25, center_thr: float = 0.75) -> list[dict[str, Any]]:
    """Greedy nearest association of per-frame detections into tracks.

    A detection joins a track when IoU with the track's last box exceeds
    ``iou_thr`` *or* the normalised centre distance is below ``center_thr``
    (handles a fast move between two scanned frames). Best match wins; ties are
    impossible because the score is continuous and frames are processed in
    order, which keeps ids stable for identical input.
    """
    tracks: list[dict[str, Any]] = []
    for frame in sorted(per_frame):
        dets = sorted(per_frame[frame], key=lambda d: (-d["score"], d["box"][0], d["box"][1]))
        for det in dets:
            best_track = None
            best_score = -1.0
            for track in tracks:
                if track["last_frame"] >= frame:
                    continue
                ov = iou(track["last_box"], det["box"])
                dist = _center_norm_dist(track["last_box"], det["box"])
                if ov < iou_thr and dist > center_thr:
                    continue
                score = ov - 0.5 * dist
                if score > best_score:
                    best_track, best_score = track, score
            if best_track is None:
                track = {
                    "id": f"f{len(tracks)}",
                    "frames": [],
                    "last_frame": frame,
                    "last_box": list(det["box"]),
                }
                tracks.append(track)
                best_track = track
            best_track["frames"].append({
                "frame": int(frame),
                "t_s": det["t_s"],
                "box": [int(v) for v in det["box"]],
                "score": round(float(det["score"]), 6),
            })
            best_track["last_frame"] = frame
            best_track["last_box"] = list(det["box"])
    return tracks


def _finish_tracks(raw_tracks: list[dict[str, Any]], *, width: int, height: int,
                   n_scan: int) -> list[dict[str, Any]]:
    finished: list[dict[str, Any]] = []
    for track in raw_tracks:
        frames = track["frames"]
        if not frames:
            continue
        xs = [f["box"][0] for f in frames]
        ys = [f["box"][1] for f in frames]
        x1 = [f["box"][0] + f["box"][2] for f in frames]
        y1 = [f["box"][1] + f["box"][3] for f in frames]
        ux, uy = max(0, min(xs)), max(0, min(ys))
        ux1, uy1 = min(width, max(x1)), min(height, max(y1))
        finished.append({
            "id": track["id"],
            "frames": frames,
            "bbox_union": [int(ux), int(uy), int(max(0, ux1 - ux)), int(max(0, uy1 - uy))],
            "coverage": round(len(frames) / float(max(1, n_scan)), 6),
            "n_frames": len(frames),
            "first_frame": int(frames[0]["frame"]),
            "last_observed_frame": int(frames[-1]["frame"]),
        })
    return finished


# --------------------------------------------------------------------------- shots

def _thumb_metrics(prev: np.ndarray, cur: np.ndarray) -> dict[str, float]:
    """Area-change and colour-distribution-change between two sampled frames.

    ``rgb_diff`` is the mean absolute per-channel difference (0..1): it answers
    "how much of the picture changed" and is insensitive to what the new
    colours are. ``hist_distance`` is the Bhattacharyya distance between
    L1-normalised 3-D HSV histograms: it answers "was the colour distribution
    replaced" and catches palette cuts whose luminance is unchanged.
    ``grey_diff`` is kept purely as a diagnostic number.
    """
    import cv2

    prev_thumb = cv2.resize(prev, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
    cur_thumb = cv2.resize(cur, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
    prev_i = prev_thumb.astype(np.int16)
    cur_i = cur_thumb.astype(np.int16)
    rgb_diff = float(np.abs(cur_i - prev_i).mean() / 255.0)
    prev_gray = cv2.cvtColor(prev_thumb, cv2.COLOR_RGB2GRAY).astype(np.int16)
    cur_gray = cv2.cvtColor(cur_thumb, cv2.COLOR_RGB2GRAY).astype(np.int16)
    grey_diff = float(np.abs(cur_gray - prev_gray).mean() / 255.0)

    prev_hsv = cv2.cvtColor(prev_thumb, cv2.COLOR_RGB2HSV)
    cur_hsv = cv2.cvtColor(cur_thumb, cv2.COLOR_RGB2HSV)
    bins = [30, 10, 10]
    ranges = [0, 180, 0, 256, 0, 256]
    h1 = cv2.calcHist([prev_hsv], [0, 1, 2], None, bins, ranges)   # CV_32F
    h2 = cv2.calcHist([cur_hsv], [0, 1, 2], None, bins, ranges)
    # L1 normalisation: compare distributions, not histogram shapes. Kept float32
    # because cv2.compareHist only accepts CV_32F.
    h1 = np.ascontiguousarray(h1 / np.float32(max(float(h1.sum()), 1.0)), dtype=np.float32)
    h2 = np.ascontiguousarray(h2 / np.float32(max(float(h2.sum()), 1.0)), dtype=np.float32)
    hist_dist = float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))
    return {
        "mean_diff": round(rgb_diff, 6),          # contract-facing "area change"
        "rgb_diff": round(rgb_diff, 6),
        "grey_diff": round(grey_diff, 6),
        "hist_distance": round(hist_dist, 6),
    }


def _is_cut(metrics: dict[str, float]) -> bool:
    rgb = metrics["rgb_diff"]
    bh = metrics["hist_distance"]
    if bh >= CUT["hist_alone"] or rgb >= CUT["rgb_alone"]:
        return True
    return rgb >= CUT["rgb_with_hist"] and bh >= CUT["hist_with_rgb"]


def _build_shots(cut_frames: list[int], *, fps: float, n_frames: int,
                 duration: float) -> list[dict[str, Any]]:
    bounds = [0] + sorted(set(cut_frames)) + [n_frames]
    shots: list[dict[str, Any]] = []
    for i in range(len(bounds) - 1):
        f0, f1 = int(bounds[i]), int(bounds[i + 1])
        if f1 <= f0:
            continue
        shots.append({
            "index": len(shots),
            "start_s": round(f0 / fps, 6),
            "end_s": round(min(f1 / fps, duration), 6),
            "start_frame": f0,
            "end_frame": min(f1, n_frames),
        })
    if shots:
        shots[-1]["end_frame"] = n_frames
        shots[-1]["end_s"] = round(duration, 6)
    return shots


# ---------------------------------------------------------------------- safe zones

def _aspect_key(width: int, height: int) -> str:
    ratio = width / float(height) if height else 1.0
    for key, target in (("9:16", 9 / 16), ("16:9", 16 / 9), ("4:5", 4 / 5), ("1:1", 1.0)):
        if abs(ratio - target) <= 0.06:
            return key
    return "9:16" if ratio < 1 else "16:9"


def _rect_overlap(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    return ix * iy


def safe_area_boxes(canvas: dict[str, Any], vision: dict[str, Any] | None = None) -> dict[str, list[float]]:
    """Normalised safe zones for a canvas, pushing the card away from faces.

    The card zone is chosen from a per-aspect candidate list: the candidate
    with the smallest face overlap (weighted by how continuously the face is
    visible) wins. With no faces the first candidate is used, so the result is
    deterministic and stable. Never invents a face.
    """
    width = int((canvas or {}).get("width") or 1080)
    height = int((canvas or {}).get("height") or 1920)
    zones = _ZONES.get(_aspect_key(width, height), _ZONES["9:16"])
    out = {
        "title": list(zones["title"]),
        "caption": list(zones["caption"]),
        "card": list(zones["cards"][0]),
    }
    faces = [t for t in ((vision or {}).get("face_tracks") or []) if t.get("bbox_union")]
    if not faces:
        return out
    best = None
    for candidate in zones["cards"]:
        cost = 0.0
        for track in faces:
            bx, by, bw, bh = (float(v) for v in track["bbox_union"])  # native px
            norm = [bx / width, by / height, bw / width, bh / height]
            cost += _rect_overlap(candidate, norm) * float(track.get("coverage") or 1.0)
        # slight preference for the default card on ties (earlier index wins)
        if best is None or cost < best[0] - 1e-12:
            best = (cost, candidate)
    out["card"] = list(best[1])
    return out


# ---------------------------------------------------------------------- public API

def analyze(video: Path, out_json: Path, *, fps_scan: float = 5.0,
            model: str | Path | None = None, force: bool = False,
            det_max_side: int = DET_MAX_SIDE) -> dict[str, Any]:
    """Analyze faces / shots / safe zones of ``video``.

    Parameters
    ----------
    video : Path
        Input video (must have a video stream).
    out_json : Path
        Where the ``semotion.vision/1`` document is written.
    fps_scan : float
        Face-detection sampling rate in Hz. The actual rate is
        ``fps / ceil(fps / fps_scan)`` so it is always frame-aligned and
        never slower than ~5 samples/second for ``fps_scan=5``.
    model : str | Path | None
        YuNet ONNX path; defaults to ``assets/models/face_detection_yunet_2023mar.onnx``.
    force : bool
        Ignore a cache hit with matching parameters.
    """
    video = Path(video)
    out_json = Path(out_json)
    mi = media.info(video)
    sha = media.sha256_file(video)
    model_path = Path(model) if model is not None else (PROJECT_ROOT / DEFAULT_MODEL)
    if not model_path.is_absolute() and not model_path.exists():
        model_path = PROJECT_ROOT / model_path

    cached = cache_path(sha)
    if not force and cached.exists():
        try:
            doc = json.loads(cached.read_text(encoding="utf-8"))
        except Exception:
            doc = None
        if (isinstance(doc, dict) and doc.get("schema") == SCHEMA
                and doc.get("source", {}).get("sha256") == sha
                and abs(float(doc.get("provenance", {}).get("scanned_fps") or 0.0)
                        - float(mi.fps) / _sample_step(mi.fps, fps_scan)) < 1e-6
                and Path(str(doc.get("detector", {}).get("model", ""))).name == model_path.name):
            _write_json(out_json, doc)
            return doc

    n_frames = int(round(mi.duration * mi.fps)) or mi.frame_count
    step = _sample_step(mi.fps, fps_scan)
    det_w, det_h = _det_size(mi.width, mi.height, det_max_side)
    frames = _decode_sampled(video, step=step, det_w=det_w, det_h=det_h)
    n_scan = int(frames.shape[0])
    if n_scan == 0:
        raise media.MediaError("E_FFMPEG", f"no frames decoded from {video}")

    sx, sy = mi.width / float(det_w), mi.height / float(det_h)
    detector = _create_detector(model_path, det_w, det_h)

    faces: list[dict[str, Any]] = []
    per_frame: dict[int, list[dict[str, Any]]] = {}
    transitions: list[dict[str, Any]] = []
    prev_rgb = None
    for i in range(n_scan):
        rgb = frames[i]
        frame_idx = min(n_frames - 1, i * step)
        t_s = round(frame_idx / mi.fps, 6)
        dets = _detect_faces(detector, rgb)
        for det in dets:
            x, y, bw, bh = det["box_det"]
            box = [
                int(max(0, min(mi.width - 2, round(x * sx)))),
                int(max(0, min(mi.height - 2, round(y * sy)))),
                int(max(2, min(mi.width, round(bw * sx)))),
                int(max(2, min(mi.height, round(bh * sy)))),
            ]
            box[2] = min(box[2], mi.width - box[0])
            box[3] = min(box[3], mi.height - box[1])
            entry = {"frame": frame_idx, "t_s": t_s, "box": box,
                     "score": round(det["score"], 6)}
            faces.append(entry)
            per_frame.setdefault(frame_idx, []).append(
                {"box": box, "score": det["score"], "t_s": t_s})
        if prev_rgb is not None:
            metrics = _thumb_metrics(prev_rgb, rgb)
            metrics["prev_frame"] = min(n_frames - 1, (i - 1) * step)
            metrics["frame"] = frame_idx
            metrics["cut"] = bool(_is_cut(metrics))
            transitions.append(metrics)
        prev_rgb = rgb

    tracks = _finish_tracks(_associate_tracks(per_frame),
                            width=mi.width, height=mi.height, n_scan=n_scan)
    cut_frames = [t["frame"] for t in transitions if t["cut"]]
    shots = _build_shots(cut_frames, fps=mi.fps, n_frames=n_frames, duration=mi.duration)

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "source": {
            "path": str(video.resolve()),
            "sha256": sha,
            "width": int(mi.width),
            "height": int(mi.height),
            "fps": round(float(mi.fps), 6),
            "n_frames": n_frames,
            "duration_s": round(float(mi.duration), 6),
        },
        "detector": {
            "name": "yunet",
            "model": model_path.name,
            "score_threshold": SCORE_THRESHOLD,
            "nms_threshold": NMS_THRESHOLD,
            "input_size": [det_w, det_h],
            "detect_scale": [round(sx, 6), round(sy, 6)],
        },
        "faces": faces,
        "face_tracks": tracks,
        "shots": shots,
        "safe_zones": {},  # filled below (needs tracks)
        "provenance": {
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "command": (f"vision.analyze fps_scan={fps_scan} step={step} "
                        f"det={det_w}x{det_h} model={model_path} :: {video}"),
            "scanned_fps": round(float(mi.fps) / step, 6),
            "requested_fps_scan": float(fps_scan),
            "sample_step_frames": step,
            "n_scanned_frames": n_scan,
            "n_frames": n_frames,
            "ndarray_dtype": "uint8 rgb24",
            "shot_detection": {
                "thresholds": dict(CUT),
                "metric": ("rgb_mean_abs_diff (area change) + "
                           "hsv_bhattacharyya on L1-normalised histogram (colour change)"),
                "thumb": [THUMB_W, THUMB_H],
                "n_transitions": len(transitions),
                "max_non_cut_rgb_diff": max(
                    [t["rgb_diff"] for t in transitions if not t["cut"]] or [0.0]),
                "max_non_cut_hist_distance": max(
                    [t["hist_distance"] for t in transitions if not t["cut"]] or [0.0]),
                "min_cut_rgb_diff": min(
                    [t["rgb_diff"] for t in transitions if t["cut"]] or [0.0]) or None,
                "min_cut_hist_distance": min(
                    [t["hist_distance"] for t in transitions if t["cut"]] or [0.0]) or None,
                "cuts": [{"frame": c["frame"], "t_s": round(c["frame"] / mi.fps, 6),
                          "mean_diff": c["mean_diff"], "hist_distance": c["hist_distance"]}
                         for c in transitions if c["cut"]],
                "transitions": transitions,
            },
            "face_detection": {
                "n_detections": len(faces),
                "n_tracks": len(tracks),
                "note": ("no face detected — downstream must use a static safe layout"
                         if not faces else "faces detected from decoded frames"),
            },
        },
    }
    doc["safe_zones"] = safe_area_boxes({"width": mi.width, "height": mi.height}, doc)

    _write_json(cached, doc)
    _write_json(out_json, doc)
    return doc


# ------------------------------------------------------------------------- helpers

def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(vision: dict[str, Any]) -> list[str]:
    """Return schema problems (empty list == valid). Mirrors INTERFACES §3.2."""
    problems: list[str] = []
    if vision.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}")
    src = vision.get("source") or {}
    for key in ("path", "sha256", "width", "height", "fps", "n_frames"):
        if key not in src:
            problems.append(f"source.{key} missing")
    width, height = int(src.get("width") or 0), int(src.get("height") or 0)
    n_frames = int(src.get("n_frames") or 0)
    duration = float(src.get("duration_s") or 0.0)
    fps = float(src.get("fps") or 25.0)

    for i, face in enumerate(vision.get("faces") or []):
        box = face.get("box") or []
        if len(box) != 4 or any(not isinstance(v, int) for v in box):
            problems.append(f"faces[{i}].box must be 4 ints (pixels)")
            continue
        x, y, w, h = box
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
            problems.append(f"faces[{i}].box outside frame: {box}")
        if not (0 <= int(face.get("frame", -1)) <= n_frames):
            problems.append(f"faces[{i}].frame out of range")

    for i, track in enumerate(vision.get("face_tracks") or []):
        box = track.get("bbox_union") or []
        if len(box) != 4:
            problems.append(f"face_tracks[{i}].bbox_union must be [x,y,w,h]")
        if not (0.0 <= float(track.get("coverage", -1)) <= 1.0):
            problems.append(f"face_tracks[{i}].coverage must be within [0,1]")
        if not track.get("frames"):
            problems.append(f"face_tracks[{i}].frames empty")

    shots = vision.get("shots") or []
    if not shots:
        problems.append("shots must contain at least one shot")
    for i, shot in enumerate(shots):
        s, e = float(shot.get("start_s", -1)), float(shot.get("end_s", -1))
        if not (0.0 <= s < e):
            problems.append(f"shots[{i}] bad interval {s}-{e}")
        if i and float(shots[i - 1]["end_s"]) > s + 1e-6:
            problems.append(f"shots[{i}] starts before previous shot ends")
        if not (0 <= int(shot.get("start_frame", -1)) <= int(shot.get("end_frame", -1)) <= n_frames):
            problems.append(f"shots[{i}] bad frame range")
    if shots:
        if float(shots[0]["start_s"]) > 1e-6:
            problems.append("first shot must start at 0.0")
        if abs(float(shots[-1]["end_s"]) - duration) > 1e-3:
            problems.append(f"last shot must end at duration ({duration}), got {shots[-1]['end_s']}")
        if int(shots[-1]["end_frame"]) != n_frames:
            problems.append("last shot must end at n_frames")
        if fps <= 0:
            problems.append("fps must be positive")

    zones = vision.get("safe_zones") or {}
    for key in ("title", "caption", "card"):
        rect = zones.get(key)
        if not rect or len(rect) != 4:
            problems.append(f"safe_zones.{key} must be [x,y,w,h]")
            continue
        x, y, w, h = (float(v) for v in rect)
        if not (0.0 <= x and 0.0 <= y and w > 0 and h > 0
                and x + w <= 1.0 + 1e-9 and y + h <= 1.0 + 1e-9):
            problems.append(f"safe_zones.{key} outside [0,1]: {rect}")

    prov = vision.get("provenance") or {}
    if float(prov.get("scanned_fps") or 0) <= 0:
        problems.append("provenance.scanned_fps missing")
    return problems


def face_tracks(vision: dict[str, Any]) -> list[dict[str, Any]]:
    """Tracks usable by layout/metrics: union box in native px + coverage."""
    return [t for t in (vision.get("face_tracks") or []) if t.get("frames")]


def box_at_frame(track: dict[str, Any], frame: int) -> list[int] | None:
    """Nearest-neighbour face box (native px) for ``frame`` in one track.

    Contract §3.2 asks for nearest-neighbour interpolation to every frame; the
    sampled observations are stored, so no synthetic values are introduced.
    """
    frames = track.get("frames") or []
    if not frames:
        return None
    if frame <= frames[0]["frame"]:
        return list(frames[0]["box"])
    if frame >= frames[-1]["frame"]:
        return list(frames[-1]["box"])
    best, best_d = None, None
    for obs in frames:
        d = abs(int(obs["frame"]) - int(frame))
        if best_d is None or d < best_d:
            best, best_d = obs, d
    return list(best["box"]) if best else None


def face_box_at(vision: dict[str, Any], t_s: float, *, min_coverage: float = 0.0) -> list[int] | None:
    """Nearest face box (union of the dominant track) at ``t_s``."""
    tracks = [t for t in face_tracks(vision) if float(t.get("coverage") or 0.0) >= min_coverage]
    if not tracks:
        return None
    track = max(tracks, key=lambda t: (t.get("n_frames", 0), t.get("coverage", 0.0)))
    frames = track.get("frames") or []
    if not frames:
        return None
    best = min(frames, key=lambda f: abs(float(f["t_s"]) - float(t_s)))
    return list(best["box"])


def face_union_box(vision: dict[str, Any], *, track_ids: Iterable[str] | None = None) -> list[int] | None:
    """Union of the requested tracks' ``bbox_union`` (native px)."""
    wanted = set(track_ids) if track_ids is not None else None
    tracks = [t for t in face_tracks(vision)
              if wanted is None or t.get("id") in wanted]
    if not tracks:
        return None
    xs = [t["bbox_union"][0] for t in tracks]
    ys = [t["bbox_union"][1] for t in tracks]
    x1 = [t["bbox_union"][0] + t["bbox_union"][2] for t in tracks]
    y1 = [t["bbox_union"][1] + t["bbox_union"][3] for t in tracks]
    return [int(min(xs)), int(min(ys)), int(max(x1) - min(xs)), int(max(y1) - min(ys))]
