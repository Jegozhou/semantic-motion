"""QA demo support: build a short real clip, demo person masks, and Paint overlays.

This file is EVIDENCE TOOLING, not the product. The person masks it writes come
from Apple Vision via pyobjc, but they are produced here, not by
``src/semotion/perception.py``; every document says
``backend.name = "qa-demo-vision-helper"`` so nobody can mistake them for the
perception module's output. The overlay painter mimics the IR's declared box,
colour, font and window so ``repair.py`` can be exercised without running the
full renderer.

Commands
--------
python build_demo.py prep
python build_demo.py paint --ir artifacts/p2/qa/motion_ir_bad.json \
    --out artifacts/p2/qa/final_bad.mp4 --overlay-out artifacts/p2/qa/overlay_bad.mp4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import Quartz
import Vision
from Foundation import NSURL

ROOT = Path(__file__).resolve().parents[4]
QA = ROOT / "artifacts/p2/qa"
FIXTURE = ROOT / "tests/fixtures/clips/fx_pd_real_16x9_a/fx_pd_real_16x9_a.mp4"
FONT = ROOT / "assets/fonts/NotoSansSC-Regular.otf"
FFMPEG = "ffmpeg"
FPS = 30.0
DURATION_S = 4.0
W, H = 1280, 720
SAMPLE_EVERY = 6  # 5 Hz at 30 fps


# --------------------------------------------------------------------- encoding
def _run(argv: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(argv, check=True, capture_output=True, text=True, **kw)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cut_base(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run([FFMPEG, "-v", "error", "-y", "-ss", "0", "-t", f"{DURATION_S:g}",
          "-i", str(FIXTURE), "-c:v", "libx264", "-preset", "medium", "-crf", "18",
          "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(dest)])
    return dest


def _frames(path: Path) -> list[np.ndarray]:
    proc = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-f", "rawvideo",
                           "-pix_fmt", "rgb24", "pipe:1"],
                          check=True, capture_output=True)
    raw = proc.stdout
    fs = W * H * 3
    data = np.frombuffer(raw, dtype=np.uint8)
    return [data[i:i + fs].reshape(H, W, 3).copy()
            for i in range(0, len(data) - fs + 1, fs)]


def encode_frames(frames: list[np.ndarray], out: Path, *, audio_from: Path | None,
                  crf: str = "18") -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(int(FPS)), "-i", "pipe:0"]
    if audio_from is not None:
        argv += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a?"]
    argv += ["-c:v", "libx264", "-preset", "medium", "-crf", crf,
             "-pix_fmt", "yuv420p"]
    if audio_from is not None:
        argv += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    argv += [str(out)]
    buf = b"".join(f.tobytes() for f in frames)
    proc = subprocess.run(argv, input=buf, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-2000:])
    return out


# ---------------------------------------------------------------------- Vision
def _cgimage(path: Path):
    url = NSURL.fileURLWithPath_(str(path))
    src = Quartz.CGImageSourceCreateWithURL(url, None)
    return Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)


def person_mask(path: Path) -> np.ndarray:
    img = _cgimage(path)
    req = Vision.VNGeneratePersonSegmentationRequest.alloc().initWithCompletionHandler_(None)
    req.setOutputPixelFormat_(0x4C303038)  # kCVPixelFormatType_OneComponent8
    req.setQualityLevel_(1)  # accurate
    Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None) \
        .performRequests_error_([req], None)
    results = req.results()
    if not results:
        raise RuntimeError(f"no segmentation result for {path}")
    pb = results[0].pixelBuffer()
    Quartz.CVPixelBufferLockBaseAddress(pb, 0)
    try:
        pw = Quartz.CVPixelBufferGetWidth(pb)
        ph = Quartz.CVPixelBufferGetHeight(pb)
        bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
        buf = Quartz.CVPixelBufferGetBaseAddress(pb).as_buffer(bpr * ph)
        return np.frombuffer(buf, dtype=np.uint8).reshape(ph, bpr)[:, :pw].copy()
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pb, 0)


def face_box(path: Path) -> tuple[list[int], float] | None:
    img = _cgimage(path)
    req = Vision.VNDetectFaceRectanglesRequest.alloc().initWithCompletionHandler_(None)
    Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None) \
        .performRequests_error_([req], None)
    best = None
    for obs in req.results() or []:
        bb = obs.boundingBox()
        x = bb.origin.x * W
        w = bb.size.width * W
        y = (1.0 - bb.origin.y - bb.size.height) * H
        h = bb.size.height * H
        cand = ([int(round(x)), int(round(y)), int(round(w)), int(round(h))],
                float(obs.confidence()))
        if best is None or cand[1] > best[1]:
            best = cand
    return best


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask > 127)
    if xs.size == 0:
        return None
    mx = mask.shape[1] / float(W)
    my = mask.shape[0] / float(H)
    return [int(round(xs.min() / mx)), int(round(ys.min() / my)),
            int(round((xs.max() + 1 - xs.min()) / mx)),
            int(round((ys.max() + 1 - ys.min()) / my))]


def _union_box(boxes: list[list[int]]) -> list[int] | None:
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return [x0, y0, x1 - x0, y1 - y0]


# ------------------------------------------------------------------ quiet zones
def _edge_density(gray: np.ndarray, box: list[int]) -> float:
    x, y, w, h = box
    sub = gray[max(0, y):y + h, max(0, x):x + w].astype(np.float32)
    if sub.size == 0:
        return 1.0
    gx = np.abs(np.diff(sub, axis=1)).mean() if sub.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(sub, axis=0)).mean() if sub.shape[0] > 1 else 0.0
    return float(min(1.0, (gx + gy) / 255.0 * 2.0))


def _overlap_frac(box: list[int], mask: np.ndarray) -> float:
    x, y, w, h = box
    mx = mask.shape[1] / float(W)
    my = mask.shape[0] / float(H)
    x0, y0 = int(math.floor(x * mx)), int(math.floor(y * my))
    x1, y1 = int(math.ceil((x + w) * mx)), int(math.ceil((y + h) * my))
    sub = mask[y0:y1, x0:x1] > 127
    if sub.size == 0:
        return 0.0
    return float(sub.mean())


def quiet_zones(person_union: np.ndarray, base_frame: np.ndarray) -> list[dict]:
    gray = (0.2126 * base_frame[..., 0] + 0.7152 * base_frame[..., 1]
            + 0.0722 * base_frame[..., 2]).astype(np.float32)
    sizes = [(320, 300), (400, 340), (360, 340), (480, 300), (560, 360),
             (320, 200), (640, 360), (900, 300), (400, 220), (280, 180)]
    margin = 32
    found: list[dict] = []
    for bw, bh in sizes:
        if bw < 0.2 * W or bh < 0.1 * H:
            continue
        for y in range(margin, H - bh - margin + 1, 24):
            for x in range(margin, W - bw - margin + 1, 24):
                box = [x, y, bw, bh]
                po = _overlap_frac(box, person_union)
                if po > 0.0:
                    continue
                edge = _edge_density(gray, box)
                luma = float(np.abs(gray[y:y + bh, x:x + bw] - 128.0).mean() / 128.0)
                area_bonus = (bw * bh) / float(W * H)
                score = 0.65 * (1.0 - edge) + 0.25 * (1.0 - min(1.0, luma)) + 0.10 * area_bonus
                found.append({"bbox": box, "kind": "background",
                              "score": round(float(score), 4),
                              "person_overlap": 0.0, "text_overlap": 0.0,
                              "edge_density": round(edge, 4),
                              "contrast_luma": round(luma, 4)})
    # deterministic ranking, then drop heavy overlaps between selected zones.
    # Large zones are selected first (one per size class) so a repair can always
    # try its full box size before shrinking; the remaining slots take the best
    # scoring candidates.
    found.sort(key=lambda z: (-z["score"], z["bbox"][1], z["bbox"][0],
                              z["bbox"][2], z["bbox"][3]))
    by_size: dict[tuple[int, int], list[dict]] = {}
    for z in found:
        by_size.setdefault((z["bbox"][2], z["bbox"][3]), []).append(z)

    def _clash(z: dict, selected: list[dict]) -> bool:
        zx, zy, zw, zh = z["bbox"]
        for s in selected:
            sx, sy, sw, sh = s["bbox"]
            ix = max(0, min(zx + zw, sx + sw) - max(zx, sx))
            iy = max(0, min(zy + zh, sy + sh) - max(zy, sy))
            if ix * iy > 0.4 * min(zw * zh, sw * sh):
                return True
        return False

    selected: list[dict] = []
    for size in sorted(by_size, key=lambda s: (-s[0] * s[1], -s[1], -s[0])):
        for z in by_size[size]:
            if not _clash(z, selected):
                selected.append(z)
                break
        if len(selected) >= 12:
            break
    for z in found:
        if len(selected) >= 12:
            break
        if not _clash(z, selected):
            selected.append(z)
    return selected


# ------------------------------------------------------------------------ prep
def prep() -> dict:
    QA.mkdir(parents=True, exist_ok=True)
    base = cut_base(QA / "base_clip.mp4")
    frames = _frames(base)
    n_frames = len(frames)
    masks_dir = QA / "masks"
    if masks_dir.exists():
        shutil.rmtree(masks_dir)
    masks_dir.mkdir(parents=True)
    tmp = QA / "_tmp"
    tmp.mkdir(exist_ok=True)
    samples = list(range(0, n_frames, SAMPLE_EVERY))
    frames_rows = []
    person_union = None
    for n, frame_idx in enumerate(samples):
        img_path = tmp / f"{n:04d}.png"
        Image.fromarray(frames[frame_idx]).save(img_path)
        m = person_mask(img_path)
        small = np.array(Image.fromarray(m).resize((W // 4, H // 4), Image.BILINEAR))
        mask_rel = f"masks/{n:04d}.png"
        Image.fromarray(small).save(QA / mask_rel)
        person_union = small if person_union is None else np.maximum(person_union, small)
        face = face_box(img_path)
        frames_rows.append({
            "frame": int(frame_idx),
            "t_s": round(frame_idx / FPS, 6),
            "person_bbox": _mask_bbox(small),
            "mask": mask_rel,
            "mask_scale": 4.0,
            "mask_coverage": round(float((small > 127).mean()), 6),
            "pose": None,
            "hands": [],
            "face": ({"bbox": face[0], "conf": round(face[1], 4)} if face else None),
        })
        img_path.unlink(missing_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    assert person_union is not None
    zones = quiet_zones(person_union, frames[0])
    if len(zones) < 3:
        raise RuntimeError(f"only {len(zones)} quiet zones found; demo needs >= 3")
    faces = [r["face"]["bbox"] for r in frames_rows if r["face"]]
    face_union = _union_box(faces)
    person_union_box = _union_box([r["person_bbox"] for r in frames_rows
                                   if r["person_bbox"]])
    margin = 32
    face_safe = ([face_union[0] - margin, face_union[1] - margin,
                  face_union[2] + 2 * margin, face_union[3] + 2 * margin]
                 if face_union else None)
    person_safe = ([person_union_box[0] - margin, person_union_box[1] - margin,
                    person_union_box[2] + 2 * margin, person_union_box[3] + 2 * margin]
                   if person_union_box else None)
    doc = {
        "schema": "semotion.perception/1",
        "ir_compatible": "semotion.vision/1",
        "source": {"path": str(base), "sha256": _sha256(base), "width": W, "height": H,
                   "fps": FPS, "n_frames": n_frames,
                   "duration_s": round(n_frames / FPS, 6)},
        "backend": {"name": "qa-demo-vision-helper", "os": "macOS-arm64",
                    "requests": ["segmentation", "face"],
                    "note": ("QA demo masks produced by build_demo.py (Apple Vision via "
                             "pyobjc). NOT src/semotion/perception.py output; do not cite "
                             "as the perception module.")},
        "capabilities": {"person_mask": True, "body_pose": False, "hands": False,
                         "existing_text": False, "scene_cuts": False},
        "sample_hz": round(FPS / SAMPLE_EVERY, 6),
        "frames": frames_rows,
        "regions": {
            "frame_size": [W, H],
            "face_union": face_union,
            "person_union": person_union_box,
            "hands_union": None,
            "face_safe": face_safe,
            "person_safe": person_safe,
            "safe_margin_px": margin,
            "existing_text": [],
            "quiet_zones": zones,
            "person_coverage_curve": [[r["t_s"], r["mask_coverage"]] for r in frames_rows],
            "shots": [{"start_s": 0.0, "end_s": round(n_frames / FPS, 6)}],
        },
    }
    (QA / "perception_demo.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ir = build_ir(base, n_frames, doc)
    (QA / "motion_ir_bad.json").write_text(
        json.dumps(ir, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "base": str(base), "n_frames": n_frames, "samples": len(samples),
        "quiet_zones": len(zones), "face_union": face_union,
        "person_union": person_union_box,
        "base_sha256": _sha256(base),
        "first_zones": zones[:3],
    }


def build_ir(base: Path, n_frames: int, perception: dict) -> dict:
    from semotion import ir as ir_mod

    badge = "QA CHECK"
    ir = {
        "schema": "semotion.motion-ir/1",
        "ir_version": "1.0.0",
        "source": {"path": str(base.resolve()), "sha256": _sha256(base),
                   "width": W, "height": H, "fps": FPS,
                   "duration_s": round(n_frames / FPS, 6), "n_frames": n_frames,
                   "frame_source": "nb_frames",
                   "timebase": {"num": 1, "den": int(FPS)}},
        "canvas": {"width": W, "height": H, "fps": FPS},
        "style": "smart_balanced",
        "semantic": {
            "transcript_schema": "semotion.transcript/1",
            "transcript_text": badge,
            "anchors": [{
                "id": "a_demo", "kind": "keyword", "text": badge,
                "start_s": 1.0, "end_s": 2.5, "confidence": 1.0,
                "source": "QA_DEMO", "segment_id": "s0", "word_ids": [],
            }],
            "face_safe": ({"bbox_union": perception["regions"]["face_safe"],
                           "margin_px": 32}
                          if perception["regions"]["face_safe"] else None),
        },
        "tracks": [{"id": "t_content", "z": 20, "elements": ["el_demo"]}],
        "elements": [{
            "id": "el_demo", "type": "keyword",
            "anchor_ref": "a_demo", "anchor_refs": ["a_demo"],
            "window": {"start_s": 1.0, "end_s": 2.5, "start_frame": 30,
                       "end_frame": 75, "hold_s": 0.0,
                       "anchor_start_s": 1.0, "anchor_end_s": 2.5},
            "style": {"font_family": "NotoSansSC",
                      "font_file": "NotoSansSC-Regular.otf",
                      "font_path": "assets/fonts/NotoSansSC-Regular.otf",
                      "weight": "bold", "color": "#F5F7FA",
                      "bg": "rgba(10,13,22,0.82)", "font_size_px": 64,
                      "padding_px": 24, "radius_px": 20, "align": "center"},
            "motion": {"enter": {"type": "fade", "dur_s": 0.2, "ease": "out_cubic"},
                       "exit": {"type": "fade", "dur_s": 0.2, "ease": "in_quad"}},
            "content": {"text": badge},
            "layout": {"mode": "qa_demo", "zone": "face_center_bad",
                       "box": [500, 90, 340, 340], "avoid_face": True,
                       "allow_face_overlap": True,
                       "allow_face_reason": "QA demo: deliberately over the face",
                       "reason": "deliberately painted over the face",
                       "intrinsic": {"width": 340, "height": 340}},
        }],
        "collision_ok": [],
        "provenance": {
            "generator": "artifacts/p2/qa/scripts/build_demo.py (QA evidence only)",
            "policy": ("a deliberately bad overlay for the v2 QA demo; the IR is a real "
                       "semotion.motion-ir/1 document so repair.py can mutate it"),
            "host": "macOS-arm64",
        },
    }
    return ir_mod.seal(ir)


# ----------------------------------------------------------------------- paint
def _parse_rgba(value: str | None) -> tuple[tuple[int, int, int], int]:
    if not value:
        return (10, 13, 22), 210
    s = str(value).strip().lower()
    if s.startswith("rgba"):
        parts = s[s.find("(") + 1:s.rfind(")")].split(",")
        return ((int(float(parts[0])), int(float(parts[1])), int(float(parts[2]))),
                int(round(float(parts[3]) * 255)))
    if s.startswith("#") and len(s) == 7:
        return ((int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)), 255)
    return (10, 13, 22), 210


def _parse_rgb(value: str | None) -> tuple[int, int, int]:
    if not value:
        return (245, 247, 250)
    s = str(value).strip()
    if s.startswith("#") and len(s) == 7:
        return (int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16))
    return (245, 247, 250)


def paint_elements(frame: np.ndarray, elements: list[dict], frame_idx: int,
                   ink_only: bool = False) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGBA")
    for el in elements:
        w = el.get("window") or {}
        if not (int(w.get("start_frame", 0)) <= frame_idx < int(w.get("end_frame", 0))):
            continue
        box = (el.get("layout") or {}).get("box")
        if not (isinstance(box, (list, tuple)) and len(box) >= 4):
            continue
        x, y, bw, bh = (int(round(v)) for v in box[:4])
        style = el.get("style") or {}
        bg_rgb, bg_a = _parse_rgba(style.get("bg"))
        fg = _parse_rgb(style.get("color"))
        radius = int(style.get("radius_px") or 18)
        pad = int(style.get("padding_px") or 20)
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        if not ink_only:
            d.rounded_rectangle([x, y, x + bw, y + bh], radius=radius,
                                fill=(*bg_rgb, bg_a))
        text = str((el.get("content") or {}).get("text") or "")
        size = int(style.get("font_size_px") or 56)
        font = ImageFont.truetype(str(FONT), size)
        bbox = d.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        while (tw > bw - 2 * pad or th > bh - 2 * pad) and size > 12:
            size -= 2
            font = ImageFont.truetype(str(FONT), size)
            bbox = d.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text((x + (bw - tw) / 2 - bbox[0], y + (bh - th) / 2 - bbox[1]),
               text, font=font, fill=(*fg, 255))
        img = Image.alpha_composite(img, layer)
    return np.array(img.convert("RGB"))


def paint(ir_path: Path, out: Path, overlay_out: Path | None, *, base: Path,
          repeats: int = 1, ink_out: Path | None = None) -> dict:
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    elements = ir.get("elements") or []
    src_frames = _frames(base)
    finals = []
    for r in range(repeats):
        painted = [paint_elements(f, elements, i) for i, f in enumerate(src_frames)]
        dest = out if r == 0 else out.with_name(f"{out.stem}_run{r + 1}{out.suffix}")
        encode_frames(painted, dest, audio_from=base)
        finals.append(str(dest))
    if overlay_out is not None:
        black = np.zeros_like(src_frames[0])
        ovl = [paint_elements(black, elements, i) for i in range(len(src_frames))]
        encode_frames(ovl, overlay_out, audio_from=None)
    if ink_out is not None:
        black = np.zeros_like(src_frames[0])
        ink = [paint_elements(black, elements, i, ink_only=True)
               for i in range(len(src_frames))]
        encode_frames(ink, ink_out, audio_from=None)
    return {"outputs": finals, "overlay": str(overlay_out) if overlay_out else None,
            "ink": str(ink_out) if ink_out else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    p = sub.add_parser("paint")
    p.add_argument("--ir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--overlay-out")
    p.add_argument("--ink-out")
    p.add_argument("--base", default=str(QA / "base_clip.mp4"))
    p.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "prep":
        info = prep()
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0
    info = paint(Path(args.ir), Path(args.out),
                 Path(args.overlay_out) if args.overlay_out else None,
                 base=Path(args.base), repeats=args.repeats,
                 ink_out=Path(args.ink_out) if args.ink_out else None)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
