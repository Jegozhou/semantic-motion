"""Semantic Motion — v2 QA: measure the rendered MP4 against real pixels and a
real person mask, then report PASS/FAIL with numbers.

Why v1 was not enough
--------------------
``verify.py`` compares element boxes with *face boxes* and (when it can) a
changed-pixel bounding box. A box can sit fully beside the face and still cover
hair, shoulders or a raised hand, and a changed-pixel bounding box says nothing
about how many overlay pixels land on the person. There is also no contracted
person mask and no measured glyph contrast.

What v2 measures (all numbers come from decoded frames of the delivered files)
-----------------------------------------------------------------------------
footprint
    Per-pixel difference between the final MP4 and the source MP4 at identical
    geometry (or an overlay-only pass, when supplied), thresholded with a noise
    floor measured on frames that have no element. Per-element footprint is the
    part of that mask inside the IR ``layout.box``.
person occlusion
    footprint ∩ person mask (``docs/PERCEPTION-CONTRACT.md``). Face overlap must
    be exactly 0. Person overlap must stay <= 0.2 % of the frame area **and**
    <= 0.5 % of the person's own pixels on every frame. Worst frame + a crop are
    written for human inspection.
existing text
    footprint ∩ burned-in text boxes must be 0 for every frame of the window.
legibility
    WCAG 2.x contrast between the rendered glyph pixels of an element and the
    rendered local background underneath them (scrim included), measured in
    blocks; minimum block contrast must be >= 4.5. The contrast against the raw
    source background (no scrim) is reported too, as an honesty check.
layout
    on-canvas, safe margins, concurrent-element collision, font-size floors.
timing
    appearance / disappearance frames measured from the footprint, compared with
    the IR window and the anchor onset; tolerance 1 output frame; signed errors.
determinism
    two decodes of the same IR (or a second render) compared with an explicit
    mean/max per-pixel tolerance.
render integrity
    geometry, frame count, duration and audio statistics of the delivered file.

Honest scope: the person mask is only as good as its producer. This module
labels the mask source it consumed and reports ``NOT RUN`` (never PASS) when the
mask document has no person capability. Face "mask" is the face bounding box,
which is conservative (it can include background next to the face).

CLI
---
``PYTHONPATH=src .venv/bin/python -m semotion.verify2 \\
    --ir artifacts/run/motion_ir.json --source artifacts/run/source.mp4 \\
    --rendered artifacts/run/final.mp4 --masks artifacts/perc/perception.json \\
    --out-dir artifacts/run/qa``

Writes ``qa_metrics.json`` and an annotated ``contact_sheet.png``; exits 0 when
the overall verdict is PASS, 1 otherwise. No network, no model calls.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import media

SCHEMA = "semotion.verify2/1"
PASS, FAIL, NOT_RUN = "PASS", "FAIL", "NOT RUN"

# ---- thresholds (docs/QA.md §Thresholds explains each one) ------------------
PERSON_FRAME_MAX = 0.002          # 0.2 % of the frame area, every frame
PERSON_PERSON_MAX = 0.005         # 0.5 % of the person's own pixels, every frame
FACE_MAX_PX = 0                 # exactly zero face pixels
TEXT_MAX_PX = 0                   # exactly zero burned-in-text pixels
MIN_CONTRAST = 4.5                # WCAG AA normal text
TIMING_TOLERANCE_FRAMES = 1
SAFE_MARGIN_PX = 32               # docs/PERCEPTION-CONTRACT.md safe_margin_px
COLLISION_AREA_RATIO = 0.01       # >1 % of the smaller box = collision
FOOTPRINT_MARGIN_FRAC = 0.025    # lead's rule: rendered footprint >= 2.5 % canvas margin
FOOTPRINT_COLLISION_RATIO = 0.005  # >0.5 % of the smaller box = footprint collision
FONT_FLOOR_FRAC = 0.045          # legacy-v2 warning floor: 4.5 % of min(canvas)
FONT_FLOOR_PX = {"keyword": 72, "number": 60, "list": 56, "card": 64, "caption": 48}
REF_W, REF_H = 1080, 1920
OVERLAY_FOOTPRINT_THRESHOLD = 4   # overlay pass is on black; dark scrims matter
FOOTPRINT_MIN_THRESHOLD = 12      # codec noise below this is never an overlay
FOOTPRINT_MAX_THRESHOLD = 80      # above this the diff is too noisy to trust
DET_MEAN_TOL = 0.05               # decoded-frame mean |Δ| (prior x264 ~3.5e-05)
DET_MAX_TOL = 32                  # isolated codec ringing pixels (0-255 scale)
GLYPH_COLOR_DISTANCE = 50         # max channel distance from the declared ink colour
GLYPH_BLOCK = 24                  # px block for blockwise WCAG contrast
GLYPH_MIN_BLOCK_PIXELS = 12


class Verify2Error(RuntimeError):
    """Bad input document (not a metric failure)."""


# --------------------------------------------------------------------- helpers
def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def _worst(values: Iterable[str]) -> str:
    vals = list(values)
    for level in (FAIL, NOT_RUN, PASS):
        if level in vals:
            return level
    return PASS


def parse_color(value: str | None, default: tuple[int, int, int] = (245, 247, 250)
                ) -> tuple[int, int, int]:
    """Parse ``#RRGGBB`` / ``#RGB`` / ``rgb(...)`` into 0-255 RGB."""
    if not value:
        return default
    s = str(value).strip()
    try:
        if s.startswith("#"):
            h = s[1:]
            if len(h) == 3:
                h = "".join(c * 2 for c in h)
            if len(h) >= 6:
                return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore
        if s.lower().startswith("rgb"):
            nums = s[s.find("(") + 1:s.rfind(")")].split(",")
            if len(nums) >= 3:
                return tuple(max(0, min(255, int(round(float(n.strip()))))) for n in nums[:3])  # type: ignore
    except (ValueError, TypeError):
        pass
    return default


def rgba_alpha(value: Any) -> float:
    """Alpha of ``rgba(r,g,b,a)`` / ``#RRGGBBAA`` / a float; 0 when absent."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    try:
        if s.startswith("rgba"):
            parts = s[s.find("(") + 1:s.rfind(")")].split(",")
            if len(parts) == 4:
                return float(parts[3].strip())
        if s.startswith("#") and len(s) == 9:
            return int(s[7:9], 16) / 255.0
    except (ValueError, TypeError):
        pass
    return 0.0


def relative_luminance(rgb: np.ndarray | Sequence[float]) -> np.ndarray | float:
    """WCAG relative luminance. ``rgb`` is 0-255, shape (..., 3)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    lum = 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]
    if isinstance(lum, np.ndarray) and lum.ndim == 0:
        return float(lum)
    return lum


def wcag_contrast(l1: float, l2: float) -> float:
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def inter_area(a: Sequence[float], b: Sequence[float]) -> float:
    w = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return w * h


def clip_box(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x0 = max(0, int(round(box[0])))
    y0 = max(0, int(round(box[1])))
    x1 = min(width, int(round(box[0] + box[2])))
    y1 = min(height, int(round(box[1] + box[3])))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def canvas_scale(width: int, height: int) -> float:
    return max(0.45, min(2.0, math.sqrt((width * height) / float(REF_W * REF_H))))


def font_floor_px(kind: str, width: int, height: int) -> int:
    return max(12, int(round(FONT_FLOOR_PX.get(kind, 56) * canvas_scale(width, height))))


# ------------------------------------------------------------------ frame I/O
def _iter_rgb_frames(path: Path, width: int, height: int,
                     frames: Sequence[int] | None = None) -> Iterator[np.ndarray]:
    """Yield decoded RGB frames (H, W, 3) uint8, streaming, bounded memory.

    ``frames`` selects exact 0-based frame indices (used for the contrast pass and
    the contact sheet); otherwise every frame is yielded in order.
    """
    cmd = [media.FFMPEG, "-v", "error", "-i", str(path)]
    if frames is not None:
        if not frames:
            return
        expr = "+".join(f"eq(n\\,{int(f)})" for f in sorted(set(int(f) for f in frames)))
        cmd += ["-vf", f"select={expr}", "-fps_mode", "passthrough"]
        expected = len(set(int(f) for f in frames))
    else:
        expected = None
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    fs = width * height * 3
    produced = 0
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(fs)
            if len(buf) < fs:
                break
            produced += 1
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
            if expected is not None and produced >= expected:
                break
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()
    err = (proc.stderr.read() if proc.stderr else b"") or b""
    if proc.returncode not in (0, None) and produced == 0:
        raise Verify2Error(f"ffmpeg could not decode {path}: {err.decode('utf-8', 'replace')[-800:]}")


def _decode_selected(path: Path, width: int, height: int,
                     frames: Sequence[int]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    uniq = sorted(set(int(f) for f in frames))
    for idx, frame in zip(uniq, _iter_rgb_frames(path, width, height, uniq)):
        out[idx] = frame
    return out


def _audio_stats(path: Path) -> dict[str, float]:
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".s16le", delete=False) as tf:
        raw = Path(tf.name)
    try:
        try:
            media.run([media.FFMPEG, "-v", "error", "-y", "-i", str(path), "-vn",
                       "-ac", "2", "-ar", "48000", "-f", "s16le", str(raw)])
        except media.MediaError:
            # a silent video (no audio stream) is legal input; stats are then zero
            return {"n_samples": 0, "duration_s": 0.0, "rms_db": -120.0,
                    "note": "no audio stream"}
        data = np.frombuffer(raw.read_bytes(), dtype=np.int16).astype(np.float32) / 32768.0
        if data.size == 0:
            return {"n_samples": 0, "duration_s": 0.0, "rms_db": -120.0}
        rms = float(np.sqrt(np.mean(data ** 2)))
        return {"n_samples": int(data.size),
                "duration_s": round(data.size / 2 / 48000, 6),
                "rms_db": round(20 * math.log10(max(rms, 1e-9)), 3)}
    finally:
        raw.unlink(missing_ok=True)


# ---------------------------------------------------------------- mask source
@dataclass
class MaskDoc:
    """Person/face masks + existing text from the perception contract."""

    path: Path
    document: dict[str, Any]
    frames: list[dict[str, Any]]
    regions: dict[str, Any]
    cache: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def has_person_mask(self) -> bool:
        caps = self.document.get("capabilities") or {}
        return bool(caps.get("person_mask")) and self._mask_size() is not None

    @property
    def backend(self) -> str:
        return str((self.document.get("backend") or {}).get("name") or "unknown")

    def _mask_size(self) -> tuple[int, int] | None:
        for f in self.frames:
            m = f.get("mask")
            if m:
                p = self.path.parent / str(m)
                if p.exists():
                    with Image.open(p) as im:
                        return im.size
        return None

    def _load(self, idx: int) -> np.ndarray | None:
        if idx in self.cache:
            return self.cache[idx]
        f = self.frames[idx]
        m = f.get("mask")
        if not m:
            return None
        p = self.path.parent / str(m)
        if not p.exists():
            return None
        with Image.open(p) as im:
            arr = np.array(im.convert("L"))
        self.cache[idx] = arr
        return arr

    def _bracket(self, frame: int, fps: float) -> tuple[int, int]:
        """Two sample indices whose masks bound ``frame`` (same index if exact)."""
        if not self.frames:
            return 0, 0
        times = []
        for f in self.frames:
            t = f.get("t_s")
            if t is None:
                t = float(f.get("frame", 0)) / max(fps, 1e-9)
            times.append(float(t))
        t = frame / max(fps, 1e-9)
        idx = int(np.searchsorted(times, t))
        if idx <= 0:
            return 0, min(1, len(times) - 1)
        if idx >= len(times):
            return len(times) - 1, len(times) - 1
        return idx - 1, idx

    def at(self, frame: int, fps: float) -> tuple[np.ndarray | None,
                                                 list[list[int]], list[list[int]],
                                                 tuple[int, int]]:
        """Union person mask + face rects + hand rects (pixels) for ``frame``."""
        if not self.frames:
            return None, [], [], (0, 0)
        i0, i1 = self._bracket(frame, fps)
        masks = [self._load(i) for i in (i0, i1)]
        masks = [m for m in masks if m is not None]
        if not masks:
            return None, [], [], (0, 0)
        h = min(m.shape[0] for m in masks)
        w = min(m.shape[1] for m in masks)
        union = np.zeros((h, w), dtype=np.uint8)
        for m in masks:
            union = np.maximum(union, m[:h, :w])
        faces: list[list[int]] = []
        hands: list[list[int]] = []
        for i in sorted({i0, i1}):
            face = self.frames[i].get("face") or None
            if face and face.get("bbox"):
                faces.append([int(round(v)) for v in face["bbox"][:4]])
            elif self.frames[i].get("face_bbox"):
                faces.append([int(round(v)) for v in self.frames[i]["face_bbox"][:4]])
            for hand in (self.frames[i].get("hands") or []):
                bbox = hand.get("bbox") if isinstance(hand, dict) else None
                if bbox:
                    hands.append([int(round(v)) for v in bbox[:4]])
        return union, faces, hands, (w, h)

    def existing_text(self, fps: float) -> list[dict[str, Any]]:
        out = []
        for r in (self.regions.get("existing_text") or []):
            try:
                box = [int(round(v)) for v in r["bbox"][:4]]
                out.append({"t0": float(r.get("t0", 0.0)), "t1": float(r.get("t1", 1e9)),
                            "bbox": box, "kind": r.get("kind", "other"),
                            "text": r.get("text", "")})
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def quiet_zones(self) -> list[dict[str, Any]]:
        out = []
        for z in (self.regions.get("quiet_zones") or []):
            try:
                out.append({"bbox": [int(round(v)) for v in z["bbox"][:4]],
                            "kind": z.get("kind", "background"),
                            "score": float(z.get("score", 0.0)),
                            "person_overlap": float(z.get("person_overlap", 0.0)),
                            "text_overlap": float(z.get("text_overlap", 0.0))})
            except (KeyError, TypeError, ValueError):
                continue
        return out


def _protected_text_regions(ir: dict, mask_doc: MaskDoc | None, fps: float
                            ) -> tuple[list[dict[str, Any]], str | None]:
    """Protected burned-in text: IR-filtered obstacles first, else the mask doc.

    design2 filters venue signage out of ``ir.perception.existing_text`` while
    keeping captions/logos that must not be covered; that filtered list is the
    contract for v2. ``None`` source means the question could not be asked.
    """
    doc = (ir.get("perception") or {}).get("existing_text")
    if isinstance(doc, list):
        regions = []
        for r in doc:
            try:
                regions.append({"t0": float(r.get("t0", 0.0)),
                                "t1": float(r.get("t1", 1e9)),
                                "bbox": [int(round(v)) for v in r["bbox"][:4]],
                                "kind": str(r.get("kind") or "other"),
                                "text": str(r.get("text") or "")})
            except (KeyError, TypeError, ValueError):
                continue
        return regions, "ir.perception.existing_text"
    if mask_doc is not None:
        caps = mask_doc.document.get("capabilities") or {}
        if caps.get("existing_text"):
            return mask_doc.existing_text(fps), "perception_document.existing_text"
    return [], None


def rects_overlap_px(footprint: np.ndarray, box: Sequence[float],
                     rects: Sequence[Sequence[int]]) -> int:
    """Footprint pixels inside a box that also fall inside any of ``rects``."""
    x0, y0, bw, bh = clip_box(box, footprint.shape[1], footprint.shape[0])
    total = 0
    for rect in rects:
        rx, ry, rw, rh = (int(round(v)) for v in rect[:4])
        x1, y1 = max(x0, rx), max(y0, ry)
        x2, y2 = min(x0 + bw, rx + rw), min(y0 + bh, ry + rh)
        if x2 > x1 and y2 > y1:
            total += int(footprint[y1:y2, x1:x2].sum())
    return total


REQUIRED_MASK_KEYS = ("schema", "source", "frames", "regions")


def load_masks(path: str | Path) -> MaskDoc:
    p = Path(path)
    if not p.exists():
        raise Verify2Error(f"mask document not found: {p}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_MASK_KEYS if k not in doc]
    if missing:
        raise Verify2Error(f"mask document {p} missing keys {missing}; expected "
                           "semotion.perception/1 (docs/PERCEPTION-CONTRACT.md)")
    frames = sorted((f for f in (doc.get("frames") or []) if isinstance(f, dict)),
                    key=lambda f: float(f.get("frame", 0)))
    if not frames:
        raise Verify2Error(f"mask document {p} has no frames")
    return MaskDoc(path=p, document=doc, frames=frames,
                   regions=doc.get("regions") or {})


def _maxpool_bool(mask: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Conservative (max-pooled) boolean downsample: any covered pixel counts."""
    h, w = mask.shape
    out_w, out_h = int(out_w), int(out_h)
    if out_w <= 0 or out_h <= 0:
        return np.zeros((1, 1), dtype=bool)
    if (w, h) == (out_w, out_h):
        return mask
    u8 = mask.astype(np.uint8)
    if w % out_w == 0 and h % out_h == 0:
        fy, fx = h // out_h, w // out_w
        return u8.reshape(out_h, fy, out_w, fx).any(axis=(1, 3))
    import cv2

    small = cv2.resize(u8 * 255, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return small > 0


def _box_on_grid(box: Sequence[float], grid: tuple[int, int], frame_w: int,
                 frame_h: int) -> tuple[int, int, int, int]:
    """Map a pixel box to the mask grid (clamped, conservative outward)."""
    mw, mh = grid
    sx, sy = mw / float(frame_w), mh / float(frame_h)
    x = max(0, min(mw, int(math.floor(box[0] * sx))))
    y = max(0, min(mh, int(math.floor(box[1] * sy))))
    x1 = max(0, min(mw, int(math.ceil((box[0] + box[2]) * sx))))
    y1 = max(0, min(mh, int(math.ceil((box[1] + box[3]) * sy))))
    return x, y, max(0, x1 - x), max(0, y1 - y)


# --------------------------------------------------------------- footprint ops
def diff_footprint(final_rgb: np.ndarray, base_rgb: np.ndarray, threshold: float) -> np.ndarray:
    """Per-pixel overlay footprint (max channel |Δ| > threshold)."""
    diff = np.abs(final_rgb.astype(np.int16) - base_rgb.astype(np.int16)).max(axis=2)
    return diff > float(threshold)


def overlay_footprint(overlay_rgb: np.ndarray,
                      threshold: float = OVERLAY_FOOTPRINT_THRESHOLD) -> np.ndarray:
    """Footprint from an overlay-only pass (black background)."""
    return overlay_rgb.astype(np.int16).max(axis=2) > float(threshold)


def occlusion_metrics(footprint: np.ndarray, box: Sequence[float],
                      person: np.ndarray | None, face_rects: Sequence[Sequence[int]],
                      frame_area: int) -> dict[str, Any]:
    """Occlusion numbers for one element box in one frame.

    ``footprint`` is the full-frame boolean overlay mask; only the part inside
    ``box`` is attributed to the element. Person overlap is computed on the
    mask's native grid with max-pooling (conservative); face overlap is exact
    full-resolution pixels.
    """
    x0, y0, bw, bh = clip_box(box, footprint.shape[1], footprint.shape[0])
    sub = footprint[y0:y0 + bh, x0:x0 + bw] if bw and bh else np.zeros((0, 0), dtype=bool)
    face_px = 0
    for rect in face_rects:
        rx, ry, rw, rh = (int(round(v)) for v in rect[:4])
        x1, y1 = max(x0, rx), max(y0, ry)
        x2, y2 = min(x0 + bw, rx + rw), min(y0 + bh, ry + rh)
        if x2 > x1 and y2 > y1:
            face_px += int(sub[y1 - y0:y2 - y0, x1 - x0:x2 - x0].sum())
    result = {"person_overlap_px": 0, "person_px": 0, "frame_px": int(frame_area),
              "person_frame_fraction": 0.0, "person_person_fraction": 0.0,
              "face_overlap_px": face_px, "mask_grid": None, "person_mask": None}
    if person is None or not bw or not bh:
        return result
    mh, mw = person.shape
    gx, gy, gw, gh = _box_on_grid([x0, y0, bw, bh], (mw, mh),
                                  footprint.shape[1], footprint.shape[0])
    if gw <= 0 or gh <= 0:
        result["mask_grid"] = [mw, mh]
        result["person_mask"] = "present"
        return result
    sub_small = _maxpool_bool(sub, gw, gh)
    person_sub = person[gy:gy + gh, gx:gx + gw] > 127
    overlap = int(np.logical_and(sub_small, person_sub).sum())
    person_bool = person > 127
    person_px = int(person_bool.sum())
    frame_small = int(mw * mh)
    result.update({
        "person_mask": "present",
        "person_overlap_px": overlap,
        "person_px": person_px,
        "person_frame_fraction": overlap / float(max(1, frame_small)),
        "person_person_fraction": overlap / float(max(1, person_px)),
        "mask_grid": [mw, mh],
    })
    return result


# --------------------------------------------------------------------- contrast
def _otsu(values: np.ndarray, bins: int = 64) -> float:
    """Deterministic Otsu luminance split for two-population pixel samples."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return 0.0
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-9:
        return hi
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return hi
    prob = hist / total
    centres = (edges[:-1] + edges[1:]) / 2.0
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centres)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    sigma = np.where(denom > 1e-12, (mu_t * omega - mu) ** 2 / np.maximum(denom, 1e-12), 0.0)
    k = int(np.argmax(sigma))
    return float(edges[k + 1])


def glyph_and_contrast(rendered: np.ndarray, base: np.ndarray, box: Sequence[float],
                       glyph_color: tuple[int, int, int], *,
                       color_distance: int = GLYPH_COLOR_DISTANCE,
                       block: int = GLYPH_BLOCK) -> dict[str, Any] | None:
    """WCAG contrast of an element's glyph pixels vs their rendered local background.

    Glyph pixels are overlay pixels whose rendered colour is close to the IR's
    declared text colour. Background pixels are the remaining rendered pixels
    around them (scrim included), so the number reflects what a viewer sees.
    The contrast against the raw source underneath (no scrim) is also returned.
    """
    x0, y0, bw, bh = clip_box(box, rendered.shape[1], rendered.shape[0])
    if bw <= 0 or bh <= 0:
        return None
    r = rendered[y0:y0 + bh, x0:x0 + bw].astype(np.int16)
    b = base[y0:y0 + bh, x0:x0 + bw].astype(np.int16)
    changed = np.abs(r - b).max(axis=2) > 6
    if not changed.any():
        return None
    dist = np.abs(r - np.array(glyph_color, dtype=np.int16)).max(axis=2)
    luma = 0.2126 * r[..., 0] + 0.7152 * r[..., 1] + 0.0722 * r[..., 2]
    changed_n = int(changed.sum())
    glyph = changed & (dist <= color_distance)
    # If "close to the declared colour" swallows most of the overlay, the text and
    # its background are too similar for colour matching to separate them (this is
    # exactly the low-contrast case). Split the changed pixels by luminance instead.
    ambiguous = changed_n > 0 and int(glyph.sum()) > 0.6 * changed_n
    if ambiguous or int(glyph.sum()) < 20:
        vals = luma[changed]
        if vals.size < 4:
            return None
        cut = _otsu(vals)
        text_is_bright = luma[changed][dist[changed] <= color_distance].mean() >= \
            float(np.median(vals)) if (dist[changed] <= color_distance).any() else True
        glyph = changed & (luma >= cut) if text_is_bright else changed & (luma <= cut)
        if int(glyph.sum()) < 20 or int(glyph.sum()) > 0.8 * changed_n:
            cut = float(np.percentile(vals, 90)) if text_is_bright else float(np.percentile(vals, 10))
            glyph = changed & (luma >= cut) if text_is_bright else changed & (luma <= cut)
    if int(glyph.sum()) < 4:
        return None
    glyph_mean_dist = float(np.abs(r[glyph].mean(axis=0) - np.array(glyph_color)).max())
    if glyph_mean_dist > 120:
        # No pixels close to the declared text colour: this is a solid block /
        # image element, not text. Legibility is NOT RUN, never a fake fail.
        return None

    # background = rendered pixels around glyphs, excluding the glyphs themselves
    grown = glyph.copy()
    for dy in (-2, -1, 0, 1, 2):
        for dx in (-2, -1, 0, 1, 2):
            grown |= np.roll(np.roll(glyph, dy, axis=0), dx, axis=1)
    background = (~grown)
    bg_fallback = None
    if int(background.sum()) < 20:
        background = (~glyph) & (np.abs(r - b).max(axis=2) > 6)
    if int(background.sum()) < 20:
        # the element's glyphs fill its box; there is no rendered background
        # sample. Measure against the source pixels underneath and say so.
        background = np.ones(glyph.shape, dtype=bool)
        bg_fallback = "source (no rendered background pixels in the box)"

    glyph_rgb = r[glyph]
    bg_rgb = r[background]
    src_rgb = b[glyph]
    l_glyph = float(relative_luminance(glyph_rgb.mean(axis=0)))
    l_bg = float(relative_luminance(bg_rgb.mean(axis=0)))
    l_src = float(relative_luminance(src_rgb.mean(axis=0)))
    contrast = wcag_contrast(l_glyph, l_bg)
    contrast_source = wcag_contrast(l_glyph, l_src)

    # blockwise minimum: a single glyph stroke over a bright spot must fail
    min_block = contrast
    block_min_px = 10 ** 9
    gh, gw = glyph.shape
    for by in range(0, gh, block):
        for bx in range(0, gw, block):
            gblk = glyph[by:by + block, bx:bx + block]
            if int(gblk.sum()) < GLYPH_MIN_BLOCK_PIXELS:
                continue
            bblk = background[by:by + block, bx:bx + block]
            # widen the background ring around a small glyph block
            y0b, y1b = max(0, by - block), min(gh, by + 2 * block)
            x0b, x1b = max(0, bx - block), min(gw, bx + 2 * block)
            ring = background[y0b:y1b, x0b:x1b]
            bg_pixels = r[y0b:y1b, x0b:x1b][ring] if ring.any() else bg_rgb
            lb = (float(relative_luminance(bg_pixels.mean(axis=0)))
                  if bg_pixels.size else l_bg)
            lg = float(relative_luminance(r[by:by + block, bx:bx + block][gblk].mean(axis=0)))
            c = wcag_contrast(lg, lb)
            if c < min_block:
                min_block = c
                block_min_px = int(gblk.sum())
    return {
        "contrast": round(float(min_block), 4),
        "contrast_global": round(float(contrast), 4),
        "contrast_vs_source": round(float(contrast_source), 4),
        "glyph_pixels": int(glyph.sum()),
        "background_pixels": int(background.sum()),
        "glyph_rgb": [int(v) for v in glyph_rgb.mean(axis=0).round()],
        "background_rgb": [int(v) for v in bg_rgb.mean(axis=0).round()],
        "source_rgb": [int(v) for v in src_rgb.mean(axis=0).round()],
        "min_block_glyph_pixels": None if block_min_px == 10 ** 9 else block_min_px,
        "background_fallback": bg_fallback,
        "glyph_color_distance": round(glyph_mean_dist, 2),
        "glyph_assignment": "luminance-split" if ambiguous else "colour-match",
    }


# ------------------------------------------------------------- IR shape helpers
def element_box(el: dict) -> list[int] | None:
    """Element box for both IR shapes: v1 ``layout.box``, v2 top-level ``box``."""
    lay = el.get("layout") or {}
    box = lay.get("box")
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        return [int(round(float(v))) for v in box[:4]]
    box = el.get("box")
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        return [int(round(float(v))) for v in box[:4]]
    return None


def element_font_size(el: dict) -> int:
    """Main display size.

    v1: ``style.font_size_px``. v2: ``fit.font_size_px`` (max of display/title/body
    as emitted by design2), else ``display_px`` (keyword/number) or ``title_px``
    (card/list). ``body_px`` is body copy and is never the main size.
    """
    style = el.get("style") or {}
    fit = el.get("fit") or {}
    for value in (style.get("font_size_px"), fit.get("font_size_px"),
                  fit.get("display_px"), fit.get("title_px")):
        try:
            size = int(value or 0)
        except (TypeError, ValueError):
            size = 0
        if size > 0:
            return size
    return 0


def element_min_font(el: dict) -> int:
    """Module-owned typography floor when the IR carries one (v2 min_font_px)."""
    style = el.get("style") or {}
    fit = el.get("fit") or {}
    for value in (fit.get("min_font_px"), style.get("min_font_px")):
        try:
            size = int(value or 0)
        except (TypeError, ValueError):
            size = 0
        if size > 0:
            return size
    return 0


def element_text(el: dict) -> str:
    """Reference string as rendered: text, title, or the rows' text joined.

    design2 stores card copy in ``content.title`` and list rows in
    ``content.rows[i].text``; older IRs may still use ``content.text``/``items``.
    """
    content = el.get("content") or {}
    for key in ("text", "title"):
        value = str(content.get(key) or "").strip()
        if value:
            return value
    rows = content.get("rows") or content.get("items") or []
    parts = []
    for row in rows:
        if isinstance(row, dict):
            value = str(row.get("text") or row.get("title") or "").strip()
        else:
            value = str(row or "").strip()
        if value:
            parts.append(value)
    return "\n".join(parts)


def element_treatment(el: dict) -> str:
    lay = el.get("layout") or {}
    for value in (lay.get("treatment"), (el.get("backdrop") or {}).get("kind"),
                  lay.get("zone")):
        if value:
            return str(value)
    return ""


def is_lower_third(el: dict) -> bool:
    return "lower_third" in element_treatment(el)


def element_ink(el: dict, ir: dict) -> tuple[int, int, int]:
    """Ink colour used for the strict text-onset check."""
    style = el.get("style") or {}
    if style.get("color"):
        return parse_color(style.get("color"))
    theme = ir.get("theme") or {}
    for key in ("ink", "text", "fg", "primary"):
        if theme.get(key):
            return parse_color(theme.get(key))
    return (245, 247, 250)


def ink_pixel_count(overlay_rgb: np.ndarray, box: Sequence[float],
                    ink: tuple[int, int, int], *, color_distance: int = 64,
                    luma_gap: int = 30) -> tuple[int, float]:
    """Count glyph-coloured pixels in the overlay pass (scrim excluded by luma).

    Returns ``(count, scrim_luma)``. Pixels must be close to the declared ink
    colour *and* separated in luminance from the local scrim/background, which is
    how a dark scrim gradient is excluded without hard-coding backdrop boxes.
    """
    x0, y0, bw, bh = clip_box(box, overlay_rgb.shape[1], overlay_rgb.shape[0])
    if bw <= 0 or bh <= 0:
        return 0, 0.0
    crop = overlay_rgb[y0:y0 + bh, x0:x0 + bw].astype(np.int16)
    dist = np.abs(crop - np.array(ink, dtype=np.int16)).max(axis=2)
    luma = 0.2126 * crop[..., 0] + 0.7152 * crop[..., 1] + 0.0722 * crop[..., 2]
    ink_luma = float(relative_luminance(np.array(ink, dtype=np.float64) / 255.0))
    # background/scrim estimate: the darker half of the box when the ink is light
    bg = float(np.median(luma)) if luma.size else 0.0
    if ink_luma * 255.0 >= bg:
        mask = (dist <= color_distance) & (luma >= bg + luma_gap)
    else:
        mask = (dist <= color_distance) & (luma <= bg - luma_gap)
    return int(mask.sum()), round(bg, 2)


def element_backdrop(el: dict) -> dict[str, Any] | None:
    bd = el.get("backdrop")
    if isinstance(bd, dict) and isinstance(bd.get("box"), (list, tuple)) \
            and len(bd["box"]) >= 4:
        return bd
    return None


def is_scrim_backdrop(kind: Any) -> bool:
    k = str(kind or "")
    return k.startswith("scrim") or k.startswith("band")


_T2S = None


def to_simplified(text: str) -> str:
    """Fold traditional Chinese OCR output to simplified when opencc is present.

    Vision occasionally returns traditional glyphs for a simplified string
    (e.g. 停滯 vs 停滞); that is an OCR script variant, not a text mismatch.
    """
    global _T2S
    if _T2S is None:
        try:
            import opencc
            _T2S = opencc.OpenCC("t2s")
        except Exception:  # pragma: no cover - optional dependency
            _T2S = False
    return _T2S.convert(str(text)) if _T2S else str(text)


def normalise_display_text(text: str) -> str:
    """Whitespace/punctuation-insensitive form used for OCR- vs IR-text equality."""
    out = []
    for ch in to_simplified(str(text)):
        if ch.isalnum():
            out.append(ch.lower())
    return "".join(out)


def is_subsequence(needle: str, haystack: str) -> bool:
    if not needle:
        return False
    it = iter(haystack)
    return all(ch in it for ch in needle)


def ocr_frame_text(rgb: np.ndarray) -> dict[str, Any]:
    """Apple Vision OCR of one RGB array (accurate level, zh-Hans + en-US).

    Returns ``{"available": False}`` when pyobjc/Vision is not installed. The
    engine is deterministic for identical input bytes (same OS build).
    """
    try:
        import Quartz
        import Vision
    except Exception as exc:  # pragma: no cover - platform guard
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    import tempfile
    from PIL import Image as _Image
    from Foundation import NSURL

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tmp = Path(tf.name)
    try:
        _Image.fromarray(rgb).save(tmp)
        url = NSURL.fileURLWithPath_(str(tmp))
        cg = Quartz.CGImageSourceCreateImageAtIndex(
            Quartz.CGImageSourceCreateWithURL(url, None), 0, None)
        req = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(None)
        req.setRecognitionLevel_(0)  # 0 = accurate (1 is "fast" and misreads CJK)
        req.setRecognitionLanguages_(["zh-Hans", "en-US"])
        req.setUsesLanguageCorrection_(True)
        Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None) \
            .performRequests_error_([req], None)
        lines = []
        for obs in req.results() or []:
            cand = obs.topCandidates_(1)
            if cand:
                lines.append({"text": str(cand[0].string()),
                              "confidence": float(cand[0].confidence())})
        return {"available": True, "lines": lines,
                "text": "".join(l["text"] for l in lines),
                "confidence": (sum(l["confidence"] for l in lines) / len(lines)
                               if lines else 0.0)}
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------- layout
def layout_checks(ir: dict, width: int, height: int, *,
                  safe_margin_px: int = SAFE_MARGIN_PX,
                  footprint_bboxes: dict[str, list[int] | None] | None = None,
                  footprint_margin_frac: float = FOOTPRINT_MARGIN_FRAC,
                  footprint_collisions: list[dict[str, Any]] | None = None
                  ) -> dict[str, Any]:
    """On-canvas, safe margins, font floors, and the lead's footprint rules.

    Two margin views are reported:
    * declared box: fully on canvas with ``safe_margin_px`` from every edge;
    * rendered footprint: the measured changed-pixel bbox must also keep
      ``footprint_margin_frac`` (2.5 %) of the canvas width/height.
    ``footprint_collisions`` comes from the rendered frames (per-frame overlap of
    two elements' footprints > 0.5 % of the smaller box = FAIL).
    """
    elements = [e for e in (ir.get("elements") or []) if isinstance(e, dict)]
    rows = []
    all_ok = True
    for el in elements:
        eid = str(el.get("id"))
        box = element_box(el)
        row: dict[str, Any] = {"id": eid, "type": el.get("type")}
        if box is None:
            row.update({"verdict": FAIL, "problems": ["element box missing"]})
            all_ok = False
            rows.append(row)
            continue
        x, y, bw, bh = box
        margins = {"left": x, "top": y, "right": width - (x + bw),
                   "bottom": height - (y + bh)}
        on_canvas = x >= 0 and y >= 0 and bw > 0 and bh > 0 \
            and x + bw <= width and y + bh <= height
        safe = on_canvas and min(margins.values()) >= safe_margin_px
        floor = font_floor_px(str(el.get("type") or ""), width, height)
        font_size = element_font_size(el)
        declared = int((el.get("style") or {}).get("font_size_px") or 0)
        min_font = element_min_font(el)
        v2_shape = bool(el.get("fit"))
        problems: list[str] = []
        font_floor_note = None
        font_floor_warning = None
        floor_source = None
        if not on_canvas:
            problems.append("box off canvas")
        if on_canvas and not safe:
            problems.append(f"breaks {safe_margin_px}px safe margin (margins {margins})")
        if min_font > 0:
            # the module's own floor is authoritative (v2 min_font_px)
            floor = min_font
            floor_source = "ir_min_font_px"
            if font_size < min_font:
                problems.append(f"font {font_size}px < module min_font_px {min_font}px")
        elif declared > 0:
            floor_source = "v1_legacy_floor"
            if font_size < floor:
                problems.append(f"font {font_size}px < floor {floor}px")
        elif v2_shape:
            # legacy v1 floor is not a hard contract for v2 multi-size typography:
            # warn only, using the lead's 0.045 * min(W,H) legibility heuristic.
            floor = max(12, int(round(FONT_FLOOR_FRAC * min(width, height))))
            floor_source = "v2_legacy_warning"
            if font_size < floor:
                font_floor_warning = (f"main font {font_size}px < {floor}px "
                                      f"(0.045 * min(canvas)) — warning only; the IR "
                                      f"carries no min_font_px")
        row_extra = {"font_floor_note": font_floor_note,
                     "font_floor_warning": font_floor_warning,
                     "font_floor_source": floor_source}
        # rendered footprint margins (2.5 % of canvas per axis)
        req_x, req_y = int(round(footprint_margin_frac * width)), \
            int(round(footprint_margin_frac * height))
        bbox = (footprint_bboxes or {}).get(eid)
        fp_margins = None
        fp_margin_ok = None
        if bbox is not None:
            fx, fy, fw, fh = bbox
            fp_margins = {"left": fx, "top": fy, "right": width - (fx + fw),
                          "bottom": height - (fy + fh)}
            fp_margin_ok = (fp_margins["left"] >= req_x and fp_margins["right"] >= req_x
                            and fp_margins["top"] >= req_y
                            and fp_margins["bottom"] >= req_y)
            if not fp_margin_ok:
                problems.append(f"rendered footprint breaks the "
                                f"{footprint_margin_frac:.1%} margin "
                                f"(needs >= {req_x}px horizontal / {req_y}px vertical, "
                                f"got {fp_margins})")
        row.update({"box": [x, y, bw, bh], "margins_px": margins,
                    "on_canvas": on_canvas, "safe_margin_px": safe_margin_px,
                    "safe": safe, "font_size_px": font_size,
                    "font_floor_px": floor, **row_extra,
                    "footprint_bbox": bbox, "footprint_margins_px": fp_margins,
                    "footprint_margin_requirement_px": [req_x, req_y],
                    "footprint_margin_ok": fp_margin_ok,
                    "verdict": FAIL if problems else PASS, "problems": problems})
        if problems:
            all_ok = False
        rows.append(row)

    collisions = []
    for i, a in enumerate(elements):
        for b in elements[i + 1:]:
            wa, wb = a.get("window") or {}, b.get("window") or {}
            try:
                overlap_time = (int(wa.get("start_frame", 0)) < int(wb.get("end_frame", 0))
                                and int(wb.get("start_frame", 0)) < int(wa.get("end_frame", 0)))
            except (TypeError, ValueError):
                overlap_time = True
            if not overlap_time:
                continue
            ba, bb = element_box(a), element_box(b)
            if ba is None or bb is None:
                continue
            inter = inter_area(ba, bb)
            smaller = min(float(ba[2]) * float(ba[3]), float(bb[2]) * float(bb[3])) or 1.0
            if inter / smaller > COLLISION_AREA_RATIO:
                collisions.append({"a": a.get("id"), "b": b.get("id"),
                                   "intersection_px": int(round(inter)),
                                   "fraction_of_smaller": round(inter / smaller, 4)})
    measured = list(footprint_collisions or [])
    return {
        "safe_margin_px": safe_margin_px,
        "footprint_margin_frac": footprint_margin_frac,
        "elements": rows,
        "collisions": collisions,
        "footprint_collisions": measured,
        "collision_threshold": COLLISION_AREA_RATIO,
        "footprint_collision_threshold": FOOTPRINT_COLLISION_RATIO,
        "verdict": PASS if all_ok and not collisions and not measured else FAIL,
    }


# ---------------------------------------------------------------------- timing
def timing_series_row(counts: Sequence[int], window: dict, fps: float,
                      anchor_start_s: float | None, n_frames: int,
                      box_area: int, pre_counts: Sequence[int] | None = None,
                      tolerance_frames: int = TIMING_TOLERANCE_FRAMES) -> dict[str, Any]:
    """Appearance/disappearance frames from per-frame footprint counts."""
    if box_area <= 0 or len(counts) == 0:
        return {"verdict": NOT_RUN, "note": "empty box or no frames"}
    sf, ef = int(window.get("start_frame", 0)), int(window.get("end_frame", 0))
    gate = max(1, int(round(0.005 * box_area)))
    pre = list(pre_counts or [])
    if pre:
        gate = max(gate, int(np.percentile(pre, 90)) * 3 + 3)
    lo = max(0, sf - 8)
    hi = min(n_frames, ef + 8)
    series = np.array([counts[f] if f < len(counts) else 0 for f in range(lo, hi)], dtype=float)
    visible = np.nonzero(series >= gate)[0]
    if visible.size == 0:
        return {"verdict": FAIL, "appearance_frame": None, "disappearance_frame": None,
                "window": [sf, ef], "gate_pixels": int(gate),
                "problems": ["no frame inside/around the window exceeds the visibility gate"],
                "note": "element never became visible in the rendered output"}
    appearance = int(lo + visible[0])
    last_visible = int(lo + visible[-1])
    disappearance = last_visible + 1
    start_err = appearance - sf
    exit_err = disappearance - ef
    anchor_frame = None if anchor_start_s is None else int(round(float(anchor_start_s) * fps))
    anchor_err = None if anchor_frame is None else appearance - anchor_frame
    problems = []
    if abs(start_err) > tolerance_frames:
        problems.append(f"appearance {appearance} vs window start {sf} ({start_err:+d} frames)")
    if abs(exit_err) > tolerance_frames:
        problems.append(f"disappearance {disappearance} vs window end {ef} ({exit_err:+d} frames)")
    if anchor_err is not None and abs(anchor_err) > tolerance_frames:
        problems.append(f"appearance {appearance} vs anchor onset {anchor_frame} "
                        f"({anchor_err:+d} frames)")
    return {
        "verdict": FAIL if problems else PASS,
        "appearance_frame": appearance,
        "disappearance_frame": disappearance,
        "window_start_frame": sf,
        "window_end_frame": ef,
        "anchor_onset_frame": anchor_frame,
        "signed_error_start_frames": int(start_err),
        "signed_error_anchor_frames": None if anchor_err is None else int(anchor_err),
        "signed_error_exit_frames": int(exit_err),
        "signed_error_start_ms": round(1000.0 * start_err / fps, 2),
        "signed_error_anchor_ms": None if anchor_err is None else round(1000.0 * anchor_err / fps, 2),
        "signed_error_exit_ms": round(1000.0 * exit_err / fps, 2),
        "tolerance_frames": tolerance_frames,
        "gate_pixels": int(gate),
        "problems": problems,
    }


# ----------------------------------------------------------------- determinism
def compare_renders(first: str | Path, second: str | Path, width: int, height: int, *,
                    mean_tolerance: float = DET_MEAN_TOL,
                    max_tolerance: float = DET_MAX_TOL) -> dict[str, Any]:
    """Decode two renders and compare with explicit tolerances."""
    n = 0
    mean_sum = 0.0
    worst_mean = 0.0
    worst_max = 0.0
    worst_frame = None
    it_a = _iter_rgb_frames(Path(first), width, height)
    it_b = _iter_rgb_frames(Path(second), width, height)
    for a, b in zip(it_a, it_b):
        d = np.abs(a.astype(np.int16) - b.astype(np.int16))
        mean_d = float(d.mean())
        max_d = float(d.max())
        mean_sum += mean_d
        n += 1
        if mean_d > worst_mean:
            worst_mean, worst_frame = mean_d, n - 1
        worst_max = max(worst_max, max_d)
    mean = (mean_sum / n) if n else 0.0
    ok = n > 0 and mean <= mean_tolerance and worst_max <= max_tolerance
    return {
        "frames_compared": n,
        "mean_pixel_diff": round(mean, 8),
        "worst_frame_mean_diff": round(worst_mean, 8),
        "worst_frame": worst_frame,
        "max_pixel_diff": round(worst_max, 4),
        "tolerance_mean": mean_tolerance,
        "tolerance_max": max_tolerance,
        "verdict": PASS if ok else FAIL,
        "policy": ("decoded frames must have mean |Δ| <= tolerance_mean and "
                   "max |Δ| <= tolerance_max; byte-identical files are never required"),
    }


# ------------------------------------------------------------------ contact sheet
def _label_font(size: int = 16) -> ImageFont.ImageFont:
    for candidate in ("/System/Library/Fonts/Helvetica.ttc",
                      "/System/Library/Fonts/Supplemental/Arial.ttf",
                      "assets/fonts/NotoSansSC-Regular.otf"):
        try:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _contour(mask: np.ndarray) -> np.ndarray:
    m = mask > 127
    edge = m ^ np.roll(m, 1, axis=0)
    edge |= m ^ np.roll(m, 1, axis=1)
    return edge


def write_contact_sheet(path: Path, frames: Sequence[tuple[int, np.ndarray]],
                        elements: Sequence[dict], person_masks: dict[int, np.ndarray | None],
                        element_verdicts: dict[str, str], overall: str,
                        face_rects: dict[int, list[list[int]]] | None = None,
                        title: str = "verify2 contact sheet") -> Path:
    """Annotated frames: element boxes, person contour, face box, verdict labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tile_w, tile_h, label_h, cols = 640, 360, 26, 3
    rows = max(1, math.ceil(len(frames) / cols))
    header = 64
    sheet = Image.new("RGB", (cols * tile_w, header + rows * (tile_h + label_h)), (18, 20, 26))
    draw = ImageDraw.Draw(sheet)
    font = _label_font(16)
    small = _label_font(13)
    big = _label_font(22)
    color = (60, 220, 120) if overall == PASS else (240, 80, 80)
    draw.text((14, 12), f"{title} — overall {overall}", fill=color, font=big)
    draw.text((14, 40), "green box = readable placement, red = failing check; "
                        "green contour = person mask; cyan = face box", fill=(200, 205, 215),
              font=small)
    for idx, (frame, rgb) in enumerate(frames):
        crop = Image.fromarray(rgb).resize((tile_w, tile_h), Image.BILINEAR)
        arr = np.array(crop)
        scale_x, scale_y = tile_w / rgb.shape[1], tile_h / rgb.shape[0]
        mask = person_masks.get(frame)
        if mask is not None:
            m = Image.fromarray(mask).resize((tile_w, tile_h), Image.NEAREST)
            edge = _contour(np.array(m))
            arr[edge] = [60, 230, 120]
        tile = Image.fromarray(arr)
        td = ImageDraw.Draw(tile)
        for rect in (face_rects or {}).get(frame, []):
            fx, fy, fw, fh = rect
            td.rectangle([fx * scale_x, fy * scale_y, (fx + fw) * scale_x,
                          (fy + fh) * scale_y], outline=(80, 210, 235), width=2)
        for el in elements:
            box = element_box(el)
            if box is None:
                continue
            x, y, w, h = box
            sf = int((el.get("window") or {}).get("start_frame", 0))
            ef = int((el.get("window") or {}).get("end_frame", 0))
            if not (sf <= frame < ef):
                continue
            verdict = element_verdicts.get(str(el.get("id")), PASS)
            col = (60, 220, 120) if verdict == PASS else (255, 70, 70)
            td.rectangle([x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y],
                         outline=col, width=3)
            td.text((x * scale_x + 4, y * scale_y + 2), f"{el.get('id')} {verdict}",
                    fill=col, font=small)
        draw.text((idx % cols * tile_w + 8, header + (idx // cols) * (tile_h + label_h)
                   + tile_h + 5), f"frame {frame}", fill=(210, 215, 225), font=font)
        sheet.paste(tile, (idx % cols * tile_w, header + (idx // cols) * (tile_h + label_h)))
    sheet.save(path)
    return path


def write_worst_crop(path: Path, rgb: np.ndarray, box: Sequence[float],
                     label: str) -> Path:
    """Crop around the worst occlusion box for human inspection."""
    x, y, w, h = clip_box(box, rgb.shape[1], rgb.shape[0])
    pad = int(max(w, h) * 0.6)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(rgb.shape[1], x + w + pad), min(rgb.shape[0], y + h + pad)
    crop = Image.fromarray(rgb[y0:y1, x0:x1])
    if crop.width > 900:
        crop = crop.resize((900, int(round(crop.height * 900 / crop.width))), Image.BILINEAR)
    canvas = Image.new("RGB", (crop.width, crop.height + 34), (18, 20, 26))
    canvas.paste(crop, (0, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), label, fill=(255, 120, 120), font=_label_font(16))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


# --------------------------------------------------------------------- verifier
def verify(*, ir: dict, source: str | Path, rendered: str | Path,
           masks: str | Path | MaskDoc | None = None,
           overlay: str | Path | None = None,
           ink: str | Path | None = None,
           second_render: str | Path | None = None,
           out_dir: str | Path | None = None,
           safe_margin_px: int = SAFE_MARGIN_PX,
           footprint_threshold: int | None = None,
           make_sheet: bool = True,
           write_metrics: bool = True) -> dict[str, Any]:
    """Measure one rendered file and return the ``semotion.verify2/1`` metrics."""
    source_p, rendered_p = Path(source), Path(rendered)
    mask_doc = masks if isinstance(masks, MaskDoc) else (load_masks(masks) if masks else None)
    width = int((ir.get("canvas") or {}).get("width") or 0)
    height = int((ir.get("canvas") or {}).get("height") or 0)
    if width <= 0 or height <= 0:
        raise Verify2Error("IR canvas width/height missing")
    fps = float((ir.get("source") or {}).get("fps")
                or (ir.get("canvas") or {}).get("fps") or 25.0)
    elements = [e for e in (ir.get("elements") or []) if isinstance(e, dict)]
    if not elements:
        raise Verify2Error("IR has no elements to verify")

    # ---- render integrity (geometry must match before pixels are comparable)
    src_info, out_info = media.info(source_p), media.info(rendered_p)
    try:
        src_frames_n = int(src_info.stream_frames or 0) or int(round(src_info.duration * fps))
        out_frames_n = int(out_info.stream_frames or 0) or int(round(out_info.duration * fps))
    except (TypeError, ValueError):
        src_frames_n = out_frames_n = 0
    src_audio = _audio_stats(source_p)
    out_audio = _audio_stats(rendered_p)
    integrity_problems = []
    if (out_info.width, out_info.height) != (width, height):
        integrity_problems.append(f"rendered geometry {out_info.width}x{out_info.height} "
                                  f"!= canvas {width}x{height}")
    if (src_info.width, src_info.height) != (width, height):
        integrity_problems.append(f"source geometry {src_info.width}x{src_info.height} "
                                  f"!= canvas {width}x{height}")
    if abs(src_info.fps - fps) > 1e-3 or abs(out_info.fps - fps) > 1e-3:
        integrity_problems.append(f"fps source={src_info.fps} rendered={out_info.fps} "
                                  f"expected={fps}")
    if abs(out_frames_n - src_frames_n) > 1:
        integrity_problems.append(f"frame count rendered={out_frames_n} "
                                  f"source={src_frames_n}")
    duration_delta = abs(out_info.duration - src_info.duration)
    if duration_delta > 1.0 / fps + 0.01:
        integrity_problems.append(f"duration delta {duration_delta:.3f}s > 1 frame")
    has_src_audio = src_info.n_audio_streams > 0
    has_out_audio = out_info.n_audio_streams > 0
    if has_src_audio != has_out_audio:
        integrity_problems.append(
            f"audio stream mismatch (source={src_info.n_audio_streams}, "
            f"rendered={out_info.n_audio_streams})")
    if has_src_audio and has_out_audio \
            and abs(out_audio["rms_db"] - src_audio["rms_db"]) > 0.5:
        integrity_problems.append(
            f"audio RMS delta {abs(out_audio['rms_db'] - src_audio['rms_db']):.2f} dB")
    render_integrity = {
        "source": {"path": str(source_p), "width": src_info.width, "height": src_info.height,
                   "fps": src_info.fps, "frames": src_frames_n,
                   "duration_s": round(src_info.duration, 6),
                   "sha256": media.sha256_file(source_p)},
        "rendered": {"path": str(rendered_p), "width": out_info.width, "height": out_info.height,
                     "fps": out_info.fps, "frames": out_frames_n,
                     "duration_s": round(out_info.duration, 6),
                     "sha256": media.sha256_file(rendered_p)},
        "duration_delta_s": round(duration_delta, 6),
        "audio_rms_delta_db": round(abs(out_audio["rms_db"] - src_audio["rms_db"]), 3),
        "problems": integrity_problems,
        "verdict": FAIL if integrity_problems else PASS,
    }

    if src_frames_n and out_frames_n:
        n_frames = min(src_frames_n, out_frames_n)
    elif out_frames_n:
        n_frames = out_frames_n
    elif src_frames_n:
        n_frames = src_frames_n
    else:
        n_frames = 0
    if n_frames <= 0:
        raise Verify2Error(f"cannot determine frame count (source={src_frames_n}, "
                           f"rendered={out_frames_n})")
    if integrity_problems and (src_info.width != width or out_info.width != width
                               or src_info.height != height or out_info.height != height):
        # Pixels cannot be compared at different geometry: fail closed, no guess.
        metrics = {
            "schema": SCHEMA, "verdict": FAIL,
            "render_integrity": render_integrity,
            "checks": {"render_integrity": FAIL},
            "notes": ["pixel checks skipped: geometry mismatch makes every pixel metric invalid"],
        }
        return _finish(metrics, out_dir, write_metrics)

    # ---- pass 1: footprint, occlusion, text, timing counts
    elements_by_id = {str(e.get("id")): e for e in elements}
    boxes: dict[str, list[int]] = {}
    backdrop_boxes: dict[str, list[int]] = {}
    for el in elements:
        box = element_box(el)
        if box is not None:
            x, y, w, h = clip_box(box, width, height)
            boxes[str(el.get("id"))] = [x, y, w, h]
        bd = element_backdrop(el)
        if bd is not None:
            bx, by, bw, bh = clip_box(bd["box"], width, height)
            if bw > 0 and bh > 0:
                backdrop_boxes[str(el.get("id"))] = [bx, by, bw, bh]
    count_series = {eid: np.zeros(n_frames, dtype=np.int64) for eid in boxes}
    ink_series = {eid: np.zeros(n_frames, dtype=np.int64) for eid in boxes}
    ink_meta: dict[str, dict[str, Any]] = {}
    ink_mode = "ink_pass" if ink is not None else (
        "overlay_heuristic" if overlay is not None else None)
    union_fp = {eid: np.zeros((height, width), dtype=bool) for eid in boxes}
    per_element: dict[str, dict[str, Any]] = {}
    for eid in boxes:
        per_element[eid] = {
            "id": eid, "type": elements_by_id[eid].get("type"),
            "box": boxes[eid],
            "window": {k: (elements_by_id[eid].get("window") or {}).get(k)
                       for k in ("start_s", "end_s", "start_frame", "end_frame",
                                 "anchor_start_s", "anchor_end_s")},
            "footprint_visible_frames": 0, "footprint_max_frame_px": 0,
            "footprint_union_px": 0, "footprint_bbox": None,
            "person_overlap_px": 0, "max_person_frame_fraction": 0.0,
            "max_person_fraction": 0.0, "worst_person_frame": None,
            "face_overlap_px": 0, "face_overlap_frames": [],
            "text_overlap_px": 0, "text_overlap_frames": [],
            "hand_overlap_px": 0, "hand_overlap_frames": [],
            "occlusion_verdict": PASS, "text_verdict": PASS,
            "scrim_person_overlap_px": 0,
            "scrim_face_overlap_px": 0, "scrim_hand_overlap_px": 0,
            "scrim_max_person_frame_fraction": 0.0,
            "scrim_worst_frame": None, "scrim_visible_frames": 0,
            "scrim_text_overlap_px": 0,
        }
    text_regions, text_regions_source = _protected_text_regions(ir, mask_doc, fps)
    person_checked = bool(mask_doc and mask_doc.has_person_mask)
    text_checked = text_regions_source is not None
    global_worst = {"frame": None, "element": None, "person_overlap_px": 0,
                    "person_frame_fraction": 0.0, "person_person_fraction": 0.0}
    global_face_worst = {"frame": None, "element": None, "face_overlap_px": 0}
    text_worst = {"frame": None, "element": None, "overlap_px": 0, "frames": 0}
    pair_stats: dict[tuple[str, str], dict[str, Any]] = {}
    pair_frames = 0
    pair_ids = [eid for eid in boxes]
    pair_candidates = []
    for i, a in enumerate(pair_ids):
        for b in pair_ids[i + 1:]:
            wa = elements_by_id[a].get("window") or {}
            wb = elements_by_id[b].get("window") or {}
            if int(wa.get("start_frame", 0)) < int(wb.get("end_frame", 0)) and \
                    int(wb.get("start_frame", 0)) < int(wa.get("end_frame", 0)):
                pair_candidates.append((a, b))

    windows = []
    for el in elements:
        w = el.get("window") or {}
        try:
            windows.append((int(w.get("start_frame", 0)), int(w.get("end_frame", 0))))
        except (TypeError, ValueError):
            continue
    noise_frames: list[int] = []
    for i in range(n_frames):
        if not any(sf - 2 <= i < ef + 2 for sf, ef in windows):
            noise_frames.append(i)
            if len(noise_frames) >= 40:
                break
    noise_samples: list[float] = []
    if overlay is None and noise_frames:
        picked = noise_frames[:: max(1, len(noise_frames) // 12)][:12]
        src_n = _decode_selected(source_p, width, height, picked)
        out_n = _decode_selected(rendered_p, width, height, picked)
        for f in picked:
            if f in src_n and f in out_n:
                d = np.abs(out_n[f].astype(np.int16) - src_n[f].astype(np.int16)).max(axis=2)
                noise_samples.append(float(np.percentile(d, 99.9)))
    threshold = (OVERLAY_FOOTPRINT_THRESHOLD if overlay is not None
                 else _footprint_threshold(noise_samples, footprint_threshold))

    it_src = _iter_rgb_frames(source_p, width, height)
    it_out = _iter_rgb_frames(rendered_p, width, height)
    it_ovl = (_iter_rgb_frames(Path(overlay), width, height)
              if overlay is not None else None)
    it_ink = (_iter_rgb_frames(Path(ink), width, height)
              if ink is not None else None)
    used_frames = 0
    try:
        for frame_idx, (src_rgb, out_rgb) in enumerate(zip(it_src, it_out)):
            if frame_idx >= n_frames:
                break
            used_frames += 1
            ovl_rgb = next(it_ovl) if it_ovl is not None else None
            ink_rgb = next(it_ink) if it_ink is not None else None
            if ovl_rgb is not None:
                fp = overlay_footprint(ovl_rgb, threshold)
            else:
                fp = diff_footprint(out_rgb, src_rgb, threshold)
            person = face_rects = hand_rects = None
            if person_checked:
                person, face_rects, hand_rects, _grid = mask_doc.at(  # type: ignore[union-attr]
                    frame_idx, fps)
            for eid, (bx, by, bw, bh) in boxes.items():
                wnd = elements_by_id[eid].get("window") or {}
                if not (int(wnd.get("start_frame", 0)) <= frame_idx
                        < int(wnd.get("end_frame", 0))):
                    continue
                sub = fp[by:by + bh, bx:bx + bw]
                count = int(sub.sum())
                count_series[eid][frame_idx] = count
                if count > 0:
                    per_element[eid]["footprint_visible_frames"] += 1
                    per_element[eid]["footprint_max_frame_px"] = max(
                        per_element[eid]["footprint_max_frame_px"], count)
                    union_fp[eid][by:by + bh, bx:bx + bw] |= sub
                if person_checked and count > 0:
                    occ = occlusion_metrics(fp, boxes[eid], person, face_rects or [],
                                            width * height)
                    per_element[eid]["person_overlap_px"] += occ["person_overlap_px"]
                    per_element[eid]["face_overlap_px"] += occ["face_overlap_px"]
                    if occ["person_frame_fraction"] > per_element[eid]["max_person_frame_fraction"]:
                        per_element[eid]["max_person_frame_fraction"] = occ["person_frame_fraction"]
                        per_element[eid]["worst_person_frame"] = frame_idx
                    per_element[eid]["max_person_fraction"] = max(
                        per_element[eid]["max_person_fraction"], occ["person_person_fraction"])
                    if occ["person_overlap_px"] > global_worst["person_overlap_px"]:
                        global_worst.update({"frame": frame_idx, "element": eid,
                                             "person_overlap_px": occ["person_overlap_px"],
                                             "person_frame_fraction": occ["person_frame_fraction"],
                                             "person_person_fraction": occ["person_person_fraction"]})
                    if occ["face_overlap_px"] > 0:
                        per_element[eid]["face_overlap_frames"].append(frame_idx)
                        if occ["face_overlap_px"] > global_face_worst["face_overlap_px"]:
                            global_face_worst.update({"frame": frame_idx, "element": eid,
                                                      "face_overlap_px": occ["face_overlap_px"]})
                    hand_px = rects_overlap_px(fp, boxes[eid], hand_rects or [])
                    if hand_px > 0:
                        per_element[eid]["hand_overlap_px"] += hand_px
                        per_element[eid]["hand_overlap_frames"].append(frame_idx)
                if text_checked and count > 0:
                    t = frame_idx / fps
                    active = [r for r in text_regions if r["t0"] <= t <= r["t1"]]
                    if active:
                        text_mask = np.zeros((bh, bw), dtype=bool)
                        for r in active:
                            rx, ry, rw, rh = r["bbox"]
                            x0t, y0t = max(bx, rx), max(by, ry)
                            x1t, y1t = min(bx + bw, rx + rw), min(by + bh, ry + rh)
                            if x1t > x0t and y1t > y0t:
                                text_mask[y0t - by:y1t - by, x0t - bx:x1t - bx] = True
                        overlap = int(np.logical_and(sub, text_mask).sum())
                        if overlap:
                            per_element[eid]["text_overlap_px"] += overlap
                            per_element[eid]["text_overlap_frames"].append(frame_idx)
                            if overlap > text_worst["overlap_px"]:
                                text_worst.update({"frame": frame_idx, "element": eid,
                                                   "overlap_px": overlap})
                wnd_sf = int(wnd.get("start_frame", 0))
                if frame_idx >= wnd_sf - 2:
                    if ink_rgb is not None:
                        # deterministic ink-only pass: glyphs on black, no scrim
                        bxi, byi, bwi, bhi = boxes[eid]
                        crop_i = ink_rgb[byi:byi + bhi, bxi:bxi + bwi].astype(np.int16)
                        lum_i = (0.2126 * crop_i[..., 0] + 0.7152 * crop_i[..., 1]
                                 + 0.0722 * crop_i[..., 2])
                        ink_series[eid][frame_idx] = int((lum_i > 12.0).sum())
                        if eid not in ink_meta:
                            ink_meta[eid] = {"ink_pass": str(ink)}
                    elif ovl_rgb is not None:
                        ink_colour = element_ink(elements_by_id[eid], ir)
                        ink_px, bg_luma = ink_pixel_count(ovl_rgb, boxes[eid], ink_colour)
                        ink_series[eid][frame_idx] = ink_px
                        if eid not in ink_meta:
                            ink_meta[eid] = {"ink_rgb": list(ink_colour),
                                             "scrim_luma_at_first": bg_luma}
                bd_box = backdrop_boxes.get(eid)
                if bd_box is not None:
                    bdx, bdy, bdw, bdh = bd_box
                    bd_count = int(fp[bdy:bdy + bdh, bdx:bdx + bdw].sum())
                    if bd_count > 0:
                        per_element[eid]["scrim_visible_frames"] += 1
                        if person_checked:
                            occ_bd = occlusion_metrics(fp, bd_box, person,
                                                       face_rects or [], width * height)
                            per_element[eid]["scrim_person_overlap_px"] += \
                                occ_bd["person_overlap_px"]
                            per_element[eid]["scrim_max_person_frame_fraction"] = max(
                                per_element[eid]["scrim_max_person_frame_fraction"],
                                occ_bd["person_frame_fraction"])
                            if occ_bd["person_overlap_px"] > 0 and \
                                    per_element[eid]["scrim_worst_frame"] is None:
                                per_element[eid]["scrim_worst_frame"] = frame_idx
                            per_element[eid]["scrim_face_overlap_px"] += \
                                rects_overlap_px(fp, bd_box, face_rects or [])
                            per_element[eid]["scrim_hand_overlap_px"] += \
                                rects_overlap_px(fp, bd_box, hand_rects or [])
                        if text_checked:
                            t = frame_idx / fps
                            active = [r for r in text_regions if r["t0"] <= t <= r["t1"]]
                            bd_text = np.zeros((bdh, bdw), dtype=bool)
                            for r in active:
                                rx, ry, rw, rh = r["bbox"]
                                x0t, y0t = max(bdx, rx), max(bdy, ry)
                                x1t, y1t = min(bdx + bdw, rx + rw), min(bdy + bdh, ry + rh)
                                if x1t > x0t and y1t > y0t:
                                    bd_text[y0t - bdy:y1t - bdy, x0t - bdx:x1t - bdx] = True
                            if bd_text.any():
                                per_element[eid]["scrim_text_overlap_px"] += int(
                                    np.logical_and(fp[bdy:bdy + bdh, bdx:bdx + bdw],
                                                   bd_text).sum())
            # rendered footprint collisions (lead rule: >0.5 % of the smaller box)
            active = [eid for eid, (bx, by, bw, bh) in boxes.items()
                      if int((elements_by_id[eid].get("window") or {}).get("start_frame", 0))
                      <= frame_idx
                      < int((elements_by_id[eid].get("window") or {}).get("end_frame", 0))]
            for a, b in pair_candidates:
                if a not in active or b not in active:
                    continue
                ba, bb_ = boxes[a], boxes[b]
                ix0 = max(ba[0], bb_[0])
                iy0 = max(ba[1], bb_[1])
                ix1 = min(ba[0] + ba[2], bb_[0] + bb_[2])
                iy1 = min(ba[1] + ba[3], bb_[1] + bb_[3])
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                overlap_n = int(fp[iy0:iy1, ix0:ix1].sum())
                smaller = min(ba[2] * ba[3], bb_[2] * bb_[3]) or 1
                ratio = overlap_n / float(smaller)
                pair_frames += 1
                st = pair_stats.setdefault((a, b), {"max_overlap_px": 0, "max_ratio": 0.0,
                                                    "worst_frame": None, "frames": 0})
                st["frames"] += 1
                if overlap_n > st["max_overlap_px"]:
                    st["max_overlap_px"] = overlap_n
                if ratio > st["max_ratio"]:
                    st["max_ratio"] = ratio
                    st["worst_frame"] = frame_idx
    finally:
        for it in (it_src, it_out, it_ovl, it_ink):
            if it is not None:
                it.close()

    s_n = int(min(used_frames, n_frames or used_frames))
    if s_n:
        for eid in count_series:
            count_series[eid] = count_series[eid][:s_n]
    for eid, fp in union_fp.items():
        ys, xs = np.nonzero(fp)
        if xs.size:
            per_element[eid]["footprint_bbox"] = [int(xs.min()), int(ys.min()),
                                                  int(xs.max() - xs.min() + 1),
                                                  int(ys.max() - ys.min() + 1)]
            per_element[eid]["footprint_union_px"] = int(xs.size)

    # ---- thresholds for occlusion + text, per element
    occlusion_rows = []
    for eid, row in per_element.items():
        verdict = PASS
        problems = []
        lower_third = is_lower_third(elements_by_id[eid])
        policy = "lower_third" if lower_third else "silhouette"
        if not person_checked:
            verdict = NOT_RUN
            problems.append("no person mask document supplied (capability person_mask=false)")
        else:
            if row["face_overlap_px"] > FACE_MAX_PX:
                verdict = FAIL
                problems.append(f"face overlap {row['face_overlap_px']} px > {FACE_MAX_PX}")
            if row["hand_overlap_px"] > FACE_MAX_PX:
                verdict = FAIL
                problems.append(f"hand overlap {row['hand_overlap_px']} px > {FACE_MAX_PX}")
            if not lower_third:
                if row["max_person_frame_fraction"] > PERSON_FRAME_MAX:
                    verdict = FAIL
                    problems.append(f"person overlap "
                                    f"{row['max_person_frame_fraction']:.5f} of frame "
                                    f"> {PERSON_FRAME_MAX}")
                if row["max_person_fraction"] > PERSON_PERSON_MAX:
                    verdict = FAIL
                    problems.append(f"person overlap {row['max_person_fraction']:.5f} of person "
                                    f"> {PERSON_PERSON_MAX}")
        row["occlusion_verdict"] = verdict
        row["occlusion_policy"] = policy
        row["occlusion_problems"] = problems
        text_verdict = PASS
        text_problems: list[str] = []
        if not text_checked:
            text_verdict = NOT_RUN
            text_problems.append("no protected-text source (IR perception and mask "
                                 "document both absent)")
        elif row["text_overlap_px"] > TEXT_MAX_PX:
            text_verdict = FAIL
            text_problems.append(f"footprint covers {row['text_overlap_px']} px of burned-in text")
        row["text_verdict"] = text_verdict
        row["text_problems"] = text_problems
        occlusion_rows.append({"id": eid, "verdict": verdict, "problems": problems,
                               "policy": row["occlusion_policy"],
                               "face_overlap_px": row["face_overlap_px"],
                               "hand_overlap_px": row["hand_overlap_px"],
                               "max_person_frame_fraction": row["max_person_frame_fraction"],
                               "max_person_fraction": row["max_person_fraction"],
                               "worst_person_frame": row["worst_person_frame"]})

    occlusion_check = {
        "verdict": _worst([r["verdict"] for r in occlusion_rows]),
        "face_overlap_px_total": sum(r["face_overlap_px"] for r in per_element.values()),
        "face_threshold_px": FACE_MAX_PX,
        "max_person_frame_fraction": max((r["max_person_frame_fraction"]
                                          for r in per_element.values()), default=0.0),
        "max_person_fraction": max((r["max_person_fraction"]
                                    for r in per_element.values()), default=0.0),
        "person_frame_threshold": PERSON_FRAME_MAX,
        "person_person_threshold": PERSON_PERSON_MAX,
        "worst_frame": global_worst["frame"],
        "worst_element": global_worst["element"],
        "worst_frame_face": global_face_worst["frame"],
        "worst_element_face": global_face_worst["element"],
        "by_element": occlusion_rows,
        "mask": None if mask_doc is None else {
            "path": str(mask_doc.path), "backend": mask_doc.backend,
            "capabilities": mask_doc.document.get("capabilities"),
            "sample_hz": mask_doc.document.get("sample_hz"),
            "grid": mask_doc._mask_size(),
            "honesty": ("mask producer is external; verify2 does not validate its "
                        "segmentation quality, only consumes it"),
        },
        "method": ("footprint ∩ person mask on the mask's native grid (max-pooled, "
                   "conservative); face mask = face bounding box, exact pixels"),
    }

    text_check = {
        "verdict": _worst([r["text_verdict"] for r in per_element.values()]),
        "regions": len(text_regions),
        "regions_source": text_regions_source,
        "max_overlap_px": max((r["text_overlap_px"] for r in per_element.values()), default=0),
        "threshold_px": TEXT_MAX_PX,
        "worst_frame": text_worst["frame"], "worst_element": text_worst["element"],
        "by_element": [{"id": eid, "verdict": r["text_verdict"],
                        "overlap_px": r["text_overlap_px"],
                        "frames": r["text_overlap_frames"][:20]}
                       for eid, r in per_element.items()],
        "honesty": ("0 regions with an IR-filtered list is a measured pass (no protected "
                    "text); no protected-text source at all is NOT RUN"),
    }

    # ---- pass 2: legibility on selected frames
    legibility = _legibility(ir, rendered_p, source_p, width, height, fps, boxes,
                             count_series, per_element)
    legibility_check = {
        "verdict": legibility["verdict"],
        "min_contrast": legibility["min_contrast"],
        "offending_element": legibility["offending_element"],
        "threshold": MIN_CONTRAST,
        "frames_sampled": legibility["frames_sampled"],
        "by_element": legibility["by_element"],
        "method": ("glyph pixels = overlay pixels close to style.color; background = "
                   "rendered pixels around them (scrim included); WCAG ratio per "
                   f"{GLYPH_BLOCK}px block, minimum block wins"),
    }

    # ---- strict first-ink-frame timing (product promise: word on the anchor frame)
    text_onset = _text_onset_check(
        elements_by_id, boxes, ink_series, ink_meta, s_n, fps,
        ink_mode=ink_mode, ink_source=str(ink) if ink is not None else None)

    # ---- displayed text (OCR of the overlay-only pass at the mid-hold frame)
    displayed = _displayed_text_check(elements_by_id, boxes, overlay, width, height, fps)

    # ---- scrim / band backdrops: alpha region must not touch the person
    scrim_ids = [eid for eid in boxes
                 if backdrop_boxes.get(eid) is not None
                 and is_scrim_backdrop((elements_by_id[eid].get("backdrop") or {}).get("kind"))]
    scrim_rows: list[dict[str, Any]] = []
    for eid in scrim_ids:
        row = per_element[eid]
        bd = elements_by_id[eid].get("backdrop") or {}
        leg = (legibility["by_element"].get(eid) or {})
        contrast = leg.get("min_contrast")
        lower_third = is_lower_third(elements_by_id[eid])
        problems = []
        verdict = PASS
        if not person_checked:
            verdict = NOT_RUN
            problems.append("no person mask document supplied")
        elif lower_third:
            # lower_third may sit over clothing, never over face or hands
            if row["scrim_face_overlap_px"] > 0:
                verdict = FAIL
                problems.append(f"scrim covers {row['scrim_face_overlap_px']} px of the face")
            if row["scrim_hand_overlap_px"] > 0:
                verdict = FAIL
                problems.append(f"scrim covers {row['scrim_hand_overlap_px']} px of the hands")
        elif row["scrim_person_overlap_px"] > 0:
            verdict = FAIL
            problems.append(f"scrim alpha region covers {row['scrim_person_overlap_px']} px "
                            f"of the person mask")
        if row["scrim_text_overlap_px"] > 0:
            verdict = FAIL
            problems.append(f"scrim covers {row['scrim_text_overlap_px']} px of burned-in "
                            f"text (existing_text overlap must be 0)")
        if contrast is None:
            problems.append("text contrast not measurable for this element")
            if verdict == PASS:
                verdict = NOT_RUN
        elif contrast < MIN_CONTRAST:
            verdict = FAIL
            problems.append(f"text contrast {contrast:.2f} < {MIN_CONTRAST} on the "
                            f"rendered frame")
        scrim_rows.append({
            "id": eid, "kind": bd.get("kind"), "alpha": bd.get("alpha"),
            "backdrop_box": backdrop_boxes.get(eid),
            "visible_frames": row["scrim_visible_frames"],
            "person_overlap_px": row["scrim_person_overlap_px"],
            "face_overlap_px": row["scrim_face_overlap_px"],
            "hand_overlap_px": row["scrim_hand_overlap_px"],
            "policy": "lower_third (body allowed)" if lower_third else "silhouette",
            "body_overlap_allowed": bool(lower_third),
            "max_person_frame_fraction": round(row["scrim_max_person_frame_fraction"], 6),
            "worst_person_frame": row["scrim_worst_frame"],
            "text_overlap_px": row["scrim_text_overlap_px"],
            "text_contrast": contrast,
            "text_contrast_vs_source": leg.get("contrast_vs_source"),
            "verdict": verdict, "problems": problems,
        })
    scrim_verdicts = [r["verdict"] for r in scrim_rows]
    scrim_check = {
        "verdict": (FAIL if FAIL in scrim_verdicts
                    else (PASS if scrim_rows else NOT_RUN)),
        "elements": len(scrim_rows),
        "by_element": scrim_rows,
        "method": ("backdrop kind scrim*/band*: rendered footprint inside the projected "
                   "backdrop box ∩ person mask must be 0 px; text contrast is measured "
                   "on rendered pixels (legibility check), not on the scrim colour"),
        "honesty": ("0 elements means the IR declares no scrim/band backdrop; that is "
                    "reported as NOT RUN, not as a pass"),
    }

    # ---- layout
    footprint_collision_rows = [
        {"a": a, "b": b, "max_overlap_px": st["max_overlap_px"],
         "max_ratio": round(st["max_ratio"], 6),
         "worst_frame": st["worst_frame"], "frames": st["frames"]}
        for (a, b), st in sorted(pair_stats.items())
    ]
    violating_collisions = [r for r in footprint_collision_rows
                            if r["max_ratio"] > FOOTPRINT_COLLISION_RATIO]
    layout = layout_checks(
        ir, width, height, safe_margin_px=safe_margin_px,
        footprint_bboxes={eid: r["footprint_bbox"] for eid, r in per_element.items()},
        footprint_collisions=violating_collisions)

    # ---- timing
    timing_rows = []
    for eid, row in per_element.items():
        el = elements_by_id[eid]
        w = el.get("window") or {}
        anchor_s = w.get("anchor_start_s")
        anchor = None
        if anchor_s is None:
            anchor_id = el.get("anchor_ref")
            for a in ((ir.get("semantic") or {}).get("anchors") or []):
                if a.get("id") == anchor_id:
                    anchor = float(a.get("start_s")) if a.get("start_s") is not None else None
                    break
        else:
            anchor = float(anchor_s)
        box_area = row["box"][2] * row["box"][3]
        sf = int(w.get("start_frame", 0))
        pre = list(count_series[eid][max(0, sf - 8):max(0, sf - 1)])
        t_row = timing_series_row(count_series[eid], w, fps, anchor, s_n, box_area,
                                  pre_counts=pre)
        row["timing"] = t_row
        timing_rows.append({"id": eid, **t_row})
    timing_check = {
        "verdict": _worst([r["verdict"] for r in timing_rows]),
        "tolerance_frames": TIMING_TOLERANCE_FRAMES,
        "frame_quantum_ms": round(1000.0 / fps, 2),
        "by_element": timing_rows,
        "method": ("appearance = first frame whose in-box footprint exceeds max(0.5 % "
                   "of the box, 3x pre-window noise); disappearance = last visible + 1"),
    }

    # ---- determinism
    if second_render is not None:
        det = compare_renders(rendered_p, second_render, width, height)
    else:
        det = {"verdict": NOT_RUN, "note": "no second render supplied",
               "tolerance_mean": DET_MEAN_TOL, "tolerance_max": DET_MAX_TOL}

    footprint_check = {
        "verdict": _worst([PASS if r["footprint_visible_frames"] > 0 else FAIL
                           for r in per_element.values()]),
        "threshold_pixel_delta": threshold,
        "noise_floor_p99_9": round(float(np.median(noise_samples)), 4) if noise_samples else None,
        "noise_frames_sampled": len(noise_samples),
        "overlay_only": overlay is not None,
        "by_element": [{"id": eid, "union_px": r["footprint_union_px"],
                        "max_frame_px": r["footprint_max_frame_px"],
                        "visible_frames": r["footprint_visible_frames"],
                        "bbox": r["footprint_bbox"]}
                       for eid, r in per_element.items()],
        "concurrent_pairs_evaluated": pair_frames,
        "concurrent_footprint_collisions": footprint_collision_rows,
    }

    checks = {
        "render_integrity": render_integrity,
        "footprint": footprint_check,
        "person_occlusion": occlusion_check,
        "existing_text_collision": text_check,
        "legibility": legibility_check,
        "text_onset": text_onset,
        "displayed_text": displayed,
        "scrim": scrim_check,
        "layout": layout,
        "timing": timing_check,
        "determinism": det,
    }
    verdicts = {k: (v.get("verdict") if isinstance(v, dict) else v) for k, v in checks.items()}
    overall = FAIL if FAIL in verdicts.values() else PASS
    failing_elements = {}
    layout_by_id = {str(r.get("id")): r for r in layout["elements"]}
    displayed_by_id = displayed["by_element"]
    onset_by_id = text_onset["by_element"]
    scrim_by_id = {str(r["id"]): r for r in scrim_rows}
    for eid, row in per_element.items():
        el_checks = [
            row["occlusion_verdict"], row["text_verdict"],
            legibility["by_element"].get(eid, {}).get("verdict", NOT_RUN),
            (layout_by_id.get(eid) or {}).get("verdict", NOT_RUN),
            (row.get("timing") or {}).get("verdict", NOT_RUN),
            (displayed_by_id.get(eid) or {}).get("verdict", NOT_RUN),
            (onset_by_id.get(eid) or {}).get("verdict", NOT_RUN),
            (scrim_by_id.get(eid) or {}).get("verdict", NOT_RUN),
        ]
        # NOT RUN is neutral for the colour of the box (it is still listed per
        # check in the JSON); only a FAIL paints the element red.
        el_verdict = FAIL if FAIL in el_checks else PASS
        row["verdict"] = el_verdict
        if el_verdict == FAIL:
            failing_elements[eid] = {
                "occlusion": row["occlusion_verdict"], "text": row["text_verdict"],
                "legibility": legibility["by_element"].get(eid, {}).get("verdict"),
                "timing": (row.get("timing") or {}).get("verdict"),
                "displayed_text": (displayed_by_id.get(eid) or {}).get("verdict"),
                "text_onset": (onset_by_id.get(eid) or {}).get("verdict"),
                "scrim": (scrim_by_id.get(eid) or {}).get("verdict"),
                "layout": (layout_by_id.get(eid) or {}).get("verdict"),
            }
    elements_out = []
    for eid, row in per_element.items():
        row.pop("occlusion_problems", None)
        row.pop("text_problems", None)
        elements_out.append({
            "id": eid, "type": row["type"], "box": row["box"], "window": row["window"],
            "verdict": row["verdict"],
            "footprint": {"union_px": row["footprint_union_px"],
                          "max_frame_px": row["footprint_max_frame_px"],
                          "visible_frames": row["footprint_visible_frames"],
                          "bbox": row["footprint_bbox"]},
            "occlusion": {"verdict": row["occlusion_verdict"],
                          "policy": row["occlusion_policy"],
                          "face_overlap_px": row["face_overlap_px"],
                          "face_overlap_frames": row["face_overlap_frames"][:20],
                          "hand_overlap_px": row["hand_overlap_px"],
                          "hand_overlap_frames": row["hand_overlap_frames"][:20],
                          "person_overlap_px": row["person_overlap_px"],
                          "max_person_frame_fraction": round(row["max_person_frame_fraction"], 6),
                          "max_person_fraction": round(row["max_person_fraction"], 6),
                          "worst_person_frame": row["worst_person_frame"]},
            "existing_text": {"verdict": row["text_verdict"],
                              "overlap_px": row["text_overlap_px"],
                              "frames": row["text_overlap_frames"][:20]},
            "legibility": legibility["by_element"].get(eid, {"verdict": NOT_RUN}),
            "displayed_text": displayed_by_id.get(eid, {"verdict": NOT_RUN}),
            "text_onset": onset_by_id.get(eid, {"verdict": NOT_RUN}),
            "scrim": scrim_by_id.get(eid),
            "layout": layout_by_id.get(eid),
            "timing": row.get("timing"),
        })
    metrics = {
        "schema": SCHEMA,
        "verdict": overall,
        "checks": checks,
        "verdicts": verdicts,
        "failing_elements": failing_elements,
        "elements": elements_out,
        "inputs": {
            "source": str(source_p), "rendered": str(rendered_p),
            "overlay_only": str(overlay) if overlay else None,
            "ink_only": str(ink) if ink else None,
            "second_render": str(second_render) if second_render else None,
            "masks": str(mask_doc.path) if mask_doc else None,
            "ir_sha256": (ir.get("provenance") or {}).get("ir_sha256"),
        },
        "thresholds": {
            "person_frame_max": PERSON_FRAME_MAX, "person_person_max": PERSON_PERSON_MAX,
            "face_max_px": FACE_MAX_PX, "text_max_px": TEXT_MAX_PX,
            "min_contrast": MIN_CONTRAST,
            "timing_tolerance_frames": TIMING_TOLERANCE_FRAMES,
            "safe_margin_px": safe_margin_px,
            "collision_area_ratio": COLLISION_AREA_RATIO,
            "footprint_margin_frac": FOOTPRINT_MARGIN_FRAC,
            "footprint_collision_ratio": FOOTPRINT_COLLISION_RATIO,
            "displayed_text": "IR text must be a subsequence of normalised overlay OCR; "
                              "ellipsis = FAIL",
            "font_floor_frac": FONT_FLOOR_FRAC,
            "font_floor": "IR fit.min_font_px is authoritative; legacy v1 floor is a hard "
                          "check for style.font_size_px; otherwise a warning",
            "lower_third": "face/hand overlap = 0; body overlap reported, not a failure",
            "text_onset_tolerance_frames": TIMING_TOLERANCE_FRAMES,
            "text_onset": "first ink pixel frame on the overlay-only pass must be within "
                          "1 frame of the window start",
            "determinism_mean": DET_MEAN_TOL, "determinism_max": DET_MAX_TOL,
        },
        "notes": [
            "every number is measured on decoded frames of the delivered MP4",
            "face mask = face bounding box (conservative); person mask comes from the "
            "perception contract and is not re-validated by verify2",
            "contrast is measured against the rendered local background (scrim included); "
            "contrast_vs_source is reported as an honesty check",
            "determinism tolerance is explicit: mean |Δ| and max |Δ| on decoded frames",
            "displayed text is OCR of the isolated overlay pass (mid-hold); OCR errors "
            "are possible and ocr_text is included for inspection",
            "rendered footprint must keep 2.5 % of canvas width/height from the edges and "
            "two elements' footprints may overlap at most 0.5 % of the smaller box",
            "scrim/band backdrops are checked separately: alpha region ∩ person must be 0 "
            "(lower_third: face/hands only) and text contrast is measured on rendered pixels",
            "text_onset is strict (ink, not scrim): overlay-only pass required; a dark scrim "
            "does not satisfy it",
            "protected burned-in text comes from ir.perception.existing_text when present "
            "(design2 filters venue signage); otherwise from the mask document",
        ],
        "tool_versions": media.tool_versions(),
    }
    metrics["contact_sheet"] = None
    if make_sheet:
        sheet_frames = _sheet_frames(elements, per_element, global_worst, layout)
        try:
            sheet_path, crop_path = _write_sheets(
                (Path(out_dir) if out_dir else Path(".")), elements, sheet_frames,
                width, height, rendered_p, mask_doc, fps,
                {eid: r["verdict"] for eid, r in per_element.items()},
                overall, global_worst)
            metrics["contact_sheet"] = str(sheet_path)
            metrics["worst_crop"] = str(crop_path) if crop_path else None
        except Exception as exc:  # a sheet failure must not hide the numbers
            metrics["contact_sheet_error"] = f"{type(exc).__name__}: {exc}"
    return _finish(metrics, out_dir, write_metrics)


def _footprint_threshold(noise_samples: Sequence[float], override: int | None) -> float:
    if override is not None:
        return float(override)
    if not noise_samples:
        return float(FOOTPRINT_MIN_THRESHOLD)
    floor = float(np.median(noise_samples))
    return float(max(FOOTPRINT_MIN_THRESHOLD,
                     min(FOOTPRINT_MAX_THRESHOLD, floor * 1.5 + 4.0)))


def _text_onset_check(elements_by_id: dict[str, dict], boxes: dict[str, list[int]],
                      ink_series: dict[str, np.ndarray], ink_meta: dict[str, dict],
                      n_frames: int, fps: float, *,
                      ink_mode: str | None,
                      ink_source: str | None = None,
                      tolerance_frames: int = TIMING_TOLERANCE_FRAMES) -> dict[str, Any]:
    """Strict first-ink-frame check.

    Preferred input is the renderer's ``final_ink.mp4`` pass (glyphs only on
    black, no scrim/captions): first frame in [window_start-2, window_end) whose
    element box has more than 32 pixels with luminance > 12. FAIL when that frame
    is later than ``window_start + 1``; early ink is reported as a warning.
    Without the ink pass a colour/luminance heuristic on the overlay pass is
    reported for diagnosis, but the check is NOT RUN (no deterministic source).
    """
    rows: dict[str, dict[str, Any]] = {}
    for eid, el in elements_by_id.items():
        if eid not in boxes:
            continue
        text = element_text(el).strip()
        if not text:
            rows[eid] = {"verdict": NOT_RUN, "note": "element has no text"}
            continue
        w = el.get("window") or {}
        try:
            sf = int(w.get("start_frame", 0))
            ef = int(w.get("end_frame", 0))
        except (TypeError, ValueError):
            rows[eid] = {"verdict": NOT_RUN, "note": "window frames missing"}
            continue
        meta = ink_meta.get(eid) or {}
        if ink_mode not in ("ink_pass", "overlay_heuristic"):
            rows[eid] = {"verdict": NOT_RUN,
                         "note": "no ink-only pass and no overlay pass supplied"}
            continue
        lo = max(0, sf - 2)
        hi = min(n_frames, ef)
        series = ink_series.get(eid)
        if series is None or lo >= hi:
            rows[eid] = {"verdict": NOT_RUN, "note": "no frames to measure"}
            continue
        gate = 32 if ink_mode == "ink_pass" else max(12, int(round(
            0.0005 * boxes[eid][2] * boxes[eid][3])))
        hits = [f for f in range(lo, min(hi, len(series))) if int(series[f]) >= gate]
        diagnostic = ink_mode == "overlay_heuristic"
        method = ("ink-only pass (final_ink.mp4): element-box pixels with luma > 12, "
                  "gate 32 px" if ink_mode == "ink_pass" else
                  "overlay-pass heuristic: glyph-coloured pixels above scrim luminance "
                  "(diagnostic only)")
        base = {"window_start_frame": sf,
                "anchor_onset_frame": (int(round(float(w.get("anchor_start_s")) * fps))
                                       if w.get("anchor_start_s") is not None else None),
                "gate_pixels": gate,
                "ink_source": ink_source if ink_mode == "ink_pass" else None,
                "method": method}
        if not hits:
            rows[eid] = {**base,
                         "text_onset_frame": None,
                         "ink_onset_delta_frames": None,
                         "max_ink_pixels": int(series[lo:hi].max()) if hi > lo else 0,
                         "verdict": NOT_RUN if diagnostic else FAIL,
                         "note": ("heuristic found no ink (diagnostic)" if diagnostic
                                  else "no measurable ink anywhere in the window"),
                         "problems": [] if diagnostic else [
                             "no ink pixel in the element box across the window"]}
            continue
        onset = int(hits[0])
        delta = onset - sf
        warning = None
        if delta > tolerance_frames:
            verdict = NOT_RUN if diagnostic else FAIL
            problems = [f"first ink at frame {onset} vs window start {sf} "
                        f"({delta:+d} frames > {tolerance_frames}-frame tolerance)"]
        else:
            verdict = NOT_RUN if diagnostic else PASS
            problems = []
            if delta < -tolerance_frames:
                warning = (f"ink appears {-delta} frames before the window start")
        rows[eid] = {
            **base,
            "verdict": verdict,
            "text_onset_frame": onset,
            "ink_onset_delta_frames": int(delta),
            "ink_pixels_at_onset": int(series[onset]),
            "max_ink_pixels": int(series[lo:hi].max()) if hi > lo else 0,
            "warning": warning,
            "problems": problems,
            **({"note": "heuristic only; no final_ink.mp4 passed — verdict NOT RUN"}
               if diagnostic else
               {"ink_rgb": meta.get("ink_rgb"),
                "scrim_luma_at_first": meta.get("scrim_luma_at_first")}),
        }
    verdicts = [r.get("verdict") for r in rows.values()]
    return {
        "verdict": (FAIL if FAIL in verdicts
                    else (NOT_RUN if (not verdicts or NOT_RUN in verdicts) else PASS)),
        "tolerance_frames": tolerance_frames,
        "ink_mode": ink_mode,
        "ink_source": ink_source if ink_mode == "ink_pass" else None,
        "by_element": rows,
        "method": ("strict first-ink-frame measurement: renderer ink-only pass, "
                   "element-box luma > 12, count > 32 px; scrim/captions are not in "
                   "that pass, so a scrim can never satisfy this check"),
        "honesty": ("without final_ink.mp4 the overlay-pass heuristic is reported but "
                    "the verdict is NOT RUN"),
    }


def _displayed_text_check(elements_by_id: dict[str, dict], boxes: dict[str, list[int]],
                          overlay: str | Path | None, width: int, height: int,
                          fps: float) -> dict[str, Any]:
    """OCR the overlay-only pass and prove the delivered text is the IR text.

    Each element is sampled at 50 %, 70 % and 85 % of its window; the best
    matching frame wins. Multiple samples are needed because v2 row-by-row reveals
    legitimately show only a prefix at mid-hold — comparing the *full* IR string at
    one arbitrary frame would report a false truncation. A real truncation never
    matches at any sampled frame, and an ellipsis is still an explicit failure.
    """
    FRACTIONS = (0.5, 0.7, 0.85)
    plan: list[tuple[str, str, list[int]]] = []
    pre_rows: dict[str, dict[str, Any]] = {}
    for eid, el in elements_by_id.items():
        if eid not in boxes:
            continue
        text = element_text(el).strip()
        if not text:
            pre_rows[eid] = {"verdict": NOT_RUN, "note": "element has no text"}
            continue
        w = el.get("window") or {}
        try:
            sf, ef = int(w.get("start_frame", 0)), int(w.get("end_frame", 0))
        except (TypeError, ValueError):
            pre_rows[eid] = {"verdict": NOT_RUN, "note": "window frames missing"}
            continue
        span = max(1, ef - sf)
        frames = sorted({min(ef - 1, sf + int(round(span * frac)))
                         for frac in FRACTIONS})
        plan.append((eid, text, frames))
    if overlay is None:
        rows = {eid: {"verdict": NOT_RUN,
                      "note": "no overlay-only pass supplied; text equality needs the "
                              "isolated overlay layer"} for eid in elements_by_id
                if eid in boxes}
        rows.update(pre_rows)
        return {"verdict": NOT_RUN, "elements": len(plan),
                "reason": "no --overlay-only pass supplied", "by_element": rows,
                "method": "Vision OCR of the isolated overlay layer"}
    needed = sorted({f for _, _, frames in plan for f in frames})
    decoded = _decode_selected(Path(overlay), width, height, needed) if needed else {}
    rows: dict[str, dict[str, Any]] = {}
    for eid, text, frames in plan:
        x, y, w, h = boxes[eid]
        pad = 10
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(width, x + w + pad), min(height, y + h + pad)
        ir_norm = normalise_display_text(text)
        best: dict[str, Any] | None = None
        tried: list[dict[str, Any]] = []
        for mid in frames:
            rgb = decoded.get(mid)
            if rgb is None:
                tried.append({"frame": mid, "status": "not decoded"})
                continue
            crop = rgb[y0:y1, x0:x1]
            if crop.size == 0 or int(crop.max()) < 20:
                tried.append({"frame": mid, "status": "no overlay pixels"})
                continue
            if crop.shape[1] < 700:
                from PIL import Image as _Image
                scaled = _Image.fromarray(crop).resize(
                    (crop.shape[1] * 2, crop.shape[0] * 2), _Image.LANCZOS)
                crop = np.array(scaled)
            ocr = ocr_frame_text(crop)
            if not ocr.get("available"):
                rows[eid] = {"verdict": NOT_RUN, "ir_text": text,
                             "note": f"OCR unavailable: {ocr.get('reason', 'unknown')}"}
                break
            raw = str(ocr.get("text") or "")
            norm_ocr = normalise_display_text(raw)
            ellipsis = ("…" in raw) or ("..." in raw)
            matched = bool(ir_norm) and is_subsequence(ir_norm, norm_ocr)
            prefix = 0
            it = iter(norm_ocr)
            for ch in ir_norm:
                if ch in it:
                    prefix += 1
                else:
                    break
            tried.append({"frame": mid, "ocr_chars": len(norm_ocr),
                          "matched_chars": prefix, "ellipsis": ellipsis})
            if best is None or (prefix, 0 if ellipsis else 1) > (
                    best["matched_chars"], 0 if best["ellipsis"] else 1):
                best = {"frame": mid, "ocr_text": raw, "ocr_normalised": norm_ocr,
                        "ocr_confidence": round(float(ocr.get("confidence") or 0.0), 4),
                        "matched_chars": prefix, "ellipsis": ellipsis,
                        "matched": matched}
        if eid in rows:
            continue
        if best is None:
            rows[eid] = {"ir_text": text, "verdict": FAIL, "frames_tried": tried,
                         "problems": ["no informative frame in the overlay pass"]}
            continue
        problems: list[str] = []
        if not best["ocr_normalised"]:
            problems.append("OCR found no text on the overlay layer")
        if best["ellipsis"]:
            problems.append("ellipsis detected in the displayed text (truncation)")
        if not best["matched"]:
            missing = ir_norm[best["matched_chars"]:]
            problems.append(f"IR text is not a subsequence of the displayed text "
                            f"(matched {best['matched_chars']}/{len(ir_norm)} chars; "
                            f"missing tail {missing[:12]!r})")
        rows[eid] = {
            "ir_text": text,
            "verdict": FAIL if problems else PASS,
            "frame": best["frame"],
            "frames_tried": tried,
            "ocr_text": best["ocr_text"][:240],
            "ocr_normalised": best["ocr_normalised"][:240],
            "ir_normalised": ir_norm,
            "ocr_confidence": best["ocr_confidence"],
            "matched_chars": best["matched_chars"],
            "char_recall": round(best["matched_chars"] / max(1, len(ir_norm)), 4),
            "ellipsis": best["ellipsis"],
            "problems": problems,
        }
    rows.update(pre_rows)
    verdicts = [r.get("verdict", NOT_RUN) for r in rows.values()]
    not_run = [eid for eid, r in rows.items() if r.get("verdict") == NOT_RUN]
    return {
        "verdict": FAIL if FAIL in verdicts else PASS,
        "not_run_elements": not_run,
        "elements": len(plan),
        "by_element": rows,
        "method": ("Apple Vision OCR (accurate, zh-Hans+en-US, traditional->simplified "
                   "folded) on the overlay-only layer at 50/70/85 % of the window; the "
                   "best-matching frame must contain the whole IR text as a subsequence "
                   "with no ellipsis"),
        "honesty": ("OCR can misread glyphs; a FAIL here is a measured discrepancy, not "
                    "proof of a rendering bug — inspect ocr_text/frames_tried. Elements "
                    "with no text are NOT RUN."),
    }


def _legibility(ir: dict, rendered: Path, source: Path, width: int, height: int,
                fps: float, boxes: dict[str, list[int]],
                count_series: dict[str, np.ndarray],
                per_element: dict[str, dict]) -> dict[str, Any]:
    sampled: dict[str, list[int]] = {}
    for eid, el in {str(e.get("id")): e for e in ir.get("elements") or []}.items():
        if eid not in boxes:
            continue
        w = el.get("window") or {}
        sf, ef = int(w.get("start_frame", 0)), int(w.get("end_frame", 0))
        span = max(1, ef - sf)
        frames = sorted({sf + int(round(span * frac)) for frac in (0.2, 0.5, 0.8)})
        n_known = len(count_series.get(eid, []))
        frames = [f for f in frames if 0 <= f < (n_known or 10 ** 9)]
        sampled[eid] = frames
    all_frames = sorted({f for fs in sampled.values() for f in fs})
    rendered_frames = _decode_selected(rendered, width, height, all_frames) if all_frames else {}
    source_frames = _decode_selected(source, width, height, all_frames) if all_frames else {}
    by_element: dict[str, Any] = {}
    min_found = None
    offending = None
    total_frames = 0
    for eid, el in {str(e.get("id")): e for e in ir.get("elements") or []}.items():
        if eid not in boxes or not sampled.get(eid):
            by_element[eid] = {"verdict": NOT_RUN, "note": "no sampled frame"}
            continue
        color = parse_color((el.get("style") or {}).get("color"))
        best = None
        worst_row = None
        frames_used = 0
        for f in sampled[eid]:
            if f not in rendered_frames or f not in source_frames:
                continue
            row = glyph_and_contrast(rendered_frames[f], source_frames[f], boxes[eid], color)
            if row is None:
                continue
            frames_used += 1
            total_frames += 1
            if best is None or row["contrast"] < best:
                best = row["contrast"]
                worst_row = {"frame": f, **row}
        if worst_row is None:
            by_element[eid] = {"verdict": NOT_RUN,
                               "note": "no glyph pixels found in sampled frames (element may "
                                       "be a scrim-only or empty element)"}
            continue
        verdict = PASS if best is not None and best >= MIN_CONTRAST else FAIL
        by_element[eid] = {"verdict": verdict, "min_contrast": round(best, 4),
                           "frames_sampled": frames_used, "worst_frame": worst_row,
                           "glyph_pixels": worst_row["glyph_pixels"],
                           "contrast_global": worst_row["contrast_global"],
                           "contrast_vs_source": worst_row["contrast_vs_source"]}
        if min_found is None or best < min_found:
            min_found, offending = best, eid
    verdict = _worst([r.get("verdict", NOT_RUN) for r in by_element.values()])
    return {"verdict": verdict, "min_contrast": None if min_found is None else round(min_found, 4),
            "offending_element": offending, "frames_sampled": total_frames,
            "by_element": by_element}


def _sheet_frames(elements: list[dict], per_element: dict[str, dict],
                  global_worst: dict[str, Any], layout: dict[str, Any]) -> list[int]:
    """Deterministic frame picks: mid-window per element + the worst occlusion."""
    picks: set[int] = set()
    for el in elements:
        w = el.get("window") or {}
        try:
            sf, ef = int(w.get("start_frame", 0)), int(w.get("end_frame", 0))
        except (TypeError, ValueError):
            continue
        picks.add(sf)
        picks.add(sf + (ef - sf) // 2)
        picks.add(max(sf, ef - 1))
    if global_worst.get("frame") is not None:
        picks.add(int(global_worst["frame"]))
    frames = sorted(picks)
    if len(frames) > 9:
        frames = frames[:9]
    return frames


def _write_sheets(out_dir: Path, elements: list[dict], frames: list[int],
                  width: int, height: int, rendered: Path, mask_doc: MaskDoc | None,
                  fps: float, verdicts: dict[str, str], overall: str,
                  global_worst: dict[str, Any]) -> tuple[Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    decoded = _decode_selected(rendered, width, height, frames)
    person_masks: dict[int, np.ndarray | None] = {}
    face_rects: dict[int, list[list[int]]] = {}
    for f in frames:
        if mask_doc is not None and mask_doc.has_person_mask:
            m, faces, _hands, _grid = mask_doc.at(f, fps)
            if m is not None:
                person_masks[f] = np.array(Image.fromarray(m).resize((width, height), Image.BILINEAR))
            else:
                person_masks[f] = None
            if faces:
                face_rects[f] = [list(r) for r in faces]
        else:
            person_masks[f] = None
    pairs = [(f, decoded[f]) for f in frames if f in decoded]
    sheet = write_contact_sheet(out_dir / "contact_sheet.png", pairs, elements,
                                person_masks, verdicts, overall, face_rects=face_rects)
    crop_path = None
    wf = global_worst.get("frame")
    we = global_worst.get("element")
    if wf is not None and wf in decoded and we in {str(e.get("id")) for e in elements}:
        box = next((element_box(e) for e in elements
                    if str(e.get("id")) == we), None)
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            crop_path = write_worst_crop(
                out_dir / "worst_occlusion_crop.png", decoded[wf], box,
                f"frame {wf} — {we} — person overlap "
                f"{global_worst.get('person_overlap_px')} px")
    return sheet, crop_path


def _finish(metrics: dict[str, Any], out_dir: str | Path | None, write: bool) -> dict[str, Any]:
    metrics = _jsonable(metrics)
    if write and out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "qa_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


# -------------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m semotion.verify2",
                                 description="measure a rendered MP4 against source pixels "
                                             "and a person mask (semotion.verify2/1)")
    ap.add_argument("--ir", required=True, help="Motion IR JSON")
    ap.add_argument("--source", required=True, help="source MP4 (no overlay)")
    ap.add_argument("--rendered", "--final", dest="rendered", required=True,
                    help="delivered final MP4 (alias: --final)")
    ap.add_argument("--masks", "--perception", dest="masks",
                    help="perception document semotion.perception/1 "
                         "(alias: --perception)")
    ap.add_argument("--overlay-only", "--overlay", dest="overlay_only",
                    help="optional overlay-only pass MP4 (alias: --overlay)")
    ap.add_argument("--ink-only", dest="ink_only",
                    help="renderer ink-only pass MP4 (glyphs on black, no scrim or "
                         "captions); required for a deterministic text_onset verdict")
    ap.add_argument("--second-render", help="second render of the same IR (determinism)")
    ap.add_argument("--out-dir", "--out", dest="out",
                    help="directory for qa_metrics.json + contact_sheet.png, or a "
                         "*.json file to write the metrics to (alias: --out)")
    ap.add_argument("--safe-margin-px", type=int, default=SAFE_MARGIN_PX)
    ap.add_argument("--footprint-threshold", type=int, default=None,
                    help="override the adaptive per-pixel delta threshold")
    ap.add_argument("--no-sheet", action="store_true", help="skip the contact sheet")
    args = ap.parse_args(list(argv) if argv is not None else None)
    ir = json.loads(Path(args.ir).read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else None
    out_json = out if (out is not None and out.suffix.lower() == ".json") else None
    out_dir = (out.parent if out_json is not None else out)
    metrics = verify(ir=ir, source=args.source, rendered=args.rendered,
                     masks=args.masks, overlay=args.overlay_only,
                     ink=args.ink_only,
                     second_render=args.second_render, out_dir=out_dir,
                     safe_margin_px=args.safe_margin_px,
                     footprint_threshold=args.footprint_threshold,
                     make_sheet=not args.no_sheet,
                     write_metrics=out_json is None)
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    for name, check in metrics.get("checks", {}).items():
        print(f"{name:28s} {check.get('verdict') if isinstance(check, dict) else check}")
    print(f"OVERALL {metrics.get('verdict')}")
    if out_json is not None:
        print(f"metrics: {out_json}")
    elif out_dir is not None:
        print(f"metrics: {out_dir / 'qa_metrics.json'}")
    if metrics.get("contact_sheet"):
        print(f"contact sheet: {metrics['contact_sheet']}")
    return 0 if metrics.get("verdict") == PASS else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
