"""Semantic Motion — perception v2 (person mask, pose, hands, burned-in text).

Contract: ``semotion.perception/1`` (docs/PERCEPTION-CONTRACT.md). The document
also carries the legacy ``semotion.vision/1`` fields (``source``, ``faces``,
``face_tracks``, ``shots``, ``safe_zones``) so v1 consumers keep working.

Why this layer exists
---------------------
Layout v1 only knew face boxes, so overlays were pasted over hair, shoulders,
raised hands and burned-in subtitles. This module answers *what the frame
already contains* with pixel-accurate geometry:

* **person mask** — Apple ``VNGeneratePersonSegmentationRequest``; the
  silhouette including hair and hands, which a face box cannot describe;
* **body pose / hands** — ``VNDetectHumanBodyPoseRequest`` /
  ``VNDetectHumanHandPoseRequest``; shoulder/wrist geometry and gesture boxes;
* **face boxes** — ``VNDetectFaceRectanglesRequest`` (also the legacy layer);
* **burned-in text** — ``VNRecognizeTextRequest`` (zh-Hans + en-US, accurate),
  grouped over time into sub/lower-third/logo regions;
* **ranked quiet zones** — candidate overlay boxes derived from the union
  person mask, the existing-text mask and measured edge density. Each box is
  validated pixel-wise: ``person_overlap == 0`` and ``text_overlap == 0``.

Measured on this machine (Apple M4, macOS 26.2, Python 3.12, pyobjc 12.2.2;
evidence in ``artifacts/p2/perception/``):

===========================  =============================================
clip                         perception runtime (125 / 120 samples @ 5 Hz)
===========================  =============================================
``inputs/zhengqinwen_25s``    see ``artifacts/p2/perception/*/probe_report.json``
``fx_pd_real_16x9_a``         see ``artifacts/p2/perception/*/probe_report.json``
===========================  =============================================

Implementation facts that drove the design (all measured, see
``docs/PERCEPTION.md``):

* **Vision geometry is normalized, bottom-left origin.** Every observation box
  / joint is converted to pixel ``[x, y, w, h]`` with top-left origin here, once.
* **The segmentation mask has its own output size.** It is *not* the input size:
  with the balanced quality level the buffer is 512x384 for a 1920x920 frame
  (and 384x512 for a 1080x1920 portrait). It is a plain resize of the frame —
  we verified alignment by compositing the mask back over the source frame —
  so the mask is resampled to the frame grid before any geometry is derived.
  The mask is then emitted at ``1/mask_scale`` (default 4) with values 0/255.
* **One Vision pass per sampled frame.** All four/five requests share one
  ``VNImageRequestHandler`` and one ``performRequests_error_`` call: that is
  ~20 % faster than one call per request and avoids repeated image setup.
* **Quantities that must be stable are rounded and sorted.** JSON is written
  with ``sort_keys=True``; floats are rounded (6 dp for times/coverage, 4 dp
  for scores/overlaps/confidences). There are no wall-clock values inside the
  hashed document — ``provenance.created_utc`` is run metadata only, and the
  cache key is derived from the video sha256 + the tuned options only.
* **No fabricated results.** A synthetic / designed-artwork clip yields
  ``mask_coverage == 0`` and ``person_bbox is null`` — Apple's person
  segmenter only fires on real people. That is reported, not papered over, and
  ``quiet_zones`` then falls back to edge/contrast-only ranking (with
  ``person_overlap == 0`` trivially true), which ``docs/PERCEPTION.md`` marks
  as "unverified person occupancy".

Honest limits (also in docs/PERCEPTION.md): burned-in Chinese subtitles are
*located* reliably but their *text* is usually misread (the clip's subtitle
line reads ``'走的巴黎纪…'`` at confidence 0.30); the OCR box covers only the
fragment that was recognized, so a small padding is applied and consumers
should still keep the bottom ~8 % clear. Long clips are sampled at 5 Hz, so
sub-200 ms gestures can be missed between samples. Multi-shot clips with
different framings share one clip-global zone list, which is conservative by
construction.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import media, vision

__all__ = [
    "SCHEMA",
    "LEGACY_SCHEMA",
    "Options",
    "analyze",
    "load",
    "validate",
    "mask_to_grid",
    "grid_bbox",
    "clamp_box",
    "grow_box",
    "group_existing_text",
    "quiet_zones",
    "build_debug_sheet",
    "options_key",
    "cache_root",
    "cache_dir",
    "vision_available",
]

SCHEMA = "semotion.perception/1"
LEGACY_SCHEMA = "semotion.vision/1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MASK_SCALE = 4
DEFAULT_SAMPLE_HZ = 5.0
DEFAULT_DET_MAX_SIDE = 1280
SAFE_MARGIN_PX = 32
POSE_MIN_CONF = 0.30
HAND_MIN_CONF = 0.25
OCCUPIED = 128                 # mask grey level >= this is person
ZONE_MIN_W_FRAC = 0.20
ZONE_MIN_H_FRAC = 0.10
ZONE_MAX = 12
ZONE_MIN = 3
ZONE_MARGIN_PX = 8             # quiet zones keep this gap from any person pixel
ZONE_NMS_IOU = 0.45
ZONE_STEP_FRAC = 0.02
EDGE_REF = 0.18                # mean |Sobel|/2040 that counts as fully "busy"
ZONE_SIZE_FRACS = (
    (0.46, 0.34), (0.36, 0.24), (0.28, 0.16), (0.22, 0.11),
    (0.22, 0.24), (0.28, 0.34), (0.46, 0.16), (0.36, 0.11),
    (0.26, 0.20), (0.34, 0.14),
)
SCORE_WEIGHTS = {"quiet": 0.52, "capacity": 0.22, "placement": 0.12, "clear": 0.14}

# Text grouping rules (see docs/PERCEPTION.md for the measured basis).
TEXT_MAX_GAP_SAMPLES = 2       # a lineage survives a 1-sample OCR dropout
TEXT_MIN_RUN = 3               # "stable across >= 3 consecutive samples"
TEXT_MIN_PERSISTENCE = 0.40    # "... or present on >= 40 % of the clip"
TEXT_IOU_LINK = 0.25
TEXT_PAD_FRAC = 0.006          # box padding when rasterizing the text mask
TEXT_MERGE_TIME_GAP_S = 1.0    # same-kind merge across a short OCR dropout
TEXT_CONSTANT_SIMILARITY = 0.50  # >= this => the words did not change (banner/logo)

_POSE_JOINTS: dict[str, tuple[str, str]] = {
    # contract key -> (Vision constant suffix, measured legacy joint string)
    "neck": ("Neck", "neck_1_joint"),
    "left_shoulder": ("LeftShoulder", "left_shoulder_1_joint"),
    "right_shoulder": ("RightShoulder", "right_shoulder_1_joint"),
    "left_elbow": ("LeftElbow", "left_forearm_joint"),
    "right_elbow": ("RightElbow", "right_forearm_joint"),
    "left_wrist": ("LeftWrist", "left_hand_joint"),
    "right_wrist": ("RightWrist", "right_hand_joint"),
    "left_hip": ("LeftHip", "left_upLeg_joint"),
    "right_hip": ("RightHip", "right_upLeg_joint"),
    # Vision's 19-joint model has no nose; head_joint is the nose/head proxy.
    "nose": ("Nose", "head_joint"),
}

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
)


# --------------------------------------------------------------------------- options

@dataclass(frozen=True)
class Options:
    """Everything that changes the document, and therefore the cache key."""

    sample_hz: float = DEFAULT_SAMPLE_HZ
    mask_scale: int = MASK_SCALE
    det_max_side: int = DEFAULT_DET_MAX_SIDE
    seg_quality: str = "balanced"
    ocr_languages: tuple[str, ...] = ("zh-Hans", "en-US")
    pose_min_conf: float = POSE_MIN_CONF
    hand_min_conf: float = HAND_MIN_CONF
    text_pad_frac: float = TEXT_PAD_FRAC
    safe_margin_px: int = SAFE_MARGIN_PX

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "sample_hz": round(float(self.sample_hz), 6),
            "mask_scale": int(self.mask_scale),
            "det_max_side": int(self.det_max_side),
            "seg_quality": self.seg_quality,
            "ocr_languages": list(self.ocr_languages),
            "pose_min_conf": round(float(self.pose_min_conf), 4),
            "hand_min_conf": round(float(self.hand_min_conf), 4),
            "text_pad_frac": round(float(self.text_pad_frac), 6),
            "safe_margin_px": int(self.safe_margin_px),
            # Tuning constants are part of the identity: changing a threshold
            # must invalidate cached geometry.
            "tuning": {
                "occupied": OCCUPIED,
                "zone_min_w_frac": ZONE_MIN_W_FRAC,
                "zone_min_h_frac": ZONE_MIN_H_FRAC,
                "zone_max": ZONE_MAX,
                "zone_margin_px": ZONE_MARGIN_PX,
                "zone_nms_iou": ZONE_NMS_IOU,
                "zone_step_frac": ZONE_STEP_FRAC,
                "zone_sizes": [list(s) for s in ZONE_SIZE_FRACS],
                "edge_ref": EDGE_REF,
                "score_weights": dict(SCORE_WEIGHTS),
                "text_max_gap_samples": TEXT_MAX_GAP_SAMPLES,
                "text_min_run": TEXT_MIN_RUN,
                "text_min_persistence": TEXT_MIN_PERSISTENCE,
                "text_iou_link": TEXT_IOU_LINK,
                "text_merge_time_gap_s": TEXT_MERGE_TIME_GAP_S,
                "text_constant_similarity": TEXT_CONSTANT_SIMILARITY,
            },
        }


def options_key(options: Options | None = None) -> dict[str, Any]:
    """The canonical, sorted option payload used for hashing."""
    return (options or Options()).as_dict()


# --------------------------------------------------------------------------- paths

def cache_root() -> Path:
    """Cache root (same env override as :mod:`semotion.vision`)."""
    return Path(os.environ.get("SEMOTION_CACHE_ROOT") or (PROJECT_ROOT / "artifacts" / "cache"))


def cache_key(source_sha256: str, options: Options | None = None) -> str:
    payload = json.dumps(
        {"sha256": source_sha256, "options": options_key(options)},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def cache_dir(source_sha256: str, options: Options | None = None) -> Path:
    return cache_root() / "perception" / cache_key(source_sha256, options)


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _r(value: float, ndigits: int = 6) -> float:
    return round(float(value), ndigits)


# --------------------------------------------------------------------------- boxes

def clamp_box(box: Sequence[float], frame_w: int, frame_h: int) -> list[int] | None:
    """Clamp ``[x, y, w, h]`` to the frame; ``None`` if nothing remains."""
    if box is None or len(box) != 4 or frame_w <= 0 or frame_h <= 0:
        return None
    x, y, w, h = (float(v) for v in box)
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(frame_w, int(round(x + w)))
    y1 = min(frame_h, int(round(y + h)))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return [x0, y0, x1 - x0, y1 - y0]


def grow_box(box: Sequence[float] | None, margin: int,
             frame_w: int, frame_h: int) -> list[int] | None:
    if box is None:
        return None
    return clamp_box([box[0] - margin, box[1] - margin,
                      box[2] + 2 * margin, box[3] + 2 * margin], frame_w, frame_h)


def _union_box(boxes: Sequence[Sequence[float]], frame_w: int, frame_h: int) -> list[int] | None:
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    xs = [float(b[0]) for b in boxes]
    ys = [float(b[1]) for b in boxes]
    x1 = [float(b[0]) + float(b[2]) for b in boxes]
    y1 = [float(b[1]) + float(b[3]) for b in boxes]
    return clamp_box([min(xs), min(ys), max(x1) - min(xs), max(y1) - min(ys)],
                     frame_w, frame_h)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, aw, ah = (float(v) for v in a[:4])
    bx0, by0, bw, bh = (float(v) for v in b[:4])
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, aw, ah = (float(v) for v in a[:4])
    bx0, by0, bw, bh = (float(v) for v in b[:4])
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    return ix * iy


# --------------------------------------------------------------------------- masks

def mask_to_grid(mask: np.ndarray, frame_w: int, frame_h: int, mask_scale: int) -> np.ndarray:
    """Resample a Vision person mask to the 1/``mask_scale`` output grid.

    Vision returns the segmentation mask in its own buffer size (512x384 for a
    16:9-ish frame, 384x512 for portrait; *not* the input size). The buffer is a
    plain resize of the full frame — verified by compositing it back over the
    source — so a bilinear resize to the frame grid keeps the geometry aligned.
    Output is uint8, binary 0/255 (contract: 0 = background, 255 = person).
    """
    import cv2

    if mask is None or getattr(mask, "size", 0) == 0:
        mask = np.zeros((2, 2), dtype=np.uint8)
    mask = np.asarray(mask, dtype=np.uint8)
    gw = max(1, int(math.ceil(frame_w / float(mask_scale))))
    gh = max(1, int(math.ceil(frame_h / float(mask_scale))))
    resized = cv2.resize(mask, (gw, gh), interpolation=cv2.INTER_LINEAR)
    return np.where(resized >= OCCUPIED, 255, 0).astype(np.uint8)


def grid_bbox(grid: np.ndarray, mask_scale: int,
              frame_w: int | None = None, frame_h: int | None = None) -> list[int] | None:
    """Pixel ``[x, y, w, h]`` of the occupied pixels of a mask grid."""
    occupied = np.asarray(grid) > (OCCUPIED - 1)
    if not occupied.any():
        return None
    rows = np.flatnonzero(occupied.any(axis=1))
    cols = np.flatnonzero(occupied.any(axis=0))
    x0, y0 = int(cols[0]) * mask_scale, int(rows[0]) * mask_scale
    x1, y1 = (int(cols[-1]) + 1) * mask_scale, (int(rows[-1]) + 1) * mask_scale
    if frame_w is not None and frame_h is not None:
        return clamp_box([x0, y0, x1 - x0, y1 - y0], frame_w, frame_h)
    return [x0, y0, x1 - x0, y1 - y0]


def _grid_span(start: float, end: float, extent: int, n: int) -> tuple[int, int]:
    i0 = int(math.floor(start * n / float(extent)))
    i1 = int(math.ceil(end * n / float(extent)))
    i0 = max(0, min(n - 1, i0))
    i1 = max(i0 + 1, min(n, i1))
    return i0, i1


def _box_mean(grid: np.ndarray, box: Sequence[float],
              frame_w: int, frame_h: int) -> float:
    gh, gw = grid.shape
    x, y, w, h = (float(v) for v in box[:4])
    gx0, gx1 = _grid_span(x, x + w, frame_w, gw)
    gy0, gy1 = _grid_span(y, y + h, frame_h, gh)
    sub = grid[gy0:gy1, gx0:gx1]
    return float(sub.mean()) if sub.size else 0.0


def _rasterize(boxes: Sequence[Sequence[float]], gw: int, gh: int,
               frame_w: int, frame_h: int) -> np.ndarray:
    grid = np.zeros((gh, gw), dtype=bool)
    for box in boxes:
        gx0, gx1 = _grid_span(box[0], box[0] + box[2], frame_w, gw)
        gy0, gy1 = _grid_span(box[1], box[1] + box[3], frame_h, gh)
        grid[gy0:gy1, gx0:gx1] = True
    return grid


# --------------------------------------------------------------------------- text

def _text_lineage_score(det: dict[str, Any], lineage: dict[str, Any]) -> float:
    """How well a detection continues a lineage (0 = no link)."""
    box, last = det["box"], lineage["last_box"]
    iou = _iou(box, last)
    h_det, h_last = float(box[3]), float(last[3])
    if iou >= TEXT_IOU_LINK:
        return 1.0 + iou
    ratio = h_det / max(1.0, h_last)
    if 0.5 <= ratio <= 2.0:
        cy_det = box[1] + h_det / 2.0
        cy_last = last[1] + h_last / 2.0
        h_gap = max(0.0, max(box[0], last[0]) - min(box[0] + box[2], last[0] + last[2]))
        union_w = max(box[0] + box[2], last[0] + last[2]) - min(box[0], last[0])
        max_w = max(box[2], last[2])
        # Same row *and* close/compact: two fragments of one line, not two
        # neighbouring elements (a mic tag next to a banner) that happen to sit
        # on the same baseline.
        if ((min(box[1] + h_det, last[1] + h_last) - max(box[1], last[1])) > 0
                and abs(cy_det - cy_last) <= 0.5 * max(h_det, h_last)
                and h_gap <= max(0.5 * max(h_det, h_last), 6.0)
                and union_w <= 1.5 * max_w):
            return 0.8
    # Same baseline, horizontally adjacent fragments (a subtitle split by OCR).
    if abs((box[1] + h_det / 2.0) - (last[1] + h_last / 2.0)) <= 0.5 * max(h_det, h_last):
        x0 = max(box[0], last[0])
        x1 = min(box[0] + box[2], last[0] + last[2])
        overlap = max(0.0, x1 - x0)
        union = max(box[0] + box[2], last[0] + last[2]) - min(box[0], last[0])
        if union > 0 and overlap / union >= 0.6:
            return 0.7
    return 0.0


def _text_similarity(texts: Sequence[str]) -> float:
    """Mean character-level similarity of each detection to the representative.

    Burned-in subtitles change their words between samples (low similarity,
    typically < 0.35 for Chinese lines that share only particles). A scene
    banner, watermark or mic tag keeps the same words even when OCR garbles a
    character here and there (measured > 0.7 on the test clip).
    """
    if not texts:
        return 0.0
    counts: dict[str, int] = {}
    first: dict[str, int] = {}
    for pos, text in enumerate(texts):
        counts[text] = counts.get(text, 0) + 1
        first.setdefault(text, pos)
    rep = max(counts, key=lambda t: (counts[t], -first[t]))
    ratios = [difflib.SequenceMatcher(None, rep, t).ratio() for t in texts]
    return float(sum(ratios) / len(ratios))


def _classify_text(box: Sequence[float], *, width: int, height: int,
                   constant: bool, persistence: float) -> str:
    """Classify a stable text region as ``subtitle`` / ``logo`` / ``other``.

    * **subtitle** — the classic lower-third, horizontally centred band whose
      *words change* over time (``constant`` false): this is the semantic
      definition of a burned subtitle, and it is what the layout must avoid.
    * **logo** — text anchored to a frame edge or corner (channel bug, gold
      lettering in the corner, mic tag in the bottom band), or a constant
      lower-third text that is clearly off-centre (a mic/watermark tag).
    * **other** — constant centred lower-third scene text (a banner/chyron in
      the shot, as in the test clip), credits, and anything else persistent
      that fits neither pattern.

    The distinction matters for the debug sheet and for downstream policy, but
    every reported region is an exclusion zone regardless of kind.
    """
    x, y, w, h = (float(v) for v in box)
    cx = (x + w / 2.0) / max(1.0, width)
    cy = (y + h / 2.0) / max(1.0, height)
    hf = h / max(1.0, height)
    subtitle_shape = (
        0.72 <= cy <= 1.0
        and 0.015 <= hf <= 0.12
        and 0.22 <= cx <= 0.78
        and (w / max(1.0, width)) <= 0.75
    )
    touches_edge = x <= 6 or y <= 6 or x + w >= width - 6 or y + h >= height - 6
    corner_band = (cy <= 0.15 or cy >= 0.90) and (cx <= 0.22 or cx >= 0.78)
    if subtitle_shape and not constant:
        return "subtitle"
    if subtitle_shape and abs(cx - 0.5) <= 0.08:
        # constant, centred lower-third text: a banner/chyron inside the shot
        return "other"
    if touches_edge or corner_band:
        # Edge/corner text that is only present briefly is usually an end
        # credit or a transient graphic, not a persistent channel bug.
        return "logo" if persistence >= 0.5 else "other"
    if subtitle_shape:
        # constant, off-centre lower-third text: a mic tag / watermark
        return "logo"
    return "other"


def group_existing_text(samples: Sequence[Sequence[dict[str, Any]]], *,
                        width: int, height: int, duration_s: float,
                        fps: float, sample_step: int,
                        keep_stats: bool = False) -> list[dict[str, Any]]:
    """Group per-sample OCR detections into time-ranged text regions.

    ``samples[i]`` is the detection list of sampled index ``i`` (detections are
    ``{"text", "conf", "box"}`` in pixel coordinates). Detections are linked
    into lineages by geometry (IoU, same baseline, adjacent fragments), because
    a hard-burned subtitle changes its *text* every line while staying in the
    same place. A lineage is reported when it is stable across
    ``TEXT_MIN_RUN`` consecutive samples or present on
    ``TEXT_MIN_PERSISTENCE`` of the clip (contract rule). Same-kind lineages
    that overlap in time and are spatially adjacent are merged (a subtitle
    split into two OCR fragments, or a two-line logo).
    """
    n_samples = len(samples)
    if n_samples == 0:
        return []
    lineages: list[dict[str, Any]] = []
    for i, dets in enumerate(samples):
        for det in sorted(dets, key=lambda d: (d["box"][1], d["box"][0], d["text"])):
            best, best_score = None, 0.0
            for lineage in lineages:
                if i - lineage["last_i"] > TEXT_MAX_GAP_SAMPLES:
                    continue
                score = _text_lineage_score(det, lineage)
                if score > best_score:
                    best, best_score = lineage, score
            if best is None:
                lineage = {
                    "first_i": i, "last_i": i, "last_box": list(det["box"]),
                    "box": list(det["box"]), "indices": [i],
                    "texts": [], "confs": [],
                }
                lineages.append(lineage)
                best = lineage
            best["last_i"] = i
            best["last_box"] = list(det["box"])
            best["indices"].append(i)
            best["texts"].append(str(det["text"]))
            best["confs"].append(float(det["conf"]))
            best["box"] = _union_box([best["box"], det["box"]], width, height) or best["box"]

    sample_period = sample_step / float(fps) if fps > 0 else 0.0
    entries: list[dict[str, Any]] = []
    for lineage in lineages:
        idxs = sorted(set(int(v) for v in lineage["indices"]))
        if not idxs:
            continue
        max_run, run = 1, 1
        for prev, cur in zip(idxs, idxs[1:]):
            run = run + 1 if cur == prev + 1 else 1
            max_run = max(max_run, run)
        persistence = len(idxs) / float(n_samples)
        # Contract rule: stable across >= TEXT_MIN_RUN consecutive samples, or
        # present on >= 40 % of the clip. The persistence branch additionally
        # needs TEXT_MIN_RUN observations, otherwise a 2-sample clip would make
        # every one-off detection "100 % of the clip".
        if max_run < TEXT_MIN_RUN and (persistence < TEXT_MIN_PERSISTENCE
                                       or len(idxs) < TEXT_MIN_RUN):
            continue
        texts = lineage["texts"]
        counts: dict[str, int] = {}
        first_seen: dict[str, int] = {}
        for pos, text in enumerate(texts):
            counts[text] = counts.get(text, 0) + 1
            first_seen.setdefault(text, pos)
        rep = max(counts, key=lambda t: (counts[t], -first_seen[t]))
        stable_ratio = counts[rep] / float(len(texts))
        mean_conf = sum(lineage["confs"]) / float(len(lineage["confs"]))
        t0 = idxs[0] * sample_period
        t1 = min(float(duration_s), idxs[-1] * sample_period + sample_period)
        entries.append({
            "t0": _r(t0, 3),
            "t1": _r(t1, 3),
            "bbox": list(lineage["box"]),
            "text": rep,
            "conf": _r(mean_conf, 4),
            "_texts": list(texts),
            "_confs": list(lineage["confs"]),
            "_persistence": _r(persistence, 4),
            "_stable_ratio": _r(stable_ratio, 4),
            "_n_samples": len(idxs),
        })

    entries.sort(key=lambda e: (e["t0"], e["bbox"][1], e["bbox"][0]))
    # Merge the *same element* seen in disjoint time ranges (OCR dropouts)
    # before classification, so persistence/similarity are measured over the
    # whole appearance rather than one fragment of it.
    merged = _merge_text_entries(entries, width=width, height=height)

    final: list[dict[str, Any]] = []
    for entry in merged:
        texts = entry.get("_texts") or [entry["text"]]
        similarity = _text_similarity(texts)
        constant = similarity >= TEXT_CONSTANT_SIMILARITY
        persistence = min(1.0, float(entry.get("_n_samples", 0)) / float(n_samples))
        kind = _classify_text(entry["bbox"], width=width, height=height,
                              constant=constant, persistence=persistence)
        confs = entry.get("_confs") or [entry["conf"]]
        final.append({
            "t0": _r(float(entry["t0"]), 3),
            "t1": _r(float(entry["t1"]), 3),
            "bbox": list(entry["bbox"]),
            "text": entry["text"],
            "conf": _r(sum(confs) / float(len(confs)), 4),
            "kind": kind,
            "_persistence": _r(persistence, 4),
            "_stable_ratio": _r(sum(1 for t in texts if t == entry["text"]) / float(len(texts)), 4),
            "_similarity": _r(similarity, 4),
            "_n_samples": int(entry.get("_n_samples", 0)),
        })
    final.sort(key=lambda e: (e["t0"], e["bbox"][1], e["bbox"][0], e["kind"]))
    if not keep_stats:
        for entry in final:
            entry.pop("_persistence", None)
            entry.pop("_stable_ratio", None)
            entry.pop("_similarity", None)
            entry.pop("_n_samples", None)
    return final


def _merge_text_entries(entries: list[dict[str, Any]], *, width: int, height: int
                        ) -> list[dict[str, Any]]:
    merged = [dict(e) for e in entries]
    changed = True
    while changed:
        changed = False
        out: list[dict[str, Any]] = []
        for entry in merged:
            target = None
            for existing in out:
                overlap = min(existing["t1"], entry["t1"]) - max(existing["t0"], entry["t0"])
                shorter = min(existing["t1"] - existing["t0"], entry["t1"] - entry["t0"])
                time_gap = max(0.0, max(existing["t0"], entry["t0"])
                               - min(existing["t1"], entry["t1"]))
                if shorter <= 0:
                    continue
                a, b = existing["bbox"], entry["bbox"]
                iou = _iou(a, b)
                same_time = overlap / shorter >= 0.5 or time_gap <= TEXT_MERGE_TIME_GAP_S
                # Only merge boxes that are the *same element* (IoU) across a
                # time gap. Adjacent-but-different text (a mic tag next to a
                # banner) must stay separate boxes: a union would invent an
                # exclusion area covering the safe space between them.
                if same_time and iou >= 0.2:
                    target = existing
                    break
            if target is None:
                out.append(entry)
                continue
            box = _union_box([target["bbox"], entry["bbox"]], width, height)
            target["bbox"] = box or target["bbox"]
            target["t0"] = min(target["t0"], entry["t0"])
            target["t1"] = max(target["t1"], entry["t1"])
            target["_texts"] = list(target.get("_texts") or []) + list(entry.get("_texts") or [])
            target["_confs"] = list(target.get("_confs") or []) + list(entry.get("_confs") or [])
            target["_n_samples"] = target.get("_n_samples", 0) + entry.get("_n_samples", 0)
            if entry.get("_n_samples", 0) > target.get("_n_samples", 0):
                target["text"] = entry["text"]
            changed = True
        merged = out
    return merged


def _text_mask(entries: Sequence[dict[str, Any]], *, width: int, height: int,
               gw: int, gh: int, pad_frac: float) -> np.ndarray:
    pad = max(4, int(round(pad_frac * max(width, height))))
    boxes = []
    for entry in entries:
        b = entry["bbox"]
        boxes.append([b[0] - pad, b[1] - pad, b[2] + 2 * pad, b[3] + 2 * pad])
    return _rasterize(boxes, gw, gh, width, height)


# --------------------------------------------------------------------------- zones

def _score_parts(quiet: float, capacity: float, placement: float, clear: float) -> dict[str, float]:
    return {
        "quiet": _r(quiet, 4),
        "capacity": _r(capacity, 4),
        "placement": _r(placement, 4),
        "clear": _r(clear, 4),
    }


def _zone_score(parts: dict[str, float]) -> float:
    score = sum(SCORE_WEIGHTS[key] * parts[key] for key in SCORE_WEIGHTS)
    return _r(max(0.0, min(1.0, score)), 4)


def _placement(center_x: float, center_y: float, width: int, height: int,
               face_center: Sequence[float] | None) -> float:
    if face_center is None:
        return 0.6
    dx = (center_x - float(face_center[0])) / max(1.0, float(width))
    dy = (center_y - float(face_center[1])) / max(1.0, float(height))
    distance = math.hypot(dx, dy)
    # peak at ~0.30 of the frame diagonal: close enough to feel related, far
    # enough not to crowd the face.
    return max(0.0, 1.0 - abs(distance - 0.30) / 0.42)


def quiet_zones(*, person_union: np.ndarray, text_grid: np.ndarray,
                edge_grid: np.ndarray, luma_grid: np.ndarray,
                frame_w: int, frame_h: int, mask_scale: int,
                face_center: Sequence[float] | None = None,
                person_safe: Sequence[float] | None = None,
                max_zones: int = ZONE_MAX) -> list[dict[str, Any]]:
    """Rank candidate overlay boxes by quietness, rejecting person/text overlap.

    A candidate qualifies only when ``person_overlap <= 0.005`` **and**
    ``text_overlap <= 0.01`` on the 1/``mask_scale`` grids, and it keeps
    ``ZONE_MARGIN_PX`` clear of every person pixel (a dilation of the union
    mask is used as a hard gate; the reported ``person_overlap`` is measured on
    the undilated union, so qualified zones always report 0.0).

    Score = 0.52 * quiet + 0.22 * capacity + 0.12 * placement + 0.14 * clear,
    where ``quiet = 1 - clamp(edge_density / EDGE_REF)``, ``capacity`` grows
    with the box area, ``placement`` prefers a box ~0.30 of the diagonal away
    from the face, and ``clear`` is 1 when the box misses the ``person_safe``
    rectangle entirely (the coarse rectangle v2 layout rejects on).

    Returns 0–``max_zones`` entries sorted by score descending. Fewer than 3 can
    happen when a person fills the frame; that is reported as-is rather than
    inventing safe space.
    """
    import cv2

    gh, gw = person_union.shape
    person = np.asarray(person_union) > 0
    text = np.asarray(text_grid) > 0
    edge = np.asarray(edge_grid, dtype=np.float64)
    luma = np.asarray(luma_grid, dtype=np.float64)
    if edge.shape != person.shape or luma.shape != person.shape or text.shape != person.shape:
        raise ValueError("quiet_zones grids must share one shape")

    margin_cells = max(1, int(round(ZONE_MARGIN_PX / float(mask_scale))))
    kernel = np.ones((2 * margin_cells + 1, 2 * margin_cells + 1), dtype=np.uint8)
    person_margin = cv2.dilate(person.astype(np.uint8), kernel, iterations=1) > 0

    person_f = person.astype(np.float64)
    text_f = text.astype(np.float64)
    margin_f = person_margin.astype(np.float64)

    def integral(a: np.ndarray) -> np.ndarray:
        return np.pad(a.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))

    ip, it = integral(person_f), integral(text_f)
    im, ie = integral(margin_f), integral(edge)
    min_w = ZONE_MIN_W_FRAC * frame_w
    min_h = ZONE_MIN_H_FRAC * frame_h
    step_x = max(1, int(round(ZONE_STEP_FRAC * frame_w)))
    step_y = max(1, int(round(ZONE_STEP_FRAC * frame_h)))

    xs_all: list[np.ndarray] = []
    ys_all: list[np.ndarray] = []
    ws_all: list[np.ndarray] = []
    hs_all: list[np.ndarray] = []
    ps_all: list[np.ndarray] = []
    ts_all: list[np.ndarray] = []
    es_all: list[np.ndarray] = []

    for wf, hf in ZONE_SIZE_FRACS:
        bw = max(1, int(round(wf * frame_w)))
        bh = max(1, int(round(hf * frame_h)))
        if bw < min_w or bh < min_h or bw > frame_w or bh > frame_h:
            continue
        xs = np.unique(np.concatenate([np.arange(0, frame_w - bw + 1, step_x), [frame_w - bw]]))
        ys = np.unique(np.concatenate([np.arange(0, frame_h - bh + 1, step_y), [frame_h - bh]]))
        X0, Y0 = np.meshgrid(xs, ys, indexing="ij")
        X1, Y1 = X0 + bw, Y0 + bh
        gx0 = np.clip(np.floor(X0 * gw / frame_w).astype(int), 0, gw - 1)
        gx1 = np.clip(np.ceil(X1 * gw / frame_w).astype(int), gx0 + 1, gw)
        gy0 = np.clip(np.floor(Y0 * gh / frame_h).astype(int), 0, gh - 1)
        gy1 = np.clip(np.ceil(Y1 * gh / frame_h).astype(int), gy0 + 1, gh)

        def sums(image: np.ndarray) -> np.ndarray:
            return (image[gy1, gx1] - image[gy0, gx1]
                    - image[gy1, gx0] + image[gy0, gx0])

        cells = (gy1 - gy0) * (gx1 - gx0)
        person_frac = sums(ip) / cells
        text_frac = sums(it) / cells
        margin_frac = sums(im) / cells
        edge_mean = sums(ie) / cells
        ok = (person_frac <= 0.005) & (text_frac <= 0.01) & (margin_frac <= 0.0)
        if not ok.any():
            continue
        xs_all.append(X0[ok])
        ys_all.append(Y0[ok])
        ws_all.append(np.full(int(ok.sum()), bw))
        hs_all.append(np.full(int(ok.sum()), bh))
        ps_all.append(person_frac[ok])
        ts_all.append(text_frac[ok])
        es_all.append(edge_mean[ok])

    if not xs_all:
        return []
    xs = np.concatenate(xs_all).astype(np.int64)
    ys = np.concatenate(ys_all).astype(np.int64)
    ws = np.concatenate(ws_all).astype(np.int64)
    hs = np.concatenate(hs_all).astype(np.int64)
    ps = np.concatenate(ps_all)
    ts = np.concatenate(ts_all)
    es = np.concatenate(es_all)

    quiet = 1.0 - np.clip(es / EDGE_REF, 0.0, 1.0)
    cap_norm = np.sqrt((ws * hs) / (0.30 * frame_w * 0.26 * frame_h))
    capacity = np.clip(cap_norm, 0.0, 1.0)
    if face_center is None:
        placement = np.full(xs.shape, 0.6)
    else:
        dx = (xs + ws / 2.0 - float(face_center[0])) / max(1.0, float(frame_w))
        dy = (ys + hs / 2.0 - float(face_center[1])) / max(1.0, float(frame_h))
        placement = np.clip(1.0 - np.abs(np.hypot(dx, dy) - 0.30) / 0.42, 0.0, 1.0)
    if person_safe is not None:
        sx, sy, sw, sh = (float(v) for v in person_safe)
        ix = np.clip(np.minimum(xs + ws, sx + sw) - np.maximum(xs, sx), 0.0, None)
        iy = np.clip(np.minimum(ys + hs, sy + sh) - np.maximum(ys, sy), 0.0, None)
        clear = (ix * iy == 0).astype(np.float64)
    else:
        clear = np.ones(xs.shape)
    scores = (SCORE_WEIGHTS["quiet"] * quiet + SCORE_WEIGHTS["capacity"] * capacity
              + SCORE_WEIGHTS["placement"] * placement + SCORE_WEIGHTS["clear"] * clear)

    order = np.lexsort((ys, xs, -scores))
    kept: list[dict[str, Any]] = []
    kept_boxes: list[list[int]] = []
    for idx in order:
        box = [int(xs[idx]), int(ys[idx]), int(ws[idx]), int(hs[idx])]
        if not _zone_qualifies(box, person_f, text_f, margin_f, frame_w, frame_h):
            continue
        if any(_overlaps_kept(box, other) for other in kept_boxes):
            continue
        parts = _zone_parts(box, edge, luma, frame_w, frame_h, face_center, clear[idx])
        entry = {
            "bbox": box,
            "kind": _zone_kind(box, frame_w, frame_h),
            "score": _zone_score(parts),
            "person_overlap": _r(_box_mean(person_f, box, frame_w, frame_h), 4),
            "text_overlap": _r(_box_mean(text_f, box, frame_w, frame_h), 4),
            "edge_density": _r(_box_mean(edge, box, frame_w, frame_h), 4),
            "contrast_luma": _r(float(np.abs(
                _box_mean(luma, box, frame_w, frame_h) - 0.5)), 4),
            "score_parts": parts,
        }
        kept.append(entry)
        kept_boxes.append(box)
        if len(kept) >= max_zones:
            break
    kept.sort(key=lambda z: (-z["score"], z["bbox"][1], z["bbox"][0]))
    return kept


def _overlaps_kept(box: Sequence[int], other: Sequence[int]) -> bool:
    """Near-duplicate test: real IoU or half of the smaller box covered."""
    if _iou(box, other) >= ZONE_NMS_IOU:
        return True
    inter = _intersection_area(box, other)
    smaller = min(box[2] * box[3], other[2] * other[3])
    return smaller > 0 and inter / smaller >= 0.5


def _zone_qualifies(box: Sequence[int], person_f: np.ndarray, text_f: np.ndarray,
                    margin_f: np.ndarray, frame_w: int, frame_h: int) -> bool:
    if box[2] < ZONE_MIN_W_FRAC * frame_w or box[3] < ZONE_MIN_H_FRAC * frame_h:
        return False
    if _box_mean(person_f, box, frame_w, frame_h) > 0.005:
        return False
    if _box_mean(text_f, box, frame_w, frame_h) > 0.01:
        return False
    if _box_mean(margin_f, box, frame_w, frame_h) > 0.0:
        return False
    return True


def _zone_parts(box: Sequence[int], edge: np.ndarray, luma: np.ndarray,
                frame_w: int, frame_h: int,
                face_center: Sequence[float] | None, clear: float) -> dict[str, float]:
    edge_mean = _box_mean(edge, box, frame_w, frame_h)
    quiet = 1.0 - min(1.0, max(0.0, edge_mean / EDGE_REF))
    capacity = min(1.0, math.sqrt((box[2] * box[3]) / (0.30 * frame_w * 0.26 * frame_h)))
    placement = _placement(box[0] + box[2] / 2.0, box[1] + box[3] / 2.0,
                           frame_w, frame_h, face_center)
    return _score_parts(quiet, capacity, placement, float(clear))


def _zone_kind(box: Sequence[int], frame_w: int, frame_h: int) -> str:
    cx = box[0] + box[2] / 2.0
    cy = box[1] + box[3] / 2.0
    if (cx < 0.30 * frame_w or cx > 0.70 * frame_w) and box[3] >= 0.22 * frame_h:
        return "side"
    if cy > 0.60 * frame_h:
        return "lower_third"
    return "background"


# --------------------------------------------------------------------------- backend

def vision_available() -> bool:
    """True when Apple Vision (pyobjc) can be imported and a request handler built."""
    try:
        import Quartz  # noqa: F401
        import Vision  # noqa: F401
    except Exception:
        return False
    return True


def _cgimage(rgb: np.ndarray):
    """CGImage from a contiguous uint8 RGB array (no disk round-trip)."""
    import CoreFoundation
    import Quartz

    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    data = CoreFoundation.CFDataCreate(None, rgb.tobytes(), int(w * h * 3))
    provider = Quartz.CGDataProviderCreateWithCFData(data)
    colorspace = Quartz.CGColorSpaceCreateDeviceRGB()
    image = Quartz.CGImageCreate(
        w, h, 8, 24, int(w * 3), colorspace,
        Quartz.kCGImageAlphaNone | Quartz.kCGBitmapByteOrderDefault,
        provider, None, False, 0,
    )
    if image is None:
        raise media.MediaError("E_VISION", "CGImageCreate failed")
    return image


def _pixel_buffer_u8(pb) -> np.ndarray:
    import Quartz

    Quartz.CVPixelBufferLockBaseAddress(pb, 1)
    try:
        w = int(Quartz.CVPixelBufferGetWidth(pb))
        h = int(Quartz.CVPixelBufferGetHeight(pb))
        stride = int(Quartz.CVPixelBufferGetBytesPerRow(pb))
        base = Quartz.CVPixelBufferGetBaseAddress(pb)
        buf = base.as_buffer(stride * h)
        arr = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(h, stride)[:, :w]
        return np.ascontiguousarray(arr)
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pb, 1)


def _vision_rect_to_box(rect, width: int, height: int) -> list[int] | None:
    """Vision normalized bottom-left rect -> pixel top-left ``[x, y, w, h]``."""
    x = float(rect.origin.x) * width
    y = (1.0 - float(rect.origin.y) - float(rect.size.height)) * height
    w = float(rect.size.width) * width
    h = float(rect.size.height) * height
    return clamp_box([x, y, w, h], width, height)


def _vision_point_to_px(point, width: int, height: int) -> list[int]:
    return [int(round(float(point.location().x) * width)),
            int(round((1.0 - float(point.location().y)) * height))]


class _AppleVisionBackend:
    """One request set, reused for every sampled frame."""

    name = "apple-vision"

    def __init__(self, options: Options):
        import Vision

        self._vision = Vision
        self.options = options
        self.requests: list[Any] = []
        self.face_request = Vision.VNDetectFaceRectanglesRequest.alloc().init()
        self.pose_request = Vision.VNDetectHumanBodyPoseRequest.alloc().init()
        self.seg_request = (
            Vision.VNGeneratePersonSegmentationRequest
            .alloc().initWithCompletionHandler_(None)
        )
        quality = {"accurate": 0, "balanced": 1, "fast": 2}[options.seg_quality]
        self.seg_request.setQualityLevel_(quality)
        self.ocr_request = (
            Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(None)
        )
        # 0 == VNRequestTextRecognitionLevelAccurate
        self.ocr_request.setRecognitionLevel_(0)
        try:
            self.ocr_request.setRecognitionLanguages_(list(options.ocr_languages))
        except Exception:
            self.ocr_request.setRecognitionLanguages_(["en-US"])
        try:
            self.ocr_request.setAutomaticallyDetectsLanguage_(True)
        except Exception:
            pass
        self.hand_request = Vision.VNDetectHumanHandPoseRequest.alloc().init()
        try:
            self.hand_request.setMaximumHandCount_(4)
        except Exception:
            pass
        self.requests = [self.face_request, self.pose_request, self.seg_request,
                         self.ocr_request, self.hand_request]
        self.joint_keys = {}
        for key, (constant, legacy) in _POSE_JOINTS.items():
            value = getattr(Vision, f"VNHumanBodyPoseObservationJointName{constant}", None)
            self.joint_keys[key] = str(value) if value is not None else legacy
        self.warm = False

    # -- individual observations ------------------------------------------------

    def _faces(self, width: int, height: int) -> list[dict[str, Any]]:
        out = []
        for obs in (self.face_request.results() or []):
            box = _vision_rect_to_box(obs.boundingBox(), width, height)
            if box is None:
                continue
            out.append({"box": box, "conf": _r(obs.confidence(), 4)})
        out.sort(key=lambda f: (-f["conf"], f["box"][0], f["box"][1]))
        return out

    def _pose(self, width: int, height: int) -> tuple[dict[str, Any] | None, dict[str, list[int]]]:
        observations = list(self.pose_request.results() or [])
        if not observations:
            return None, {}
        best_obs, best_score = None, -1.0
        for obs in observations:
            score = 0.0
            for key in self.joint_keys:
                point, _err = obs.recognizedPointForJointName_error_(self.joint_keys[key], None)
                if point is not None:
                    score += float(point.confidence())
            if score > best_score:
                best_obs, best_score = obs, score
        if best_obs is None:
            return None, {}
        pose: dict[str, Any] = {}
        confs: dict[str, list[int]] = {}
        for key, joint in self.joint_keys.items():
            point, _err = best_obs.recognizedPointForJointName_error_(joint, None)
            if point is None or float(point.confidence()) < self.options.pose_min_conf:
                pose[key] = None
                continue
            px = _vision_point_to_px(point, width, height)
            pose[key] = [int(max(0, min(width - 1, px[0]))),
                         int(max(0, min(height - 1, px[1])))]
            confs[key] = [int(px[0]), int(px[1]), _r(float(point.confidence()), 4)]
        return pose, confs

    def _hands(self, width: int, height: int, pose_confs: dict[str, list[int]]
               ) -> list[dict[str, Any]]:
        vision_mod = self._vision
        left = vision_mod.VNChiralityLeft
        right = vision_mod.VNChiralityRight
        out: list[dict[str, Any]] = []
        for obs in (self.hand_request.results() or []):
            points = []
            confs = []
            for joint in obs.availableJointNames():
                point, _err = obs.recognizedPointForJointName_error_(joint, None)
                if point is None:
                    continue
                conf = float(point.confidence())
                if conf < self.options.hand_min_conf:
                    continue
                points.append(_vision_point_to_px(point, width, height))
                confs.append(conf)
            if len(points) < 3:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            pad = max(2, int(round(0.15 * max(max(xs) - min(xs), max(ys) - min(ys), 4))))
            box = clamp_box([min(xs) - pad, min(ys) - pad,
                             (max(xs) - min(xs)) + 1 + 2 * pad,
                             (max(ys) - min(ys)) + 1 + 2 * pad], width, height)
            if box is None:
                continue
            chirality = int(obs.chirality())
            if chirality == left:
                side = "left"
            elif chirality == right:
                side = "right"
            else:
                side = self._infer_hand_side(box, pose_confs)
            out.append({"side": side, "bbox": box, "conf": _r(sum(confs) / len(confs), 4)})
        out.sort(key=lambda h: (h["bbox"][0], h["bbox"][1], h["side"]))
        return out

    @staticmethod
    def _infer_hand_side(box: Sequence[int], pose_confs: dict[str, list[int]]) -> str:
        cx = box[0] + box[2] / 2.0
        cy = box[1] + box[3] / 2.0
        candidates = []
        for side in ("left", "right"):
            wrist = pose_confs.get(f"{side}_wrist")
            if wrist is None:
                continue
            candidates.append((side, math.hypot(cx - wrist[0], cy - wrist[1])))
        if not candidates:
            return "unknown"
        candidates.sort(key=lambda item: item[1])
        return candidates[0][0]

    def _ocr(self, width: int, height: int) -> list[dict[str, Any]]:
        out = []
        for obs in (self.ocr_request.results() or []):
            candidates = obs.topCandidates_(1)
            if not candidates:
                continue
            candidate = candidates[0]
            text = str(candidate.string())
            if not text:
                continue
            box = _vision_rect_to_box(obs.boundingBox(), width, height)
            if box is None:
                continue
            out.append({"text": text, "conf": _r(candidate.confidence(), 4), "box": box})
        out.sort(key=lambda d: (d["box"][1], d["box"][0], d["text"]))
        return out

    def _mask(self) -> np.ndarray | None:
        results = self.seg_request.results()
        if not results:
            return None
        try:
            return _pixel_buffer_u8(results[0].pixelBuffer())
        except Exception:
            return None

    # -- one frame --------------------------------------------------------------

    def run(self, rgb: np.ndarray, out_width: int | None = None,
            out_height: int | None = None) -> dict[str, Any]:
        # Vision boxes/joints are normalized, so they are converted straight to
        # *source* pixel coordinates (the decode may be downscaled).
        height, width = rgb.shape[:2]
        width = int(out_width or width)
        height = int(out_height or height)
        handler = (self._vision.VNImageRequestHandler
                   .alloc().initWithCGImage_options_(_cgimage(rgb), {}))
        ok, error = handler.performRequests_error_(self.requests, None)
        self.warm = True
        if not ok:
            raise media.MediaError("E_VISION", f"Vision requests failed: {error}")
        pose, pose_confs = self._pose(width, height)
        return {
            "faces": self._faces(width, height),
            "pose": pose,
            "pose_confs": pose_confs,
            "hands": self._hands(width, height, pose_confs),
            "ocr": self._ocr(width, height),
            "mask": self._mask(),
        }

    def backend_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "os": f"macOS {platform.mac_ver()[0]}",
            "requests": ["face", "pose", "segmentation", "ocr", "hands"],
            "segmentation_quality": self.options.seg_quality,
            "ocr_languages": list(self.options.ocr_languages),
            "ocr_recognition_level": "accurate",
            "detect_max_side": int(self.options.det_max_side),
            "revisions": {
                "face": int(self.face_request.revision()),
                "pose": int(self.pose_request.revision()),
                "segmentation": int(self.seg_request.revision()),
                "ocr": int(self.ocr_request.revision()),
                "hands": int(self.hand_request.revision()),
            },
            "notes": [
                "segmentation output is model-sized (512x384 / 384x512), resampled to the frame grid",
                "nose is Vision head_joint (the 19-joint model has no nose)",
                "hand side is inferred from the nearest pose wrist when chirality is unknown",
            ],
        }


# --------------------------------------------------------------------------- decode

def _decode_sampled(video: Path, *, step: int, det_w: int, det_h: int,
                    tmp_dir: Path | None = None) -> Path:
    """Decode every ``step``-th frame at ``det_w x det_h`` into a raw RGB file.

    Returns the temporary file path (the caller deletes it). Going through a
    file keeps memory at one frame even for long clips, and the file lives on
    the artifacts volume so the small system disk is untouched.
    """
    raw_path: Path | None = None
    try:
        try:
            if tmp_dir is not None:
                tmp_dir.mkdir(parents=True, exist_ok=True)
                handle = tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=".raw",
                                                     delete=False)
            else:
                raise OSError("no tmp dir")
        except OSError:
            handle = tempfile.NamedTemporaryFile(suffix=".raw", delete=False)
        raw_path = Path(handle.name)
        handle.close()
        vf = (f"select='not(mod(n\\,{step}))',"
              f"scale={det_w}:{det_h}:flags=bilinear")
        media.run([media.FFMPEG, "-v", "error", "-y", "-i", str(video),
                   "-vf", vf, "-fps_mode", "passthrough",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", str(raw_path)])
        frame_bytes = det_w * det_h * 3
        if raw_path.stat().st_size // frame_bytes <= 0:
            raise media.MediaError("E_FFMPEG", f"no frames decoded from {video}")
        return raw_path
    except Exception:
        if raw_path is not None:
            raw_path.unlink(missing_ok=True)
        raise


def _read_frames(raw_path: Path, det_w: int, det_h: int) -> tuple[np.memmap, int]:
    frame_bytes = det_w * det_h * 3
    count = raw_path.stat().st_size // frame_bytes
    if count <= 0:
        raise media.MediaError("E_FFMPEG", f"empty raw decode: {raw_path}")
    arr = np.memmap(raw_path, dtype=np.uint8, mode="r",
                    shape=(count, det_h, det_w, 3))
    return arr, count


# --------------------------------------------------------------------------- frames

def _sample_step(fps: float, sample_hz: float) -> int:
    if sample_hz <= 0:
        raise media.MediaError("E_INPUT", f"sample_hz must be positive, got {sample_hz}")
    return max(1, int(math.ceil(float(fps) / float(sample_hz) - 1e-9)))


def _shot_cuts(frames: np.ndarray, n: int, step: int, fps: float) -> tuple[list[int], list[dict[str, Any]]]:
    """Cut detection reusing the measured v1 rule (vision._thumb_metrics/_is_cut)."""
    cuts: list[int] = []
    transitions: list[dict[str, Any]] = []
    prev = None
    for i in range(n):
        cur = frames[i]
        if prev is not None:
            metrics = vision._thumb_metrics(prev, cur)
            cut = bool(vision._is_cut(metrics))
            frame_idx = i * step
            transitions.append({
                "frame": int(frame_idx),
                "t_s": _r(frame_idx / fps, 6),
                "rgb_diff": metrics["rgb_diff"],
                "hist_distance": metrics["hist_distance"],
                "cut": cut,
            })
            if cut:
                cuts.append(int(frame_idx))
        prev = cur
    return cuts, transitions


# --------------------------------------------------------------------------- debug sheet

_FONT_CACHE: dict[int, Any] = {}


def _font(size: int):
    from PIL import ImageFont

    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                font = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _label(draw, xy, text: str, size: int, fill, *, bg=(0, 0, 0, 190), pad: int = 2):
    from PIL import ImageColor

    font = _font(size)
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = xy
    draw.rectangle([x - pad, y - pad, x + w + pad, y + h + 3 * pad + 2], fill=bg)
    draw.text((x, y + pad), text, font=font, fill=fill)


def build_debug_sheet(path: Path, doc: dict[str, Any], frames_rgb: dict[int, np.ndarray],
                      *, cell_width: int = 820, max_cells: int = 4,
                      mask_grids: dict[int, np.ndarray] | None = None,
                      picks: Sequence[int] | None = None) -> Path:
    """Write the 2x2 debug sheet (masks, face, pose, hands, text, quiet zones)."""
    from PIL import Image, ImageDraw

    width, height = doc["regions"]["frame_size"]
    frames = doc["frames"]
    n = len(frames)
    if n == 0 or not frames_rgb:
        raise ValueError("debug sheet needs at least one frame")
    if picks is None:
        picks = _debug_picks(n, available=list(frames_rgb.keys()))
    picks = [int(p) for p in picks if p in frames_rgb][:max_cells]
    if not picks:
        picks = sorted(frames_rgb.keys())[:max_cells]

    mono_mode = "L"
    cell_pad = 14
    header_h = 76
    cell_header_h = 26
    text_entries = doc["regions"].get("existing_text") or []
    text_rows = min(7, int(math.ceil(len(text_entries) / 2.0)))
    legend_h = 150 + text_rows * 19
    rows = 2 if len(picks) > 1 else 1
    cols = 2 if len(picks) > 1 else 1

    cells: list[dict[str, Any]] = []
    for i in picks:
        frame_rgb = np.asarray(frames_rgb[i])
        scale = min((cell_width - 2 * cell_pad) / float(frame_rgb.shape[1]),
                    (cell_width - 2 * cell_pad) / float(frame_rgb.shape[0]) * 0.72,
                    0.9)
        img_w = max(80, int(round(frame_rgb.shape[1] * scale)))
        img_h = max(60, int(round(frame_rgb.shape[0] * scale)))
        image = Image.fromarray(frame_rgb.astype(np.uint8), "RGB").resize(
            (img_w, img_h), Image.LANCZOS)
        cells.append({"index": i, "image": image, "scale": img_w / float(width),
                      "img_w": img_w, "img_h": img_h})
        cells[-1]["cell_h"] = cell_header_h + img_h + 2 * cell_pad

    sheet_w = cols * cell_width + (cols + 1) * cell_pad
    row_heights = []
    for r in range(rows):
        row_cells = [c for c in cells if cells.index(c) // cols == r]
        row_heights.append(max((c["cell_h"] for c in row_cells), default=0))
    sheet_h = header_h + sum(row_heights) + (rows + 1) * cell_pad + legend_h
    sheet = Image.new("RGB", (sheet_w, sheet_h), (18, 20, 24))
    draw = ImageDraw.Draw(sheet, "RGBA")

    source = doc.get("source", {})
    title = (f"semotion.perception/1 debug — {Path(str(source.get('path', 'video'))).name}"
             f"  {width}x{height} @ {source.get('fps')} fps"
             f"  {n} samples @ {doc.get('sample_hz')} Hz  backend={doc.get('backend', {}).get('name')}")
    draw.text((cell_pad, 16), title, font=_font(20), fill=(245, 245, 245))
    subtitle = (f"quiet zones: blue boxes, #rank / score / WxH   |   person mask: green tint   |   "
                f"face: yellow   |   pose: cyan   |   hands: magenta   |   existing text: orange")
    draw.text((cell_pad, 44), subtitle, font=_font(15), fill=(170, 176, 188))

    y_cursor = header_h
    for row in range(rows):
        x_cursor = cell_pad
        for col in range(cols):
            if row * cols + col >= len(cells):
                break
            cell = cells[row * cols + col]
            index = cell["index"]
            frame = frames[index]
            x0, y0 = x_cursor, y_cursor + cell_pad
            img = cell["image"].copy()
            s = cell["scale"]
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            odraw = ImageDraw.Draw(overlay, "RGBA")

            # person mask tint
            mask_grid = (mask_grids or {}).get(index)
            if mask_grid is not None:
                tint = Image.fromarray(np.where(mask_grid >= OCCUPIED, 255, 0).astype(np.uint8),
                                       mono_mode).resize((cell["img_w"], cell["img_h"]), Image.BILINEAR)
                green = Image.new("RGBA", img.size, (40, 230, 120, 90))
                overlay.paste(green, (0, 0), tint)

            # quiet zones (all of them, rank labels for the top 6)
            for rank, zone in enumerate(doc["regions"].get("quiet_zones") or [], start=1):
                bx, by, bw, bh = zone["bbox"]
                rect = [bx * s, by * s, (bx + bw) * s, (by + bh) * s]
                color = (70, 140, 255, 235) if rank <= 6 else (70, 140, 255, 130)
                odraw.rectangle(rect, outline=color, width=2 if rank <= 6 else 1)
                if rank <= 6:
                    label = f"Z{rank} {zone['score']:.2f} {bw}x{bh}"
                    _label(odraw, (rect[0] + 4, rect[1] + 4), label, 14,
                           (200, 220, 255, 255))

            # existing text (only regions that exist at this sample time)
            period = 1.0 / max(0.001, float(doc.get("sample_hz") or 5.0))
            frame_t = float(frame["t_s"])
            for entry in text_entries:
                if not (float(entry["t0"]) - period <= frame_t < float(entry["t1"]) + period):
                    continue
                b = entry["bbox"]
                rect = [b[0] * s, b[1] * s, (b[0] + b[2]) * s, (b[1] + b[3]) * s]
                odraw.rectangle(rect, outline=(255, 150, 30, 245), width=2)
                text = str(entry.get("text", ""))[:10]
                _label(odraw, (rect[0], max(0, rect[1] - 19)),
                       f"{entry['kind']} {entry['t0']:.1f}-{entry['t1']:.1f}s", 13,
                       (255, 190, 120, 255))

            # hands
            for hand in frame.get("hands") or []:
                b = hand["bbox"]
                rect = [b[0] * s, b[1] * s, (b[0] + b[2]) * s, (b[1] + b[3]) * s]
                odraw.rectangle(rect, outline=(255, 60, 210, 245), width=2)
                _label(odraw, (rect[0], rect[1] + 2),
                       f"hand {hand['side']} {hand['conf']:.2f}", 13, (255, 170, 240, 255))

            # face
            face = frame.get("face")
            if face:
                fx, fy, fw, fh = face["bbox"]
                rect = [fx * s, fy * s, (fx + fw) * s, (fy + fh) * s]
                odraw.rectangle(rect, outline=(255, 220, 0, 245), width=3)
                _label(odraw, (rect[0], rect[1] + 2),
                       f"face {face['conf']:.2f}", 14, (255, 235, 140, 255))

            # pose skeleton
            pose = frame.get("pose") or {}
            joints = {k: v for k, v in pose.items() if v}
            bones = [("neck", "left_shoulder"), ("neck", "right_shoulder"),
                     ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
                     ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
                     ("neck", "nose"), ("left_shoulder", "left_hip"),
                     ("right_shoulder", "right_hip"), ("left_hip", "right_hip")]
            for a, b in bones:
                if a in joints and b in joints:
                    pa, pb = joints[a], joints[b]
                    odraw.line([pa[0] * s, pa[1] * s, pb[0] * s, pb[1] * s],
                               fill=(0, 210, 255, 235), width=2)
            for key, point in joints.items():
                px, py = point[0] * s, point[1] * s
                odraw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(0, 230, 255, 245))
            if "left_wrist" in joints:
                px, py = joints["left_wrist"]
                _label(odraw, (px * s + 6, py * s - 8), "L.wrist", 12, (150, 240, 255, 255))
            if "right_wrist" in joints:
                px, py = joints["right_wrist"]
                _label(odraw, (px * s + 6, py * s - 8), "R.wrist", 12, (150, 240, 255, 255))

            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
            sheet.paste(img, (x0 + cell_pad, y0 + cell_header_h))
            dr = ImageDraw.Draw(sheet, "RGBA")
            info = (f"S{index + 1}  sample={index}  frame={frame['frame']}  t={frame['t_s']:.2f}s"
                    f"  person={frame['mask_coverage'] * 100:.1f}%"
                    f"  face={'Y' if face else '-'}"
                    f"  hands={len(frame.get('hands') or [])}"
                    f"  zones={len(doc['regions'].get('quiet_zones') or [])}")
            dr.rectangle([x0, y0, x0 + cell_width, y0 + cell_header_h - 2], fill=(32, 36, 44, 255))
            dr.text((x0 + 8, y0 + 4), info, font=_font(15), fill=(235, 238, 245))
            dr.rectangle([x0, y0, x0 + cell_width, y0 + cell["cell_h"]],
                         outline=(60, 66, 78, 255), width=1)
            x_cursor += cell_width + cell_pad
        y_cursor += row_heights[row] + cell_pad

    legend_x = cell_pad
    legend_y = sheet_h - legend_h + 6
    items = [
        ((40, 230, 120), "person mask (segmentation)"),
        ((255, 220, 0), "face box"),
        ((0, 210, 255), "pose skeleton / wrists"),
        ((255, 60, 210), "hand box"),
        ((255, 150, 30), "existing burned-in text"),
        ((70, 140, 255), "quiet zone (rank, score, size)"),
    ]
    for i, (color, label) in enumerate(items):
        col = i % 3
        row = i // 3
        x = legend_x + col * 540
        y = legend_y + row * 28
        draw.rectangle([x, y + 4, x + 18, y + 22], fill=color + (255,))
        draw.text((x + 26, y + 2), label, font=_font(15), fill=(210, 215, 225))
    draw.text((legend_x, legend_y + 64),
              "Zones: score = 0.52*quiet + 0.22*capacity + 0.12*placement + 0.14*clear; "
              "a zone qualifies only with person_overlap<=0.005 and text_overlap<=0.01.",
              font=_font(14), fill=(160, 168, 180))
    draw.text((legend_x, legend_y + 84),
              "Person tint is that sample's mask; zones use the clip-wide mask union, so a zone is quiet "
              "only if no sampled frame has the person there. Mask PNGs are 1/4 scale, 0=bg / 255=person.",
              font=_font(14), fill=(160, 168, 180))
    draw.text((legend_x, legend_y + 108),
              "Existing text regions (exclusion zones for the whole window):",
              font=_font(14), fill=(210, 215, 225))
    for idx, entry in enumerate(text_entries[:14]):
        col = 0 if idx < text_rows else 1
        row = idx if idx < text_rows else idx - text_rows
        x = legend_x + col * (sheet_w // 2 - 10)
        y = legend_y + 128 + row * 19
        text = str(entry.get("text", ""))
        if len(text) > 26:
            text = text[:26] + "…"
        draw.text((x, y),
                  f"{entry['kind']:8s} {entry['t0']:5.2f}-{entry['t1']:5.2f}s "
                  f"bbox={tuple(entry['bbox'])}  {text}",
                  font=_font(13), fill=(200, 205, 215))

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG")
    return path


# --------------------------------------------------------------------------- assemble

def _assemble_document(*, movie: media.MediaInfo, sha256: str, options: Options,
                       step: int, det_w: int, det_h: int,
                       frames_entries: list[dict[str, Any]],
                       per_frame_faces: dict[int, list[dict[str, Any]]],
                       face_entries: list[dict[str, Any]],
                       existing_text: list[dict[str, Any]],
                       zones: list[dict[str, Any]],
                       person_union: list[int] | None,
                       person_safe: list[int] | None,
                       face_union: list[int] | None,
                       face_safe: list[int] | None,
                       hands_union: list[int] | None,
                       coverage_curve: list[list[float]],
                       shots: list[dict[str, Any]],
                       transitions: list[dict[str, Any]],
                       backend: dict[str, Any],
                       capabilities: dict[str, bool],
                       provenance_extra: dict[str, Any],
                       cache_key_value: str) -> dict[str, Any]:
    width, height = int(movie.width), int(movie.height)
    n_frames = int(movie.frame_count)
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "ir_compatible": LEGACY_SCHEMA,
        "source": {
            "path": str(Path(movie.path).resolve()),
            "sha256": sha256,
            "width": width,
            "height": height,
            "fps": _r(movie.fps, 6),
            "n_frames": n_frames,
            "duration_s": _r(movie.duration, 6),
        },
        "backend": backend,
        "capabilities": capabilities,
        "sample_hz": _r(options.sample_hz, 6),
        "frames": frames_entries,
        "regions": {
            "frame_size": [width, height],
            "face_union": face_union,
            "person_union": person_union,
            "hands_union": hands_union,
            "face_safe": face_safe,
            "person_safe": person_safe,
            "safe_margin_px": int(options.safe_margin_px),
            "existing_text": existing_text,
            "quiet_zones": zones,
            "person_coverage_curve": coverage_curve,
            "shots": [{"start_s": s["start_s"], "end_s": s["end_s"]} for s in shots],
        },
        # ---- legacy semotion.vision/1 fields (v1 consumers) ----
        "faces": face_entries,
        "face_tracks": [],
        "shots": shots,
        "safe_zones": {},
        "provenance": {
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "command": (f"semotion.perception video={Path(movie.path).name} "
                        f"fps={options.sample_hz} det={det_w}x{det_h} "
                        f"mask_scale={options.mask_scale} seg={options.seg_quality}"),
            "cache_key": cache_key_value,
            "scanned_fps": _r(float(movie.fps) / step, 6),
            "requested_sample_hz": _r(options.sample_hz, 6),
            "sample_step_frames": int(step),
            "n_samples": len(frames_entries),
            "decode_size": [int(det_w), int(det_h)],
            "mask_scale": int(options.mask_scale),
            "shot_detection": {
                "metric": "v1 rule: rgb mean abs diff + HSV Bhattacharyya (vision._is_cut)",
                "n_transitions": len(transitions),
                "cuts": [{"frame": t["frame"], "t_s": t["t_s"]} for t in transitions if t["cut"]],
            },
        },
    }
    doc["provenance"].update(provenance_extra)
    doc["face_tracks"] = _face_tracks(per_frame_faces, width=width, height=height,
                                      n_samples=len(frames_entries))
    doc["safe_zones"] = vision.safe_area_boxes({"width": width, "height": height}, doc)
    return doc


def _face_tracks(per_frame_faces: dict[int, list[dict[str, Any]]],
                 *, width: int, height: int, n_samples: int) -> list[dict[str, Any]]:
    if not per_frame_faces:
        return []
    tracks = vision._associate_tracks(per_frame_faces)
    return vision._finish_tracks(tracks, width=width, height=height, n_scan=n_samples)


def _frame_entry(*, frame_idx: int, fps: float, mask_name: str | None,
                 mask_scale: int, coverage: float, person_bbox: list[int] | None,
                 pose: dict[str, Any] | None, hands: list[dict[str, Any]],
                 face: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "frame": int(frame_idx),
        "t_s": _r(frame_idx / fps, 6),
        "person_bbox": person_bbox,
        "mask": mask_name,
        "mask_scale": float(mask_scale),
        "mask_coverage": _r(coverage, 6),
        "pose": pose,
        "hands": hands,
        "face": face,
    }


# --------------------------------------------------------------------------- analyze

def analyze(video: str | Path, out_json: str | Path, *, mask_dir: str | Path | None = None,
            debug_path: str | Path | None = None, options: Options | None = None,
            sample_hz: float | None = None, force: bool = False,
            write_debug: bool = True, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run perception on ``video`` and write the ``semotion.perception/1`` document.

    Cached by ``sha256(video) + options``; ``force=True`` recomputes. A cache
    hit copies the mask PNGs (and the debug sheet) from the cache into the
    requested locations and writes an identical JSON document.
    """
    t_start = time.perf_counter()
    video = Path(video)
    out_json = Path(out_json)
    if options is None:
        options = Options(sample_hz=float(sample_hz) if sample_hz else DEFAULT_SAMPLE_HZ)
    elif sample_hz is not None:
        options = Options(**{**options.__dict__, "sample_hz": float(sample_hz)})
    if mask_dir is None:
        mask_dir = out_json.parent / f"{out_json.stem}_masks"
    mask_dir = Path(mask_dir)
    if debug_path is None:
        debug_path = out_json.with_name(out_json.name + ".debug.png")
    debug_path = Path(debug_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    movie = media.info(video)
    sha256 = media.sha256_file(video)
    key = cache_key(sha256, options)
    cache = cache_dir(sha256, options)
    cache_json = cache / "perception.json"
    cache_masks = cache / "masks"
    cache_debug = cache / "debug.png"
    phases: dict[str, float] = {}

    if (not force and cache_json.exists() and (cache_masks / "0000.png").exists()
            and (not write_debug or cache_debug.exists())):
        try:
            cached = json.loads(cache_json.read_text(encoding="utf-8"))
        except Exception:
            cached = None
        if (isinstance(cached, dict) and cached.get("schema") == SCHEMA
                and cached.get("source", {}).get("sha256") == sha256):
            doc = _materialize(cached, out_json=out_json, mask_dir=mask_dir,
                               cache_masks=cache_masks)
            if write_debug and cache_debug.exists():
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cache_debug, debug_path)
            _write_json(out_json, doc)
            if meta is not None:
                meta.update({"cache": "hit", "cache_key": key,
                             "runtime_s": _r(time.perf_counter() - t_start, 3),
                             "phase_s": {}, "n_samples": len(doc.get("frames") or [])})
            return doc

    t0 = time.perf_counter()
    step = _sample_step(movie.fps, options.sample_hz)
    n_frames = int(movie.frame_count)
    det_w, det_h = vision._det_size(movie.width, movie.height, options.det_max_side)
    raw_path = _decode_sampled(video, step=step, det_w=det_w, det_h=det_h,
                               tmp_dir=cache_root() / "tmp")
    frames, n = _read_frames(raw_path, det_w, det_h)
    phases["decode_s"] = _r(time.perf_counter() - t0, 3)
    try:
        if vision_available():
            doc, cell_frames, mask_grids, phases_vision = _vision_pass(
                video=video, movie=movie, frames=frames, n=n, step=step,
                det_w=det_w, det_h=det_h, options=options, sha256=sha256,
                cache_key_value=key,
            )
            phases.update(phases_vision)
        else:
            doc, cell_frames = _fallback_document(
                video=video, movie=movie, frames=frames, n=n, step=step,
                options=options, sha256=sha256, cache_key_value=key,
            )
            mask_grids = []
            phases["vision_s"] = _r(time.perf_counter() - (t0 + phases["decode_s"]), 3)
    finally:
        del frames
        raw_path.unlink(missing_ok=True)

    t_masks = time.perf_counter()
    if mask_grids:
        _save_masks(mask_grids, cache_masks)
    cache.mkdir(parents=True, exist_ok=True)
    canonical = _canonical_masks(doc)
    _write_json(cache_json, canonical)

    if write_debug and cell_frames:
        t_debug = time.perf_counter()
        build_debug_sheet(cache_debug, canonical, cell_frames,
                          mask_grids=dict(zip(cell_frames.keys(),
                                              (mask_grids[i] for i in cell_frames.keys()))))
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache_debug, debug_path)
        phases["debug_sheet_s"] = _r(time.perf_counter() - t_debug, 3)

    if mask_grids:
        _save_masks(mask_grids, mask_dir)
    phases["mask_write_s"] = _r(time.perf_counter() - t_masks, 3)

    out_doc = _materialize(canonical, out_json=out_json, mask_dir=mask_dir,
                           cache_masks=None)
    _write_json(out_json, out_doc)
    if meta is not None:
        runtime = time.perf_counter() - t_start
        meta.update({"cache": "miss", "cache_key": key, "runtime_s": _r(runtime, 3),
                     "phase_s": phases, "n_samples": len(doc.get("frames") or [])})
    return out_doc


def _vision_pass(*, video: Path, movie: media.MediaInfo, frames: np.memmap, n: int,
                 step: int, det_w: int, det_h: int, options: Options, sha256: str,
                 cache_key_value: str) -> tuple[dict[str, Any], dict[int, np.ndarray],
                                                 list[np.ndarray], dict[str, float]]:
    import cv2

    width, height = int(movie.width), int(movie.height)
    fps = float(movie.fps)
    backend = _AppleVisionBackend(options)
    gw = max(1, int(math.ceil(width / float(options.mask_scale))))
    gh = max(1, int(math.ceil(height / float(options.mask_scale))))

    masks: list[np.ndarray] = []
    coords: list[tuple[int, int]] = []       # (sample_index, source_frame)
    faces_per_frame: dict[int, list[dict[str, Any]]] = {}
    face_entries: list[dict[str, Any]] = []
    text_samples: list[list[dict[str, Any]]] = []
    hand_rows: list[list[dict[str, Any]]] = []
    pose_rows: list[dict[str, Any] | None] = []
    face_rows: list[dict[str, Any] | None] = []
    coverage_rows: list[float] = []
    empty_mask_frames: list[int] = []
    face_without_mask: list[int] = []
    pose_failures: list[int] = []
    edge_sum = np.zeros((gh, gw), dtype=np.float64)
    luma_sum = np.zeros((gh, gw), dtype=np.float64)

    stride = max(1, int(round(n / 16.0)))
    cell_frames: dict[int, np.ndarray] = {}

    t_vision = time.perf_counter()
    for i in range(n):
        rgb = np.asarray(frames[i])
        result = backend.run(rgb, width, height)
        frame_idx = min(int(movie.frame_count) - 1, i * step)
        coords.append((i, frame_idx))
        mask = result["mask"]
        grid = (mask_to_grid(mask, width, height, options.mask_scale) if mask is not None
                else np.zeros((gh, gw), dtype=np.uint8))
        masks.append(grid)
        coverage = float((grid >= OCCUPIED).mean())
        coverage_rows.append(coverage)
        if mask is None or coverage <= 0.0:
            empty_mask_frames.append(frame_idx)
        if coverage <= 0.0 and result["faces"]:
            face_without_mask.append(frame_idx)
        if result["pose"] is None:
            pose_failures.append(frame_idx)
        face_rows.append(result["faces"][0] if result["faces"] else None)
        pose_rows.append(result["pose"])
        hand_rows.append(result["hands"])
        text_samples.append(result["ocr"])
        for face in result["faces"]:
            entry = {"frame": int(frame_idx), "t_s": _r(frame_idx / fps, 6),
                     "box": list(face["box"]), "score": face["conf"]}
            face_entries.append(entry)
            faces_per_frame.setdefault(frame_idx, []).append(
                {"box": list(face["box"]), "score": face["conf"],
                 "t_s": _r(frame_idx / fps, 6)})

        # edge / luma accumulation at mask resolution
        small = cv2.resize(rgb, (gw, gh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_sum += (np.abs(gx) + np.abs(gy)) / 2040.0
        luma_sum += gray / 255.0

        if i % stride == 0 or i == n - 1:
            cell_frames[i] = rgb.copy()
    phases = {"vision_s": _r(time.perf_counter() - t_vision, 3)}
    picks = _debug_picks(n, coverage_rows, list(cell_frames.keys()))
    cell_frames = {i: cell_frames[i] for i in picks if i in cell_frames}

    if not masks:
        raise media.MediaError("E_EMPTY", f"no frames sampled from {video}")
    mask_stack = np.stack(masks, axis=0)
    person_any = (mask_stack >= OCCUPIED).any(axis=0)
    coverage_curve = [
        [_r(frame_idx / fps, 6), _r(coverage, 6)]
        for (_, frame_idx), coverage in zip(coords, coverage_rows)
    ]
    person_union = grid_bbox(np.where(person_any, 255, 0).astype(np.uint8),
                             options.mask_scale, width, height)
    face_union = _union_box([f["box"] for f in _faces_from_rows(face_rows)], width, height)
    hands_union = _union_box([h["bbox"] for row in hand_rows for h in row], width, height)

    t_group = time.perf_counter()
    existing_text = group_existing_text(
        text_samples, width=width, height=height, duration_s=movie.duration,
        fps=fps, sample_step=step,
    )
    text_grid = _text_mask(existing_text, width=width, height=height, gw=gw, gh=gh,
                           pad_frac=options.text_pad_frac)
    edge_grid = edge_sum / float(n)
    luma_grid = luma_sum / float(n)
    face_center = None
    if face_union is not None:
        face_center = [face_union[0] + face_union[2] / 2.0,
                       face_union[1] + face_union[3] / 2.0]
    person_safe = grow_box(person_union, options.safe_margin_px, width, height)
    face_safe = grow_box(face_union, options.safe_margin_px, width, height)
    zones = quiet_zones(
        person_union=person_any,
        text_grid=text_grid,
        edge_grid=edge_grid,
        luma_grid=luma_grid,
        frame_w=width,
        frame_h=height,
        mask_scale=options.mask_scale,
        face_center=face_center,
        person_safe=person_safe,
    )
    phases["group_zones_s"] = _r(time.perf_counter() - t_group, 3)

    # frames[] entries
    frames_entries: list[dict[str, Any]] = []
    for i, ((_, frame_idx), grid, coverage) in enumerate(zip(coords, masks, coverage_rows)):
        frames_entries.append(_frame_entry(
            frame_idx=frame_idx, fps=fps,
            mask_name=f"masks/{i:04d}.png",
            mask_scale=options.mask_scale,
            coverage=coverage,
            person_bbox=grid_bbox(grid, options.mask_scale, width, height),
            pose=pose_rows[i],
            hands=hand_rows[i],
            face=({"bbox": list(face_rows[i]["box"]), "conf": face_rows[i]["conf"]}
                  if face_rows[i] else None),
        ))

    cuts, transitions = _shot_cuts(frames, n, step, fps)
    shots = vision._build_shots(cuts, fps=fps, n_frames=int(movie.frame_count),
                                duration=movie.duration)
    capabilities = {"person_mask": True, "body_pose": True, "hands": True,
                    "existing_text": True, "scene_cuts": True}
    provenance_extra = {
        "mask_failures": empty_mask_frames[:64],
        "mask_failures_count": len(empty_mask_frames),
        "face_without_mask": face_without_mask[:64],
        "face_without_mask_count": len(face_without_mask),
        "pose_failures_count": len(pose_failures),
        "hand_samples": int(sum(1 for row in hand_rows if row)),
        "text_detections": int(sum(len(row) for row in text_samples)),
        "zone_count": len(zones),
    }
    doc = _assemble_document(
        movie=movie, sha256=sha256, options=options, step=step,
        det_w=det_w, det_h=det_h,
        frames_entries=frames_entries, per_frame_faces=faces_per_frame,
        face_entries=sorted(face_entries, key=lambda f: (f["frame"], f["box"][0], f["box"][1])),
        existing_text=existing_text, zones=zones,
        person_union=person_union, person_safe=person_safe,
        face_union=face_union, face_safe=face_safe, hands_union=hands_union,
        coverage_curve=coverage_curve, shots=shots, transitions=transitions,
        backend=backend.backend_info(), capabilities=capabilities,
        provenance_extra=provenance_extra, cache_key_value=cache_key_value,
    )
    return doc, cell_frames, masks, phases


def _faces_from_rows(rows: Sequence[dict[str, Any] | None]) -> list[dict[str, Any]]:
    return [row for row in rows if row]


def _fallback_document(*, video: Path, movie: media.MediaInfo, frames: np.memmap, n: int,
                        step: int, options: Options, sha256: str,
                        cache_key_value: str) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    """Face-only fallback when Apple Vision is unavailable (no mask/pose/OCR)."""
    legacy_dir = cache_root() / "perception" / cache_key_value
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_json = legacy_dir / "fallback_vision.json"
    legacy = vision.analyze(video, legacy_json, fps_scan=options.sample_hz)
    width, height = int(movie.width), int(movie.height)
    fps = float(movie.fps)
    n_frames = int(movie.frame_count)
    faces = legacy.get("faces") or []
    per_frame: dict[int, list[dict[str, Any]]] = {}
    for face in faces:
        per_frame.setdefault(int(face["frame"]), []).append(
            {"box": list(face["box"]), "score": float(face["score"]),
             "t_s": float(face["t_s"])})
    frames_entries = []
    for i in range(n):
        frame_idx = min(n_frames - 1, i * step)
        frames_entries.append(_frame_entry(
            frame_idx=frame_idx, fps=fps, mask_name=None,
            mask_scale=options.mask_scale, coverage=0.0, person_bbox=None,
            pose=None, hands=[], face=None,
        ))
    face_union = _union_box([f["box"] for f in faces], width, height)
    doc = _assemble_document(
        movie=movie, sha256=sha256, options=options, step=step,
        det_w=0, det_h=0,
        frames_entries=frames_entries, per_frame_faces=per_frame,
        face_entries=faces, existing_text=[], zones=[],
        person_union=None, person_safe=None,
        face_union=face_union, face_safe=grow_box(face_union, options.safe_margin_px, width, height),
        hands_union=None, coverage_curve=[],
        shots=legacy.get("shots") or [],
        transitions=[],
        backend={"name": "opencv-yunet-fallback",
                 "os": f"macOS {platform.mac_ver()[0]}",
                 "requests": ["face"],
                 "notes": ["Apple Vision unavailable: face boxes only; no mask/pose/hands/OCR; "
                           "quiet_zones are empty rather than guessed"]},
        capabilities={"person_mask": False, "body_pose": False, "hands": False,
                      "existing_text": False, "scene_cuts": True},
        provenance_extra={"fallback": True},
        cache_key_value=cache_key_value,
    )
    return doc, {}


def _canonical_masks(doc: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(doc, ensure_ascii=False))
    for i, frame in enumerate(out.get("frames") or []):
        if frame.get("mask"):
            frame["mask"] = f"masks/{i:04d}.png"
    return out


def _materialize(doc: dict[str, Any], *, out_json: Path, mask_dir: Path,
                 cache_masks: Path | None) -> dict[str, Any]:
    out = json.loads(json.dumps(doc, ensure_ascii=False))
    if cache_masks is not None:
        _copy_masks(cache_masks, mask_dir)
    rel = os.path.relpath(str(mask_dir), str(out_json.parent))
    for i, frame in enumerate(out.get("frames") or []):
        if frame.get("mask"):
            frame["mask"] = f"{rel}/{i:04d}.png"
    return out


def _copy_masks(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return
    for png in sorted(source.glob("*.png")):
        shutil.copyfile(png, target / png.name)


def _debug_picks(n: int, coverage: Sequence[float] | None = None,
                 available: Sequence[int] | None = None) -> list[int]:
    """The sample indices shown on the debug sheet.

    Spread over the clip, but prefer frames where a person was actually
    segmented (a black tail frame is honest data, yet useless as a visual).
    """
    if n <= 0:
        return []
    candidates = sorted(set(int(v) for v in (available if available is not None else range(n))))
    if coverage is not None:
        good = [i for i in candidates if float(coverage[i]) > 0.0]
        if good:
            candidates = good
    if len(candidates) <= 4:
        return candidates
    idx = [int(round(k * (len(candidates) - 1) / 3.0)) for k in range(4)]
    return sorted({candidates[i] for i in idx})


def _save_masks(mask_grids: Sequence[np.ndarray], target: Path) -> list[str]:
    """Save each mask grid as an 8-bit L PNG; return the relative names."""
    from PIL import Image

    target.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for i, grid in enumerate(mask_grids):
        name = f"{i:04d}.png"
        Image.fromarray(np.asarray(grid, dtype=np.uint8), mode="L").save(target / name,
                                                                        format="PNG")
        names.append(name)
    return names


# --------------------------------------------------------------------------- validate

def validate(doc: dict[str, Any]) -> list[str]:
    """Manual schema validation for ``semotion.perception/1`` (no jsonschema).

    Returns a list of problems; empty means valid. Also checks the legacy
    ``semotion.vision/1`` fields via :func:`semotion.vision.validate` (schema
    name excluded, since this document intentionally reports its own schema).
    """
    problems: list[str] = []
    if not isinstance(doc, dict):
        return ["document must be an object"]
    if doc.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}")
    if doc.get("ir_compatible") != LEGACY_SCHEMA:
        problems.append(f"ir_compatible must be {LEGACY_SCHEMA!r}")
    source = doc.get("source") or {}
    for key in ("path", "sha256", "width", "height", "fps", "n_frames", "duration_s"):
        if key not in source:
            problems.append(f"source.{key} missing")
    width = int(source.get("width") or 0)
    height = int(source.get("height") or 0)
    fps = float(source.get("fps") or 0.0)
    n_frames = int(source.get("n_frames") or 0)
    duration = float(source.get("duration_s") or 0.0)
    if width <= 0 or height <= 0:
        problems.append("source.width/height must be positive")
    sha = str(source.get("sha256") or "")
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        problems.append("source.sha256 must be 64 lowercase hex chars")
    backend = doc.get("backend") or {}
    if not backend.get("name"):
        problems.append("backend.name missing")
    caps = doc.get("capabilities") or {}
    for key in ("person_mask", "body_pose", "hands", "existing_text", "scene_cuts"):
        if key not in caps or not isinstance(caps[key], bool):
            problems.append(f"capabilities.{key} must be a bool")
    if float(doc.get("sample_hz") or 0.0) <= 0:
        problems.append("sample_hz must be positive")

    frames = doc.get("frames") or []
    if not frames:
        problems.append("frames must be non-empty")
    last_t = -1.0
    for i, frame in enumerate(frames):
        for key in ("frame", "t_s", "mask", "mask_scale", "mask_coverage", "pose", "hands", "face"):
            if key not in frame:
                problems.append(f"frames[{i}].{key} missing")
        t = float(frame.get("t_s", -1.0))
        if t < last_t - 1e-6:
            problems.append(f"frames[{i}].t_s must be non-decreasing")
        last_t = max(last_t, t)
        if fps > 0 and abs(t - float(frame.get("frame", 0)) / fps) > 1e-4:
            problems.append(f"frames[{i}].t_s does not equal frame/fps")
        if float(frame.get("mask_scale") or 0.0) <= 0:
            problems.append(f"frames[{i}].mask_scale must be positive")
        coverage = float(frame.get("mask_coverage", -1.0))
        if not (0.0 <= coverage <= 1.0):
            problems.append(f"frames[{i}].mask_coverage outside [0,1]")
        person_box = frame.get("person_bbox")
        if person_box is not None:
            if len(person_box) != 4 or not all(isinstance(v, int) for v in person_box):
                problems.append(f"frames[{i}].person_bbox must be 4 ints")
            elif person_box[2] <= 0 or person_box[3] <= 0 or \
                    person_box[0] + person_box[2] > width or person_box[1] + person_box[3] > height:
                problems.append(f"frames[{i}].person_bbox outside frame: {person_box}")
        pose = frame.get("pose")
        if pose is not None:
            if not isinstance(pose, dict) or not set(pose).issubset(set(_POSE_JOINTS)):
                problems.append(f"frames[{i}].pose has unexpected joints")
            else:
                for joint, point in pose.items():
                    if point is None:
                        continue
                    if len(point) != 2 or not (0 <= point[0] <= width and 0 <= point[1] <= height):
                        problems.append(f"frames[{i}].pose.{joint} outside frame")
        hands = frame.get("hands")
        if not isinstance(hands, list):
            problems.append(f"frames[{i}].hands must be a list")
        else:
            for j, hand in enumerate(hands):
                if hand.get("side") not in ("left", "right", "unknown"):
                    problems.append(f"frames[{i}].hands[{j}].side invalid")
                b = hand.get("bbox") or []
                if len(b) != 4 or b[2] <= 0 or b[3] <= 0:
                    problems.append(f"frames[{i}].hands[{j}].bbox invalid")
                if not (0.0 <= float(hand.get("conf", -1)) <= 1.0):
                    problems.append(f"frames[{i}].hands[{j}].conf outside [0,1]")
        face = frame.get("face")
        if face is not None:
            b = face.get("bbox") or []
            if len(b) != 4 or not (0.0 <= float(face.get("conf", -1)) <= 1.0):
                problems.append(f"frames[{i}].face invalid")

    regions = doc.get("regions") or {}
    if list(regions.get("frame_size") or []) != [width, height]:
        problems.append("regions.frame_size must match source width/height")
    for key in ("face_union", "person_union", "hands_union", "face_safe", "person_safe"):
        box = regions.get(key)
        if box is None:
            continue
        if len(box) != 4 or box[2] <= 0 or box[3] <= 0:
            problems.append(f"regions.{key} invalid")
        elif box[0] < 0 or box[1] < 0 or box[0] + box[2] > width or box[1] + box[3] > height:
            problems.append(f"regions.{key} outside frame")
    if not isinstance(regions.get("safe_margin_px"), int) or regions["safe_margin_px"] < 0:
        problems.append("regions.safe_margin_px must be a non-negative int")
    entries = regions.get("existing_text")
    if not isinstance(entries, list):
        problems.append("regions.existing_text must be a list")
    else:
        for i, entry in enumerate(entries):
            for key in ("t0", "t1", "bbox", "text", "conf", "kind"):
                if key not in entry:
                    problems.append(f"existing_text[{i}].{key} missing")
            if entry.get("kind") not in ("subtitle", "logo", "other"):
                problems.append(f"existing_text[{i}].kind invalid")
            if not (0.0 <= float(entry.get("t0", -1)) <= float(entry.get("t1", -1)) <= duration + 1e-6):
                problems.append(f"existing_text[{i}] bad time range")
            b = entry.get("bbox") or []
            if len(b) != 4 or b[0] < 0 or b[1] < 0 or \
                    b[0] + b[2] > width or b[1] + b[3] > height:
                problems.append(f"existing_text[{i}].bbox outside frame")
    zones = regions.get("quiet_zones")
    if not isinstance(zones, list):
        problems.append("regions.quiet_zones must be a list")
    else:
        if len(zones) > ZONE_MAX:
            problems.append(f"quiet_zones has {len(zones)} entries (max {ZONE_MAX})")
        allow_zero = not bool(caps.get("person_mask", True))
        if len(zones) < ZONE_MIN and not allow_zero and len(frames) and caps.get("person_mask", True):
            problems.append(f"quiet_zones has {len(zones)} entries (contract wants >= {ZONE_MIN})")
        prev = 2.0
        for i, zone in enumerate(zones):
            for key in ("bbox", "kind", "score", "person_overlap", "text_overlap",
                        "edge_density", "contrast_luma"):
                if key not in zone:
                    problems.append(f"quiet_zones[{i}].{key} missing")
            if zone.get("kind") not in ("background", "lower_third", "side"):
                problems.append(f"quiet_zones[{i}].kind invalid")
            score = float(zone.get("score", -1))
            if not (0.0 <= score <= 1.0):
                problems.append(f"quiet_zones[{i}].score outside [0,1]")
            if score > prev + 1e-9:
                problems.append("quiet_zones must be sorted by score descending")
            prev = min(prev, score)
            if float(zone.get("person_overlap", 1)) > 0.005:
                problems.append(f"quiet_zones[{i}].person_overlap {zone.get('person_overlap')} > 0.005")
            if float(zone.get("text_overlap", 1)) > 0.01:
                problems.append(f"quiet_zones[{i}].text_overlap {zone.get('text_overlap')} > 0.01")
            b = zone.get("bbox") or []
            if len(b) != 4 or b[2] <= 0 or b[3] <= 0:
                problems.append(f"quiet_zones[{i}].bbox invalid")
                continue
            if b[2] < ZONE_MIN_W_FRAC * width or b[3] < ZONE_MIN_H_FRAC * height:
                problems.append(f"quiet_zones[{i}].bbox smaller than the contract floor")
            if b[0] < 0 or b[1] < 0 or b[0] + b[2] > width or b[1] + b[3] > height:
                problems.append(f"quiet_zones[{i}].bbox outside frame")
            for key in ("edge_density", "contrast_luma"):
                if not (0.0 <= float(zone.get(key, -1)) <= 1.0):
                    problems.append(f"quiet_zones[{i}].{key} outside [0,1]")
    curve = regions.get("person_coverage_curve")
    if not isinstance(curve, list):
        problems.append("regions.person_coverage_curve must be a list")
    else:
        prev_t = -1.0
        for i, pair in enumerate(curve):
            if len(pair) != 2:
                problems.append(f"person_coverage_curve[{i}] must be [t, coverage]")
                continue
            t, cov = float(pair[0]), float(pair[1])
            if t < prev_t - 1e-6 or not (0.0 <= cov <= 1.0):
                problems.append(f"person_coverage_curve[{i}] invalid")
            prev_t = max(prev_t, t)
    shots = regions.get("shots")
    if not isinstance(shots, list) or not shots:
        problems.append("regions.shots must be a non-empty list")

    # legacy fields
    for key in ("faces", "face_tracks", "shots", "safe_zones"):
        if key not in doc:
            problems.append(f"legacy field {key} missing")
    try:
        legacy_problems = [p for p in vision.validate(doc) if not p.startswith("schema")]
    except Exception as exc:  # pragma: no cover - defensive
        legacy_problems = [f"legacy validation raised: {exc}"]
    problems.extend(f"legacy: {p}" for p in legacy_problems)
    return problems


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- CLI

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m semotion.perception",
        description="Perception v2: person mask, pose, hands, burned-in text and "
                    "ranked quiet zones for talking-head layout (semotion.perception/1).",
    )
    parser.add_argument("--video", default=None, help="input video (mp4/mov)")
    parser.add_argument("--out", default=None, help="output JSON path")
    parser.add_argument("--mask-dir", default=None,
                        help="output mask directory (default: <out stem>_masks)")
    parser.add_argument("--fps", type=float, default=DEFAULT_SAMPLE_HZ,
                        help=f"sample rate in Hz (default {DEFAULT_SAMPLE_HZ})")
    parser.add_argument("--mask-scale", type=int, default=MASK_SCALE,
                        help=f"mask downscale factor (default {MASK_SCALE})")
    parser.add_argument("--det-max-side", type=int, default=DEFAULT_DET_MAX_SIDE,
                        help=f"decode/detection long side in px (default {DEFAULT_DET_MAX_SIDE})")
    parser.add_argument("--seg-quality", choices=["accurate", "balanced", "fast"],
                        default="balanced",
                        help="person segmentation quality (default balanced; measured "
                             "IoU 0.99 vs accurate at 1/16 the cost)")
    parser.add_argument("--debug-sheet", default=None,
                        help="debug sheet PNG (default: <out>.debug.png)")
    parser.add_argument("--no-debug", action="store_true", help="skip the debug sheet")
    parser.add_argument("--force", action="store_true", help="ignore the cache")
    parser.add_argument("--validate-only", metavar="JSON", default=None,
                        help="validate an existing document and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.validate_only and (not args.video or not args.out):
        parser.error("--video and --out are required unless --validate-only is used")
    if args.validate_only:
        problems = validate(load(args.validate_only))
        if problems:
            for problem in problems:
                print(f"FAIL {problem}")
            return 1
        print(f"OK {args.validate_only}")
        return 0
    options = Options(sample_hz=float(args.fps), mask_scale=int(args.mask_scale),
                      det_max_side=int(args.det_max_side), seg_quality=args.seg_quality)
    meta: dict[str, Any] = {}
    doc = analyze(args.video, args.out, mask_dir=args.mask_dir,
                  debug_path=args.debug_sheet, options=options, force=args.force,
                  write_debug=not args.no_debug, meta=meta)
    problems = validate(doc)
    regions = doc.get("regions") or {}
    zones = regions.get("quiet_zones") or []
    zero_overlap = sum(1 for z in zones if z["person_overlap"] == 0.0)
    print(f"video      : {args.video}")
    print(f"out        : {Path(args.out).resolve()}")
    print(f"backend    : {doc['backend']['name']}  cache={meta.get('cache')}  "
          f"runtime={meta.get('runtime_s')}s")
    print(f"samples    : {len(doc.get('frames') or [])} @ {doc.get('sample_hz')} Hz "
          f"({doc['source']['width']}x{doc['source']['height']} @ {doc['source']['fps']} fps)")
    print(f"mask       : coverage mean "
          f"{_r(float(np.mean([f['mask_coverage'] for f in doc['frames']])), 4) if doc['frames'] else 0}"
          f"  failures={doc['provenance'].get('mask_failures_count', 0)}")
    print(f"text       : {len(regions.get('existing_text') or [])} regions")
    for entry in regions.get("existing_text") or []:
        print(f"             [{entry['t0']:6.2f}-{entry['t1']:6.2f}s] {entry['kind']:8s} "
              f"{entry['bbox']} conf={entry['conf']} {entry['text']!r}")
    print(f"quiet zones: {len(zones)} (person_overlap==0: {zero_overlap})")
    for rank, zone in enumerate(zones, 1):
        print(f"             Z{rank:<2d} score={zone['score']:.3f} bbox={zone['bbox']} "
              f"{zone['kind']:11s} person={zone['person_overlap']} text={zone['text_overlap']} "
              f"edge={zone['edge_density']}")
    if problems:
        print("VALIDATE   : FAIL")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("VALIDATE   : OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
