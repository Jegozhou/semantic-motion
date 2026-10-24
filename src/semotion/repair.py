"""Semantic Motion — deterministic repair for failed v2 QA checks.

Input: a Motion IR, the ``semotion.verify2/1`` metrics of its rendered output,
and the perception mask document (person mask, face, existing text, quiet zones).
Output: a repaired copy of the IR + ``repairs.json`` explaining every change with
the numbers that justified it.

Repair ladder (in this fixed order, per element)
-----------------------------------------------
1. **retime** to the anchor onset when the measured appearance/exit is off by more
   than one frame — the window length in frames is preserved;
2. **relocate** to the best quiet zone with zero person/face/text overlap; if the
   box does not fit at its current size, **shrink** towards the module's font
   floor (never below it);
3. **scrim** (raise the background alpha to >= 0.88) when measured contrast is
   below 4.5 while the placement is otherwise safe — a scrim never fixes person
   overlap;
4. **drop** the element when no safe placement exists (and say why).

Guarantees
----------
* deterministic: no wall clock, no randomness; ties broken by sorted zone/box order;
* preserves semantics: text, effect type, anchors and motion get no new content.
  Only layout box, font size (down to the floor), background alpha and window
  timing can change;
* idempotent: a second run on the repaired IR (even with the same stale metrics)
  produces no changes — every action is skipped when the IR already satisfies it;
* never invents a quiet zone: placement candidates come from the perception
  document, then from a fixed fallback grid that is still checked against the
  real person mask.

CLI
---
``PYTHONPATH=src .venv/bin/python -m semotion.repair \\
    --ir artifacts/run/motion_ir.json --metrics artifacts/run/qa/qa_metrics.json \\
    --masks artifacts/perc/perception.json --out-dir artifacts/run/repair``
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import ir as ir_mod
from . import media
from .verify2 import (SAFE_MARGIN_PX, MaskDoc, element_backdrop, element_box,
                      element_font_size, font_floor_px, is_scrim_backdrop,
                      load_masks, parse_color, rgba_alpha)

SCHEMA = "semotion.repair/1"
SHRINK_RATIOS = (1.0, 0.92, 0.84, 0.76, 0.68, 0.60, 0.52, 0.46, 0.40)
SCRIM_ALPHA_TARGET = 0.88
# Fixed fallback anchors (fractions of the free canvas area). Zones from the
# perception document are tried first; these only keep the repair possible when
# the document lists no usable zone, and every one is still mask-checked.
FALLBACK_POSITIONS = (
    ("top_right", 1.0, 0.0), ("top_left", 0.0, 0.0),
    ("bottom_right", 1.0, 1.0), ("bottom_left", 0.0, 1.0),
    ("mid_right", 1.0, 0.5), ("mid_left", 0.0, 0.5),
    ("top_center", 0.5, 0.0), ("bottom_center", 0.5, 1.0),
)


class RepairError(RuntimeError):
    """Bad input document (not a QA failure)."""


# ---------------------------------------------------------------------- geometry
def _box_on_grid(box: Sequence[float], grid: tuple[int, int], frame_w: int,
                 frame_h: int) -> tuple[int, int, int, int]:
    mw, mh = grid
    sx, sy = mw / float(frame_w), mh / float(frame_h)
    x0 = max(0, min(mw, int(math.floor(box[0] * sx))))
    y0 = max(0, min(mh, int(math.floor(box[1] * sy))))
    x1 = max(0, min(mw, int(math.ceil((box[0] + box[2]) * sx))))
    y1 = max(0, min(mh, int(math.ceil((box[1] + box[3]) * sy))))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def _overlap_px(box: Sequence[float], mask: np.ndarray | None, grid: tuple[int, int],
                frame_w: int, frame_h: int) -> int:
    if mask is None:
        return 0
    x, y, w, h = _box_on_grid(box, grid, frame_w, frame_h)
    if w <= 0 or h <= 0:
        return 0
    return int((mask[y:y + h, x:x + w] > 127).sum())


def _canvas(ir: dict) -> tuple[int, int, float]:
    canvas = ir.get("canvas") or {}
    width = int(canvas.get("width") or 0)
    height = int(canvas.get("height") or 0)
    fps = float((ir.get("source") or {}).get("fps") or canvas.get("fps") or 25.0)
    return width, height, fps


def _n_frames(ir: dict) -> int:
    src = ir.get("source") or {}
    for key in ("n_frames", "frames"):
        try:
            value = int(src.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return 0


def _set_element_box(el: dict, box: Sequence[float]) -> None:
    lay = el.get("layout") or {}
    if isinstance(lay.get("box"), (list, tuple)) or "layout" in el:
        lay["box"] = [int(round(v)) for v in box[:4]]
        el["layout"] = lay
    else:
        el["box"] = [int(round(v)) for v in box[:4]]


def _backdrop_box(kind: str, box: Sequence[float], width: int, height: int,
                  pad: int) -> list[int]:
    x, y, w, h = (int(round(v)) for v in box[:4])
    if kind == "scrim_right":
        x0 = max(0, x - pad)
        return [x0, 0, width - x0, height]
    if kind == "scrim_left":
        x1 = min(width, x + w + pad)
        return [0, 0, x1, height]
    if kind == "band_top":
        y1 = min(height, y + h + pad)
        return [0, 0, width, y1]
    if kind == "band_bottom":
        y0 = max(0, y - pad)
        return [0, y0, width, height - y0]
    if kind == "card":
        return [max(0, x - pad), max(0, y - pad),
                min(width, x + w + pad) - max(0, x - pad),
                min(height, y + h + pad) - max(0, y - pad)]
    return [x, y, w, h]


def _backdrop_kinds(kind: str) -> list[str]:
    order = [kind]
    flip = {"scrim_right": "scrim_left", "scrim_left": "scrim_right",
            "band_top": "band_bottom", "band_bottom": "band_top"}
    if kind in flip:
        order.append(flip[kind])
    return order


def _boxes_intersect(a: Sequence[float], b: Sequence[float]) -> float:
    w = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return w * h


def _collides(box: Sequence[float], other: Sequence[float]) -> bool:
    inter = _boxes_intersect(box, other)
    smaller = min(box[2] * box[3], other[2] * other[3]) or 1.0
    return inter / smaller > 0.01


# -------------------------------------------------------------------- mask usage
def _union_in_window(mask_doc: MaskDoc | None, sf: int, ef: int, fps: float,
                     frame_w: int, frame_h: int) -> dict[str, Any]:
    """Union person mask + face/text geometry over an element's window."""
    if mask_doc is None or not mask_doc.frames:
        return {"person": None, "grid": None, "faces": [], "texts": []}
    t0, t1 = sf / fps, ef / fps
    selected = []
    for idx, f in enumerate(mask_doc.frames):
        t = float(f.get("t_s", float(f.get("frame", 0)) / fps))
        if t0 - 0.25 <= t <= t1 + 0.25:
            selected.append(idx)
    if not selected:
        # window between samples: use the bracketing samples
        i0, i1 = mask_doc._bracket(sf, fps)
        selected = sorted({i0, i1})
    masks = [mask_doc._load(i) for i in selected]
    masks = [m for m in masks if m is not None]
    person = None
    grid = mask_doc._mask_size()
    if masks:
        h = min(m.shape[0] for m in masks)
        w = min(m.shape[1] for m in masks)
        person = np.zeros((h, w), dtype=np.uint8)
        for m in masks:
            person = np.maximum(person, m[:h, :w])
        grid = (w, h)
    faces: list[list[int]] = []
    for i in selected:
        face = mask_doc.frames[i].get("face") or None
        if face and face.get("bbox"):
            faces.append([int(round(v)) for v in face["bbox"][:4]])
        elif mask_doc.frames[i].get("face_bbox"):
            faces.append([int(round(v)) for v in mask_doc.frames[i]["face_bbox"][:4]])
    texts = [r for r in mask_doc.existing_text(fps)
             if r["t1"] >= t0 - 0.25 and r["t0"] <= t1 + 0.25]
    return {"person": person, "grid": grid, "faces": faces,
            "texts": [r["bbox"] for r in texts]}


def _face_safe_boxes(faces: Sequence[Sequence[int]], margin: int) -> list[list[int]]:
    out = []
    for f in faces:
        out.append([int(f[0]) - margin, int(f[1]) - margin,
                    int(f[2]) + 2 * margin, int(f[3]) + 2 * margin])
    return out


# ---------------------------------------------------------------------- assess
def assess_elements(ir: dict, mask_doc: MaskDoc | None, *, safe_margin_px: int,
                    fps: float | None = None,
                    width: int | None = None, height: int | None = None
                    ) -> dict[str, dict[str, Any]]:
    """Static, mask-based assessment of every element (no rendered pixels needed).

    This is deliberately independent of the metrics file so the planner can prove
    a repaired IR is already fixed (idempotence) even if stale metrics are passed.
    """
    canvas = ir.get("canvas") or {}
    width = int(width or canvas.get("width") or 0)
    height = int(height or canvas.get("height") or 0)
    fps = float(fps or (ir.get("source") or {}).get("fps") or 25.0)
    elements = [e for e in (ir.get("elements") or []) if isinstance(e, dict)]
    out: dict[str, dict[str, Any]] = {}
    for el in elements:
        eid = str(el.get("id"))
        w = el.get("window") or {}
        sf, ef = int(w.get("start_frame", 0)), int(w.get("end_frame", 0))
        box = element_box(el)
        row: dict[str, Any] = {"id": eid, "issues": {}, "box": box}
        if not (isinstance(box, (list, tuple)) and len(box) >= 4):
            row["issues"]["layout"] = "layout.box missing"
            out[eid] = row
            continue
        box = [int(round(v)) for v in box[:4]]
        row["box"] = box
        window = _union_in_window(mask_doc, sf, ef, fps, width, height)
        person, grid = window["person"], window["grid"]
        overlap = _overlap_px(box, person, grid or (1, 1), width, height) if grid else 0
        if overlap > 0:
            row["issues"]["person"] = (f"{overlap} px of the person mask overlap the box "
                                       f"over its window (grid {grid})")
            row["person_overlap_px"] = overlap
        face_hit = False
        for face in window["faces"]:
            if _boxes_intersect(box, face) > 0:
                face_hit = True
        for safe in _face_safe_boxes(window["faces"], safe_margin_px):
            if _boxes_intersect(box, safe) > 0:
                face_hit = True
        if face_hit:
            row["issues"]["face"] = "box intersects the face / face-safe area"
        text_hit = any(_boxes_intersect(box, t) > 0 for t in window["texts"])
        if text_hit:
            row["issues"]["text"] = "box intersects burned-in text"
        bd = element_backdrop(el)
        if bd is not None and is_scrim_backdrop(bd.get("kind")) and person is not None \
                and grid is not None:
            bd_overlap = _overlap_px(bd["box"], person, grid, width, height)
            if bd_overlap > 0:
                row["issues"]["scrim_backdrop"] = (
                    f"scrim alpha region covers {bd_overlap} px of the person mask")
                row["backdrop_person_overlap_px"] = bd_overlap
        margins = {"left": box[0], "top": box[1],
                   "right": width - (box[0] + box[2]),
                   "bottom": height - (box[1] + box[3])}
        if min(margins.values()) < safe_margin_px or box[0] < 0 or box[1] < 0 \
                or box[0] + box[2] > width or box[1] + box[3] > height:
            row["issues"]["layout"] = f"breaks {safe_margin_px}px safe margin {margins}"
        floor = font_floor_px(str(el.get("type") or ""), width, height)
        font_size = int((el.get("style") or {}).get("font_size_px") or 0)
        if font_size < floor:
            row["issues"]["font"] = f"font {font_size}px < floor {floor}px"
        row["font_floor_px"] = floor
        row["margins_px"] = margins
        out[eid] = row

    # collisions among simultaneously visible elements
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
            ba = (a.get("layout") or {}).get("box")
            bb = (b.get("layout") or {}).get("box")
            if isinstance(ba, (list, tuple)) and isinstance(bb, (list, tuple)) \
                    and _collides(ba, bb):
                for eid, other in ((str(a.get("id")), str(b.get("id"))),
                                   (str(b.get("id")), str(a.get("id")))):
                    if eid in out:
                        out[eid].setdefault("issues", {})["collision"] = \
                            f"overlaps concurrently visible element {other}"
    return out


# ---------------------------------------------------------------------- placement
def _candidate_positions(box_size: tuple[int, int], width: int, height: int,
                         zones: Sequence[dict[str, Any]], safe_margin: int
                         ) -> list[tuple[str, int, int]]:
    """Deterministic candidate origins: quiet zones first, then fallback grid."""
    bw, bh = box_size
    out: list[tuple[str, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for z in sorted(zones, key=lambda z: (-float(z.get("score", 0.0)),
                                          z["bbox"][1], z["bbox"][0],
                                          z["bbox"][2], z["bbox"][3])):
        if float(z.get("person_overlap", 0.0)) > 0.005:
            continue
        if float(z.get("text_overlap", 0.0)) > 0.01:
            continue
        zx, zy, zw, zh = z["bbox"]
        if zw >= bw and zh >= bh:
            x = zx + (zw - bw) // 2
            y = zy + (zh - bh) // 2
            if (x, y) not in seen:
                seen.add((x, y))
                out.append((f"zone:{z.get('kind', 'background')}", x, y))
        else:
            # zone smaller than the box at this size: try its corners only if the
            # box stays inside the zone's safe upper-left; otherwise skip.
            continue
    for name, fx, fy in FALLBACK_POSITIONS:
        x = int(round(safe_margin + fx * (width - bw - 2 * safe_margin)))
        y = int(round(safe_margin + fy * (height - bh - 2 * safe_margin)))
        x = max(safe_margin, min(width - bw - safe_margin, x))
        y = max(safe_margin, min(height - bh - safe_margin, y))
        if (x, y) not in seen:
            seen.add((x, y))
            out.append((f"grid:{name}", x, y))
    return out


def _box_is_safe(box: Sequence[float], width: int, height: int, safe_margin: int,
                 person: np.ndarray | None, grid: tuple[int, int] | None,
                 faces: Sequence[Sequence[int]], texts: Sequence[Sequence[int]],
                 others: Sequence[Sequence[float]]) -> tuple[bool, str]:
    x, y, w, h = (int(round(v)) for v in box)
    if w <= 0 or h <= 0:
        return False, "non-positive box"
    if x < safe_margin or y < safe_margin or x + w > width - safe_margin \
            or y + h > height - safe_margin:
        return False, "safe margin"
    if person is not None and grid:
        if _overlap_px(box, person, grid, width, height) > 0:
            return False, "person"
    for face in faces:
        if _boxes_intersect(box, face) > 0:
            return False, "face"
    for safe in _face_safe_boxes(faces, safe_margin):
        if _boxes_intersect(box, safe) > 0:
            return False, "face_safe"
    if any(_boxes_intersect(box, t) > 0 for t in texts):
        return False, "text"
    if any(_collides(box, o) for o in others):
        return False, "collision"
    return True, ""


def find_placement(box: Sequence[float], font_size: int, kind: str,
                   width: int, height: int, *, safe_margin: int,
                   person: np.ndarray | None, grid: tuple[int, int] | None,
                   faces: Sequence[Sequence[int]], texts: Sequence[Sequence[int]],
                   zones: Sequence[dict[str, Any]],
                   others: Sequence[Sequence[float]],
                   backdrop_kind: str | None = None) -> dict[str, Any] | None:
    """Best (largest, then zone-ranked) safe placement, or None."""
    floor = font_floor_px(kind, width, height)
    x, y, bw, bh = (int(round(v)) for v in box[:4])
    for ratio in SHRINK_RATIOS:
        new_font = max(floor, int(round(font_size * ratio)))
        actual = (new_font / float(font_size)) if font_size > 0 else 1.0
        nw, nh = max(24, int(round(bw * actual))), max(24, int(round(bh * actual)))
        if nw > width - 2 * safe_margin or nh > height - 2 * safe_margin:
            continue
        for zone_kind, px, py in _candidate_positions((nw, nh), width, height,
                                                      zones, safe_margin):
            candidate = [px, py, nw, nh]
            ok, _reason = _box_is_safe(candidate, width, height, safe_margin,
                                       person, grid, faces, texts, others)
            if not ok:
                continue
            chosen_backdrop = None
            if backdrop_kind:
                pad = max(16, int(round(0.015 * max(width, height))))
                for bk in _backdrop_kinds(backdrop_kind):
                    bd_box = _backdrop_box(bk, candidate, width, height, pad)
                    if person is not None and grid:
                        if _overlap_px(bd_box, person, grid, width, height) > 0:
                            continue
                    chosen_backdrop = {"kind": bk, "box": bd_box}
                    break
                if chosen_backdrop is None:
                    continue
            return {"box": candidate, "font_size_px": new_font,
                    "scale": round(actual, 4), "zone": zone_kind,
                    "shrank": new_font < font_size,
                    "backdrop": chosen_backdrop}
    return None


# ---------------------------------------------------------------------- repairs
def _scrim_value(style: dict) -> tuple[str | None, float]:
    bg = style.get("bg")
    alpha = rgba_alpha(bg)
    if bg is None:
        color = parse_color(style.get("color"), (16, 21, 33))
        return f"rgba({color[0]},{color[1]},{color[2]},{SCRIM_ALPHA_TARGET})", alpha
    if str(bg).lower().startswith("#") and len(str(bg)) == 7:
        rgb = parse_color(str(bg))
        return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{SCRIM_ALPHA_TARGET})", alpha
    if str(bg).lower().startswith("rgba"):
        parts = str(bg)[str(bg).find("(") + 1:str(bg).rfind(")")].split(",")
        if len(parts) == 4:
            r, g, b = (int(round(float(v))) for v in parts[:3])
            return f"rgba({r},{g},{b},{SCRIM_ALPHA_TARGET})", alpha
    if str(bg).lower().startswith("rgb"):
        parts = str(bg)[str(bg).find("(") + 1:str(bg).rfind(")")].split(",")
        if len(parts) >= 3:
            r, g, b = (int(round(float(v))) for v in parts[:3])
            return f"rgba({r},{g},{b},{SCRIM_ALPHA_TARGET})", alpha
    return None, alpha


def _anchor_onset(el: dict, ir: dict) -> float | None:
    w = el.get("window") or {}
    anchor_s = w.get("anchor_start_s")
    if anchor_s is None:
        anchor_id = el.get("anchor_ref")
        for a in ((ir.get("semantic") or {}).get("anchors") or []):
            if a.get("id") == anchor_id and a.get("start_s") is not None:
                anchor_s = float(a["start_s"])
                break
    return None if anchor_s is None else float(anchor_s)


def _retime_to_anchor(el: dict, ir: dict, fps: float) -> dict[str, Any] | None:
    """Return a new window snapped to the anchor onset, or None if already there."""
    w = el.get("window") or {}
    try:
        sf, ef = int(w.get("start_frame", 0)), int(w.get("end_frame", 0))
    except (TypeError, ValueError):
        return None
    anchor_s = _anchor_onset(el, ir)
    if anchor_s is None:
        return None
    target_sf = int(round(float(anchor_s) * fps))
    length = max(1, ef - sf)
    n_frames = _n_frames(ir)
    target_sf = max(0, target_sf)
    target_ef = target_sf + length
    if n_frames and target_ef > n_frames:
        target_ef = n_frames
        target_sf = max(0, target_ef - length)
    new_start_s, new_end_s = target_sf / fps, target_ef / fps
    if target_sf == sf and target_ef == ef:
        return None
    anchor_end = w.get("anchor_end_s")
    hold = 0.0
    if anchor_end is not None:
        hold = max(0.0, round(new_end_s - float(anchor_end), 6))
        if hold > 0.8:
            hold = 0.0
    new_window = dict(w)
    new_window.update({"start_s": round(new_start_s, 6), "end_s": round(new_end_s, 6),
                       "start_frame": target_sf, "end_frame": target_ef,
                       "hold_s": hold})
    return {"old": dict(w), "new": new_window}


def plan_repairs(ir: dict, metrics: dict | None, mask_doc: MaskDoc | None, *,
                 safe_margin_px: int = SAFE_MARGIN_PX) -> tuple[dict, dict]:
    """Compute (repaired_ir, repairs_document) without writing files."""
    canvas = ir.get("canvas") or {}
    width, height = int(canvas.get("width") or 0), int(canvas.get("height") or 0)
    if width <= 0 or height <= 0:
        raise RepairError("IR canvas width/height missing")
    fps = float((ir.get("source") or {}).get("fps") or canvas.get("fps") or 25.0)
    repaired = deepcopy(ir)
    elements = [e for e in (repaired.get("elements") or []) if isinstance(e, dict)]
    metrics_by_el = {}
    if metrics:
        for row in metrics.get("elements") or []:
            if isinstance(row, dict) and row.get("id") is not None:
                metrics_by_el[str(row["id"])] = row
    zones = mask_doc.quiet_zones() if mask_doc else []

    # static assessment of the current IR (mask-based, independent of metrics)
    assessment = assess_elements(repaired, mask_doc, safe_margin_px=safe_margin_px,
                                 fps=fps, width=width, height=height)
    repairs: list[dict[str, Any]] = []
    unfixable: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    kept_boxes: dict[str, list[float]] = {
        str(e.get("id")): list(element_box(e) or []) for e in elements}

    for el in list(elements):
        eid = str(el.get("id"))
        w = el.get("window") or {}
        sf, ef = int(w.get("start_frame", 0)), int(w.get("end_frame", 0))
        style = el.setdefault("style", {})
        lay = el.setdefault("layout", {}) if "layout" in el else {}
        box = element_box(el)
        if box is None:
            dropped.append({"element_id": eid, "reason": "element box missing"})
            _remove_element(repaired, eid)
            kept_boxes.pop(eid, None)
            continue
        font_now = element_font_size(el)
        before = {"box": [int(round(v)) for v in box[:4]],
                  "font_size_px": font_now,
                  "bg": style.get("bg"),
                  "backdrop": (element_backdrop(el) or {}).get("kind"),
                  "window": {k: w.get(k) for k in
                             ("start_s", "end_s", "start_frame", "end_frame", "hold_s")}}
        actions: list[str] = []
        evidence: dict[str, Any] = {}
        issues = dict((assessment.get(eid) or {}).get("issues") or {})

        # metric-driven failures (rendered evidence) ---------------------------
        mrow = metrics_by_el.get(eid) or {}
        timing = mrow.get("timing") or {}
        timing_failed = timing.get("verdict") == "FAIL"
        leg = mrow.get("legibility") or {}
        contrast_failed = leg.get("verdict") == "FAIL"

        # 1. timing: snap to the anchor onset ----------------------------------
        timing_unfixable = False
        contrast_unfixable = False
        if timing_failed:
            evidence["timing_before"] = {
                "verdict": timing.get("verdict"),
                "appearance_frame": timing.get("appearance_frame"),
                "window_start_frame": timing.get("window_start_frame"),
                "anchor_onset_frame": timing.get("anchor_onset_frame"),
                "signed_error_anchor_frames": timing.get("signed_error_anchor_frames"),
            }
            retime = _retime_to_anchor(el, repaired, fps)
            if retime is not None:
                el["window"] = retime["new"]
                actions.append("retime")
                evidence["tiny"] = {"old_window": retime["old"], "new_window": retime["new"]}
                w = el["window"]
                sf, ef = int(w["start_frame"]), int(w["end_frame"])
            else:
                # already snapped (or no anchor): no mutation; a missing anchor is
                # the only genuinely unfixable timing case.
                if _anchor_onset(el, repaired) is None:
                    timing_unfixable = True
                evidence["timing_already_at_anchor"] = _anchor_onset(el, repaired) is not None

        # 2. placement: person/face/text/layout/collision ----------------------
        placement_issue = any(k in issues for k in ("person", "face", "text",
                                                     "layout", "collision",
                                                     "scrim_backdrop"))
        if placement_issue:
            window = _union_in_window(mask_doc, sf, ef, fps, width, height)
            # only consider other elements whose windows overlap in time
            others = []
            for e2 in elements:
                if str(e2.get("id")) == eid:
                    continue
                w2 = e2.get("window") or {}
                if int(w2.get("start_frame", 0)) < ef and sf < int(w2.get("end_frame", 0)):
                    b2 = element_box(e2)
                    if b2 is not None:
                        others.append([float(v) for v in b2])
            bd = element_backdrop(el)
            bd_kind = (str(bd.get("kind")) if bd is not None
                       and is_scrim_backdrop(bd.get("kind")) else None)
            placement = find_placement(
                box, font_now, str(el.get("type") or ""),
                width, height, safe_margin=safe_margin_px,
                person=window["person"], grid=window["grid"],
                faces=window["faces"], texts=window["texts"], zones=zones,
                others=others, backdrop_kind=bd_kind)
            if placement is None:
                dropped.append({
                    "element_id": eid,
                    "reason": "no zero-person-overlap placement exists down to the "
                              "font floor",
                    "evidence": {"issues": issues,
                                 "window": [sf, ef],
                                 "font_floor_px": font_floor_px(str(el.get("type") or ""),
                                                                width, height)},
                })
                _remove_element(repaired, eid)
                kept_boxes.pop(eid, None)
                continue
            _set_element_box(el, placement["box"])
            if "layout" in el:
                lay = el["layout"]
                lay["zone"] = placement["zone"]
                lay["repaired"] = True
            if placement.get("backdrop"):
                bd = el.setdefault("backdrop", {})
                bd.update(placement["backdrop"])
            if placement["font_size_px"] != font_now:
                if int(style.get("font_size_px") or 0) > 0:
                    style["font_size_px"] = placement["font_size_px"]
                elif "fit" in el:
                    el["fit"]["display_px"] = placement["font_size_px"]
                else:
                    style["font_size_px"] = placement["font_size_px"]
            before_box = before["box"]
            evidence["placement"] = {
                "issues": issues,
                "person_overlap_px_before": int((assessment.get(eid) or {})
                                                .get("person_overlap_px") or 0),
                "chosen_zone": placement["zone"],
                "box_before": before_box, "box_after": placement["box"],
                "scale": placement["scale"],
                "font_before": before["font_size_px"],
                "font_after": element_font_size(el),
                "backdrop_after": (element_backdrop(el) or {}).get("box"),
            }
            actions.append("relocate" if not placement["shrank"] else "relocate+shrink")
            kept_boxes[eid] = [float(v) for v in placement["box"]]

        # 3. font floor --------------------------------------------------------
        floor = font_floor_px(str(el.get("type") or ""), width, height)
        font_now = element_font_size(el)
        v2_display = int((el.get("fit") or {}).get("display_px") or 0)
        declared = int(style.get("font_size_px") or 0)
        if font_now < floor and (declared > 0 or v2_display > 0):
            ratio = floor / float(font_now) if font_now > 0 else 1.0
            if declared > 0:
                style["font_size_px"] = floor
            elif "fit" in el:
                el["fit"]["display_px"] = floor
            else:
                style["font_size_px"] = floor
            current_box = element_box(el) or [0, 0, 0, 0]
            bx, by, bw, bh = current_box
            _set_element_box(el, [bx, by, max(24, int(round(bw * ratio))),
                                  max(24, int(round(bh * ratio)))])
            actions.append("font_floor")
            evidence["font_floor"] = {"before": font_now, "after": floor,
                                      "scope": "declared display size"}

        # 4. contrast: add / raise the scrim ----------------------------------
        if contrast_failed:
            new_bg, old_alpha = _scrim_value(style)
            evidence["contrast_before"] = {
                "verdict": leg.get("verdict"), "min_contrast": leg.get("min_contrast"),
                "offending": leg.get("offending_element") == eid,
                "bg_before": style.get("bg"), "bg_alpha_before": round(old_alpha, 3),
            }
            if new_bg is not None and old_alpha < SCRIM_ALPHA_TARGET - 1e-6:
                style["bg"] = new_bg
                actions.append("scrim")
                evidence["scrim"] = {"bg_after": new_bg,
                                     "alpha_after": SCRIM_ALPHA_TARGET}
            else:
                evidence["contrast_already_scrimmed"] = True
                if new_bg is None:
                    contrast_unfixable = True

        if actions:
            after = {"box": list(element_box(el) or []),
                     "font_size_px": element_font_size(el),
                     "bg": style.get("bg"),
                     "backdrop": (element_backdrop(el) or {}).get("kind"),
                     "window": {k: (el.get("window") or {}).get(k) for k in
                                ("start_s", "end_s", "start_frame", "end_frame", "hold_s")}}
            reasons = []
            if "retime" in actions:
                reasons.append("rendered appearance/exit was outside the 1-frame tolerance")
            if "relocate" in actions or "relocate+shrink" in actions:
                reasons.append("overlay footprint touched the person/face/text or violated layout")
            if "scrim" in actions:
                reasons.append("measured WCAG contrast was below 4.5")
            if "font_floor" in actions:
                reasons.append("font was below the module floor")
            repairs.append({
                "element_id": eid,
                "checks_failed": sorted(set(
                    [k for k in ("person", "face", "text", "layout", "collision")
                     if k in issues] + (["timing"] if timing_failed else [])
                    + (["contrast"] if contrast_failed else []))),
                "action": "+".join(dict.fromkeys(actions)),
                "reason": "; ".join(reasons) or "repair",
                "before": before,
                "after": after,
                "evidence": evidence,
            })
        else:
            # genuinely unfixable metric failures (no anchor / unparseable scrim)
            for label, flag in (("contrast", contrast_unfixable),
                                ("timing", timing_unfixable)):
                if flag:
                    unfixable.append({
                        "element_id": eid, "check": label,
                        "reason": ("no deterministic repair exists: the element has no "
                                   "anchor onset / no parseable background to raise; "
                                   "re-render and re-verify"),
                        "evidence": evidence.get(f"{label}_before"),
                    })

    # tracks: drop references to removed elements (v1-shaped IR only)
    kept_ids = {str(e.get("id")) for e in repaired.get("elements") or []}
    v1_shape = "tracks" in repaired and isinstance(repaired.get("tracks"), list)
    if v1_shape:
        for track in repaired.get("tracks") or []:
            track["elements"] = [e for e in (track.get("elements") or [])
                                 if str(e) in kept_ids]
        repaired["tracks"] = [t for t in (repaired.get("tracks") or [])
                              if t.get("elements")]
    # collision declarations: keep only live pairs, add newly overlapping windows
    if v1_shape:
        collision_ok = [list(p) for p in (repaired.get("collision_ok") or [])
                        if isinstance(p, (list, tuple)) and len(p) >= 2
                        and str(p[0]) in kept_ids and str(p[1]) in kept_ids]
        live = {str(e.get("id")): e for e in repaired.get("elements") or []}
        declared = {tuple(sorted((str(p[0]), str(p[1])))) for p in collision_ok}
        ids = sorted(live)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                wa, wb = live[a].get("window") or {}, live[b].get("window") or {}
                try:
                    overlap_time = (int(wa.get("start_frame", 0))
                                    < int(wb.get("end_frame", 0))
                                    and int(wb.get("start_frame", 0))
                                    < int(wa.get("end_frame", 0)))
                except (TypeError, ValueError):
                    overlap_time = False
                if overlap_time and (a, b) not in declared:
                    collision_ok.append([a, b])
                    declared.add((a, b))
        repaired["collision_ok"] = collision_ok

    # reseal so the repaired IR is renderable (render.py refuses a stale hash)
    try:
        ir_mod.seal(repaired)
    except Exception:  # pragma: no cover - seal is deterministic and cannot fail
        pass
    validation_errors: list[str] = []
    if v1_shape:
        try:
            validation_errors = ir_mod.validate(repaired)
        except Exception as exc:  # pragma: no cover - defensive
            validation_errors = [f"validate raised {type(exc).__name__}: {exc}"]
    else:
        validation_errors = []  # v2 shape (element.box, no tracks): render2 owns validation

    doc = {
        "schema": SCHEMA,
        "noop": not repairs and not dropped,
        "policy": ("deterministic ladder: retime -> relocate/shrink -> scrim -> drop; "
                   "text/effect/anchor semantics are never changed"),
        "inputs": {
            "ir_sha256_in": (ir.get("provenance") or {}).get("ir_sha256"),
            "ir_sha256_out": (repaired.get("provenance") or {}).get("ir_sha256"),
            "metrics": None if metrics is None else {
                "verdict": metrics.get("verdict"),
                "failing_elements": sorted((metrics.get("failing_elements") or {}).keys()),
            },
            "masks": None if mask_doc is None else {
                "path": str(mask_doc.path), "backend": mask_doc.backend,
                "quiet_zones": len(zones),
                "sample_hz": mask_doc.document.get("sample_hz"),
            },
            "canvas": [width, height],
            "safe_margin_px": safe_margin_px,
        },
        "repairs": repairs,
        "dropped": dropped,
        "unfixable": unfixable,
        "summary": {
            "elements_in": len(ir.get("elements") or []),
            "elements_out": len(repaired.get("elements") or []),
            "repaired": len(repairs),
            "dropped": len(dropped),
            "unfixable": len(unfixable),
            "idempotent_noop": not repairs and not dropped,
        },
        "validation_errors": validation_errors,
    }
    return repaired, doc


def _remove_element(ir: dict, element_id: str) -> None:
    ir["elements"] = [e for e in (ir.get("elements") or [])
                      if str(e.get("id")) != element_id]


def write_repair(repaired_ir: dict, doc: dict, *, out_ir: Path | str,
                 out_json: Path | str) -> tuple[Path, Path]:
    out_ir = Path(out_ir)
    out_json = Path(out_json)
    out_ir.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_ir.write_text(json.dumps(repaired_ir, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    out_json.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    return out_ir, out_json


# -------------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m semotion.repair",
                                 description="deterministically repair failed verify2 checks "
                                             "(semotion.repair/1)")
    ap.add_argument("--ir", required=True, help="Motion IR to repair")
    ap.add_argument("--metrics", help="qa_metrics.json from semotion.verify2")
    ap.add_argument("--masks", help="perception document (semotion.perception/1)")
    ap.add_argument("--out-dir", required=True, help="destination for repaired IR + repairs.json")
    ap.add_argument("--out-ir", help="override repaired IR path")
    ap.add_argument("--safe-margin-px", type=int, default=SAFE_MARGIN_PX)
    args = ap.parse_args(list(argv) if argv is not None else None)
    ir = json.loads(Path(args.ir).read_text(encoding="utf-8"))
    metrics = (json.loads(Path(args.metrics).read_text(encoding="utf-8"))
               if args.metrics else None)
    mask_doc = load_masks(args.masks) if args.masks else None
    repaired, doc = plan_repairs(ir, metrics, mask_doc,
                                 safe_margin_px=args.safe_margin_px)
    out_dir = Path(args.out_dir)
    out_ir = Path(args.out_ir) if args.out_ir else out_dir / "motion_ir.repaired.json"
    write_repair(repaired, doc, out_ir=out_ir, out_json=out_dir / "repairs.json")
    print(f"noop={doc['noop']} repaired={doc['summary']['repaired']} "
          f"dropped={doc['summary']['dropped']} unfixable={doc['summary']['unfixable']}")
    for r in doc["repairs"]:
        print(f"  {r['element_id']}: {r['action']}  {r['reason']}")
    print(f"repaired IR: {out_ir}")
    print(f"repairs: {out_dir / 'repairs.json'}")
    if doc["validation_errors"]:
        print("WARNING repaired IR has validation errors:")
        for e in doc["validation_errors"][:8]:
            print("  -", e)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
