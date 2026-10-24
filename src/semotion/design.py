"""Safe-area layout for motion elements.

The layout engine never invents coordinates from a language model: it takes the
measured element size (``effects.measure``), the vision safe zones and the face
union box, and picks the first position that satisfies all hard constraints:

* fully on the canvas (with a small edge margin, so the metric sampler never sees
  a changed pixel on the border);
* at most 1 % of the face-safe box covered (the face-safe box is the union of
  tracked faces grown by ``margin_px``);
* at most 5 % of the smaller box covered by an already-placed element.

If nothing fits it degrades step by step (zone -> smaller font -> static safe
slot) and reports ``degraded: true`` with a reason. It returns canvas-valid boxes
even then; only the caller may decide to abort (``E_LAYOUT_NO_SPACE`` belongs to
the pipeline, not to a per-element helper).
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from .effects import base, card, keyword, list as list_effect, number

EFFECT_MODULES = {"keyword": keyword, "number": number, "list": list_effect,
                  "card": card}

# zone preference per element type — works for 9:16 and 16:9 because zones come
# from vision and are normalised.
ZONE_ORDER = {
    "keyword": ("title", "card", "caption"),
    "number": ("card", "title", "caption"),
    "list": ("card", "caption", "title"),
    "card": ("card", "title", "caption"),
    "caption": ("caption", "card", "title"),
}

# typography degradation ladder, applied to the style font size
FONT_RATIOS = (1.0, 0.92, 0.84, 0.76, 0.68)
FACE_OVERLAP_RATIO = 0.01
COLLISION_AREA_RATIO = 0.05
EDGE_MARGIN_PX = 16
DEFAULT_MARGIN_PX = 32

# used when vision produced no safe zones at all (still canvas-relative)
DEFAULT_ZONES = {
    "title": (0.08, 0.09, 0.84, 0.14),
    "card": (0.06, 0.30, 0.88, 0.42),
    "caption": (0.08, 0.74, 0.84, 0.12),
}


def _zone_px(canvas: dict, norm: Sequence[float]) -> list[int]:
    w, h = int(canvas.get("width") or 1080), int(canvas.get("height") or 1920)
    try:
        nx, ny, nw, nh = (float(v) for v in norm[:4])
    except (TypeError, ValueError):
        return [0, 0, w, h]
    x = int(round(nx * w))
    y = int(round(ny * h))
    bw = int(round(nw * w))
    bh = int(round(nh * h))
    # keep the zone itself on-canvas
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    bw = max(1, min(bw, w - x))
    bh = max(1, min(bh, h - y))
    return [x, y, bw, bh]


def safe_area_boxes(canvas: dict, vision: dict | None) -> dict:
    """Resolve pixel zones and the face-safe box for one canvas."""
    vision = vision or {}
    w, h = int(canvas.get("width") or 1080), int(canvas.get("height") or 1920)
    zones_norm = dict(DEFAULT_ZONES)
    provided = vision.get("safe_zones") or {}
    for name, norm in provided.items():
        if isinstance(norm, (list, tuple)) and len(norm) >= 4:
            zones_norm[str(name)] = norm
    zones = {name: _zone_px(canvas, norm) for name, norm in zones_norm.items()}

    unions: list[Sequence[float]] = []
    for tr in vision.get("face_tracks") or []:
        b = tr.get("bbox_union")
        if b and len(b) >= 4 and float(b[2]) > 0 and float(b[3]) > 0:
            unions.append(b)
    if not unions:
        for f in vision.get("faces") or []:
            b = f.get("box")
            if b and len(b) >= 4 and float(b[2]) > 0 and float(b[3]) > 0:
                unions.append(b)
    raw_union = base.union_boxes(unions)
    margin = int(vision.get("face_safe_margin_px") or DEFAULT_MARGIN_PX)
    face_safe = base.expand_box(raw_union, margin) if raw_union else None
    usable = [z for z in zones.values() if z[2] >= 0.40 * w]
    return {
        "canvas": {"width": w, "height": h},
        "zones": zones,
        "zones_norm": zones_norm,
        "face_union": raw_union,
        "face_box": face_safe,
        "margin_px": margin,
        "usable_zones": usable,
        "has_space": bool(usable),
    }


def _inter(a: Sequence[float], b: Sequence[float]) -> float:
    return base.rect_inter_area(a, b)


def _face_ratio(box: Sequence[float], face: Sequence[float] | None) -> float:
    if not face:
        return 0.0
    area = float(face[2]) * float(face[3])
    if area <= 0:
        return 0.0
    return _inter(box, face) / area


def _overlaps_in_time(a: dict | None, b: dict | None) -> bool:
    if not a or not b:
        return True
    try:
        asf, aef = int(a.get("start_frame")), int(a.get("end_frame"))
        bsf, bef = int(b.get("start_frame")), int(b.get("end_frame"))
    except (TypeError, ValueError):
        return True
    return asf < bef and bsf < aef


def _occupied_ratio(box: Sequence[float], occupied: Iterable[dict],
                    window: dict | None = None) -> float:
    """Overlap with elements that are actually on screen at the same time."""
    worst = 0.0
    for o in occupied:
        if isinstance(o, dict):
            ob = o.get("box")
            if window is not None and not _overlaps_in_time(window, o.get("window")):
                continue
        else:
            ob = o
        if not ob or len(ob) < 4:
            continue
        small = min(float(box[2]) * float(box[3]), float(ob[2]) * float(ob[3])) or 1.0
        worst = max(worst, _inter(box, ob) / small)
    return worst


def _inside(box: Sequence[int], canvas: dict, margin: int = EDGE_MARGIN_PX) -> bool:
    w, h = int(canvas.get("width")), int(canvas.get("height"))
    x, y, bw, bh = (int(v) for v in box[:4])
    return (bw > 0 and bh > 0 and x >= margin and y >= margin
            and x + bw <= w - margin and y + bh <= h - margin)


def _clamp_box(box: Sequence[float], canvas: dict) -> list[int]:
    w, h = int(canvas.get("width")), int(canvas.get("height"))
    x, y, bw, bh = (int(round(v)) for v in box[:4])
    bw = max(1, min(bw, w))
    bh = max(1, min(bh, h))
    x = max(0, min(x, w - bw))
    y = max(0, min(y, h - bh))
    return [x, y, bw, bh]


def _measure(element: dict, canvas: dict, max_text_width: float) -> dict:
    mod = EFFECT_MODULES.get(str(element.get("type")))
    if mod is None:
        raise ValueError(f"no effect module for type {element.get('type')!r}")
    return mod.measure(element, canvas, max_text_width=max_text_width)


def _measure_stable(element: dict, canvas: dict, max_text_width: float) -> dict:
    """Re-measure at the resulting content width until the wrap is a fixed point.

    ``raster.py`` passes ``box_w - 2*padding`` as the text width, so the spec
    recorded here must be the one produced at exactly that width — otherwise the
    painter would re-wrap and paint outside the box.
    """
    width = float(max_text_width)
    spec = _measure(element, canvas, width)
    for _ in range(4):
        content_w = float(spec.get("content_w") or 0)
        if content_w <= 0 or abs(content_w - width) < 1.0:
            break
        width = content_w
        spec = _measure(element, canvas, width)
    return spec


def _candidates(zone: Sequence[int], bw: int, bh: int, kind: str) -> list[list[int]]:
    zx, zy, zw, zh = (int(v) for v in zone[:4])
    x_center = zx + (zw - bw) // 2
    x_left = zx
    slack = zh - bh
    if slack < 0:
        return []
    y_center = zy + slack // 2
    if kind in ("keyword", "number", "caption"):
        ys = [y_center, zy + int(slack * 0.18), zy + int(slack * 0.82), zy + slack, zy]
    elif kind == "list":
        ys = [zy + int(slack * 0.25), y_center, zy + int(slack * 0.08), zy, zy + slack]
    else:  # card
        ys = [y_center, zy + int(slack * 0.30), zy + int(slack * 0.70), zy]
    xs = [x_center] if x_center == zx else [x_center, x_left, zx + zw - bw]
    out: list[list[int]] = []
    for y in ys:
        for x in xs:
            out.append([int(x), int(y), int(bw), int(bh)])
    return out


def layout(element: dict, canvas: dict, vision: dict, occupied: list[dict]) -> dict:
    """Place one element; always returns a canvas-valid box (never raises)."""
    canvas = {"width": int(canvas.get("width") or 1080),
              "height": int(canvas.get("height") or 1920),
              "fps": float(canvas.get("fps") or 25.0)}
    info = safe_area_boxes(canvas, vision)
    zones: dict[str, list[int]] = info["zones"]
    face = info["face_box"]
    kind = str(element.get("type") or "keyword")
    order = ZONE_ORDER.get(kind, ("card", "title", "caption"))
    window = element.get("window") or None
    base_style = dict(element.get("style") or {})
    base_font = int(base_style.get("font_size_px") or base.scaled(84, canvas))
    pad = int(base_style.get("padding_px") or base.scaled(36, canvas))
    warnings: list[str] = []

    best: dict[str, Any] | None = None      # most tolerable degraded candidate
    for ratio in FONT_RATIOS:
        style = dict(base_style)
        style["font_size_px"] = max(12, int(round(base_font * ratio)))
        probe = dict(element)
        probe["style"] = style
        for zone_name in order:
            zone = zones.get(zone_name)
            if not zone:
                continue
            max_text_w = max(80.0, zone[2] - 2 * pad)
            try:
                spec = _measure_stable(probe, canvas, max_text_w)
            except Exception as exc:  # pragma: no cover - defensive
                warnings.append(f"measure failed in {zone_name}: {exc}")
                continue
            for box in _candidates(zone, int(spec["width"]), int(spec["height"]), kind):
                face_ratio = _face_ratio(box, face)
                occ_ratio = _occupied_ratio(box, occupied, window)
                if not _inside(box, canvas):
                    continue
                if face_ratio <= FACE_OVERLAP_RATIO and occ_ratio <= COLLISION_AREA_RATIO:
                    return _result(zone_name, box, style, spec, info, degraded=False,
                                   reason="", warnings=warnings)
                # remember the least-bad slot in case everything fails
                score = face_ratio * 4.0 + occ_ratio
                if best is None or score < best["score"]:
                    best = {"score": score, "zone": zone_name, "box": box,
                            "style": style, "spec": spec, "face_ratio": face_ratio,
                            "occ_ratio": occ_ratio}

    # ---- degraded: static safe slot, smallest damage, still fully on canvas
    if best is not None:
        box = _clamp_box(best["box"], canvas)
        reason = (f"no conflict-free slot; best zone={best['zone']} "
                  f"face_overlap={best['face_ratio']:.1%} "
                  f"occupied_overlap={best['occ_ratio']:.1%}")
        warnings.append("layout degraded: " + reason)
        return _result(best["zone"], box, best["style"], best["spec"], info,
                       degraded=True, reason=reason, warnings=warnings)

    # no zone / nothing measurable: fall back to a centred static box
    zone_name, zone = _first_zone(order, zones)
    if zone is None:
        zone = _zone_px(canvas, DEFAULT_ZONES["card"])
        zone_name = "card"
    ratio = FONT_RATIOS[-1]
    style = dict(base_style)
    style["font_size_px"] = max(12, int(round(base_font * ratio)))
    probe = {"style": style, "type": kind, "content": element.get("content") or {}}
    try:
        spec = _measure_stable(probe, canvas, max(80.0, zone[2] - 2 * pad))
        box = _clamp_box([zone[0], zone[1], min(spec["width"], zone[2]),
                          min(spec["height"], zone[3])], canvas)
    except Exception:
        spec = {"width": zone[2], "height": zone[3]}
        box = _clamp_box(zone, canvas)
    reason = "no usable safe zone measured; static centred fallback"
    warnings.append("layout degraded: " + reason)
    return _result(zone_name, box, style, spec, info, degraded=True, reason=reason,
                   warnings=warnings)


def _first_zone(order: Sequence[str], zones: dict) -> tuple[str | None, list[int] | None]:
    for name in order:
        if zones.get(name):
            return name, zones[name]
    for name, z in zones.items():
        return name, z
    return None, None


def _result(zone: str, box: Sequence[int], style: dict, spec: dict, info: dict, *,
            degraded: bool, reason: str, warnings: list[str]) -> dict:
    return {
        "mode": "static_safe" if not degraded else "degraded_static",
        "zone": zone,
        "box": [int(v) for v in box[:4]],
        "avoid_face": True,
        "face_safe_margin_px": int(info.get("margin_px") or DEFAULT_MARGIN_PX),
        "face_union": info.get("face_union"),
        "degraded": bool(degraded),
        "reason": reason,
        "warnings": warnings,
        "style": {"font_size_px": int(style.get("font_size_px")),
                  "padding_px": int(style.get("padding_px") or 0),
                  "radius_px": int(style.get("radius_px") or 0)},
        "intrinsic": {"width": int(spec.get("width") or 0),
                      "height": int(spec.get("height") or 0)},
    }


def collisions(boxes: Sequence[dict | Sequence[int]]) -> list[tuple[str, str, float]]:
    """Overlapping pairs with ratio = intersection / smaller element area."""
    norm: list[tuple[str, Sequence[float]]] = []
    for i, b in enumerate(boxes):
        if isinstance(b, dict):
            norm.append((str(b.get("id") or i), b.get("box") or [0, 0, 0, 0]))
        else:
            norm.append((str(i), b))
    out: list[tuple[str, str, float]] = []
    for i in range(len(norm)):
        for j in range(i + 1, len(norm)):
            ida, ba = norm[i]
            idb, bb = norm[j]
            inter = _inter(ba, bb)
            if inter <= 0:
                continue
            small = min(float(ba[2]) * float(ba[3]), float(bb[2]) * float(bb[3])) or 1.0
            out.append((ida, idb, inter / small))
    return out
