"""Spatial layout solver.

For each element we know its lifetime [t_in, t_end] and a few candidate shapes (width variants).
We build a per-element cost field on the analysis grid from what the camera sees during that
lifetime (face/hair, body, hands, motion, existing on-screen text, background clutter), reject
any box touching the face or leaving the platform safe area or colliding with other visible
graphics, and minimise  area-cost + preference-distance + size/continuity terms.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .vision import VisionData

W_HEAD, W_PERSON, W_MOTION, W_HANDS, W_TEXT, W_CLUTTER = 60.0, 8.0, 3.0, 4.0, 10.0, 0.6


@dataclass
class Fields:
    cost: np.ndarray       # [gh,gw] float cost density
    hard: np.ndarray       # [gh,gw] bool forbidden (face/head, safe area)
    head: np.ndarray
    person: np.ndarray
    text: np.ndarray
    lum: np.ndarray
    clutter: np.ndarray
    face_box: Optional[list]   # normalized [x,y,w,h] median face over window
    person_box: Optional[list]
    open_side: str


def _warp(m: np.ndarray, zoom: float, focus: tuple[float, float]) -> np.ndarray:
    """Apply a camera zoom about `focus` (normalized) to a grid map (what the viewer sees)."""
    if abs(zoom - 1.0) < 1e-3:
        return m
    gh, gw = m.shape[:2]
    fx, fy = focus[0] * gw, focus[1] * gh
    M = np.float32([[zoom, 0, fx - zoom * fx], [0, zoom, fy - zoom * fy]])
    return cv2.warpAffine(m.astype(np.float32), M, (gw, gh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def build_fields(vd: VisionData, t0: float, t1: float, safe: tuple, camera: Optional[list] = None,
                 face_margin: float = 0.18, chin_margin: Optional[float] = None) -> Fields:
    """Cost field + hard obstacle mask for the lifetime [t0, t1]. Face boxes get a margin (more above for hair);
    `chin_margin` (fraction of face height) relaxes the margin below the chin, used for portrait video where
    text over the chest, just under the chin, is the platform convention."""
    idx = vd.samples_between(t0, t1)
    gh, gw = vd.gh, vd.gw

    def zoom_at(t: float):
        if not camera:
            return 1.0, (0.5, 0.5)
        for c in camera:
            if c["t0"] <= t < c["t1"]:
                return c["zoom"], tuple(c["focus"])
        return 1.0, (0.5, 0.5)

    heads, persons, motions = [], [], []
    face_boxes = []
    hand_pts = []
    for i in idx:
        z, f = zoom_at(float(vd.t[i]))
        heads.append(_warp(vd.head[i], z, f))
        persons.append(_warp(vd.person[i], z, f))
        motions.append(_warp(vd.motion[i], z, f))
        for fb in vd.faces[int(i)]:
            x, y, w, h = fb[:4]
            # zoom transform of the face box
            cx, cy = (x + w / 2 - f[0]) * z + f[0], (y + h / 2 - f[1]) * z + f[1]
            face_boxes.append([cx - w * z / 2, cy - h * z / 2, w * z, h * z])
        for hx, hy, hv in vd.hands[int(i)]:
            hand_pts.append(((hx - f[0]) * z + f[0], (hy - f[1]) * z + f[1]))
    head = np.max(np.array(heads), axis=0)
    person = np.max(np.array(persons), axis=0)
    motion = np.mean(np.array(motions), axis=0)
    shots = sorted(set(int(vd.sample_shot[i]) for i in idx))
    text = np.maximum(vd.overlay_text, np.max(vd.shot_text[shots], axis=0))
    clutter = np.mean(vd.shot_clutter[shots], axis=0)
    lum = np.mean(vd.shot_lum[shots], axis=0)
    cl = clutter / (np.percentile(clutter, 95) + 1e-6)
    hands = np.zeros((gh, gw), np.float32)
    for hx, hy in hand_pts:
        cv2.circle(hands, (int(hx * gw), int(hy * gh)), max(2, gw // 18), 1.0, -1)
    if hand_pts:
        hands = cv2.GaussianBlur(hands, (0, 0), 1.5)
        hands /= max(1e-6, hands.max())
    cost = (W_HEAD * head + W_PERSON * person + W_MOTION * motion + W_HANDS * hands + W_TEXT * np.clip(text * 1.5, 0, 1)
            + W_CLUTTER * np.clip(cl, 0, 2))
    hard = cv2.dilate((head > 0.35).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    for x, y, w, h in face_boxes:
        mx, my = w * face_margin, h * face_margin
        x0, y0 = int(math.floor((x - mx) * gw)), int(math.floor((y - my * 1.6) * gh))  # extra room above (hair)
        x1, y1 = int(math.ceil((x + w + mx) * gw)), int(math.ceil((y + h + (h * chin_margin if chin_margin is not None else my)) * gh))
        hard[max(0, y0):max(0, min(gh, y1)), max(0, x0):max(0, min(gw, x1))] = True
    # existing burned-in text (subtitles, name tags) is hard too: never draw over it
    hard |= cv2.dilate((text > 0.45).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    sx0, sy0, sx1, sy1 = safe
    outside = np.ones((gh, gw), bool)
    outside[int(math.ceil(sy0 * gh)):int(math.floor(sy1 * gh)), int(math.ceil(sx0 * gw)):int(math.floor(sx1 * gw))] = False
    hard |= outside
    fb = None
    if face_boxes:
        fa = np.array(face_boxes)
        fb = [float(np.median(fa[:, k])) for k in range(4)]
    from .vision import _box_from_mask

    pb = _box_from_mask(person, 0.5)
    left = float(cost[:, : gw // 2].mean())
    right = float(cost[:, gw // 2:].mean())
    open_side = "right" if right < left * 0.85 else ("left" if left < right * 0.85 else "balanced")
    return Fields(cost=cost.astype(np.float32), hard=hard, head=head, person=person, text=text, lum=lum, clutter=cl,
                  face_box=fb, person_box=pb, open_side=open_side)


def _integral(a: np.ndarray) -> np.ndarray:
    return cv2.integral(a.astype(np.float64))


def _box_sum(ii: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    return float(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])


REGION_PREFS = {
    # normalized preference points by orientation; 'side' points are resolved against the open side
    "landscape": {"top": (0.5, 0.14), "bottom": (0.5, 0.80), "center": (0.5, 0.5), "top_left": (0.12, 0.14),
                  "top_right": (0.88, 0.14), "bottom_left": (0.12, 0.80), "bottom_right": (0.88, 0.80)},
    "portrait": {"top": (0.5, 0.16), "bottom": (0.5, 0.70), "center": (0.5, 0.55), "top_left": (0.2, 0.16),
                 "top_right": (0.8, 0.16), "bottom_left": (0.2, 0.72), "bottom_right": (0.8, 0.72)},
}


def preference_point(region: str, etype: str, f: Fields, orientation: str, prev_side: Optional[str]) -> tuple[float, float, str]:
    """Return (px, py, side) preferred centre of the element box."""
    fb = f.face_box
    face_cx = fb[0] + fb[2] / 2 if fb else 0.5
    face_cy = fb[1] + fb[3] / 2 if fb else 0.38
    side = f.open_side
    if side == "balanced":
        side = prev_side or ("right" if face_cx <= 0.5 else "left")
    if region in ("left", "right"):
        side = region
    o = "portrait" if orientation == "portrait" else "landscape"
    prefs = REGION_PREFS[o]
    if region in prefs:
        px, py = prefs[region]
        return px, py, ("left" if px < 0.45 else "right" if px > 0.55 else "center")
    if o == "portrait":
        # vertical video: graphics live above the head or in the chest band, centred
        if region == "above_head" or (region == "auto" and etype in ("title",)):
            return 0.5, max(0.14, (fb[1] if fb else 0.3) - 0.10), "center"
        if region in ("beside_face",):
            return (0.75 if face_cx < 0.5 else 0.25), face_cy, ("right" if face_cx < 0.5 else "left")
        if etype == "lower_third":
            return 0.38, 0.70, "left"
        base_y = (fb[1] + fb[3] + 0.16) if fb else 0.62
        return 0.5, min(0.72, max(0.5, base_y)), "center"
    # landscape
    if region == "above_head":
        return face_cx, max(0.12, (fb[1] if fb else 0.25) - 0.10), "center"
    if etype == "lower_third" and region == "auto":
        return (0.24, 0.80, "left") if face_cx > 0.4 else (0.76, 0.80, "right")
    if etype == "title" and region == "auto":
        x = 0.27 if side == "left" else 0.73
        return x, 0.30, side
    x_open = 0.22 if side == "left" else 0.78
    if fb:
        # sit in the middle of the open band between the face and the frame edge
        if side == "left":
            x_open = min(0.30, max(0.16, fb[0] / 2))
        else:
            x_open = max(0.70, min(0.84, (fb[0] + fb[2] + 1.0) / 2))
    if region in ("beside_face",) or (region == "auto" and etype in ("keyword", "stat", "callout")):
        return x_open, min(0.55, face_cy + 0.04), side
    return x_open, 0.48, side


def solve_box(f: Fields, variants: list[dict], W: int, H: int, pref: tuple[float, float], avoid: list[tuple],
              region: str, vd_grid: tuple[int, int], scale_steps=(1.0, 0.88, 0.76), continuity: Optional[tuple] = None,
              margin_u: float = 1.2) -> Optional[dict]:
    """Search all grid positions for every variant and scale. Returns best placement or None."""
    gw, gh = vd_grid
    cw, chh = W / gw, H / gh
    u = min(W, H) / 100.0
    ii_cost = _integral(f.cost)
    ii_hard = _integral(f.hard.astype(np.float32))
    ii_person = _integral((f.person > 0.5).astype(np.float32))
    px, py = pref
    best = None
    avoid_cells = []
    for (ax, ay, aw, ah) in avoid:
        m = margin_u * u
        avoid_cells.append((int(math.floor((ax - m) / cw)), int(math.floor((ay - m) / chh)),
                            int(math.ceil((ax + aw + m) / cw)), int(math.ceil((ay + ah + m) / chh))))
    for si, sc in enumerate(scale_steps):
        for vi, v in enumerate(variants):
            bw, bh = v["w"] * sc, v["h"] * sc
            nx, ny = int(math.ceil(bw / cw)), int(math.ceil(bh / chh))
            if nx >= gw or ny >= gh:
                continue
            for y0 in range(0, gh - ny + 1):
                for x0 in range(0, gw - nx + 1):
                    x1, y1 = x0 + nx, y0 + ny
                    if _box_sum(ii_hard, x0, y0, x1, y1) > 0:
                        continue
                    if any(not (x1 <= a0 or x0 >= a1 or y1 <= b0 or y0 >= b1) for a0, b0, a1, b1 in avoid_cells):
                        continue
                    area = nx * ny
                    c_area = _box_sum(ii_cost, x0, y0, x1, y1) / area
                    p_frac = _box_sum(ii_person, x0, y0, x1, y1) / area
                    cx, cy = (x0 + nx / 2) / gw, (y0 + ny / 2) / gh
                    d2 = ((cx - px) * (W / max(W, H))) ** 2 + ((cy - py) * (H / max(W, H))) ** 2
                    cost = c_area + 18.0 * p_frac + 40.0 * d2 + 0.35 * vi + 2.5 * si
                    if continuity is not None:
                        # stay aligned with the previous element's leading edge if it was on the same side
                        lx = continuity[0] / W
                        cost += 6.0 * min(abs(x0 / gw - lx), 0.2)
                    if best is None or cost < best["cost"]:
                        best = {"cost": round(cost, 3), "variant": vi, "scale": sc, "cells": [x0, y0, x1, y1],
                                "c_area": round(c_area, 3), "person_frac": round(p_frac, 3), "dist2": round(d2, 4)}
        if best is not None and best["scale"] == sc:
            break
    if best is None:
        return None
    x0, y0, x1, y1 = best["cells"]
    v = variants[best["variant"]]
    sc = best["scale"]
    bw, bh = v["w"] * sc, v["h"] * sc
    # centre the exact box inside its cell span, then pull toward the preference point within the span
    bx = x0 * cw + ((x1 - x0) * cw - bw) / 2
    by = y0 * chh + ((y1 - y0) * chh - bh) / 2
    best["box"] = [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)]
    best["layout_w"] = v["layout_w"]
    return best


def box_stats(f: Fields, box: list, W: int, H: int) -> dict:
    """Measured overlap / background stats under a pixel box (for QA and backing decisions)."""
    gh, gw = f.cost.shape
    x, y, w, h = box
    x0, y0 = max(0, int(x / W * gw)), max(0, int(y / H * gh))
    x1, y1 = min(gw, int(math.ceil((x + w) / W * gw))), min(gh, int(math.ceil((y + h) / H * gh)))
    if x1 <= x0 or y1 <= y0:
        return {}
    sl = (slice(y0, y1), slice(x0, x1))
    return {"head_max": round(float(f.head[sl].max()), 3), "person_frac": round(float((f.person[sl] > 0.5).mean()), 3),
            "text_max": round(float(f.text[sl].max()), 3), "lum_mean": round(float(f.lum[sl].mean()), 3),
            "lum_p90": round(float(np.percentile(f.lum[sl], 90)), 3), "clutter": round(float(f.clutter[sl].mean()), 3),
            "hard_cells": int(f.hard[sl].sum())}
