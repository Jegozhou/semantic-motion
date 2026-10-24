"""Semantic Motion director v2: semantic anchors -> an editable Motion IR.

The director decides *whether* an idea deserves motion and *which* composition
carries it. A layout solver then finds a place where the overlay does not touch
the speaker's silhouette, burned-in text, or the caption band. Nothing here
invents timestamps: every window derives from measured anchor onsets.

Placement hierarchy (best-scored candidate that satisfies every hard constraint):

1. ``scrim_right`` / ``scrim_left`` — a directional gradient column beside the
   person; the type sits against the frame edge where the scrim is strongest.
2. ``band_top`` — a top gradient band when there is headroom.
3. ``band_bottom`` — a bottom band when the person ends above the caption band.
4. ``card`` — a floating glass panel inside a measured quiet zone.
5. ``lower_third`` — last resort over the torso (never over face, hair or hands).

Output schema ``semotion.motion-ir/2`` is consumed by :mod:`semotion.render2`.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from . import effects2 as FX
from . import theme2 as T

SCHEMA = "semotion.motion-ir/2"
IR_VERSION = "2.1.0"

# Smart Balanced defaults -----------------------------------------------------
MAX_ELEMENTS_PER_10S = 4
MIN_GAP_S = 0.85
MAX_COVERAGE = 0.58
HOLD_S = 0.45
MIN_DWELL_S = 1.35
CAPTION_BAND = 0.14
ENTER_OVERLAP_LIMIT = 0.05

TYPE_ALIASES = {
    "card": "card", "keyword": "keyword", "list": "list", "list_item": "list",
    "number": "stat", "stat": "stat", "data": "stat", "chart": "barchart",
    "barchart": "barchart", "compare": "compare", "contrast": "compare",
    "quote": "quote",
}

# treatment preference per element type (higher = more editorial)
PREF = {
    "keyword": {"band_top": 0.10, "scrim_right": 0.12, "scrim_left": 0.10,
                "band_bottom": 0.08, "card": 0.04, "lower_third": 0.0},
    "card": {"scrim_right": 0.12, "scrim_left": 0.10, "card": 0.08,
             "band_top": 0.05, "band_bottom": 0.04, "lower_third": 0.0},
    "list": {"scrim_right": 0.12, "scrim_left": 0.10, "card": 0.08,
             "band_top": 0.04, "band_bottom": 0.04, "lower_third": 0.0},
    "stat": {"scrim_right": 0.12, "scrim_left": 0.10, "card": 0.08,
             "band_top": 0.04, "band_bottom": 0.04, "lower_third": 0.0},
    "barchart": {"card": 0.10, "scrim_right": 0.08, "band_top": 0.05,
                 "lower_third": 0.0},
    "compare": {"card": 0.10, "scrim_right": 0.08, "band_top": 0.05,
                "lower_third": 0.0},
    "quote": {"band_top": 0.08, "scrim_right": 0.10, "card": 0.05,
              "lower_third": 0.0},
}


# --------------------------------------------------------------------------- geo
def _clamp_box(box: Sequence[float], W: int, H: int) -> tuple[int, int, int, int]:
    x, y, w, h = [float(v) for v in box[:4]]
    x = max(0, min(int(round(x)), max(0, W - 1)))
    y = max(0, min(int(round(y)), max(0, H - 1)))
    w = max(1, min(int(round(w)), W - x))
    h = max(1, min(int(round(h)), H - y))
    return (x, y, w, h)


def _union(boxes: Iterable[Sequence[float]]) -> list[int] | None:
    bs = [b for b in boxes if b and len(b) >= 4 and float(b[2]) > 0 and float(b[3]) > 0]
    if not bs:
        return None
    x0 = min(float(b[0]) for b in bs)
    y0 = min(float(b[1]) for b in bs)
    x1 = max(float(b[0]) + float(b[2]) for b in bs)
    y1 = max(float(b[1]) + float(b[3]) for b in bs)
    return [int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0))]


def _intersect_area(a: Sequence[int], b: Sequence[int]) -> int:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy


def _intersects(a: Sequence[int], b: Sequence[int]) -> bool:
    return _intersect_area(a, b) > 0


def _erode(box: Sequence[int], dx: int, dy: int | None = None) -> tuple[int, int, int, int]:
    dy = dx if dy is None else dy
    x, y, w, h = box[:4]
    return (int(x + dx), int(y + dy), max(1, int(w - 2 * dx)), max(1, int(h - 2 * dy)))


def _overlap_frac(box: Sequence[int], other: Sequence[int] | None) -> float:
    if not other:
        return 0.0
    return _intersect_area(box, other) / max(1, box[2] * box[3])


# ------------------------------------------------------------------------ config
@dataclass
class DesignConfig:
    style: str = "smart_balanced"
    language: str = "zh"
    hold_s: float = HOLD_S
    max_elements_per_10s: int = MAX_ELEMENTS_PER_10S
    min_gap_s: float = MIN_GAP_S
    max_coverage: float = MAX_COVERAGE
    caption_band: float = CAPTION_BAND
    margin_ratio: float = 0.030
    min_col_ratio: float = 0.19        # a side column narrower than this is a card
    min_band_ratio: float = 0.115
    allow_torso: bool = True           # lower-third fallback over clothing
    text_conf_min: float = 0.50        # ignore low-confidence OCR noise (painting text)


@dataclass
class Candidate:
    treatment: str
    text_area: tuple[int, int, int, int]     # x, y, w, h available for type
    backdrop: tuple[int, int, int, int] | None
    zone_id: str = ""
    quiet: float = 0.5                        # 0..1, higher = smoother background
    align: str = "left"
    anchor: str = "center"
    category: str = "side"
    profile: list | None = None


# ------------------------------------------------------------------------ designer
class Designer:
    def __init__(self, *, source: dict, canvas: dict, anchors: list[dict],
                 transcript: dict | None = None, perception: dict | None = None,
                 vision: dict | None = None, decisions: dict | None = None,
                 frames: list[Image.Image] | None = None,
                 cfg: DesignConfig | None = None):
        self.source = source
        self.canvas = canvas
        self.W = int(canvas.get("width") or source.get("width") or 1080)
        self.H = int(canvas.get("height") or source.get("height") or 1920)
        self.fps = float(canvas.get("fps") or source.get("fps") or 25.0)
        self.duration = float(source.get("duration_s") or 0.0)
        self.anchors = list(anchors or [])
        self.transcript = transcript or {}
        self.perception = perception or {}
        self.vision = vision or {}
        self.decisions = decisions or {}
        self.cfg = cfg or DesignConfig(language=(self.transcript.get("language") or "zh"))
        self.frames = frames or []
        self.palette = T.extract_palette(self.frames)
        self.book = T.FontBook()
        self.ctx = FX.Ctx(book=self.book, palette=self.palette, canvas_w=self.W,
                          canvas_h=self.H, fps=self.fps,
                          language=self.cfg.language)
        self.p_frames: list[dict] = []
        self.person_union_all: list[int] | None = None
        self.face_union_all: list[int] | None = None
        self.hands_union_all: list[int] | None = None
        self.existing_text: list[dict] = []
        self.quiet_zones: list[dict] = []
        self.margin = int(max(48, self.cfg.margin_ratio * min(self.W, self.H)))
        self._last_zone: str | None = None
        self._last_end: float = -99.0
        self._current_start: float = 0.0
        self._last_audit: list[dict] = []
        self._quiet_cache: dict = {}
        self._mask_cache: dict = {}
        self.mask_scale = float((self.perception or {}).get("frames", [{}])[0].get("mask_scale") or 4.0) \
            if (self.perception or {}).get("frames") else 4.0
        self.perception_path = (self.perception or {}).get("_path")

    # ------------------------------------------------------------- perception
    def ingest(self):
        p = self.perception or {}
        regions = p.get("regions") or {}
        self.p_frames = [f for f in (p.get("frames") or []) if f.get("t_s") is not None]
        self.person_union_all = _union([f.get("person_bbox") for f in self.p_frames
                                        if f.get("person_bbox")]) or regions.get("person_union")
        fu = regions.get("face_union") or _union([f.get("face", {}).get("bbox")
                                                  for f in self.p_frames if f.get("face")])
        self.face_union_all = fu or self._face_from_vision()
        hand_boxes = []
        for f in self.p_frames:
            for h in f.get("hands") or []:
                if h.get("bbox"):
                    hand_boxes.append(h["bbox"])
        self.hands_union_all = _union(hand_boxes) or regions.get("hands_union")
        if not self.person_union_all and not self.p_frames:
            self.person_union_all = self._person_from_vision()

        raw_text = list(regions.get("existing_text") or []) + list(p.get("existing_text") or [])
        for t in raw_text:
            bbox = t.get("bbox")
            if not bbox:
                continue
            conf = float(t.get("conf") or t.get("confidence") or 1.0)
            kind = str(t.get("kind") or "other")
            text = str(t.get("text") or "")
            if conf < self.cfg.text_conf_min:
                continue
            if not text.strip() and kind != "logo":
                continue
            bb = [int(v) for v in bbox[:4]]
            if not _is_obstacle(bb, kind, self.W, self.H):
                continue
            self.existing_text.append({
                "bbox": bb,
                "t0": float(t.get("t0") or 0.0), "t1": float(t.get("t1") or self.duration),
                "kind": kind, "text": text, "conf": conf,
            })
        deduped: list[dict] = []
        for t in sorted(self.existing_text, key=lambda z: -z["conf"]):
            if any(_intersect_area(t["bbox"], d["bbox"]) >
                   0.6 * t["bbox"][2] * t["bbox"][3] for d in deduped):
                continue
            deduped.append(t)
        self.existing_text = deduped
        for z in (regions.get("quiet_zones") or []):
            if z.get("bbox"):
                self.quiet_zones.append(z)

    def _face_from_vision(self) -> list[int] | None:
        v = self.vision or {}
        boxes = [tr["bbox_union"] for tr in (v.get("face_tracks") or []) if tr.get("bbox_union")]
        if not boxes:
            boxes = [f["box"] for f in (v.get("faces") or []) if f.get("box")]
        return _union(boxes)

    def _person_from_vision(self) -> list[int] | None:
        u = self._face_from_vision()
        if not u:
            return None
        fx_, fy_, fw_, fh_ = u
        by = max(0, fy_ - 0.10 * fh_)
        return _clamp_box((fx_ - 0.52 * fw_, by, 2.08 * fw_, self.H - by), self.W, self.H)

    # ------------------------------------------------------ window-local unions
    def _frames_between(self, t0: float, t1: float) -> list[dict]:
        if not self.p_frames:
            return []
        lo, hi = t0 - 0.12, t1 + 0.12
        sel = [f for f in self.p_frames if lo <= float(f["t_s"]) <= hi]
        if not sel:
            sel = [min(self.p_frames, key=lambda f: abs(float(f["t_s"]) - (t0 + t1) / 2))]
        return sel

    def person_union_for(self, t0: float, t1: float) -> list[int] | None:
        fs = self._frames_between(t0, t1)
        u = _union([f.get("person_bbox") for f in fs if f.get("person_bbox")])
        return u or self.person_union_all

    def face_union_for(self, t0: float, t1: float) -> list[int] | None:
        fs = self._frames_between(t0, t1)
        u = _union([f.get("face", {}).get("bbox") for f in fs if f.get("face")])
        return u or self.face_union_all

    def hands_union_for(self, t0: float, t1: float) -> list[int] | None:
        boxes = []
        for f in self._frames_between(t0, t1):
            for h in f.get("hands") or []:
                if h.get("bbox"):
                    boxes.append(h["bbox"])
        return _union(boxes) or self.hands_union_all

    def text_obstacles(self, t0: float, t1: float) -> list[dict]:
        return [t for t in self.existing_text
                if t["t1"] >= t0 - 0.15 and t["t0"] <= t1 + 0.15]

    # ------------------------------------------------------------- silhouette
    def _mask_path(self, f: dict) -> Path | None:
        mp = f.get("mask")
        if not mp:
            return None
        p = Path(mp)
        if p.is_absolute() and p.exists():
            return p
        base = Path(self.perception_path).parent if self.perception_path else None
        if base is not None and (base / p).exists():
            return base / p
        return None

    def _union_mask(self, t0: float, t1: float):
        """Binary person mask for a window, unioned at the perception sample rate."""
        if not self.p_frames:
            return None
        key = (round(t0 * 2) / 2, round(t1 * 2) / 2)
        if key in self._mask_cache:
            return self._mask_cache[key]
        masks = []
        for f in self._frames_between(t0, t1):
            path = self._mask_path(f)
            if not path:
                continue
            try:
                im = Image.open(path).convert("L")
                if im.size[0] > 640:
                    im = im.resize((640, max(1, int(640 * im.size[1] / im.size[0]))))
                masks.append(np.asarray(im) > 24)
            except Exception:
                continue
        if not masks:
            self._mask_cache[key] = None
            return None
        h = min(m.shape[0] for m in masks); w = min(m.shape[1] for m in masks)
        u = np.zeros((h, w), dtype=bool)
        for m in masks:
            u |= m[:h, :w]
        self._mask_cache[key] = u
        return u

    def _band_edges(self, t0: float, t1: float, y0: float, y1: float):
        """Leftmost / rightmost person pixel inside a horizontal band (source px)."""
        u = self._union_mask(t0, t1)
        if u is None:
            pu = self.person_union_for(t0, t1)
            return (pu[0], pu[0] + pu[2]) if pu else None
        h, w = u.shape
        a = max(0, int(h * y0 / self.H)); b = min(h, max(a + 1, int(h * y1 / self.H)))
        band = u[a:b]
        cols = np.where(band.any(axis=0))[0]
        if cols.size == 0:
            return None
        sx = self.W / w
        return (int(cols[0] * sx), int((cols[-1] + 1) * sx))

    def edge_profile(self, t0: float, t1: float, side: str = "right",
                     samples: int = 48):
        """Per-row silhouette edge as [[y_px, x_px], ...] (source coordinates).

        A rectangular scrim would tint a moving shoulder; the renderer uses this
        profile to fade the gradient out exactly where the person begins.
        """
        u = self._union_mask(t0, t1)
        if u is None:
            return None
        h, w = u.shape
        sx = self.W / w
        out = []
        for i in range(samples):
            a = int(h * i / samples); b = max(a + 1, int(h * (i + 1) / samples))
            band = u[a:b]
            cols = np.where(band.any(axis=0))[0]
            y = int(self.H * (i + 0.5) / samples)
            if cols.size == 0:
                continue
            edge = int((cols[-1] + 1) * sx) if side == "right" else int(cols[0] * sx)
            out.append([y, edge])
        return out or None

    # ------------------------------------------------------------- quietness
    def _quietness(self, box: Sequence[int]) -> float:
        """0..1 smoothness of a region across sampled frames (1 = flat wall)."""
        if not self.frames:
            return 0.55
        key = tuple(int(v) for v in box[:4])
        if key in self._quiet_cache:
            return self._quiet_cache[key]
        x, y, w, h = key
        vals = []
        for img in self.frames[:5]:
            a = np.asarray(img.convert("L"), dtype=np.float32)
            crop = a[max(0, y):min(a.shape[0], y + h), max(0, x):min(a.shape[1], x + w)]
            if crop.size < 64:
                continue
            if crop.shape[0] > 200:
                st = max(1, crop.shape[0] // 200)
                crop = crop[::st, ::st]
            gy, gx = np.gradient(crop)
            vals.append(float(np.mean(np.abs(gx) + np.abs(gy)) / 255.0))
        q = 0.55 if not vals else float(max(0.0, min(1.0, 1.0 - float(np.mean(vals)) / 0.085)))
        self._quiet_cache[key] = q
        return q

    # ---------------------------------------------------------- candidate zones
    def candidates(self, t0: float, t1: float) -> list[Candidate]:
        W, H, m = self.W, self.H, self.margin
        top = int(0.05 * H)
        bottom = int((1 - self.cfg.caption_band) * H)
        pu = self.person_union_for(t0, t1)
        face = self.face_union_for(t0, t1)
        hands = self.hands_union_for(t0, t1)
        band_y0, band_y1 = int(0.18 * H), int(0.74 * H)
        edges = self._band_edges(t0, t1, band_y0, band_y1)
        out: list[Candidate] = []

        def quiet(box, fallback=0.5):
            q = self._quietness(box)
            zone_best = 0.0
            for z in self.quiet_zones:
                zb = z.get("bbox")
                if not zb:
                    continue
                if _intersect_area(zb, box) > 0.45 * box[2] * box[3]:
                    zone_best = max(zone_best, float(z.get("score") or 0.0))
            return max(q, zone_best * 0.95, fallback)

        col_min = int(self.cfg.min_col_ratio * W)
        if pu:
            p_right = edges[1] if edges else pu[0] + pu[2]
            p_left = edges[0] if edges else pu[0]
            right_gap = W - p_right - m
            if right_gap >= col_min:
                x0 = p_right + m
                inset = int(max(24, min(0.05 * W, right_gap * 0.14)))
                if W - m - (x0 + inset) >= 180:
                    area = (x0 + inset, band_y0, W - m - (x0 + inset), band_y1 - band_y0)
                    out.append(Candidate("scrim_right", area, (x0, 0, W - x0, H),
                                         zone_id="col_right", quiet=quiet(area),
                                         profile=self.edge_profile(t0, t1, "right")))
            left_gap = p_left - m
            if left_gap >= col_min:
                inset = int(max(24, min(0.05 * W, left_gap * 0.14)))
                if left_gap - inset - m >= 180:
                    area = (m, band_y0, left_gap - m - inset, band_y1 - band_y0)
                    out.append(Candidate("scrim_left", area, (0, 0, pu[0] - m, H),
                                         zone_id="col_left", quiet=quiet(area)))
            band_h = pu[1] - m
            if band_h >= int(self.cfg.min_band_ratio * H):
                area = (int(0.06 * W), int(0.03 * H), int(0.88 * W),
                        max(80, band_h - int(0.03 * H)))
                out.append(Candidate("band_top", area, (0, 0, W, pu[1]),
                                     zone_id="band_top", quiet=quiet((0, 0, W, pu[1])),
                                     anchor="center"))
            gap_bottom = bottom - (pu[1] + pu[3]) - m
            if gap_bottom >= int(self.cfg.min_band_ratio * H):
                y0 = pu[1] + pu[3] + m
                area = (int(0.06 * W), y0, int(0.88 * W), bottom - y0)
                out.append(Candidate("band_bottom", area, (0, y0, W, H - y0),
                                     zone_id="band_bottom",
                                     quiet=quiet((0, y0, W, H - y0)), anchor="center"))
        zones = sorted(self.quiet_zones, key=lambda z: -float(z.get("score") or 0))
        if not zones and pu:
            zx = pu[0] + pu[2] + m
            if W - zx > 0.22 * W:
                zones = [{"bbox": [zx, int(0.18 * H), W - zx - m, int(0.34 * H)],
                          "score": 0.5}]
        for i, z in enumerate(zones[:8]):
            zb0 = [int(v) for v in z["bbox"][:4]]
            x0 = max(zb0[0], m); y0 = max(zb0[1], top)
            zb = (x0, y0, min(zb0[2], W - x0 - m), min(zb0[3], bottom - y0))
            if zb[2] < 0.20 * W or zb[3] < int(0.075 * H):
                continue
            if pu and _overlap_frac(zb, pu) > 0.0:
                continue
            if face and _intersects(zb, face):
                continue
            if any(_intersects(zb, t["bbox"]) for t in self.text_obstacles(t0, t1)):
                continue
            area = _erode(zb, int(max(14, 0.02 * W)), int(max(12, 0.012 * H)))
            out.append(Candidate("card", area, zb, zone_id=f"zone{i}",
                                 quiet=quiet(zb), category="card"))
        if self.cfg.allow_torso and pu:
            band_h = int(0.26 * H)
            y0 = bottom - band_h
            if face and (face[1] + face[3]) > y0 - int(0.02 * H):
                y0 = face[1] + face[3] + int(0.02 * H)
            if bottom - y0 >= int(0.11 * H):
                area = (int(0.06 * W), y0, int(0.88 * W), bottom - y0)
                if not (hands and _intersects(area, hands)):
                    out.append(Candidate("lower_third", area, (0, y0, W, H - y0),
                                         zone_id="lower_third",
                                         quiet=quiet((0, y0, W, H - y0)),
                                         anchor="center", category="torso"))
        return out

    # ------------------------------------------------------------------ anchors
    def _groups(self) -> list[dict]:
        by_id = {a["id"]: a for a in self.anchors}
        groups: list[dict] = []
        used: set[str] = set()
        for dec in (self.decisions or {}).get("decisions") or []:
            ids = [i for i in (dec.get("anchor_ids") or [dec.get("anchor_id")]) if i in by_id]
            if not ids:
                continue
            used.update(ids)
            groups.append({
                "type": TYPE_ALIASES.get(str(dec.get("effect") or "").lower(), "keyword"),
                "anchors": [by_id[i] for i in ids],
                "hold_s": float(dec.get("hold_s") or self.cfg.hold_s),
                "emphasis": dec.get("emphasis") or "medium",
                "reason": dec.get("reason") or "",
                "confidence": float(dec.get("confidence") or 0.8),
                "params": dec.get("params") or {},
                "decision_id": dec.get("anchor_id"),
            })
        for a in self.anchors:
            if a["id"] in used:
                continue
            kind = TYPE_ALIASES.get(str(a.get("kind") or "keyword").lower(), "keyword")
            groups.append({"type": kind, "anchors": [a],
                           "hold_s": self.cfg.hold_s, "emphasis": "medium",
                           "reason": "", "confidence": float(a.get("confidence") or 0.6),
                           "params": {}, "decision_id": a["id"]})
        groups.sort(key=lambda g: float(g["anchors"][0].get("start_s") or 0.0))
        merged: list[dict] = []
        for g in groups:
            if merged and g["type"] == "list" and merged[-1]["type"] == "list":
                prev = merged[-1]
                last_end = float(prev["anchors"][-1].get("end_s") or 0)
                gap = float(g["anchors"][0].get("start_s") or 0) - last_end
                if gap <= 2.4 and len(prev["anchors"]) < 5:
                    prev["anchors"].extend(g["anchors"])
                    continue
            merged.append(g)
        return merged

    def _content_for(self, group: dict) -> dict | None:
        anchors = group["anchors"]
        gtype = group["type"]
        params = group.get("params") or {}
        first = anchors[0]
        text = str(first.get("text") or "").strip()
        if gtype == "list":
            rows = [{"marker": str(i + 1), "text": str(a.get("text") or "").strip(),
                     "at_s": max(0.0, float(a.get("start_s") or 0.0))}
                    for i, a in enumerate(anchors)]
            return {"rows": rows, "kicker": params.get("kicker") or "",
                    "items": [r["text"] for r in rows]}
        if gtype == "stat":
            disp, value, unit, dec = _anchor_number(first)
            if value is None:
                return {"text": text, "kicker": params.get("kicker") or ""}
            return {"value": value, "display": disp, "unit": unit,
                    "label": str(params.get("label") or ""), "decimals": dec,
                    "kicker": params.get("kicker") or ""}
        if gtype == "card":
            body = params.get("body") or (str(anchors[1].get("text") or "")
                                          if len(anchors) > 1 else "")
            return {"title": text, "body": body, "kicker": params.get("kicker") or "",
                    "emphasis": params.get("emphasis") or []}
        if gtype == "quote":
            return {"text": text, "attribution": params.get("attribution") or "",
                    "kicker": params.get("kicker") or ""}
        if gtype in ("barchart", "compare"):
            if params.get("bars"):
                return {"title": text, "bars": params["bars"],
                        "kicker": params.get("kicker") or ""}
            if params.get("left") or params.get("right"):
                return {"left": params.get("left") or {},
                        "right": params.get("right") or {},
                        "kicker": params.get("kicker") or ""}
            return None
        return {"text": text, "kicker": params.get("kicker") or "",
                "emphasis": params.get("emphasis") or []}

    # --------------------------------------------------------------------- build
    def build(self) -> dict:
        self.ingest()
        groups = self._groups()
        spans: list[list[float]] = []
        for g in groups:
            a0 = float(g["anchors"][0].get("start_s") or 0.0)
            a1 = float(g["anchors"][-1].get("end_s") or a0)
            e1 = min(a1 + float(g.get("hold_s") or self.cfg.hold_s),
                     (self.duration - 0.12) if self.duration else a1 + 0.45)
            if e1 - a0 < MIN_DWELL_S:
                e1 = min(a0 + MIN_DWELL_S,
                         (self.duration - 0.12) if self.duration else a0 + MIN_DWELL_S)
            spans.append([a0, e1])
        for i in range(len(spans) - 1):
            if spans[i][1] > spans[i + 1][0]:
                spans[i][1] = max(spans[i][0] + 0.9, spans[i + 1][0])

        elements: list[dict] = []
        audit: list[dict] = []
        occupied: list[tuple[float, float, tuple[int, int, int, int]]] = []
        budget: list[float] = []
        prev_start = -99.0

        for gi, g in enumerate(groups):
            content = self._content_for(g)
            if not content:
                audit.append({"group": gi, "decision": "drop", "reason": "no_content"})
                continue
            start, end = float(spans[gi][0]), float(spans[gi][1])
            budget = [t for t in budget if t > start - 10.0]
            if len(budget) >= self.cfg.max_elements_per_10s:
                audit.append({"group": gi, "decision": "drop", "reason": "density_budget",
                              "start_s": round(start, 3)})
                continue
            if elements and start - prev_start < self.cfg.min_gap_s:
                if start - elements[-1]["window"]["end_s"] < 0.35:
                    elements[-1]["window"]["end_s"] = max(
                        elements[-1]["window"]["end_s"], round(end, 3))
                    audit.append({"group": gi, "decision": "merge_into_previous",
                                  "reason": "min_gap", "start_s": round(start, 3)})
                    continue
            placed = self._place(g, content, start, end, occupied)
            if placed is None:
                audit.append({"group": gi, "decision": "drop",
                              "reason": "no_face_free_space", "start_s": round(start, 3),
                              "text": _content_label(content),
                              "candidates": self._last_audit})
                continue
            el, info = placed
            elements.append(el)
            self._last_zone = el["layout"]["zone"]
            self._last_end = el["window"]["end_s"]
            occupied.append((el["window"]["start_s"], el["window"]["end_s"], tuple(el["box"])))
            budget.append(start)
            prev_start = start
            audit.append({"group": gi, "decision": "place", "element": el["id"],
                          "treatment": info["treatment"], "zone": info["zone"],
                          "start_s": round(start, 3), "box": el["box"],
                          "text": _content_label(content)})

        coverage = self._enforce_coverage(elements, audit)
        return {
            "schema": SCHEMA,
            "ir_version": IR_VERSION,
            "style": self.cfg.style,
            "source": self.source,
            "canvas": {"width": self.W, "height": self.H, "fps": self.fps},
            "theme": self.palette.as_dict(),
            "semantic": {"language": self.cfg.language, "anchors": self.anchors,
                         "decisions_sha256": _sha(self.decisions) if self.decisions else None},
            "perception": {
                "schema": (self.perception or {}).get("schema"),
                "person_union": self.person_union_all,
                "face_union": self.face_union_all,
                "hands_union": self.hands_union_all,
                "existing_text": self.existing_text,
                "quiet_zone_count": len(self.quiet_zones),
                "used": bool(self.p_frames or self.quiet_zones),
            },
            "elements": elements,
            "layout_audit": {
                "decisions": audit,
                "coverage": round(coverage, 4),
                "perception_used": bool(self.p_frames),
                "text_obstacles": len(self.existing_text),
            },
            "provenance": {
                "director": "semotion.design2/2",
                "truth_class": ("HUMAN" if (self.decisions or {}).get("decision_model") == "human"
                                else "MACHINE"),
                "note": ("windows derive from measured anchor onsets; placement rejects any "
                         "box that touches the person silhouette, burned-in text or captions"),
            },
        }

    def _enforce_coverage(self, elements: list[dict], audit: list[dict]) -> float:
        dur = max(0.001, self.duration or max([e["window"]["end_s"] for e in elements] or [1.0]))
        total = sum(max(0.0, e["window"]["end_s"] - e["window"]["start_s"]) for e in elements)
        if total / dur <= self.cfg.max_coverage:
            return total / dur
        ranked = sorted(elements, key=lambda e: (-float(e.get("priority") or 0.5),
                                                 e["window"]["start_s"]))
        kept, used = [], 0.0
        for e in ranked:
            d = e["window"]["end_s"] - e["window"]["start_s"]
            if used + d <= self.cfg.max_coverage * dur * 1.02:
                kept.append(e); used += d
            else:
                audit.append({"decision": "drop", "reason": "coverage_budget",
                              "element": e["id"]})
        elements[:] = sorted(kept, key=lambda e: e["window"]["start_s"])
        return used / dur

    # ----------------------------------------------------------------- placement
    def _place(self, group: dict, content: dict, start: float, end: float,
               occupied: list[tuple[float, float, tuple[int, int, int, int]]]
               ) -> tuple[dict, dict] | None:
        gtype = group["type"]
        self._current_start = start
        cands = self.candidates(start, end)
        if not cands:
            self._last_audit = []
            return None
        face = self.face_union_for(start, end)
        hands = self.hands_union_for(start, end)
        pu = self.person_union_for(start, end)
        obstacles = self.text_obstacles(start, end)
        audit: list[dict] = []
        best = None
        for c in cands:
            tx, ty, tw, th = c.text_area
            rec = {"treatment": c.treatment, "zone": c.zone_id,
                   "text_area": [tx, ty, tw, th], "quiet": round(c.quiet, 3),
                   "status": "no_fit"}
            fit = None
            for tier in (1.0, 0.92, 0.84, 0.76, 0.68, 0.62):
                fit = FX.fit_element({"type": gtype, "content": content,
                                      "backdrop": {"kind": c.treatment}},
                                     self.ctx, tw, th, tier=tier)
                if fit is not None:
                    break
            if fit is None:
                rec["reason"] = "text_too_long_for_area"
                audit.append(rec)
                continue
            box = self._position(fit, c)
            rec.update(box=list(box), fit_w=fit.width, fit_h=fit.height)
            if c.treatment == "lower_third":
                body_ok = not (face and _intersects(box, face)) and \
                    not (hands and _intersects(box, hands))
            else:
                body_ok = not (pu and _intersects(box, pu))
            if not body_ok:
                rec.update(status="rejected", reason="covers_person")
                audit.append(rec)
                continue
            if any(_intersects(box, t["bbox"]) for t in obstacles):
                rec.update(status="rejected", reason="burns_in_text")
                audit.append(rec)
                continue
            if face and _intersects(box, face) and c.treatment != "lower_third":
                rec.update(status="rejected", reason="covers_face")
                audit.append(rec)
                continue
            if any(_time_hit(start, end, o[0], o[1]) and _intersects(box, o[2])
                   for o in occupied):
                rec.update(status="rejected", reason="collides_with_visible_element")
                audit.append(rec)
                continue
            if c.treatment not in ("scrim_right", "scrim_left", "band_top", "band_bottom") \
                    and c.backdrop and pu and _overlap_frac(c.backdrop, pu) > 0.02:
                rec.update(status="rejected", reason="backdrop_covers_person")
                audit.append(rec)
                continue
            score = self._score(c, fit, box, face, start)
            rec.update(status="ok", score=round(score, 4))
            audit.append(rec)
            if best is None or score > best[0]:
                best = (score, c, fit, box, rec)
        self._last_audit = sorted(audit, key=lambda r: -float(r.get("score") or 0))[:8]
        if best is None:
            return None
        score, c, fit, box, rec = best
        el = self._element(group, content, fit, box, c, start, end)
        el["layout"]["candidates"] = self._last_audit
        return el, {"treatment": c.treatment, "zone": c.zone_id}

    def _position(self, fit: FX.Fit, c: Candidate) -> tuple[int, int, int, int]:
        # keep at least one safe margin from every frame edge
        W, H = self.W, self.H
        x, y, w, h = c.text_area
        bw, bh = fit.width, fit.height
        if c.treatment == "scrim_right":
            tx = x + w - bw
            ty = y + (h - bh) // 2
        elif c.treatment == "scrim_left":
            tx = x
            ty = y + (h - bh) // 2
        elif c.treatment == "band_bottom":
            tx = x + (w - bw) // 2
            ty = y + int((h - bh) * 0.58)
        elif c.treatment in ("band_top", "lower_third"):
            tx = x + (w - bw) // 2
            ty = y + (h - bh) // 2
        else:
            tx = x + (w - bw) // 2
            ty = y + (h - bh) // 2
        tx = max(0, min(tx, W - bw))
        ty = max(0, min(ty, H - bh))
        return _clamp_box((tx, ty, bw, bh), W, H)

    def _score(self, c: Candidate, fit: FX.Fit, box: Sequence[int],
               face: Sequence[int] | None, start: float) -> float:
        s = 0.40 * c.quiet
        s += PREF.get(fit.kind, PREF["keyword"]).get(c.treatment, 0.0)
        if self._last_zone and c.zone_id == self._last_zone and 0 <= start - self._last_end < 9.0:
            s += 0.10
        if fit.kind in ("keyword", "card"):
            want = T.type_px("display" if fit.kind == "keyword" else "title", self.W, self.H)
            got = fit.display_px or fit.title_px
            s += 0.08 * max(0.0, min(1.0, (got / max(1, want) - 0.6) / 0.4))
        if face:
            dx = max(0, box[0] - (face[0] + face[2]), face[0] - (box[0] + box[2]))
            dy = max(0, box[1] - (face[1] + face[3]), face[1] - (box[1] + box[3]))
            s += 0.06 * min(1.0, math.hypot(dx, dy) / (0.22 * math.hypot(self.W, self.H)))
        if c.category == "torso":
            s -= 0.18
        if c.treatment in ("scrim_right", "scrim_left"):
            s += 0.10          # an edge column reads more editorial than a floating card
        return s

    def _element(self, group: dict, content: dict, fit: FX.Fit, box: Sequence[int],
                 c: Candidate, start: float, end: float) -> dict:
        fps = self.fps
        sf = int(round(start * fps))
        ef = max(sf + 1, int(round(end * fps)))
        eid = f"el_{len(group['anchors'])}_{_short_hash(group, content)}"
        rows = [{"marker": r["marker"], "text": r["text"],
                 "at_s": round(max(0.0, float(r.get("at_s") or 0.0) - start), 3),
                 "lines": r["lines"]} for r in fit.rows]
        t = group["type"]
        motion = {
            "enter": {"style": {"keyword": "wipe", "card": "wipe", "quote": "wipe",
                                "list": "rows", "stat": "count", "barchart": "bars",
                                "compare": "columns"}.get(t, "fade_up"),
                      "dur_s": 0.42, "ease": "out_expo", "offset_y": 12},
            "exit": {"dur_s": 0.18, "ease": "in_quad", "offset_y": -6},
            "hold": {"style": "none"},
        }
        backdrop = None
        if c.backdrop:
            kind = {"scrim_right": "scrim_right", "scrim_left": "scrim_left",
                    "band_top": "band_top", "band_bottom": "band_bottom",
                    "lower_third": "band_bottom", "card": "card"}.get(c.treatment, "card")
            backdrop = {"kind": kind, "box": [int(v) for v in c.backdrop],
                        "alpha": 0.86 if kind.startswith(("scrim", "band")) else 0.80,
                        "blur_px": 26 if kind == "card" else 0,
                        "enter_s": 0.34, "exit_s": 0.22}
            if c.profile:
                backdrop["profile"] = c.profile
                backdrop["profile_side"] = "right" if kind == "scrim_right" else "left"
        return {
            "id": eid,
            "type": t,
            "anchor_refs": [a["id"] for a in group["anchors"]],
            "window": {"start_s": round(start, 4), "end_s": round(end, 4),
                       "start_frame": sf, "end_frame": ef,
                       "hold_s": round(float(group.get("hold_s") or self.cfg.hold_s), 3),
                       "anchor_start_s": round(float(group["anchors"][0].get("start_s") or start), 4),
                       "anchor_end_s": round(float(group["anchors"][-1].get("end_s") or end), 4)},
            "box": [int(v) for v in box],
            "backdrop": backdrop,
            "content": content,
            "fit": {"kind": fit.kind, "width": fit.width, "height": fit.height,
                    "pad": fit.pad,
                    "font_size_px": max(fit.display_px, fit.title_px, fit.body_px),
                    "min_font_px": fit.min_px,
                    "display_px": fit.display_px, "title_px": fit.title_px,
                    "body_px": fit.body_px, "kicker_px": fit.kicker_px, "pitch": fit.pitch,
                    "lines": fit.lines, "body_lines": fit.body_lines, "kicker": fit.kicker,
                    "rows": rows, "bars": fit.bars, "compare": fit.compare,
                    "quote": fit.quote, "rule": fit.rule, "align": fit.align,
                    "num_str": fit.num_str, "unit": fit.unit, "label": fit.label,
                    "emphasis": fit.emphasis},
            "motion": motion,
            "priority": {"stat": 0.95, "barchart": 0.9, "compare": 0.85, "list": 0.8,
                         "card": 0.75, "keyword": 0.7, "quote": 0.65}.get(t, 0.6),
            "layout": {"treatment": c.treatment, "zone": c.zone_id,
                       "zone_box": list(c.text_area), "category": c.category,
                       "reason": group.get("reason") or "", "candidates": []},
            "provenance": {"decision_id": group.get("decision_id"),
                           "decision_confidence": group.get("confidence"),
                           "emphasis": group.get("emphasis"),
                           "reason": group.get("reason") or ""},
        }


# --------------------------------------------------------------------- utilities
def _is_obstacle(bbox: Sequence[int], kind: str, W: int, H: int) -> bool:
    """Burned-in subtitles and small credits matter; background signage does not.

    A venue wall reading "HOUSE" is part of the shot, not a caption: darkening it
    with a scrim is normal cinematography. A subtitle line or a credit must never
    be covered, so those stay hard obstacles.
    """
    x, y, w, h = [int(v) for v in bbox[:4]]
    if w < 0.02 * W or h < 0.012 * H:
        return False
    if kind in ("subtitle", "caption"):
        return True
    if h <= 0.055 * H:
        return True
    if kind == "logo" and h <= 0.16 * H and (y <= 0.16 * H or y + h >= 0.84 * H):
        return True
    return False


def _time_hit(a0: float, a1: float, b0: float, b1: float) -> bool:
    return min(a1, b1) - max(a0, b0) > ENTER_OVERLAP_LIMIT


def _content_label(content: dict) -> str:
    return str(content.get("text") or content.get("title") or content.get("value") or "")[:40]


def _short_hash(group: dict, content: dict) -> str:
    blob = json.dumps({"ids": [a["id"] for a in group["anchors"]], "type": group["type"],
                       "content": content}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


_ZH_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}
_ZH_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def _cn_number(s: str) -> float | None:
    total, section, num = 0, 0, 0
    seen = False
    for ch in s:
        if ch in _ZH_DIGITS:
            num = _ZH_DIGITS[ch]; seen = True
        elif ch in _ZH_UNITS:
            unit = _ZH_UNITS[ch]
            if unit == 10000:
                section = (section + num) * unit
                total += section
                section = 0
            else:
                section += (num or 1) * unit
            num = 0; seen = True
        else:
            return None
    return float(total + section + num) if seen else None


def _number_tokens(text: str) -> list[dict]:
    out = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(%|％|倍|万|亿|千|k|K|M|元|块|美元|分|秒|分钟|小时|天|年|个月)?",
                         text):
        raw = m.group(1)
        dec = len(raw.split(".")[1]) if "." in raw else 0
        unit = (m.group(2) or "").replace("％", "%")
        out.append({"value": float(raw), "unit": unit, "decimals": dec, "raw": m.group(0)})
    if out:
        return out
    m = re.search(r"([零〇一二两三四五六七八九十百千万]+)\s*(成|折|%|％)", text)
    if m:
        v = _cn_number(m.group(1))
        if v is not None:
            unit = m.group(2)
            if unit in ("成", "折"):
                v = v * 10
            return [{"value": v, "unit": "%", "decimals": 0, "raw": m.group(0)}]
    return []


def _anchor_number(anchor: dict) -> tuple[str, float | None, str, int]:
    text = str(anchor.get("text") or "").strip()
    for tok in _number_tokens(text):
        return text, tok["value"], tok["unit"], tok["decimals"]
    return text, None, "", 0


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def design(*, source: dict, canvas: dict, anchors: list[dict],
           transcript: dict | None = None, perception: dict | None = None,
           vision: dict | None = None, decisions: dict | None = None,
           frames: list[Image.Image] | None = None,
           cfg: DesignConfig | None = None) -> dict:
    d = Designer(source=source, canvas=canvas, anchors=anchors, transcript=transcript,
                 perception=perception, vision=vision, decisions=decisions,
                 frames=frames, cfg=cfg)
    return d.build()
