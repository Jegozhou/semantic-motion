from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import skia

from ..render.easing import ease, prog
from ..render.fonts import FontBook
from ..render.text import UnitLook

REGISTRY: dict[str, type] = {}


def register(cls):
    REGISTRY[cls.type] = cls
    return cls


def get_element_class(t: str):
    if t not in REGISTRY:
        raise KeyError(f"unknown element type '{t}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[t]


@dataclass
class RenderCtx:
    W: int                 # reference (source) frame size
    H: int
    fb: FontBook
    style: dict
    orientation: str

    @property
    def u(self) -> float:
        return min(self.W, self.H) / 100.0

    def ts(self, key: str) -> float:
        """Type size in px for a style token (portrait variant when present)."""
        t = self.style["type"]
        if self.orientation == "portrait" and f"{key}_portrait" in t:
            return t[f"{key}_portrait"] * self.u
        return t[key] * self.u

    def w(self, key: str, default: int = 400) -> int:
        return int(self.style["weights"].get(key, default))


# preferred layout widths (fraction of frame width) per orientation
PREF_WIDTHS = {
    "landscape": {"keyword": [0.30, 0.40, 0.56], "stat": [0.26, 0.36], "list": [0.30, 0.38, 0.48], "compare": [0.44, 0.56],
                  "chart": [0.32, 0.42, 0.52], "flow": [0.36, 0.56, 0.70], "quote": [0.36, 0.46, 0.60], "title": [0.40, 0.60],
                  "lower_third": [0.36], "callout": [0.24, 0.32], "custom": [1.0], "captions": [0.72]},
    "portrait": {"keyword": [0.62, 0.78], "stat": [0.56, 0.74], "list": [0.70, 0.80], "compare": [0.80],
                 "chart": [0.72, 0.80], "flow": [0.62, 0.80], "quote": [0.72, 0.80], "title": [0.70, 0.80],
                 "lower_third": [0.64], "callout": [0.50, 0.64], "custom": [1.0], "captions": [0.80]},
}


class Element:
    type = "base"
    layer = "primary"          # primary | secondary | overlay
    default_region = "side"

    def __init__(self, spec: dict, ctx: RenderCtx):
        self.spec = spec
        self.ctx = ctx
        self.c = spec.get("content", {})
        self.u = ctx.u
        self.style = ctx.style
        self.m = ctx.style["motion"]
        self.t_in = spec.get("t_in", 0.0)
        self.t_land = spec.get("t_land", self.t_in)
        self.t_out = spec.get("t_out", self.t_in + 3)
        self.t_end = spec.get("t_end", self.t_out + self.m["exit"])
        self.exit_mode = spec.get("exit_mode", "anim")
        self.energy = float((spec.get("motion") or {}).get("energy", 0.5))
        self.build(spec.get("layout_w") or self.default_width())

    # ---------------------------------------------------------------- sizing
    def default_width(self) -> float:
        return PREF_WIDTHS[self.ctx.orientation if self.ctx.orientation in PREF_WIDTHS else "landscape"].get(self.type, [0.4])[0] * self.ctx.W

    @classmethod
    def variants(cls, spec: dict, ctx: RenderCtx) -> list[dict]:
        o = ctx.orientation if ctx.orientation in PREF_WIDTHS else "landscape"
        out = []
        for f in PREF_WIDTHS[o].get(cls.type, [0.4]):
            el = cls({**spec, "layout_w": f * ctx.W}, ctx)
            w, h = el.size()
            out.append({"layout_w": f * ctx.W, "w": w, "h": h})
        # dedupe identical sizes
        uniq = []
        for v in out:
            if not any(abs(v["w"] - q["w"]) < 2 and abs(v["h"] - q["h"]) < 2 for q in uniq):
                uniq.append(v)
        return uniq

    def build(self, max_w: float) -> None:
        raise NotImplementedError

    def size(self) -> tuple[float, float]:
        raise NotImplementedError

    # ---------------------------------------------------------------- timing helpers
    @property
    def enter_dur(self) -> float:
        return float(self.spec.get("enter_dur", self.m["enter"]))

    @property
    def exit_dur(self) -> float:
        return float(self.spec.get("exit_dur", self.m["exit"]))

    def p_in(self, t: float, delay: float = 0.0, dur: Optional[float] = None, start: Optional[float] = None) -> float:
        s = (self.t_in if start is None else start) + delay
        return ease(self.m["ease_in_curve"], prog(t, s, self.enter_dur if dur is None else dur))

    def p_out(self, t: float, delay: float = 0.0) -> float:
        if self.exit_mode == "cut":
            return 1.0 if t >= self.t_end - 1e-6 else 0.0
        return ease(self.m["ease_out_curve"], prog(t, self.t_out + delay, self.exit_dur))

    def active(self, t: float) -> bool:
        return self.t_in - 1e-6 <= t < self.t_end

    def reveal_look(self, t: float, start: float, dur: Optional[float] = None) -> UnitLook:
        """Standard unit entrance (rise + fade + de-blur) and exit (sink + fade)."""
        dur = self.enter_dur if dur is None else dur
        pi = ease(self.m["ease_in_curve"], prog(t, start, dur))
        po = self.p_out(t)
        travel = self.m["travel"] * self.u * (0.7 + 0.6 * self.energy)
        a = min(1.0, pi * 1.25) * (1 - po)
        dy = (1 - pi) * travel - po * 0.55 * travel
        blur = (1 - pi) * self.m["blur_in"] * self.u + po * 0.3 * self.u
        sc = 1.0
        if self.m.get("overshoot"):
            from ..render.easing import spring
            sc = 0.9 + 0.1 * spring(prog(t, start, dur * 1.6))
        return UnitLook(alpha=a, dy=dy, blur=blur, scale=sc)

    # ---------------------------------------------------------------- drawing
    def draw(self, canvas: skia.Canvas, t: float) -> None:
        raise NotImplementedError

    def footprint(self) -> tuple[float, float, float, float]:
        x, y, w, h = self.spec["box"]
        return x, y, w, h


class CardElement(Element):
    """Elements that live on a frosted-glass card with internally choreographed content."""

    def card_open(self, canvas: skia.Canvas, t: float, W: float, H: float, glass_on: bool = True,
                  H_vis: Optional[float] = None) -> float:
        from ..render.draw import glass

        x, y, _, _ = self.spec["box"]
        s = self.spec.get("scale", 1.0)
        canvas.save()
        canvas.translate(x, y)
        canvas.scale(s, s)
        pi = self.p_in(t)
        po = self.p_out(t)
        a = min(1.0, pi * 1.3) * (1 - po)
        sc = self.m["scale_in"] + (1 - self.m["scale_in"]) * pi - 0.015 * po
        dy = (1 - pi) * self.m["travel"] * self.u * 1.2 + po * 0.4 * self.u
        canvas.translate(W / 2, H / 2 + dy)
        canvas.scale(sc, sc)
        canvas.translate(-W / 2, -H / 2)
        backing = self.spec.get("backing", "glass")
        hv = H if H_vis is None else max(1.0, min(H, H_vis))
        if glass_on and backing != "none":
            glass(canvas, 0, 0, W, hv, self.u * self.style["radius"], self.style, self.u, alpha=a, blur=(backing == "glass"))
        if H_vis is not None:
            canvas.clipRect(skia.Rect.MakeXYWH(-2 * self.u, -2 * self.u, W + 4 * self.u, hv + 2 * self.u))
        return a

    def card_close(self, canvas: skia.Canvas) -> None:
        canvas.restore()

    def item_look(self, t: float, start: float):
        return self.reveal_look(t, start, self.enter_dur * 0.9)
