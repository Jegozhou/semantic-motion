"""Captions (opt-in): phrase-chunked subtitles with a restrained spoken-word highlight."""
from __future__ import annotations

import skia

from ..render.draw import col, glass, text_shadow
from ..render.easing import ease, prog
from ..render.text import UnitLook, draw_layout, fit_layout
from .base import Element, register


@register
class Captions(Element):
    type = "captions"
    layer = "overlay"
    default_region = "bottom"

    def build(self, max_w: float) -> None:
        ctx = self.ctx
        self.max_w = max_w
        self.size_px = ctx.ts("caption") * float(self.c.get("size_mul", 1.0))
        self.weight = ctx.w("caption", 600)
        self.chunks = self.c.get("chunks", [])
        self.lays = []
        for ch in self.chunks:
            lay = fit_layout(ctx.fb, ch["text"], self.size_px, self.weight, max_w, max_lines=2, line_height=1.2, align="center")
            self.lays.append(lay)
        self.h = max([l.height for l in self.lays] or [self.size_px * 1.2])
        self.hl = self.c.get("highlight", "spoken")  # spoken | none

    def size(self):
        return self.max_w, self.h

    def draw(self, canvas: skia.Canvas, t: float) -> None:
        if not self.active(t) or not self.chunks:
            return
        x, y, w, h = self.spec["box"]
        for ch, lay in zip(self.chunks, self.lays):
            if not (ch["t0"] - 0.12 <= t < ch["t1"] + 0.15):
                continue
            a = ease("out", prog(t, ch["t0"] - 0.12, 0.12)) * (1 - ease("in", prog(t, ch["t1"], 0.15)))
            ox = x + (w - lay.width) / 2
            oy = y + h - lay.height
            if self.spec.get("backing") in ("glass", "plate"):
                pad = 0.9 * self.u
                glass(canvas, ox - 1.4 * pad, oy - pad * 0.7, lay.width + 2.8 * pad, lay.height + 1.4 * pad, 1.2 * self.u,
                      self.style, self.u, alpha=a, blur=self.spec.get("backing") == "glass")
            wt = ch.get("unit_times") or []
            acc = col(self.style["colors"]["accent"])

            def look(un, a=a, wt=wt):
                lk = UnitLook(alpha=a)
                if self.hl == "spoken" and un.idx < len(wt) and wt[un.idx] is not None:
                    spoken = t >= wt[un.idx]
                    lk.alpha = a * (1.0 if spoken else 0.62)
                return lk

            draw_layout(canvas, lay, ox, oy, self.ctx.fb, self.weight, col(self.style["colors"]["text"]), look,
                        text_shadow(self.style, self.u, 1.1))
