"""Keyword: the core semantic emphasis — a concise phrase that lands on the spoken word."""
from __future__ import annotations

import skia

from ..render.draw import col, glass, text_shadow
from ..render.easing import ease, prog
from ..render.text import UnitLook, draw_layout, fit_layout, layout, unit_rects_by_line
from .base import Element, register


@register
class Keyword(Element):
    type = "keyword"
    default_region = "beside_face"

    def build(self, max_w: float) -> None:
        ctx = self.ctx
        c = self.c
        self.align = self.spec.get("align", "left")
        size = ctx.ts("keyword") * float(c.get("size_mul", 1.0))
        self.weight = ctx.w("keyword", 800)
        self.lay = fit_layout(ctx.fb, c.get("text", ""), size, self.weight, max_w, max_lines=int(c.get("max_lines", 2)),
                              line_height=1.16, align=self.align, highlights=c.get("highlight"),
                              tracking=0.0)
        self.sub = None
        if c.get("sub"):
            self.sub = layout(ctx.fb, c["sub"], ctx.ts("keyword_sub"), 500, max(max_w, self.lay.width), line_height=1.3,
                              align=self.align)
        self.pad = self.u * self.style["pad"] if self.spec.get("backing") in ("glass", "plate") else 0.0
        self.gap = 1.1 * self.u if self.sub else 0.0

    def size(self):
        w = max(self.lay.width, self.sub.width if self.sub else 0)
        h = self.lay.height + (self.gap + self.sub.height if self.sub else 0)
        return w + 2 * self.pad, h + 2 * self.pad

    def _unit_start(self, i: int) -> float:
        times = self.c.get("unit_times")
        reveal = (self.spec.get("motion") or {}).get("reveal", "word")
        if reveal == "spoken" and times and i < len(times) and times[i] is not None:
            return times[i] - self.enter_dur * self.m["land_frac"]
        if reveal == "whole":
            return self.t_in
        n = max(1, len(self.lay.units))
        st = min(self.m["stagger"], self.m["stagger_cap"] / n)
        return self.t_in + i * st

    def draw(self, canvas: skia.Canvas, t: float) -> None:
        if not self.active(t):
            return
        x, y, w, h = self.spec["box"]
        s = self.spec.get("scale", 1.0)
        canvas.save()
        canvas.translate(x, y)
        canvas.scale(s, s)
        W, H = self.size()
        accent = col(self.style["colors"]["accent"])
        white = col(self.style["colors"]["text"])
        backing = self.spec.get("backing", "none")
        if backing in ("glass", "plate"):
            a = self.p_in(t, dur=self.enter_dur * 0.8) * (1 - self.p_out(t))
            glass(canvas, 0, 0, W, H, self.u * self.style["radius"], self.style, self.u, alpha=a, blur=(backing == "glass"))
        ox, oy = self.pad, self.pad
        if self.align == "center":
            ox = (W - self.lay.width) / 2
        elif self.align == "right":
            ox = W - self.pad - self.lay.width
        # highlight underline sweep (drawn under text)
        runs = unit_rects_by_line(self.lay)
        marker = self.c.get("marker", "underline" if self.c.get("highlight") else "none")
        if marker != "none" and runs:
            hl_units = [u for u in self.lay.units if u.tag == "hl"]
            st = max(self._unit_start(hl_units[-1].idx) + self.enter_dur * 0.55, self.t_in) if hl_units else self.t_in
            sweep = ease("inout", prog(t, st, 0.38))
            alpha = 1 - self.p_out(t)
            for (li, x0, x1) in runs:
                if marker == "underline":
                    by = oy + self.lay.ascent + li * self.lay.line_h + 0.16 * self.lay.size
                    hh = max(2.0, 0.085 * self.lay.size)
                    p = skia.Paint(AntiAlias=True)
                    p.setColor4f(col(self.style["colors"]["accent"], 0.95 * alpha))
                    canvas.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(ox + x0, by, (x1 - x0) * sweep, hh), hh / 2, hh / 2), p)
                elif marker == "bar":
                    by = oy + li * self.lay.line_h + self.lay.ascent * 0.45
                    hh = self.lay.ascent * 0.62
                    p = skia.Paint(AntiAlias=True)
                    p.setColor4f(col(self.style["colors"]["accent"], 0.38 * alpha))
                    pad = 0.06 * self.lay.size
                    canvas.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(ox + x0 - pad, by, (x1 - x0 + 2 * pad) * sweep, hh), 0.1 * hh, 0.1 * hh), p)

        def look(un):
            lk = self.reveal_look(t, self._unit_start(un.idx))
            if un.tag == "hl":
                lk.color = accent if marker != "bar" else white
            return lk

        shadow = text_shadow(self.style, self.u, 0.6 if backing != "none" else 1.0)
        draw_layout(canvas, self.lay, ox, oy, self.ctx.fb, self.weight, white, look, shadow)
        if self.sub:
            sy = oy + self.lay.height + self.gap
            sx = self.pad if self.align == "left" else ((W - self.sub.width) / 2 if self.align == "center" else W - self.pad - self.sub.width)
            last = self._unit_start(len(self.lay.units) - 1)
            sub_start = min(last + 0.08, self.t_in + 0.35)

            def look2(un):
                lk = self.reveal_look(t, sub_start, self.enter_dur)
                lk.alpha *= 0.82
                return lk

            draw_layout(canvas, self.sub, sx, sy, self.ctx.fb, 500, white, look2, text_shadow(self.style, self.u, 0.8))
        canvas.restore()
