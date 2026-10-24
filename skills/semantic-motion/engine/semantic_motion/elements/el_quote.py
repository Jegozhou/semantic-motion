"""Quote: a statement worth framing; words reveal as spoken."""
from __future__ import annotations

import skia

from ..render.draw import col, text_shadow
from ..render.easing import ease, prog
from ..render.text import draw_layout, layout
from .base import CardElement, register


@register
class Quote(CardElement):
    type = "quote"

    def build(self, max_w: float) -> None:
        ctx, c, u = self.ctx, self.c, self.u
        self.pad = self.style["pad"] * u * 1.1
        inner = max_w - 2 * self.pad
        self.mark_size = 7.5 * u
        self.lay = layout(ctx.fb, c.get("text", ""), ctx.ts("quote"), ctx.w("quote", 600), inner, line_height=1.32,
                          highlights=c.get("highlight"))
        self.by = layout(ctx.fb, "— " + c["by"], ctx.ts("byline"), 500, inner) if c.get("by") else None

    def size(self):
        u = self.u
        w = max(self.lay.width, self.by.width if self.by else 0)
        h = self.mark_size * 0.55 + self.lay.height + (1.4 * u + self.by.height if self.by else 0)
        return w + 2 * self.pad, h + 2 * self.pad

    def unit_start(self, i: int) -> float:
        times = self.c.get("unit_times")
        if times and i < len(times) and times[i] is not None:
            return times[i] - self.enter_dur * self.m["land_frac"]
        n = max(1, len(self.lay.units))
        return self.t_in + 0.15 + i * min(0.03, 0.6 / n)

    def draw(self, canvas: skia.Canvas, t: float) -> None:
        if not self.active(t):
            return
        W, H = self.size()
        u = self.u
        a = self.card_open(canvas, t, W, H)
        accent = self.style["colors"]["accent"]
        f = self.ctx.fb.font(self.mark_size, 900)
        p = skia.Paint(AntiAlias=True)
        p.setColor4f(col(accent, a))
        canvas.drawString("“", self.pad - 0.3 * u, self.pad + self.mark_size * 0.78, f, p)
        y = self.pad + self.mark_size * 0.55
        white = col(self.style["colors"]["text"])
        sh = text_shadow(self.style, u, 0.5)

        def lk(un):
            l2 = self.item_look(t, self.unit_start(un.idx))
            if un.tag == "hl":
                l2.color = col(accent)
            return l2
        draw_layout(canvas, self.lay, self.pad, y, self.ctx.fb, self.ctx.w("quote", 600), white, lk, sh)
        if self.by:
            last = self.unit_start(len(self.lay.units) - 1)
            def lb(un):
                l3 = self.item_look(t, last + 0.1)
                l3.alpha *= 0.7
                return l3
            draw_layout(canvas, self.by, self.pad, y + self.lay.height + 1.4 * u, self.ctx.fb, 500, white, lb, sh)
        self.card_close(canvas)
