"""Title / chapter card: kicker + headline + accent rule. Marks a section change."""
from __future__ import annotations

import skia

from ..render.draw import col, glass, text_shadow
from ..render.easing import ease, prog
from ..render.text import draw_layout, fit_layout, layout
from .base import Element, register


@register
class Title(Element):
    type = "title"
    layer = "secondary"

    def build(self, max_w: float) -> None:
        ctx, c, u = self.ctx, self.c, self.u
        self.align = self.spec.get("align", "left")
        self.kicker = layout(ctx.fb, c["kicker"], ctx.ts("kicker"), 700, max_w, tracking=self.style["tracking"]["kicker"]) if c.get("kicker") else None
        self.lay = fit_layout(ctx.fb, c.get("text", ""), ctx.ts("title"), ctx.w("title", 800), max_w, max_lines=2, line_height=1.14,
                              align=self.align, highlights=c.get("highlight"))
        self.sub = layout(ctx.fb, c["sub"], ctx.ts("keyword_sub"), 500, max_w, align=self.align) if c.get("sub") else None
        self.pad = self.style["pad"] * u if self.spec.get("backing") in ("glass", "plate") else 0.0

    def size(self):
        u = self.u
        w = max(self.lay.width, self.kicker.width if self.kicker else 0, self.sub.width if self.sub else 0)
        h = (self.kicker.height + 1.2 * u if self.kicker else 0) + self.lay.height + 1.4 * u + 0.5 * u + (1.0 * u + self.sub.height if self.sub else 0)
        return w + 2 * self.pad, h + 2 * self.pad

    def draw(self, canvas: skia.Canvas, t: float) -> None:
        if not self.active(t):
            return
        u = self.u
        x, y, _, _ = self.spec["box"]
        s = self.spec.get("scale", 1.0)
        canvas.save()
        canvas.translate(x, y)
        canvas.scale(s, s)
        W, H = self.size()
        if self.spec.get("backing") in ("glass", "plate"):
            glass(canvas, 0, 0, W, H, self.style["radius"] * u, self.style, u, alpha=self.p_in(t) * (1 - self.p_out(t)))
        accent = col(self.style["colors"]["accent"])
        white = col(self.style["colors"]["text"])
        sh = text_shadow(self.style, u)
        yy = self.pad

        def ax(width: float) -> float:
            return self.pad + ((W - 2 * self.pad - width) / 2 if self.align == "center" else (W - 2 * self.pad - width if self.align == "right" else 0))
        if self.kicker:
            def lk(un):
                l2 = self.reveal_look(t, self.t_in)
                l2.color = accent
                return l2
            draw_layout(canvas, self.kicker, ax(self.kicker.width), yy, self.ctx.fb, 700, white, lk, sh,
                        tracking=self.style["tracking"]["kicker"])
            yy += self.kicker.height + 1.2 * u
        n = max(1, len(self.lay.units))

        def lt(un):
            l3 = self.reveal_look(t, self.t_in + 0.08 + un.idx * min(0.035, 0.4 / n))
            if un.tag == "hl":
                l3.color = accent
            return l3
        draw_layout(canvas, self.lay, ax(self.lay.width), yy, self.ctx.fb, self.ctx.w("title", 800), white, lt, sh)
        yy += self.lay.height + 1.4 * u
        rw = min(self.lay.width, 9 * u) * ease("inout", prog(t, self.t_in + 0.25, 0.5)) * (1 - self.p_out(t))
        p = skia.Paint(AntiAlias=True)
        p.setColor4f(accent)
        rx = ax(min(self.lay.width, 9 * u))
        canvas.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(rx, yy, rw, 0.5 * u), 0.25 * u, 0.25 * u), p)
        yy += 0.5 * u + 1.0 * u
        if self.sub:
            def ls(un):
                l4 = self.reveal_look(t, self.t_in + 0.3)
                l4.alpha *= 0.8
                return l4
            draw_layout(canvas, self.sub, ax(self.sub.width), yy, self.ctx.fb, 500, white, ls, sh)
        canvas.restore()
