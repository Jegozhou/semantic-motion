"""Lower third: name + role with a masked slide reveal and accent bar."""
from __future__ import annotations

import skia

from ..render.draw import col, glass, text_shadow
from ..render.easing import ease, prog
from ..render.text import draw_layout, layout
from .base import Element, register


@register
class LowerThird(Element):
    type = "lower_third"
    layer = "secondary"

    def build(self, max_w: float) -> None:
        ctx, c, u = self.ctx, self.c, self.u
        self.name = layout(ctx.fb, c.get("name", ""), ctx.ts("name"), ctx.w("name", 700), max_w - 3 * u)
        self.role = layout(ctx.fb, c["role"], ctx.ts("role"), 400, max_w - 3 * u) if c.get("role") else None
        self.bar_w = 0.55 * u
        self.gap = 1.6 * u

    def size(self):
        u = self.u
        w = self.bar_w + self.gap + max(self.name.width, self.role.width if self.role else 0)
        h = self.name.height + (0.7 * u + self.role.height if self.role else 0)
        return w + 2.4 * u, h + 2.4 * u

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
        po = self.p_out(t)
        pb = ease("inout", prog(t, self.t_in, 0.4))
        if self.spec.get("backing", "glass") != "none":
            glass(canvas, 0, 0, W, H, 1.2 * u, self.style, u, alpha=min(1, pb * 1.2) * (1 - po))
        p = skia.Paint(AntiAlias=True)
        p.setColor4f(col(self.style["colors"]["accent"], 1 - po))
        bh = (H - 2.4 * u) * pb
        canvas.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(1.2 * u, 1.2 * u + (H - 2.4 * u - bh) / 2, self.bar_w, bh), self.bar_w / 2, self.bar_w / 2), p)
        tx = 1.2 * u + self.bar_w + self.gap
        pt = ease(self.m["ease_in_curve"], prog(t, self.t_in + 0.15, 0.5))
        canvas.save()
        canvas.clipRect(skia.Rect.MakeXYWH(tx - 0.2 * u, 0, W - tx + 0.2 * u, H))
        dx = -(1 - pt) * 6 * u
        white = col(self.style["colors"]["text"])
        sh = text_shadow(self.style, u, 0.4)
        a = (1 - po) * min(1.0, pt * 1.4)
        from ..render.text import UnitLook
        draw_layout(canvas, self.name, tx + dx, 1.2 * u, self.ctx.fb, self.ctx.w("name", 700), white, lambda un: UnitLook(alpha=a), sh)
        if self.role:
            pr = ease(self.m["ease_in_curve"], prog(t, self.t_in + 0.25, 0.5))
            draw_layout(canvas, self.role, tx - (1 - pr) * 6 * u, 1.2 * u + self.name.height + 0.7 * u, self.ctx.fb, 400, white,
                        lambda un: UnitLook(alpha=(1 - po) * min(1.0, pr * 1.4) * 0.75), sh)
        canvas.restore()
        canvas.restore()
