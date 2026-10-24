"""Callout: a short label tied to a point in the frame (an object, a gesture, the speaker) by a drawn leader line."""
from __future__ import annotations

import math

import skia

from ..render.draw import col, fill_circle, glass, stroke_path, text_shadow
from ..render.easing import ease, prog
from ..render.text import draw_layout, layout
from .base import Element, register


@register
class Callout(Element):
    type = "callout"
    default_region = "near_target"

    def build(self, max_w: float) -> None:
        ctx, c = self.ctx, self.c
        self.lay = layout(ctx.fb, c.get("text", ""), ctx.ts("callout"), 600, max_w, line_height=1.25)
        self.sub = layout(ctx.fb, c["sub"], ctx.ts("item_sub"), 400, max_w, line_height=1.25) if c.get("sub") else None
        self.pad_x = 1.6 * self.u
        self.pad_y = 1.1 * self.u
        self.gap = 0.5 * self.u if self.sub else 0.0

    def size(self):
        w = max(self.lay.width, self.sub.width if self.sub else 0) + 2 * self.pad_x
        h = self.lay.height + (self.gap + self.sub.height if self.sub else 0) + 2 * self.pad_y
        return w, h

    def target_px(self):
        tg = self.spec.get("target_px")
        if tg:
            return tg
        tn = self.c.get("target")
        if isinstance(tn, (list, tuple)) and len(tn) == 2:
            return [tn[0] * self.ctx.W, tn[1] * self.ctx.H]
        return None

    def draw(self, canvas: skia.Canvas, t: float) -> None:
        if not self.active(t):
            return
        x, y, _, _ = self.spec["box"]
        s = self.spec.get("scale", 1.0)
        W, H = self.size()
        W, H = W * s, H * s
        acc = col(self.style["colors"]["accent"])
        po = self.p_out(t)
        tg = self.target_px()
        # 1) target marker (dot + expanding ring), 2) leader line grows from target, 3) label settles
        if tg:
            tx, ty = tg
            p0 = ease("out", prog(t, self.t_in, 0.3))
            a = p0 * (1 - po)
            fill_circle(canvas, tx, ty, 0.75 * self.u * p0, col(self.style["colors"]["accent"], a))
            ring = prog(t, self.t_in, 0.9)
            if ring < 1:
                rp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=0.25 * self.u)
                rp.setColor4f(col(self.style["colors"]["accent"], (1 - ring) * 0.8 * (1 - po)))
                canvas.drawCircle(tx, ty, (0.8 + 2.6 * ease("out", ring)) * self.u, rp)
            # connect to the nearest point on the label box edge
            cx = min(max(tx, x), x + W)
            cy = min(max(ty, y), y + H)
            if x <= tx <= x + W and y <= ty <= y + H:
                cx, cy = x, y + H / 2
            # elbow: horizontal first if the label is to the side
            path = skia.Path()
            path.moveTo(tx, ty)
            if abs(cx - tx) > abs(cy - ty):
                mx = tx + (cx - tx) * 0.35
                path.lineTo(mx, cy)
                path.lineTo(cx, cy)
            else:
                path.lineTo(cx, cy)
            trim = ease("inout", prog(t, self.t_in + 0.08, 0.38))
            stroke_path(canvas, path, col("#FFFFFF", 0.9 * (1 - po)), 0.22 * self.u, trim)
        start = self.t_in + (0.3 if tg else 0.0)
        pi = ease(self.m["ease_in_curve"], prog(t, start, self.enter_dur))
        a = min(1.0, pi * 1.3) * (1 - po)
        canvas.save()
        canvas.translate(x, y + (1 - pi) * 0.8 * self.u)
        canvas.scale(s, s)
        w0, h0 = self.size()
        glass(canvas, 0, 0, w0, h0, h0 / 2 if not self.sub else self.u * self.style["radius"], self.style, self.u, alpha=a)
        # accent tick on the leading edge
        bar = skia.Paint(AntiAlias=True)
        bar.setColor4f(col(self.style["colors"]["accent"], a))

        def look(un):
            lk = self.reveal_look(t, start + 0.06)
            return lk

        draw_layout(canvas, self.lay, self.pad_x, self.pad_y, self.ctx.fb, 600, col(self.style["colors"]["text"]), look,
                    text_shadow(self.style, self.u, 0.5))
        if self.sub:
            def look2(un):
                lk = self.reveal_look(t, start + 0.14)
                lk.alpha *= 0.75
                return lk
            draw_layout(canvas, self.sub, self.pad_x, self.pad_y + self.lay.height + self.gap, self.ctx.fb, 400,
                        col(self.style["colors"]["text"]), look2)
        canvas.restore()
