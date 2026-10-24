"""Flow: process steps (A -> B -> C) with connectors drawn as each step is spoken."""
from __future__ import annotations

import math

import skia

from ..render.draw import arrow_head, col, glass, rrect, stroke_path, text_shadow
from ..render.easing import ease, prog
from ..render.text import draw_layout, layout
from .base import CardElement, register


@register
class Flow(CardElement):
    type = "flow"

    def build(self, max_w: float) -> None:
        ctx, c, u = self.ctx, self.c, self.u
        steps = [s if isinstance(s, dict) else {"text": str(s)} for s in c.get("steps", [])]
        self.steps = steps
        n = max(1, len(steps))
        d = c.get("direction", "auto")
        self.vertical = d == "vertical" or (d == "auto" and (ctx.orientation == "portrait" or max_w < 0.45 * ctx.W and n > 3))
        self.gap = 5.0 * u if not self.vertical else 3.4 * u
        self.npad = 1.6 * u
        if self.vertical:
            tw = max_w - 2 * self.npad
        else:
            tw = (max_w - (n - 1) * self.gap) / n - 2 * self.npad
        self.lays = [layout(ctx.fb, s.get("text", ""), ctx.ts("item"), 600, max(tw, 6 * u), line_height=1.2, align="center",
                            highlights=s.get("highlight")) for s in steps]
        self.subs = [layout(ctx.fb, s["sub"], ctx.ts("item_sub"), 400, max(tw, 6 * u), line_height=1.25, align="center") if s.get("sub") else None for s in steps]
        self.node_w = [l.width + 2 * self.npad for l in self.lays]
        self.node_h = [l.height + 2 * self.npad * 0.8 + ((sb.height + 0.4 * u) if sb else 0) for l, sb in zip(self.lays, self.subs)]
        if not self.vertical:
            hh = max(self.node_h)
            self.node_h = [hh] * n
            ww = max(self.node_w)
            self.node_w = [ww] * n
        else:
            ww = max(self.node_w)
            self.node_w = [ww] * n

    @classmethod
    def variants(cls, spec, ctx):
        from .base import PREF_WIDTHS

        o = ctx.orientation if ctx.orientation in PREF_WIDTHS else "landscape"
        d = (spec.get("content") or {}).get("direction", "auto")
        out = []
        dirs = ["horizontal", "vertical"] if d == "auto" else [d]
        for di in dirs:
            widths = PREF_WIDTHS[o]["flow"] if di == "horizontal" else ([0.22, 0.28] if o == "landscape" else [0.5, 0.62])
            for f in widths:
                c2 = {**(spec.get("content") or {}), "direction": di}
                el = cls({**spec, "content": c2, "layout_w": f * ctx.W}, ctx)
                w, h = el.size()
                out.append({"layout_w": f * ctx.W, "w": w, "h": h, "content_patch": {"direction": di}})
        return out

    def size(self):
        if self.vertical:
            return max(self.node_w), sum(self.node_h) + self.gap * (len(self.steps) - 1)
        return sum(self.node_w) + self.gap * (len(self.steps) - 1), max(self.node_h)

    def step_time(self, k: int) -> float:
        s = self.steps[k]
        if s.get("t_land") is not None:
            return s["t_land"] - self.enter_dur * self.m["land_frac"]
        return self.t_in + 0.45 * k

    def draw(self, canvas: skia.Canvas, t: float) -> None:
        if not self.active(t):
            return
        W, H = self.size()
        u = self.u
        x, y, _, _ = self.spec["box"]
        s = self.spec.get("scale", 1.0)
        canvas.save()
        canvas.translate(x, y)
        canvas.scale(s, s)
        accent = self.style["colors"]["accent"]
        white = col(self.style["colors"]["text"])
        sh = text_shadow(self.style, u, 0.5)
        po = self.p_out(t)
        times = [self.step_time(k) for k in range(len(self.steps))]
        latest = max([k for k, tt in enumerate(times) if t >= tt] or [-1])
        pos = []
        cx = cy = 0.0
        for k in range(len(self.steps)):
            pos.append((cx, cy))
            if self.vertical:
                cy += self.node_h[k] + self.gap
            else:
                cx += self.node_w[k] + self.gap
        for k in range(len(self.steps)):
            st = times[k]
            nx, ny = pos[k]
            nw, nh = self.node_w[k], self.node_h[k]
            if k > 0:  # connector from previous node, drawn just before this node lands
                px, py = pos[k - 1]
                if self.vertical:
                    a0 = (px + nw / 2, py + self.node_h[k - 1] + 0.6 * u)
                    a1 = (nx + nw / 2, ny - 0.6 * u)
                else:
                    a0 = (px + self.node_w[k - 1] + 0.6 * u, py + nh / 2)
                    a1 = (nx - 0.6 * u, ny + nh / 2)
                tr = ease("inout", prog(t, st - 0.28, 0.32))
                if tr > 0:
                    path = skia.Path()
                    path.moveTo(*a0)
                    path.lineTo(*a1)
                    c2 = col(accent, (1 - po))
                    stroke_path(canvas, path, c2, 0.36 * u, tr)
                    if tr > 0.98:
                        ang = math.atan2(a1[1] - a0[1], a1[0] - a0[0])
                        arrow_head(canvas, a1[0], a1[1], ang, 0.9 * u, c2, 0.36 * u)
            pi = ease(self.m["ease_in_curve"], prog(t, st, self.enter_dur))
            if pi <= 0:
                continue
            a = min(1.0, pi * 1.3) * (1 - po)
            sc = 0.92 + 0.08 * pi
            canvas.save()
            canvas.translate(nx + nw / 2, ny + nh / 2 + (1 - pi) * 0.8 * u)
            canvas.scale(sc, sc)
            canvas.translate(-(nx + nw / 2), -(ny + nh / 2))
            glass(canvas, nx, ny, nw, nh, 1.4 * u, self.style, u, alpha=a)
            if k == latest:
                p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=0.24 * u)
                p.setColor4f(col(accent, a * ease("out", prog(t, st + 0.1, 0.3))))
                canvas.drawRRect(rrect(nx, ny, nw, nh, 1.4 * u), p)
            lay = self.lays[k]
            ty = ny + self.npad * 0.8
            def lk(un, st=st):
                l2 = self.item_look(t, st + 0.04)
                if un.tag == "hl":
                    l2.color = col(accent)
                return l2
            draw_layout(canvas, lay, nx + (nw - lay.width) / 2, ty, self.ctx.fb, 600, white, lk, sh)
            if self.subs[k]:
                sb = self.subs[k]
                def lks(un, st=st):
                    l3 = self.item_look(t, st + 0.1)
                    l3.alpha *= 0.7
                    return l3
                draw_layout(canvas, sb, nx + (nw - sb.width) / 2, ty + lay.height + 0.4 * u, self.ctx.fb, 400, white, lks, sh)
            canvas.restore()
        canvas.restore()
