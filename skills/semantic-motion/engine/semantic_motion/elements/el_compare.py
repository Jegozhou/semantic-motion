"""Compare: two sides (A vs B / before -> after), each revealed on its own spoken anchor."""
from __future__ import annotations

import skia

from ..render.draw import arrow_head, col, glass, rrect, text_shadow
from ..render.easing import ease, prog
from ..render.text import draw_layout, layout
from .base import CardElement, register


@register
class Compare(CardElement):
    type = "compare"

    def build(self, max_w: float) -> None:
        ctx, c, u = self.ctx, self.c, self.u
        self.pad = self.style["pad"] * u
        self.mid = 6.0 * u
        col_w = (max_w - 2 * self.pad - self.mid) / 2
        self.sides = []
        for key in ("left", "right"):
            sd = c.get(key, {}) or {}
            t = layout(ctx.fb, sd.get("title", ""), ctx.ts("card_title") * 1.08, 700, col_w, line_height=1.2) if sd.get("title") else None
            v = layout(ctx.fb, str(sd["value"]), ctx.ts("stat") * 0.5, 800, col_w) if sd.get("value") not in (None, "") else None
            pts = [layout(ctx.fb, p, ctx.ts("item") * 0.88, 500, col_w, line_height=1.25) for p in sd.get("points", [])[:4]]
            self.sides.append((sd, t, v, pts))
        self.col_w = max(1.0, max(max([x.width for x in [t, v] + pts if x] or [0]) for _, t, v, pts in self.sides))

    def _col_h(self, side) -> float:
        u = self.u
        sd, t, v, pts = side
        h = 0.0
        for x, gap in ((t, 1.0 * u), (v, 0.8 * u)):
            if x:
                h += x.height + gap
        for p in pts:
            h += p.height + 0.8 * u
        return h

    def size(self):
        h = max(self._col_h(s) for s in self.sides)
        return 2 * self.col_w + self.mid + 2 * self.pad, h + 2 * self.pad

    def side_time(self, k: int) -> float:
        sd = self.sides[k][0]
        if sd.get("t_land") is not None:
            return sd["t_land"] - self.enter_dur * self.m["land_frac"]
        return self.t_in + (0.0 if k == 0 else 0.35)

    def draw(self, canvas: skia.Canvas, t: float) -> None:
        if not self.active(t):
            return
        W, H = self.size()
        u = self.u
        self.card_open(canvas, t, W, H)
        white = col(self.style["colors"]["text"])
        accent = self.style["colors"]["accent"]
        sh = text_shadow(self.style, u, 0.5)
        winner = self.c.get("winner")
        mode = self.c.get("mode", "vs")
        for k, side in enumerate(self.sides):
            sd, tl, vl, pts = side
            st = self.side_time(k)
            x0 = self.pad + k * (self.col_w + self.mid)
            y = self.pad
            is_win = (winner == ("left" if k == 0 else "right"))
            lk0 = self.item_look(t, st)
            if is_win and lk0.alpha > 0:
                pw = ease("out", prog(t, st + 0.25, 0.4))
                p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=0.28 * u)
                p.setColor4f(col(accent, 0.9 * lk0.alpha * pw))
                canvas.drawRRect(rrect(x0 - 1.0 * u, self.pad - 1.0 * u, self.col_w + 2.0 * u, H - 2 * self.pad + 2.0 * u, 1.4 * u), p)
            for lay, gap, wt, colr, delay, amul in ((tl, 1.0 * u, 700, None, 0.0, 1.0), (vl, 0.8 * u, 800, accent if (is_win or winner is None) else None, 0.06, 1.0)):
                if not lay:
                    continue
                def lf(un, delay=delay, colr=colr, amul=amul):
                    l2 = self.item_look(t, st + delay)
                    l2.alpha *= amul
                    if colr:
                        l2.color = col(colr)
                    return l2
                draw_layout(canvas, lay, x0, y, self.ctx.fb, wt, white, lf, sh)
                y += lay.height + gap
            for j, pl in enumerate(pts):
                def lp(un, j=j):
                    l3 = self.item_look(t, st + 0.10 + 0.07 * j)
                    l3.alpha *= 0.85
                    return l3
                draw_layout(canvas, pl, x0, y, self.ctx.fb, 500, white, lp, sh)
                y += pl.height + 0.8 * u
        # divider
        cx = self.pad + self.col_w + self.mid / 2
        pm = ease("inout", prog(t, self.t_in + 0.1, 0.5)) * (1 - self.p_out(t))
        p = skia.Paint(AntiAlias=True, StrokeWidth=max(1.0, 0.12 * u), Style=skia.Paint.kStroke_Style)
        p.setColor4f(col("#FFFFFF", 0.22 * pm))
        hh = (H - 2 * self.pad) * pm
        canvas.drawLine(cx, H / 2 - hh / 2, cx, H / 2 + hh / 2, p)
        lk = self.item_look(t, self.side_time(1) - 0.1)
        if mode == "vs":
            r = 2.3 * u
            glass(canvas, cx - r, H / 2 - r, 2 * r, 2 * r, r, self.style, u, alpha=lk.alpha, blur=False)
            f = self.ctx.fb.font(1.9 * u, 800)
            tp = skia.Paint(AntiAlias=True)
            tp.setColor4f(col(accent, lk.alpha))
            canvas.drawString("VS", cx - f.measureText("VS") / 2, H / 2 + 0.68 * u, f, tp)
        else:
            path = skia.Path()
            path.moveTo(cx - 1.6 * u, H / 2)
            path.lineTo(cx + 1.6 * u, H / 2)
            from ..render.draw import stroke_path
            stroke_path(canvas, path, col(accent, lk.alpha), 0.45 * u, ease("out", prog(t, self.side_time(1) - 0.2, 0.3)))
            arrow_head(canvas, cx + 1.6 * u, H / 2, 0.0, 1.0 * u, col(accent, lk.alpha), 0.45 * u)
        self.card_close(canvas)
