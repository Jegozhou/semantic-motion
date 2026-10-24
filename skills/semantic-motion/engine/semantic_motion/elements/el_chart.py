"""Chart: bar / hbar / line / ring. Each datum can be anchored to the moment it is spoken."""
from __future__ import annotations

import math

import skia

from ..render.draw import col, fill_rrect, stroke_path, text_shadow
from ..render.easing import ease, prog
from ..render.text import draw_layout, layout
from .base import CardElement, register
from .el_stat import format_value


@register
class Chart(CardElement):
    type = "chart"

    def build(self, max_w: float) -> None:
        ctx, c, u = self.ctx, self.c, self.u
        self.kind = c.get("kind", "bar")
        self.pad = self.style["pad"] * u
        inner = max_w - 2 * self.pad
        self.title = layout(ctx.fb, c["title"], ctx.ts("card_title"), 700, inner) if c.get("title") else None
        self.data = [d if isinstance(d, dict) else {"label": str(d[0]), "value": float(d[1])} for d in c.get("data", [])]
        if self.kind == "ring" and not self.data:
            self.data = [{"label": c.get("label", ""), "value": float(c.get("value", 0))}]
        self.unit = c.get("unit", "")
        self.decimals = int(c.get("decimals", 0 if all(float(d.get("value", 0)).is_integer() for d in self.data) else 1))
        vals = [float(d.get("value", 0)) for d in self.data] or [1.0]
        self.vmax = float(c.get("max", max(vals) if self.kind != "ring" else 100.0)) or 1.0
        lf = ctx.fb.font(ctx.ts("chart_label"), 500)
        vf = ctx.fb.font(ctx.ts("chart_value"), 700)
        self.lf, self.vf = lf, vf
        n = max(1, len(self.data))
        if self.kind == "bar":
            self.plot_w = inner
            self.plot_h = min(inner * 0.62, 30 * u)
            self.bar_w = min(7.0 * u, self.plot_w / n * 0.56)
        elif self.kind == "hbar":
            self.label_w = min(inner * 0.36, max(lf.measureText(d.get("label", "")) for d in self.data) + 1.2 * u)
            self.row_h = 3.6 * u
            self.plot_w = inner
            self.plot_h = n * self.row_h + (n - 1) * 1.0 * u
        elif self.kind == "line":
            self.plot_w = inner
            self.plot_h = min(inner * 0.55, 26 * u)
        else:  # ring
            self.ring_r = min(inner / 2, 11 * u)
            self.plot_w = 2 * self.ring_r
            self.plot_h = 2 * self.ring_r + (3.6 * u if self.data[0].get("label") else 0)
            self.ring_label = layout(ctx.fb, self.data[0].get("label", ""), ctx.ts("chart_label") * 1.1, 500, inner) if self.data[0].get("label") else None

    def size(self):
        u = self.u
        w = max(self.plot_w, self.title.width if self.title else 0)
        h = self.plot_h + (self.title.height + 1.8 * u if self.title else 0)
        if self.kind in ("bar", "line"):
            h += 3.4 * u  # labels below + values above headroom
        return w + 2 * self.pad, h + 2 * self.pad

    def datum_time(self, k: int) -> float:
        d = self.data[k]
        if d.get("t_land") is not None:
            return d["t_land"] - 0.25
        return self.t_in + 0.25 + 0.12 * k

    def draw(self, canvas: skia.Canvas, t: float) -> None:
        if not self.active(t):
            return
        W, H = self.size()
        u = self.u
        a = self.card_open(canvas, t, W, H)
        white = col(self.style["colors"]["text"])
        accent = self.style["colors"]["accent"]
        sh = text_shadow(self.style, u, 0.5)
        y = self.pad
        if self.title:
            draw_layout(canvas, self.title, self.pad, y, self.ctx.fb, 700, white, lambda un: self.item_look(t, self.t_in + 0.05), sh)
            y += self.title.height + 1.8 * u
        x0 = self.pad
        fade = 1 - self.p_out(t)
        tp = skia.Paint(AntiAlias=True)
        n = len(self.data)
        hl_any = any(d.get("highlight") for d in self.data)
        if self.kind == "bar":
            base_y = y + 1.6 * u + self.plot_h
            lp = skia.Paint(AntiAlias=True, StrokeWidth=max(1.0, 0.1 * u))
            lp.setColor4f(col("#FFFFFF", 0.25 * a))
            canvas.drawLine(x0, base_y, x0 + self.plot_w, base_y, lp)
            slot = self.plot_w / max(1, n)
            for k, d in enumerate(self.data):
                st = self.datum_time(k)
                g = ease([0.2, 0.9, 0.25, 1.0], prog(t, st, 0.65))
                v = float(d.get("value", 0))
                bh = self.plot_h * (v / self.vmax) * g
                bx = x0 + slot * k + (slot - self.bar_w) / 2
                is_hl = d.get("highlight") or (not hl_any and k == n - 1)
                cc = col(accent, fade) if is_hl else col("#FFFFFF", 0.62 * fade)
                if bh > 0.5:
                    fill_rrect(canvas, bx, base_y - bh, self.bar_w, bh, min(0.9 * u, self.bar_w / 3), cc)
                    canvas.drawRect(skia.Rect.MakeXYWH(bx, base_y - min(bh, 0.9 * u), self.bar_w, min(bh, 0.9 * u)), skia.Paint(AntiAlias=True, Color4f=cc))
                la = min(1.0, prog(t, st, 0.3) * 1.5) * fade
                tp.setColor4f(col("#FFFFFF", 0.8 * la))
                lab = d.get("label", "")
                canvas.drawString(lab, bx + self.bar_w / 2 - self.lf.measureText(lab) / 2, base_y + 2.8 * u, self.lf, tp)
                vt = format_value(v * g, self.decimals, "") + self.unit
                tp.setColor4f(col(accent if is_hl else "#FFFFFF", la))
                canvas.drawString(vt, bx + self.bar_w / 2 - self.vf.measureText(vt) / 2, base_y - bh - 0.9 * u, self.vf, tp)
        elif self.kind == "hbar":
            for k, d in enumerate(self.data):
                st = self.datum_time(k)
                g = ease([0.2, 0.9, 0.25, 1.0], prog(t, st, 0.6))
                ry = y + k * (self.row_h + 1.0 * u)
                la = min(1.0, prog(t, st, 0.3) * 1.5) * fade
                tp.setColor4f(col("#FFFFFF", 0.85 * la))
                canvas.drawString(d.get("label", ""), x0, ry + self.row_h * 0.68, self.lf, tp)
                v = float(d.get("value", 0))
                bw_max = self.plot_w - self.label_w - 7 * u
                bw = bw_max * (v / self.vmax) * g
                is_hl = d.get("highlight") or (not hl_any and k == 0)
                cc = col(accent, fade) if is_hl else col("#FFFFFF", 0.55 * fade)
                if bw > 0.5:
                    fill_rrect(canvas, x0 + self.label_w, ry + self.row_h * 0.2, bw, self.row_h * 0.6, self.row_h * 0.3, cc)
                vt = format_value(v * g, self.decimals, "") + self.unit
                tp.setColor4f(col(accent if is_hl else "#FFFFFF", la))
                canvas.drawString(vt, x0 + self.label_w + bw + 1.0 * u, ry + self.row_h * 0.7, self.vf, tp)
        elif self.kind == "line":
            base_y = y + 1.6 * u + self.plot_h
            vmin = float(self.c.get("min", min(float(d.get("value", 0)) for d in self.data)))
            span = max(1e-9, self.vmax - vmin)
            pts = []
            for k, d in enumerate(self.data):
                px = x0 + (self.plot_w * (k / max(1, n - 1)) if n > 1 else self.plot_w / 2)
                py = base_y - self.plot_h * 0.85 * ((float(d.get("value", 0)) - vmin) / span) - self.plot_h * 0.08
                pts.append((px, py))
            path = skia.Path()
            for k, (px, py) in enumerate(pts):
                (path.moveTo if k == 0 else path.lineTo)(px, py)
            t_last = self.datum_time(n - 1) if n else self.t_in
            trim = ease("inout", prog(t, self.datum_time(0), max(0.5, t_last - self.datum_time(0) + 0.5)))
            stroke_path(canvas, path, col(accent, fade), 0.55 * u, trim)
            for k, (px, py) in enumerate(pts):
                st = self.datum_time(k)
                pa = ease("outback", prog(t, st + 0.2, 0.35)) * fade
                if pa > 0:
                    canvas.drawCircle(px, py, 0.9 * u * pa, skia.Paint(AntiAlias=True, Color4f=col("#FFFFFF", fade)))
                    tp.setColor4f(col("#FFFFFF", 0.8 * min(1, pa)))
                    lab = str(self.data[k].get("label", ""))
                    canvas.drawString(lab, px - self.lf.measureText(lab) / 2, base_y + 2.8 * u, self.lf, tp)
                    if k == n - 1 or self.data[k].get("highlight"):
                        vt = format_value(float(self.data[k].get("value", 0)), self.decimals, "") + self.unit
                        tp.setColor4f(col(accent, min(1, pa)))
                        canvas.drawString(vt, px - self.vf.measureText(vt) / 2, py - 1.6 * u, self.vf, tp)
        else:  # ring
            d = self.data[0]
            st = self.datum_time(0)
            g = ease([0.2, 0.9, 0.25, 1.0], prog(t, st, 0.9))
            cx, cy, r = x0 + self.plot_w / 2, y + self.ring_r, self.ring_r * 0.84
            trackp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1.5 * u)
            trackp.setColor4f(col("#FFFFFF", 0.16 * a))
            canvas.drawCircle(cx, cy, r, trackp)
            frac = max(0.0, min(1.0, float(d.get("value", 0)) / self.vmax)) * g
            if frac > 0.001:
                arc = skia.Path()
                arc.addArc(skia.Rect.MakeLTRB(cx - r, cy - r, cx + r, cy + r), -90, 360 * frac)
                stroke_path(canvas, arc, col(accent, fade), 1.5 * u)
            vt = format_value(float(d.get("value", 0)) * g, self.decimals, "") + (self.unit or "%")
            f = self.ctx.fb.font(self.ring_r * 0.5, 800)
            tp.setColor4f(col("#FFFFFF", a))
            canvas.drawString(vt, cx - f.measureText(vt) / 2, cy + self.ring_r * 0.17, f, tp)
            if self.ring_label:
                draw_layout(canvas, self.ring_label, cx - self.ring_label.width / 2, y + 2 * self.ring_r + 1.0 * u, self.ctx.fb, 500,
                            white, lambda un: self.item_look(t, st + 0.1), sh)
        self.card_close(canvas)
