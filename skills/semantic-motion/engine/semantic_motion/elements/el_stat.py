"""Stat: a number that counts up exactly while it is being spoken, with unit and label."""
from __future__ import annotations

import skia

from ..render.draw import col, glass, text_shadow
from ..render.easing import ease, prog
from ..render.text import UnitLook, draw_layout, layout
from .base import Element, register


def format_value(v: float, decimals: int, sep: str) -> str:
    s = f"{v:,.{decimals}f}" if sep == "," else f"{v:.{decimals}f}"
    return s


@register
class Stat(Element):
    type = "stat"
    default_region = "side"

    def build(self, max_w: float) -> None:
        ctx, c = self.ctx, self.c
        self.weight = ctx.w("stat", 800)
        self.value = float(c.get("value", 0))
        self.start_value = float(c.get("from", 0))
        self.decimals = int(c.get("decimals", 0 if float(self.value).is_integer() else 1))
        self.sep = c.get("separator", "," if abs(self.value) >= 10000 else "")
        self.prefix = c.get("prefix", "")
        self.suffix = c.get("suffix", "")
        self.unit = c.get("unit", "")
        size = ctx.ts("stat") * float(c.get("size_mul", 1.0))
        self.font = ctx.fb.font(size, self.weight)
        self.size_px = size
        final = format_value(self.value, self.decimals, self.sep)
        self.digit_w = max(self.font.measureText(d) for d in "0123456789")
        self.num_w = sum(self.digit_w if ch.isdigit() else self.font.measureText(ch) for ch in final)
        # if the display is wider than allowed, shrink the number
        self.suf_size = size * 0.46
        self.suf_font = ctx.fb.font(self.suf_size, 700)
        self.sym_font = ctx.fb.font(size * 0.62, self.weight)
        self.pre_w = self.font.measureText(self.prefix) if self.prefix else 0.0
        self.sym_w = (self.sym_font.measureText(self.suffix) + 0.03 * size) if self.suffix else 0.0
        self.unit_w = (self.suf_font.measureText(self.unit) + 0.10 * size) if self.unit else 0.0
        self.suf_w = self.sym_w + self.unit_w
        total = self.pre_w + self.num_w + self.suf_w
        if total > max_w:
            k = max_w / total
            self.font = ctx.fb.font(size * k, self.weight)
            self.suf_font = ctx.fb.font(self.suf_size * k, 700)
            self.sym_font = ctx.fb.font(size * 0.62 * k, self.weight)
            self.sym_w *= k
            self.unit_w *= k
            self.size_px = size * k
            self.digit_w *= k
            self.num_w *= k
            self.pre_w *= k
            self.suf_w *= k
        m = self.font.getMetrics()
        self.cap_h = -m.fAscent * 0.78
        self.label = layout(ctx.fb, c["label"], ctx.ts("stat_label"), 500, max(max_w, total), line_height=1.3) if c.get("label") else None
        self.sub = layout(ctx.fb, c["sub"], ctx.ts("item_sub"), 400, max(max_w, total), line_height=1.3) if c.get("sub") else None
        self.pad = self.u * self.style["pad"] if self.spec.get("backing") in ("glass", "plate") else 0.0
        self.num_line_w = self.pre_w + self.num_w + self.suf_w

    def size(self):
        w = max(self.num_line_w, self.label.width if self.label else 0, self.sub.width if self.sub else 0)
        h = self.cap_h + 0.12 * self.size_px
        if self.label:
            h += 1.2 * self.u + self.label.height
        if self.sub:
            h += 0.6 * self.u + self.sub.height
        return w + 2 * self.pad, h + 2 * self.pad

    def current_value(self, t: float) -> float:
        t0 = self.c.get("count_t0", self.t_land)
        t1 = self.c.get("count_t1", self.t_land + 0.9)
        if self.c.get("count", True) is False:
            return self.value
        k = ease([0.16, 1.0, 0.3, 1.0], prog(t, t0, max(0.35, t1 - t0)))
        return self.start_value + (self.value - self.start_value) * k

    def draw(self, canvas: skia.Canvas, t: float) -> None:
        if not self.active(t):
            return
        x, y, w, h = self.spec["box"]
        s = self.spec.get("scale", 1.0)
        canvas.save()
        canvas.translate(x, y)
        canvas.scale(s, s)
        W, H = self.size()
        align = self.spec.get("align", "left")
        backing = self.spec.get("backing", "none")
        if backing in ("glass", "plate"):
            a = self.p_in(t, dur=self.enter_dur * 0.8) * (1 - self.p_out(t))
            glass(canvas, 0, 0, W, H, self.u * self.style["radius"], self.style, self.u, alpha=a, blur=(backing == "glass"))
        lk = self.reveal_look(t, self.t_in)
        ox = self.pad + ((W - 2 * self.pad - self.num_line_w) / 2 if align == "center" else (W - 2 * self.pad - self.num_line_w if align == "right" else 0))
        base_y = self.pad + self.cap_h
        accent = col(self.style["colors"]["accent"], lk.alpha)
        white = col(self.style["colors"]["text"], lk.alpha)
        sh = text_shadow(self.style, self.u, 0.7 if backing != "none" else 1.0)
        v = self.current_value(t)
        txt = format_value(v, self.decimals, self.sep)
        final_txt = format_value(self.value, self.decimals, self.sep)
        # right-align the counting digits inside the final width so nothing jitters
        cells = []
        cx = 0.0
        for ch in txt:
            cw = self.digit_w if ch.isdigit() else self.font.measureText(ch)
            cells.append((ch, cx, cw))
            cx += cw
        offset = self.num_w - cx
        canvas.save()
        canvas.translate(0, lk.dy)
        sc = 0.94 + 0.06 * self.p_in(t)
        canvas.translate(ox + self.num_line_w / 2, base_y)
        canvas.scale(sc, sc)
        canvas.translate(-(ox + self.num_line_w / 2), -base_y)
        for pass_shadow in (True, False):
            p = skia.Paint(AntiAlias=True)
            if pass_shadow:
                c0, sig, sdy = sh
                p.setColor4f(skia.Color4f(c0.fR, c0.fG, c0.fB, c0.fA * lk.alpha))
                p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, sig + lk.blur))
            else:
                p.setColor4f(accent)
                if lk.blur > 0.3:
                    p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, lk.blur))
            dy = sh[2] if pass_shadow else 0
            xx = ox
            if self.prefix:
                canvas.drawString(self.prefix, xx, base_y + dy, self.font, p)
                xx += self.pre_w
            for ch, cxx, cw in cells:
                gx = xx + offset + cxx + (cw - self.font.measureText(ch)) / 2
                canvas.drawString(ch, gx, base_y + dy, self.font, p)
            xx += self.num_w
            if self.suffix:
                xx += 0.03 * self.size_px
                canvas.drawString(self.suffix, xx, base_y + dy, self.sym_font, p)
                xx += self.sym_w - 0.03 * self.size_px
            if self.unit:
                if not pass_shadow:
                    p.setColor4f(white)
                xx += 0.10 * self.size_px
                canvas.drawString(self.unit, xx, base_y + dy, self.suf_font, p)
        canvas.restore()
        yy = base_y + 0.12 * self.size_px + 1.2 * self.u
        if self.label:
            lx = self.pad + ((W - 2 * self.pad - self.label.width) / 2 if align == "center" else (W - 2 * self.pad - self.label.width if align == "right" else 0))
            def look(un):
                l2 = self.reveal_look(t, self.t_in + 0.12)
                l2.alpha *= 0.92
                return l2
            draw_layout(canvas, self.label, lx, yy, self.ctx.fb, 500, col(self.style["colors"]["text"]), look, sh)
            yy += self.label.height + 0.6 * self.u
        if self.sub:
            lx = self.pad + ((W - 2 * self.pad - self.sub.width) / 2 if align == "center" else (W - 2 * self.pad - self.sub.width if align == "right" else 0))
            def look3(un):
                l3 = self.reveal_look(t, self.t_in + 0.22)
                l3.alpha *= 0.7
                return l3
            draw_layout(canvas, self.sub, lx, yy, self.ctx.fb, 400, col(self.style["colors"]["text"]), look3, sh)
        canvas.restore()
