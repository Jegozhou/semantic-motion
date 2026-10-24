"""List: items revealed exactly when each one is spoken (第一/第二/... or enumerations)."""
from __future__ import annotations

import skia

from ..render.draw import col, fill_circle, text_shadow
from ..render.easing import ease, prog
from ..render.text import draw_layout, layout
from .base import CardElement, register


@register
class ListCard(CardElement):
    type = "list"

    def build(self, max_w: float) -> None:
        ctx, c = self.ctx, self.c
        u = self.u
        self.pad = self.style["pad"] * u
        self.badge_r = 1.55 * u * (ctx.ts("item") / (3.3 * u))
        inner = max_w - 2 * self.pad
        self.title = layout(ctx.fb, c["title"], ctx.ts("card_title"), ctx.w("card_title", 700), inner) if c.get("title") else None
        self.items = []
        text_w = inner - 2 * self.badge_r - 1.4 * u
        for it in c.get("items", []):
            it = it if isinstance(it, dict) else {"text": str(it)}
            lt = layout(ctx.fb, it.get("text", ""), ctx.ts("item"), ctx.w("item", 500), text_w, line_height=1.25,
                        highlights=it.get("highlight"))
            ls = layout(ctx.fb, it["sub"], ctx.ts("item_sub"), 400, text_w, line_height=1.3) if it.get("sub") else None
            self.items.append((it, lt, ls))
        self.row_gap = 1.5 * u
        self.text_x = 2 * self.badge_r + 1.4 * u

    def size(self):
        u = self.u
        w = max([self.title.width if self.title else 0] + [self.text_x + max(lt.width, ls.width if ls else 0) for _, lt, ls in self.items])
        h = 0.0
        if self.title:
            h += self.title.height + 1.6 * u
        for k, (_, lt, ls) in enumerate(self.items):
            rh = max(lt.height + (0.5 * u + ls.height if ls else 0), 2 * self.badge_r)
            h += rh + (self.row_gap if k < len(self.items) - 1 else 0)
        return w + 2 * self.pad, h + 2 * self.pad

    def row_bottoms(self) -> list:
        u = self.u
        y = self.pad + ((self.title.height + 1.6 * u) if self.title else 0.0)
        out = []
        for k, (_, lt, ls) in enumerate(self.items):
            rh = max(lt.height + (0.5 * u + ls.height if ls else 0), 2 * self.badge_r)
            y += rh
            out.append(y + self.pad)
            y += self.row_gap
        return out

    def visible_height(self, t: float):
        """Card grows with the spoken items (only when items are anchored to speech)."""
        if not any(it.get("t_land") is not None for it, _, _ in self.items) or self.c.get("grow") is False:
            return None
        bottoms = self.row_bottoms()
        top = self.pad + ((self.title.height + 1.6 * self.u) if self.title else 0.0)
        h = (top + self.pad) if self.title else (bottoms[0] if bottoms else top)
        for k, b in enumerate(bottoms):
            st = self.item_time(k)
            g = ease(self.m["ease_move"], prog(t, st - 0.12, 0.38))
            if g <= 0:
                break
            prev = h if k > 0 or self.title else b
            h = prev + (b - prev) * g if k > 0 or self.title else b
        return h

    def item_time(self, k: int) -> float:
        it = self.items[k][0]
        if it.get("t_land") is not None:
            return it["t_land"] - self.enter_dur * self.m["land_frac"]
        return self.t_in + 0.18 + 0.12 * k

    def draw(self, canvas: skia.Canvas, t: float) -> None:
        if not self.active(t):
            return
        W, H = self.size()
        u = self.u
        a = self.card_open(canvas, t, W, H, H_vis=self.visible_height(t))
        white = col(self.style["colors"]["text"])
        accent = self.style["colors"]["accent"]
        sh = text_shadow(self.style, u, 0.5)
        y = self.pad
        if self.title:
            def lt_(un):
                lk = self.item_look(t, self.t_in + 0.05)
                lk.color = col(accent)
                return lk
            draw_layout(canvas, self.title, self.pad, y, self.ctx.fb, self.ctx.w("card_title", 700), white, lt_, sh)
            y += self.title.height + 1.6 * u
        style_kind = self.c.get("style", "numbered")
        times = [self.item_time(k) for k in range(len(self.items))]
        latest = max([k for k, tt in enumerate(times) if t >= tt + 0.05] or [-1])
        for k, (it, lt, ls) in enumerate(self.items):
            rh = max(lt.height + (0.5 * u + ls.height if ls else 0), 2 * self.badge_r)
            st = times[k]
            lk = self.item_look(t, st)
            dim = 1.0 if (k == latest or latest < 0 or not self.c.get("focus", True)) else 0.62
            if lk.alpha > 0.003:
                bx, by = self.pad + self.badge_r, y + min(rh, lt.line_h) / 2 + 0.05 * u
                pb = ease("outback", prog(t, st, 0.42))
                ca = lk.alpha * (1.0 if k == latest or latest < 0 else 0.75)
                if style_kind == "bullet":
                    fill_circle(canvas, bx, by + lk.dy, self.badge_r * 0.38 * pb, col(accent, ca))
                else:
                    fill_circle(canvas, bx, by + lk.dy, self.badge_r * pb, col(accent, ca))
                    if style_kind == "check":
                        p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=0.42 * u,
                                       StrokeCap=skia.Paint.kRound_Cap, StrokeJoin=skia.Paint.kRound_Join)
                        p.setColor4f(col("#111317", ca))
                        path = skia.Path()
                        r = self.badge_r * pb
                        path.moveTo(bx - 0.45 * r, by + lk.dy + 0.02 * r)
                        path.lineTo(bx - 0.1 * r, by + lk.dy + 0.36 * r)
                        path.lineTo(bx + 0.5 * r, by + lk.dy - 0.34 * r)
                        canvas.drawPath(path, p)
                    else:
                        f = self.ctx.fb.font(self.badge_r * 1.15, 800)
                        num = str(it.get("badge", k + 1))
                        p = skia.Paint(AntiAlias=True)
                        p.setColor4f(col("#111317", ca))
                        tw = f.measureText(num)
                        canvas.drawString(num, bx - tw / 2, by + lk.dy + self.badge_r * 0.40, f, p)

                def lkf(un, lk=lk, dim=dim):
                    l2 = self.item_look(t, st)
                    l2.alpha *= dim
                    if un.tag == "hl":
                        l2.color = col(accent)
                    return l2
                draw_layout(canvas, lt, self.pad + self.text_x, y, self.ctx.fb, self.ctx.w("item", 500), white, lkf, sh)
                if ls:
                    def lks(un, dim=dim):
                        l3 = self.item_look(t, st + 0.06)
                        l3.alpha *= 0.68 * dim
                        return l3
                    draw_layout(canvas, ls, self.pad + self.text_x, y + lt.height + 0.5 * u, self.ctx.fb, 400, white, lks, sh)
            y += rh + self.row_gap
        self.card_close(canvas)
