"""Custom: a procedural animation composed from primitive layers with keyframes.

This is the escape hatch for one-off visual ideas (a timeline that grows, a 3x multiplier
popping, a map-pin drop, an icon-like diagram). Coordinates are in `u` (min(W,H)/100) inside the
element box; times are seconds relative to the element start, or strings 'land+0.2' / 'out-0.1'
(relative to landing / exit) or '@12.34' (absolute, written by the compiler from anchors).
See reference/elements.md for the full layer vocabulary.
"""
from __future__ import annotations

import math
import re

import skia

from ..render.draw import col, glass, stroke_path, text_shadow
from ..render.easing import ease, prog, spring
from ..render.text import UnitLook, draw_layout, layout
from .base import Element, register

ANIMATABLE = {"x", "y", "w", "h", "r", "cx", "cy", "opacity", "scale", "rotate", "trim", "sweep", "start", "dx", "dy",
              "blur", "progress", "stroke_w", "size"}


def parse_svg_path(d: str, k: float = 1.0) -> skia.Path:
    """Minimal SVG path parser (M L H V C S Q T Z, absolute + relative); coordinates scaled by k."""
    toks = re.findall(r"[MmLlHhVvCcSsQqTtZz]|-?\d*\.?\d+(?:e-?\d+)?", d)
    p = skia.Path()
    i, cmd = 0, None
    cx = cy = sx = sy = 0.0
    lcx = lcy = None

    def num():
        nonlocal i
        v = float(toks[i]) * k
        i += 1
        return v

    while i < len(toks):
        if re.match(r"[A-Za-z]", toks[i]):
            cmd = toks[i]
            i += 1
            if cmd in "Zz":
                p.close()
                cx, cy = sx, sy
                continue
        rel = cmd.islower()
        C = cmd.upper()
        ox, oy = (cx, cy) if rel else (0.0, 0.0)
        if C == "M":
            cx, cy = ox + num(), oy + num()
            p.moveTo(cx, cy)
            sx, sy = cx, cy
            cmd = "l" if rel else "L"
        elif C == "L":
            cx, cy = ox + num(), oy + num()
            p.lineTo(cx, cy)
        elif C == "H":
            cx = (cx if rel else 0) + num()
            p.lineTo(cx, cy)
        elif C == "V":
            cy = (cy if rel else 0) + num()
            p.lineTo(cx, cy)
        elif C == "C":
            x1, y1, x2, y2, x, y = ox + num(), oy + num(), ox + num(), oy + num(), ox + num(), oy + num()
            p.cubicTo(x1, y1, x2, y2, x, y)
            lcx, lcy, cx, cy = x2, y2, x, y
        elif C == "S":
            x1, y1 = (2 * cx - lcx, 2 * cy - lcy) if lcx is not None else (cx, cy)
            x2, y2, x, y = ox + num(), oy + num(), ox + num(), oy + num()
            p.cubicTo(x1, y1, x2, y2, x, y)
            lcx, lcy, cx, cy = x2, y2, x, y
        elif C == "Q":
            x1, y1, x, y = ox + num(), oy + num(), ox + num(), oy + num()
            p.quadTo(x1, y1, x, y)
            lcx, lcy, cx, cy = x1, y1, x, y
        elif C == "T":
            x1, y1 = (2 * cx - lcx, 2 * cy - lcy) if lcx is not None else (cx, cy)
            x, y = ox + num(), oy + num()
            p.quadTo(x1, y1, x, y)
            lcx, lcy, cx, cy = x1, y1, x, y
        else:
            i += 1
    return p


def _color(spec, style, alpha=1.0):
    if spec is None:
        return None
    s = str(spec)
    a = 1.0
    if "@" in s:
        s, aa = s.split("@", 1)
        a = float(aa)
    c = style["colors"]
    if s in c and isinstance(c[s], str) and c[s].startswith("#"):
        hx = c[s]
        if s == "plate":
            a *= c.get("plate_alpha", 0.5) if "@" not in str(spec) else 1.0
    elif s.startswith("#"):
        hx = s
    else:
        hx = c["text"]
    return col(hx, a * alpha)


@register
class Custom(Element):
    type = "custom"
    default_region = "auto"

    def build(self, max_w: float) -> None:
        c = self.c
        sz = c.get("size", [40, 24])
        self.bw, self.bh = float(sz[0]) * self.u, float(sz[1]) * self.u
        self.scale_fit = 1.0
        if self.bw > max_w and not self.spec.get("box_fixed"):
            self.scale_fit = max_w / self.bw
        self.layers = c.get("layers", [])
        self._text_cache: dict = {}

    def size(self):
        return self.bw * self.scale_fit, self.bh * self.scale_fit

    # ------------------------------------------------------------------ keyframes
    def _time(self, v) -> float:
        if isinstance(v, (int, float)):
            return self.t_in + float(v)
        s = str(v).strip()
        if s.startswith("@"):
            return float(s[1:])
        m = re.match(r"^(land|out|in|end)\s*([+-]\s*\d*\.?\d+)?$", s)
        if m:
            base = {"land": self.t_land, "out": self.t_out, "in": self.t_in, "end": self.t_end}[m.group(1)]
            return base + (float(m.group(2).replace(" ", "")) if m.group(2) else 0.0)
        return self.t_in + float(s)

    def _val(self, layer: dict, prop: str, t: float, default):
        keys = (layer.get("keys") or {}).get(prop)
        if not keys:
            return layer.get(prop, default)
        pts = [(self._time(k[0]), k[1], (k[2] if len(k) > 2 else "out")) for k in keys]
        if t <= pts[0][0]:
            return pts[0][1]
        for (t0, v0, _), (t1, v1, e) in zip(pts, pts[1:]):
            if t0 <= t < t1:
                x = (t - t0) / max(1e-6, t1 - t0)
                if e == "spring":
                    y = spring(x)
                elif e == "hold":
                    y = 0.0
                else:
                    y = ease(e, x)
                return v0 + (v1 - v0) * y
        return pts[-1][1]

    # ------------------------------------------------------------------ draw
    def draw(self, canvas: skia.Canvas, t: float) -> None:
        if not self.active(t):
            return
        x, y, _, _ = self.spec["box"]
        s = self.spec.get("scale", 1.0) * self.scale_fit
        auto = self.c.get("auto_motion", True)
        pi = self.p_in(t) if auto else 1.0
        po = self.p_out(t) if auto else (1.0 if t >= self.t_end else 0.0)
        ga = min(1.0, pi * 1.3) * (1 - po)
        if ga <= 0.003:
            return
        canvas.save()
        canvas.translate(x, y)
        canvas.scale(s, s)
        if auto:
            sc = self.m["scale_in"] + (1 - self.m["scale_in"]) * pi
            canvas.translate(self.bw / 2, self.bh / 2 + (1 - pi) * self.m["travel"] * self.u)
            canvas.scale(sc, sc)
            canvas.translate(-self.bw / 2, -self.bh / 2)
        if self.spec.get("backing") in ("glass", "plate"):
            glass(canvas, 0, 0, self.bw, self.bh, self.u * self.style["radius"], self.style, self.u, alpha=ga,
                  blur=self.spec.get("backing") == "glass")
        for li, L in enumerate(self.layers):
            self._draw_layer(canvas, L, t, ga, li)
        canvas.restore()

    def _paints(self, L, t, alpha):
        u = self.u
        fill = _color(L.get("fill"), self.style, alpha) if L.get("fill") else None
        stroke = _color(L.get("stroke"), self.style, alpha) if L.get("stroke") else None
        sw = float(self._val(L, "stroke_w", t, 0.35)) * u
        return fill, stroke, sw

    def _draw_layer(self, canvas: skia.Canvas, L: dict, t: float, ga: float, li: int) -> None:
        u = self.u
        shape = L.get("shape", "rect")
        op = float(self._val(L, "opacity", t, 1.0)) * ga
        if op <= 0.003:
            return
        if "from" in L and "keys" not in L:  # convenience: {"from": 0.3} = fade/rise in at 0.3 s
            t0 = self._time(L["from"])
            p = ease(self.m["ease_in_curve"], prog(t, t0, float(L.get("dur", self.enter_dur))))
            op *= p
            L = {**L, "dy": L.get("dy", 0) + (1 - p) * 1.0}
        scale = float(self._val(L, "scale", t, 1.0))
        rot = float(self._val(L, "rotate", t, 0.0))
        dx = float(self._val(L, "dx", t, 0.0)) * u
        dy = float(self._val(L, "dy", t, 0.0)) * u
        blur = float(self._val(L, "blur", t, 0.0)) * u
        fill, stroke, sw = self._paints(L, t, op)
        # pivot
        if shape in ("circle", "arc", "ring"):
            px, py = float(self._val(L, "cx", t, 0)) * u, float(self._val(L, "cy", t, 0)) * u
        elif shape in ("line", "arrow", "polyline"):
            pts = L.get("points", [[0, 0], [1, 1]])
            px = sum(p[0] for p in pts) / len(pts) * u
            py = sum(p[1] for p in pts) / len(pts) * u
        else:
            px = (float(self._val(L, "x", t, 0)) + float(self._val(L, "w", t, 0)) / 2) * u
            py = (float(self._val(L, "y", t, 0)) + float(self._val(L, "h", t, 0)) / 2) * u
        canvas.save()
        canvas.translate(dx, dy)
        if scale != 1.0 or rot:
            canvas.translate(px, py)
            canvas.rotate(rot)
            canvas.scale(scale, scale)
            canvas.translate(-px, -py)

        def paint_fill(c):
            p = skia.Paint(AntiAlias=True)
            p.setColor4f(c)
            if blur > 0.2:
                p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
            return p

        def paint_stroke(c):
            p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=sw, StrokeCap=skia.Paint.kRound_Cap,
                           StrokeJoin=skia.Paint.kRound_Join)
            p.setColor4f(c)
            return p

        trim = float(self._val(L, "trim", t, 1.0))
        if shape in ("rect", "rrect"):
            X, Y = float(self._val(L, "x", t, 0)) * u, float(self._val(L, "y", t, 0)) * u
            Wd, Ht = float(self._val(L, "w", t, 10)) * u, float(self._val(L, "h", t, 10)) * u
            r = float(self._val(L, "r", t, 0 if shape == "rect" else 1.5)) * u
            rr = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(X, Y, max(0, Wd), max(0, Ht)), r, r)
            if L.get("glass"):
                glass(canvas, X, Y, Wd, Ht, r, self.style, u, alpha=op)
            if fill:
                canvas.drawRRect(rr, paint_fill(fill))
            if stroke:
                path = skia.Path()
                path.addRRect(rr)
                stroke_path(canvas, path, stroke, sw, trim)
        elif shape in ("circle",):
            cx, cy = float(self._val(L, "cx", t, 0)) * u, float(self._val(L, "cy", t, 0)) * u
            r = float(self._val(L, "r", t, 2)) * u
            if fill:
                canvas.drawCircle(cx, cy, r, paint_fill(fill))
            if stroke:
                path = skia.Path()
                path.addCircle(cx, cy, r)
                stroke_path(canvas, path, stroke, sw, trim)
        elif shape in ("arc", "ring"):
            cx, cy = float(self._val(L, "cx", t, 0)) * u, float(self._val(L, "cy", t, 0)) * u
            r = float(self._val(L, "r", t, 5)) * u
            st = float(self._val(L, "start", t, -90))
            sweep = float(self._val(L, "sweep", t, 360))
            if L.get("track"):
                tp = paint_stroke(_color(L["track"], self.style, op))
                canvas.drawCircle(cx, cy, r, tp)
            if abs(sweep) > 0.1:
                path = skia.Path()
                path.addArc(skia.Rect.MakeXYWH(cx - r, cy - r, 2 * r, 2 * r), st, sweep * trim)
                canvas.drawPath(path, paint_stroke(stroke or _color("accent", self.style, op)))
        elif shape in ("line", "polyline", "arrow", "path"):
            if shape == "path":
                path = parse_svg_path(L.get("d", ""), u)
            else:
                pts = L.get("points", [[0, 0], [10, 0]])
                path = skia.Path()
                path.moveTo(pts[0][0] * u, pts[0][1] * u)
                for q in pts[1:]:
                    path.lineTo(q[0] * u, q[1] * u)
            if fill and shape == "path":
                canvas.drawPath(path, paint_fill(fill))
            sc = stroke or (None if (fill and shape == "path") else _color("text", self.style, op))
            if sc is not None:
                stroke_path(canvas, path, sc, sw, trim)
            if shape == "arrow" and trim > 0.02:
                pm = skia.PathMeasure(path, False)
                L_ = pm.getLength()
                res = pm.getPosTan(L_ * trim)
                if res:
                    pos, tan = res
                    from ..render.draw import arrow_head

                    arrow_head(canvas, pos.x(), pos.y(), math.atan2(tan.y(), tan.x()), 1.6 * u * float(L.get("head", 1.0)),
                               sc or _color("text", self.style, op), sw)
        elif shape == "text":
            self._draw_text(canvas, L, t, op, li)
        canvas.restore()

    def _draw_text(self, canvas, L, t, op, li):
        u = self.u
        size = float(self._val(L, "size", t, 4.0)) * u
        weight = int(L.get("weight", 700))
        txt = L.get("text", "")
        cnt = L.get("count")
        if cnt:
            pr = float(self._val(L, "progress", t, 1.0))
            v = cnt.get("from", 0) + (cnt.get("to", 0) - cnt.get("from", 0)) * pr
            d = int(cnt.get("decimals", 0))
            txt = f"{cnt.get('prefix', '')}{v:,.{d}f}{cnt.get('suffix', '')}" if cnt.get("separator", "") == "," else \
                f"{cnt.get('prefix', '')}{v:.{d}f}{cnt.get('suffix', '')}"
        key = (li, txt, round(size, 2), weight)
        lay = self._text_cache.get(key)
        if lay is None:
            maxw = float(L.get("max_w", 1e4)) * u
            lay = layout(self.ctx.fb, txt, size, weight, maxw, line_height=1.2, align=L.get("align", "left"),
                         highlights=L.get("highlight"))
            if len(self._text_cache) > 64:
                self._text_cache.clear()
            self._text_cache[key] = lay
        X, Y = float(self._val(L, "x", t, 0)) * u, float(self._val(L, "y", t, 0)) * u
        al = L.get("align", "left")
        if al == "center":
            X -= lay.width / 2
        elif al == "right":
            X -= lay.width
        va = L.get("valign", "top")
        if va == "middle":
            Y -= lay.height / 2
        elif va == "bottom":
            Y -= lay.height
        c = _color(L.get("fill", "text"), self.style, 1.0)
        acc = col(self.style["colors"]["accent"])
        reveal = L.get("reveal", "whole")
        t0 = self._time(L.get("from", 0))

        def look(un):
            if reveal == "word":
                n = max(1, len(lay.units))
                st = min(self.m["stagger"], self.m["stagger_cap"] / n)
                lk = self.reveal_look(t, t0 + un.idx * st)
                lk.alpha *= op
            else:
                lk = UnitLook(alpha=op)
            if un.tag == "hl":
                lk.color = acc
            return lk

        draw_layout(canvas, lay, X, Y, self.ctx.fb, weight, c, look,
                    text_shadow(self.style, u, float(L.get("shadow", 0.8))) if L.get("shadow", 0.8) else None)
