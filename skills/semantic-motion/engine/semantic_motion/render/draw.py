"""Drawing primitives shared by elements: colours, glass plates, shadows, strokes with trim."""
from __future__ import annotations

import skia

from ..style import hex_to_rgba


def col(hexstr: str, alpha_mul: float = 1.0) -> skia.Color4f:
    r, g, b, a = hex_to_rgba(hexstr)
    return skia.Color4f(r, g, b, max(0.0, min(1.0, a * alpha_mul)))


def resolve_color(spec, style: dict, alpha: float = 1.0) -> skia.Color4f:
    c = style["colors"]
    if spec in (None, "", "text"):
        return col(c["text"], alpha)
    if spec == "accent":
        return col(c["accent"], alpha)
    if spec in c and isinstance(c[spec], str) and c[spec].startswith("#"):
        return col(c[spec], alpha)
    return col(spec, alpha)


def rrect(x: float, y: float, w: float, h: float, r: float) -> skia.RRect:
    return skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), r, r)


def glass(canvas: skia.Canvas, x: float, y: float, w: float, h: float, r: float, style: dict, u: float,
          alpha: float = 1.0, tint_alpha: float | None = None, blur: bool = True, stroke: bool = True) -> None:
    """Frosted-glass plate: backdrop blur + dark tint + hairline, faded as a whole by `alpha`."""
    if alpha <= 0.003 or w <= 1 or h <= 1:
        return
    c = style["colors"]
    ta = c["plate_alpha"] if tint_alpha is None else tint_alpha
    rr = rrect(x, y, w, h, r)
    # soft contact shadow
    sp = skia.Paint(AntiAlias=True, Color4f=skia.Color4f(0, 0, 0, 0.22 * alpha))
    sp.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 1.6 * u))
    canvas.drawRRect(rrect(x, y + 0.5 * u, w, h, r), sp)
    canvas.save()
    canvas.clipRRect(rr, skia.ClipOp.kIntersect, True)
    lp = skia.Paint(Alphaf=alpha)
    if blur:
        bf = skia.ImageFilters.Blur(style["blur"] * u, style["blur"] * u, skia.TileMode.kClamp)
        canvas.saveLayer(skia.Canvas.SaveLayerRec(skia.Rect.MakeXYWH(x, y, w, h), lp, bf, 0))
    else:
        canvas.saveLayer(skia.Rect.MakeXYWH(x, y, w, h), lp)
    canvas.drawColor(col(c["plate"], ta))
    # faint top sheen for depth
    sheen = skia.Paint(AntiAlias=True)
    sheen.setShader(skia.GradientShader.MakeLinear([skia.Point(x, y), skia.Point(x, y + h)],
                                                   [skia.Color4f(1, 1, 1, 0.055).toColor(), skia.Color4f(1, 1, 1, 0.0).toColor()]))
    canvas.drawRect(skia.Rect.MakeXYWH(x, y, w, h), sheen)
    canvas.restore()
    canvas.restore()
    if stroke:
        p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=max(1.0, 0.11 * u))
        p.setColor4f(col(c["stroke"], alpha))
        canvas.drawRRect(rr, p)


def text_shadow(style: dict, u: float, strength: float = 1.0) -> tuple:
    c = style["colors"]
    return (col(c["shadow"], c["shadow_alpha"] * strength), style["shadow_blur"] * u, 0.22 * u)


def stroke_path(canvas: skia.Canvas, path: skia.Path, color: skia.Color4f, width: float, trim: float = 1.0,
                cap=skia.Paint.kRound_Cap) -> None:
    if trim <= 0.001:
        return
    p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=width, StrokeCap=cap,
                   StrokeJoin=skia.Paint.kRound_Join)
    p.setColor4f(color)
    if trim >= 0.999:
        canvas.drawPath(path, p)
        return
    pm = skia.PathMeasure(path, False)
    length = pm.getLength()
    seg = skia.Path()
    pm.getSegment(0, length * trim, seg, True)
    canvas.drawPath(seg, p)


def arrow_head(canvas: skia.Canvas, x: float, y: float, angle: float, size: float, color: skia.Color4f, width: float) -> None:
    import math

    p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=width, StrokeCap=skia.Paint.kRound_Cap,
                   StrokeJoin=skia.Paint.kRound_Join)
    p.setColor4f(color)
    path = skia.Path()
    a1, a2 = angle + math.radians(150), angle - math.radians(150)
    path.moveTo(x + size * math.cos(a1), y + size * math.sin(a1))
    path.lineTo(x, y)
    path.lineTo(x + size * math.cos(a2), y + size * math.sin(a2))
    canvas.drawPath(path, p)


def fill_rrect(canvas: skia.Canvas, x, y, w, h, r, color: skia.Color4f) -> None:
    p = skia.Paint(AntiAlias=True)
    p.setColor4f(color)
    canvas.drawRRect(rrect(x, y, w, h, r), p)


def fill_circle(canvas: skia.Canvas, cx, cy, r, color: skia.Color4f) -> None:
    p = skia.Paint(AntiAlias=True)
    p.setColor4f(color)
    canvas.drawCircle(cx, cy, r, p)
