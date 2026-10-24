"""Text layout for motion typography: per-unit (CJK char / Latin word) layout with kinsoku
line-breaking, balanced wrapping, highlight tags and two-pass (shadow, fill) drawing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import skia

from ..text import NO_LINE_END, NO_LINE_START, is_cjk
from .fonts import FontBook


@dataclass
class Unit:
    text: str
    w: float
    i0: int
    i1: int
    tag: str = ""
    font: Optional[skia.Font] = None
    x: float = 0.0
    y: float = 0.0
    line: int = 0
    idx: int = 0
    trail: float = 0.0   # width of trailing whitespace


def _break_positions(text: str) -> set[int] | None:
    """Allowed line-break positions (char index before which a break may occur) for CJK text,
    from jieba word segmentation, so words such as 百老汇 are never split across lines."""
    if not any(is_cjk(c) for c in text):
        return None
    try:
        import jieba  # type: ignore

        jieba.setLogLevel(60)
        pos, k = set(), 0
        for w in jieba.cut(text, HMM=True):
            k += len(w)
            pos.add(k)
        return pos
    except Exception:
        return None


@dataclass
class TextLayout:
    units: list
    lines: int
    width: float
    height: float
    line_h: float
    ascent: float
    descent: float
    size: float
    line_widths: list = field(default_factory=list)

    def line_units(self, k: int) -> list:
        return [u for u in self.units if u.line == k]


def _tokens(text: str, tags: list[str]) -> list[tuple[str, int, int, str]]:
    toks: list[list] = []  # [text, i0, i1, tag]
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            if toks:
                toks[-1][0] += ch
                toks[-1][2] = i + 1
            i += 1
            continue
        if is_cjk(ch):
            toks.append([ch, i, i + 1, tags[i]])
            i += 1
            continue
        if ch in NO_LINE_START and toks:
            toks[-1][0] += ch
            toks[-1][2] = i + 1
            i += 1
            continue
        j = i
        while j < n and not text[j].isspace() and not is_cjk(text[j]) and not (j > i and text[j] in NO_LINE_START and text[j] not in ".,%'’-"):
            j += 1
            if text[j - 1] in NO_LINE_END and j < n:
                continue
        j = max(j, i + 1)
        seg = text[i:j]
        hl = sum(1 for k in range(i, j) if tags[k]) * 2 >= (j - i)
        toks.append([seg, i, j, "hl" if hl else ""])
        i = j
    # attach opening brackets to the following token
    out: list[list] = []
    carry = None
    for t in toks:
        if carry is not None:
            t = [carry[0] + t[0], carry[1], t[2], t[3] or carry[3]]
            carry = None
        if t[0].strip() and all(c in NO_LINE_END for c in t[0].strip()):
            carry = t
            continue
        out.append(t)
    if carry is not None:
        out.append(carry)
    return [(t[0], t[1], t[2], t[3]) for t in out]


def tag_spans(text: str, highlights) -> list[str]:
    tags = [""] * len(text)
    if not highlights:
        return tags
    if isinstance(highlights, str):
        highlights = [highlights]
    for h in highlights:
        if not h:
            continue
        start = 0
        while True:
            k = text.find(h, start)
            if k < 0:
                break
            for q in range(k, k + len(h)):
                tags[q] = "hl"
            start = k + len(h)
    return tags


def _measure(fb: FontBook, font: skia.Font, s: str, tracking: float, weight: int) -> tuple[float, Optional[skia.Font]]:
    core = s.rstrip()
    use = font
    if core and any(font.unicharToGlyph(ord(c)) == 0 for c in core if not c.isspace()):
        tf = fb.fallback_typeface(core[0], weight)
        if tf is not None:
            use = skia.Font(tf, font.getSize())
            use.setSubpixel(True)
            use.setEdging(skia.Font.Edging.kAntiAlias)
    w = use.measureText(s)
    if tracking:
        w += tracking * font.getSize() * len(s)
    return w, (use if use is not font else None)


def layout(fb: FontBook, text: str, size: float, weight: int = 400, max_w: float = 1e9, line_height: float = 1.22,
           tracking: float = 0.0, align: str = "left", balance: bool = True, highlights=None, max_lines: int = 0) -> TextLayout:
    font = fb.font(size, weight)
    tags = tag_spans(text, highlights)
    toks = _tokens(text, tags)
    brk = _break_positions(text)
    units = []
    for k, (s, i0, i1, tg) in enumerate(toks):
        w, fnt = _measure(fb, font, s, tracking, weight)
        trail = 0.0
        if s != s.rstrip():
            trail = w - _measure(fb, font, s.rstrip(), tracking, weight)[0]
        units.append(Unit(text=s, w=w, i0=i0, i1=i1, tag=tg, font=fnt, idx=k, trail=trail))

    def can_break_before(k: int) -> bool:
        if k == 0:
            return False
        u, prev = units[k], units[k - 1]
        if brk is not None and is_cjk(u.text[0]) and is_cjk(prev.text.rstrip()[-1:] or "a"):
            if u.i0 not in brk:
                return False
        if u.tag == "hl" and prev.tag == "hl" and is_cjk(u.text[0]):
            return False  # keep highlighted phrases together
        return True

    def wrap(limit: float) -> list[list[Unit]]:
        lines: list[list[Unit]] = [[]]
        cur = 0.0
        last_ok = -1  # index in current line of last break opportunity
        for k, u in enumerate(units):
            if can_break_before(k):
                last_ok = len(lines[-1])
            if lines[-1] and cur + u.w - u.trail > limit:
                ln = lines[-1]
                if 0 < last_ok < len(ln):
                    carry = ln[last_ok:]
                    lines[-1] = ln[:last_ok]
                    lines.append(carry)
                else:
                    lines.append([])
                cur = sum(x.w for x in lines[-1])
                last_ok = len(lines[-1]) if can_break_before(k) else -1
            lines[-1].append(u)
            cur += u.w
        return [ln for ln in lines if ln]

    lines = wrap(max_w)
    if balance and len(lines) > 1:
        lo, hi = 0.35 * max_w, max_w
        n = len(lines)
        for _ in range(18):
            mid = (lo + hi) / 2
            if len(wrap(mid)) > n:
                lo = mid
            else:
                hi = mid
        lines = wrap(hi)
    if max_lines and len(lines) > max_lines:
        pass  # caller decides how to shrink
    m = font.getMetrics()
    ascent, descent = -m.fAscent, m.fDescent
    lh = size * line_height
    widths = []
    for ln in lines:
        widths.append(sum(u.w for u in ln) - (ln[-1].trail if ln else 0))
    W = max(widths) if widths else 0.0
    for li, ln in enumerate(lines):
        x = 0.0
        if align == "center":
            x = (W - widths[li]) / 2
        elif align == "right":
            x = W - widths[li]
        for u in ln:
            u.x, u.y, u.line = x, ascent + li * lh, li
            x += u.w
    H = ascent + descent + (len(lines) - 1) * lh if lines else 0.0
    return TextLayout(units=units, lines=len(lines), width=W, height=H, line_h=lh, ascent=ascent, descent=descent,
                      size=size, line_widths=widths)


def fit_layout(fb: FontBook, text: str, size: float, weight: int, max_w: float, max_lines: int, min_scale: float = 0.6,
               **kw) -> TextLayout:
    s = size
    lay = layout(fb, text, s, weight, max_w, **kw)
    while lay.lines > max_lines and s > size * min_scale:
        s *= 0.93
        lay = layout(fb, text, s, weight, max_w, **kw)
    return lay


@dataclass
class UnitLook:
    alpha: float = 1.0
    dx: float = 0.0
    dy: float = 0.0
    scale: float = 1.0
    blur: float = 0.0
    color: Optional[skia.Color4f] = None


def draw_layout(canvas: skia.Canvas, lay: TextLayout, ox: float, oy: float, fb: FontBook, weight: int,
                base_color: skia.Color4f, look: Optional[Callable[[Unit], UnitLook]] = None,
                shadow: Optional[tuple] = None, tracking: float = 0.0) -> None:
    """Draw text units. shadow = (Color4f, sigma, dy) or None. Two passes so shadows never overlap fills."""
    font = fb.font(lay.size, weight)
    looks = [look(u) if look else UnitLook() for u in lay.units]
    passes = ([True] if shadow else []) + [False]
    for is_shadow in passes:
        for u, lk in zip(lay.units, looks):
            if lk.alpha <= 0.003 or not u.text.strip():
                continue
            f = u.font or font
            p = skia.Paint(AntiAlias=True)
            if is_shadow:
                c, sigma, sdy = shadow
                p.setColor4f(skia.Color4f(c.fR, c.fG, c.fB, c.fA * lk.alpha * lk.alpha))
                p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, max(0.5, sigma + lk.blur)))
            else:
                col = lk.color or base_color
                p.setColor4f(skia.Color4f(col.fR, col.fG, col.fB, col.fA * lk.alpha))
                if lk.blur > 0.25:
                    p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, lk.blur))
            x = ox + u.x + lk.dx
            y = oy + u.y + lk.dy + (shadow[2] if is_shadow else 0.0)
            txt = u.text.rstrip()
            if lk.scale != 1.0:
                canvas.save()
                cx = x + (u.w - u.trail) / 2
                cy = y - lay.size * 0.35
                canvas.translate(cx, cy)
                canvas.scale(lk.scale, lk.scale)
                canvas.translate(-cx, -cy)
            if tracking:
                xx = x
                for ch in txt:
                    canvas.drawString(ch, xx, y, f, p)
                    xx += f.measureText(ch) + tracking * f.getSize()
            else:
                canvas.drawString(txt, x, y, f, p)
            if lk.scale != 1.0:
                canvas.restore()


def unit_rects_by_line(lay: TextLayout, tag: str = "hl") -> list[tuple[int, float, float]]:
    """Consecutive tagged unit runs per line -> (line, x0, x1)."""
    runs = []
    for k in range(lay.lines):
        cur = None
        for u in lay.line_units(k):
            if u.tag == tag:
                if cur is None:
                    cur = [k, u.x, u.x + u.w - u.trail]
                else:
                    cur[2] = u.x + u.w - u.trail
            elif cur is not None:
                runs.append(tuple(cur))
                cur = None
        if cur is not None:
            runs.append(tuple(cur))
    return runs
