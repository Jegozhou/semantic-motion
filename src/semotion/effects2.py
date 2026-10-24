"""Procedural motion-graphics elements (v2).

An *element* is a declarative spec produced by :mod:`semotion.design2`; this module
measures it (``fit``) and paints one frame of it (``paint``). Everything is drawn
with Pillow at 2x and downsampled, so type stays crisp at any output size.

Design rules encoded here (see docs/DESIGN.md):

* type first, one accent, generous negative space;
* the backdrop is drawn by the renderer, not by the element — an element only
  paints glyphs, rules and marks, so the same element can sit on a scrim, a card
  or bare footage without changing its geometry;
* motion is short, eased and purposeful: nothing spins or bounces, no element
  enters twice, and list rows are revealed on the words that speak them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import theme2 as T

SS = 2  # supersampling factor for all element rasterisation


# --------------------------------------------------------------------------- ctx
@dataclass
class Ctx:
    book: T.FontBook
    palette: T.Palette
    canvas_w: int
    canvas_h: int
    fps: float
    language: str = "zh"
    ref_h: int = 0

    def __post_init__(self):
        if not self.ref_h:
            self.ref_h = T.reference_height(self.canvas_w, self.canvas_h)


# ----------------------------------------------------------------------- helpers
def _fmt_number(v: float, decimals: int, thousands: bool = True) -> str:
    if abs(v - round(v)) < 1e-9:
        s = f"{int(round(v)):,}" if thousands else str(int(round(v)))
    else:
        s = f"{v:,.{decimals}f}" if thousands else f"{v:.{decimals}f}"
    return s


def _is_stat_number(text: Any) -> tuple[bool, float, int]:
    try:
        s = str(text).strip().replace(",", "")
        if s.endswith("%"):
            s = s[:-1]
        v = float(s)
        dec = len(s.split(".")[1]) if "." in s else 0
        return True, v, dec
    except (TypeError, ValueError):
        return False, 0.0, 0


def _masked_layer(size: tuple[int, int], draw_fn) -> Image.Image:
    """Draw at SS and return an RGBA image at final size with anti-aliasing."""
    w, h = max(1, int(size[0])), max(1, int(size[1]))
    img = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(img), SS)
    return img.resize((w, h), Image.LANCZOS)


def _wipe_mask(w: int, h: int, p: float, soft: int = 14, direction: str = "up") -> Image.Image:
    """Vertical reveal mask: p=0 hidden, p=1 fully visible."""
    p = T.clamp(p)
    if p >= 1:
        return Image.new("L", (w, h), 255)
    if p <= 0:
        return Image.new("L", (w, h), 0)
    y = np.arange(h, dtype=np.float32)
    dist = (h - 1 - y) if direction == "up" else y
    edge = p * h
    arr = np.clip((edge - dist) / max(1.0, soft), 0.0, 1.0)
    m = Image.fromarray((np.repeat(arr[:, None], w, axis=1) * 255).astype(np.uint8), "L")
    return m


def _draw_rich(draw: ImageDraw.ImageDraw, x: float, y: float, line: str,
               font, base, accent_color, spans: Sequence[str], tracking: float = 0.0):
    """Draw a line character by character so substrings can be accented."""
    spans = [s for s in spans if s]
    i = 0
    for ch in line:
        hit = any(line.startswith(s, i) for s in spans)
        color = accent_color if hit else base
        draw.text((x, y), ch, font=font, fill=color)
        x += font.getlength(ch) + tracking
        i += 1
    return x


def _fitting_font(ctx: Ctx, text: str, kind: str, max_w: float, start_px: int,
                  max_lines: int) -> tuple[str, int, list[str]] | None:
    """Shrink type until the text wraps into <= max_lines inside max_w."""
    px = start_px
    floor = max(14, int(start_px * 0.58))
    flat = "".join(text.split())
    while px >= floor:
        font = ctx.book.get(kind if kind in ("display", "semibold", "regular") else "display", px)
        lines = T.wrap_text(text, font, max_w, max_lines=max_lines)
        widest = max((T.text_width(font, ln) for ln in lines), default=0)
        complete = "".join("".join(lines).split()) == flat
        if lines and complete and widest <= max_w * 1.005 and len(lines) <= max_lines:
            return kind, px, lines
        px = int(px * 0.92)
    return None


# -------------------------------------------------------------------------- fit
@dataclass
class Fit:
    kind: str                      # keyword | card | list | stat | barchart | compare | quote
    width: int
    height: int
    pad: int
    display_px: int = 0
    title_px: int = 0
    body_px: int = 0
    kicker_px: int = 0
    pitch: int = 0
    lines: list = field(default_factory=list)
    body_lines: list = field(default_factory=list)
    kicker: str = ""
    emphasis: list = field(default_factory=list)
    num_str: str = ""
    unit: str = ""
    label: str = ""
    rows: list = field(default_factory=list)      # [{marker,text,at_s,lines,marker_px}]
    bars: list = field(default_factory=list)      # [{label,value,frac}]
    compare: dict = field(default_factory=dict)
    quote: dict = field(default_factory=dict)
    rule: bool = False
    align: str = "left"
    min_px: int = 12


def _content(element: dict) -> dict:
    return element.get("content") or {}


def _kicker_for(element: dict, language: str) -> str:
    """Optional eyebrow label. Only where it adds structure, never decoration."""
    c = _content(element)
    if c.get("kicker"):
        return str(c["kicker"])
    kind = element.get("type")
    zh = {"list": "要点", "stat": "数据", "barchart": "对比", "compare": "对比",
          "quote": "引言"}
    en = {"list": "KEY POINTS", "stat": "THE NUMBER", "barchart": "BY THE NUMBERS",
          "compare": "SIDE BY SIDE", "quote": "QUOTE"}
    return (en if language == "en" else zh).get(kind, "")


def _draw_kicker(draw, x, y, text, ctx, px, color) -> float:
    """Eyebrow label: CJK uses the CJK face, Latin uses tracked caps."""
    if not text:
        return x
    cjk = any("\u4e00" <= ch <= "\u9fff" for ch in text)
    if cjk:
        f = ctx.book.get("semibold", int(px))
        t = px * 0.06
    else:
        f = ctx.book.get("latin", int(px), "semibold")
        t = px * 0.18
    return T.draw_tracked(draw, (x, y), text, f, color, t)


def _kicker_width(text, ctx, px) -> float:
    if not text:
        return 0.0
    cjk = any("\u4e00" <= ch <= "\u9fff" for ch in text)
    if cjk:
        f = ctx.book.get("semibold", int(px))
        return T.text_width(f, text, px * 0.06)
    f = ctx.book.get("latin", int(px), "semibold")
    return T.text_width(f, text, px * 0.18)


def fit_element(element: dict, ctx: Ctx, max_w: float, max_h: float,
                tier: float = 1.0) -> Fit | None:
    fit = _fit_element_inner(element, ctx, max_w, max_h, tier)
    if fit is not None:
        fit.min_px = max(10, int(round((fit.display_px or fit.title_px or fit.body_px) * 0.58)))
    return fit


def _fit_element_inner(element: dict, ctx: Ctx, max_w: float, max_h: float,
                       tier: float = 1.0) -> Fit | None:
    """Measure an element inside (max_w, max_h) and return its resolved geometry."""
    kind = element.get("type", "keyword")
    c = _content(element)
    ref = ctx.ref_h
    max_w = max(120, int(max_w))
    max_h = max(80, int(max_h))
    pad = 6 if (element.get("backdrop") or {}).get("kind", "none") != "card" else 0

    if kind == "keyword":
        text = str(c.get("text") or "").strip()
        if not text:
            return None
        start = int(T.TYPE_RATIO["display"] * ref * tier)
        got = _fitting_font(ctx, text, "display", max_w - 2 * pad, start, 2)
        if not got:
            return None
        _, dpx, lines = got
        pitch = T.line_pitch(dpx, "display")
        kicker = _kicker_for(element, ctx.language)
        kpx = max(12, int(T.TYPE_RATIO["meta"] * ref * 1.02))
        body = c.get("body")
        bpx = max(14, int(T.TYPE_RATIO["body"] * ref * 0.95))
        body_lines: list[str] = []
        if body:
            bf = ctx.book.get("semibold", bpx)
            body_lines = T.wrap_text(str(body), bf, max_w - 2 * pad, max_lines=2)
        tw = max([T.text_width(ctx.book.get("display", dpx), ln) for ln in lines] or [0])
        bw = max([T.text_width(ctx.book.get("semibold", bpx), ln) for ln in body_lines] or [0])
        kw = _kicker_width(kicker, ctx, kpx)
        width = int(max(tw, bw, kw)) + 2 * pad
        h = 0
        if kicker:
            h += int(kpx * 1.5) + int(dpx * 0.42)
        h += len(lines) * pitch
        if body_lines:
            h += int(dpx * 0.30) + len(body_lines) * T.line_pitch(bpx, "body")
        h += int(dpx * 0.62)  # breathing room for the accent rule
        height = int(min(h, max_h))
        return Fit(kind=kind, width=min(width, max_w), height=height, pad=pad,
                   display_px=dpx, body_px=bpx, kicker_px=kpx, pitch=pitch,
                   lines=lines, body_lines=body_lines, kicker=kicker,
                   emphasis=c.get("emphasis") or [], rule=True,
                   align=c.get("align", "left"))

    if kind == "card":
        title = str(c.get("title") or c.get("text") or "").strip()
        if not title:
            return None
        kicker = _kicker_for(element, ctx.language)
        bpx = max(13, int(T.TYPE_RATIO["body"] * ref * 0.95))
        kpx = max(11, int(T.TYPE_RATIO["meta"] * ref * 0.95))
        start = int(T.TYPE_RATIO["title"] * ref * tier)
        bf_all = ctx.book.get("semibold", bpx)
        body = str(c.get("body") or "").strip()
        body_lines = T.wrap_text(body, bf_all, max_w - 2 * pad, max_lines=2) if body else []
        # title may use two lines; body needs the remaining height
        avail_h = max_h - (len(body_lines) * T.line_pitch(bpx, "body")) - int(bpx * 2.6)
        got = _fitting_font(ctx, title, "display", max_w - 2 * pad, start, 2)
        if not got:
            return None
        _, tpx, lines = got
        pitch = T.line_pitch(tpx, "title")
        while len(lines) * pitch > avail_h and tpx > int(start * 0.6):
            tpx = int(tpx * 0.92)
            font = ctx.book.get("display", tpx)
            lines = T.wrap_text(title, font, max_w - 2 * pad, max_lines=2)
            pitch = T.line_pitch(tpx, "title")
        tw = max([T.text_width(ctx.book.get("display", tpx), ln) for ln in lines] or [0])
        bw = max([T.text_width(bf_all, ln) for ln in body_lines] or [0])
        kw = _kicker_width(kicker, ctx, kpx)
        width = int(max(tw, bw, kw)) + 2 * pad
        h = int(kpx * 1.5) + int(tpx * 0.52) + len(lines) * pitch
        if body_lines:
            h += int(tpx * 0.28) + len(body_lines) * T.line_pitch(bpx, "body")
        return Fit(kind=kind, width=min(width, max_w), height=int(min(h, max_h)),
                   pad=pad, title_px=tpx, body_px=bpx, kicker_px=kpx, pitch=pitch,
                   lines=lines, body_lines=body_lines, kicker=kicker,
                   emphasis=c.get("emphasis") or [], align=c.get("align", "left"))

    if kind == "list":
        rows_in = c.get("rows") or []
        if not rows_in:
            items = c.get("items") or ([c.get("text")] if c.get("text") else [])
            rows_in = [{"text": t, "at_s": 0.0} for t in items]
        if not rows_in:
            return None
        kicker = _kicker_for(element, ctx.language)
        kpx = max(11, int(T.TYPE_RATIO["meta"] * ref * 0.95))
        start_px = int(T.TYPE_RATIO["title"] * ref * 0.86 * tier)
        marker_px = max(18, int(start_px * 0.62))
        row_gap = int(start_px * 0.46)
        avail = max_w - 2 * pad - marker_px * 2.1
        texts = [str(r.get("text") or "").strip() for r in rows_in[:5]]
        texts = [t for t in texts if t]
        if not texts:
            return None
        # one type size for the whole list: the largest that fits every row
        chosen = None
        px = start_px
        floor = max(14, int(start_px * 0.55))
        while px >= floor:
            font = ctx.book.get("semibold", px)
            plan = []
            ok = True
            for text in texts:
                lines = T.wrap_text(text, font, avail, max_lines=2)
                if "".join("".join(lines).split()) != "".join(text.split()):
                    ok = False
                    break
                widest = max((T.text_width(font, ln) for ln in lines), default=0)
                if widest > avail * 1.005:
                    ok = False
                    break
                plan.append(lines)
            if ok:
                chosen = (px, plan)
                break
            px = int(px * 0.94)
        if chosen is None:
            return None
        tpx, plan = chosen
        pitch = T.line_pitch(tpx, "title")
        rows = []
        for i, (r, lines) in enumerate(zip(rows_in[:5], plan)):
            rows.append({"marker": str(r.get("marker") or (i + 1)),
                         "text": str(r.get("text") or "").strip(),
                         "at_s": float(r.get("at_s") or 0.0), "lines": lines,
                         "px": tpx})
        widest = max(T.text_width(ctx.book.get("semibold", tpx), ln)
                     for r in rows for ln in r["lines"])
        width = int(widest + marker_px * 2.1) + 2 * pad
        h = int(kpx * 1.5) + int(tpx * 0.5)
        for r in rows:
            h += len(r["lines"]) * pitch + row_gap
        h -= row_gap
        return Fit(kind=kind, width=min(width, max_w), height=int(min(h, max_h)),
                   pad=pad, title_px=tpx, kicker_px=kpx, pitch=pitch, kicker=kicker,
                   rows=rows, emphasis=c.get("emphasis") or [])

    if kind == "stat":
        raw = c.get("value", c.get("text"))
        if raw is None or str(raw).strip() == "":
            return None
        ok, value, dec = _is_stat_number(raw)
        display = str(c.get("display") or "").strip()
        unit = str(c.get("unit") or "")
        label = str(c.get("label") or "").strip()
        prefix = str(c.get("prefix") or "")
        if not ok and display:
            # non-numeric stat (e.g. "九成"): draw the display string large
            got = _fitting_font(ctx, display, "display", max_w - 2 * pad,
                                int(T.TYPE_RATIO["stat"] * ref * 0.62 * tier), 2)
            if not got:
                return None
            _, npx, lines = got
            label_lines = T.wrap_text(label, ctx.book.get("semibold", int(npx * 0.20)),
                                      max_w - 2 * pad, max_lines=2) if label else []
            kicker = _kicker_for(element, ctx.language)
            kpx = max(11, int(T.TYPE_RATIO["meta"] * ref * 0.95))
            width = int(max([T.text_width(ctx.book.get("display", npx), l) for l in lines]
                            + [T.text_width(ctx.book.get("semibold", int(npx * 0.2)), l)
                               for l in label_lines] + [10])) + 2 * pad
            h = int(kpx * 1.5) + int(npx * 0.3) + len(lines) * T.line_pitch(npx, "stat") \
                + (int(npx * 0.26) + len(label_lines) * T.line_pitch(int(npx * 0.2), "body")
                   if label_lines else 0)
            return Fit(kind=kind, width=min(width, max_w), height=int(min(h, max_h)),
                       pad=pad, display_px=npx, kicker_px=kpx, pitch=T.line_pitch(npx, "stat"),
                       lines=lines, kicker=kicker, label=label, unit=unit,
                       num_str=display, rule=True)
        npx = int(T.TYPE_RATIO["stat"] * ref * tier)
        npx = min(npx, int(max_h * 0.55))
        num_str = _fmt_number(value, dec)
        nfont = ctx.book.get("latin", npx, "bold")
        while T.text_width(nfont, f"{prefix}{num_str}{c.get('suffix', '')}") > max_w - 2 * pad \
                and npx > 24:
            npx = int(npx * 0.9)
            nfont = ctx.book.get("latin", npx, "bold")
        upx = max(12, int(npx * 0.30))
        kicker = _kicker_for(element, ctx.language)
        kpx = max(11, int(T.TYPE_RATIO["meta"] * ref * 0.95))
        lpx = max(13, int(T.TYPE_RATIO["body"] * ref * 0.95))
        label_lines = T.wrap_text(label, ctx.book.get("semibold", lpx), max_w - 2 * pad,
                                  max_lines=2) if label else []
        nw = T.text_width(nfont, f"{prefix}{num_str}") + \
            (T.text_width(ctx.book.get("semibold", upx), unit) if unit else 0)
        widths = [nw] + [T.text_width(ctx.book.get("semibold", lpx), l) for l in label_lines]
        width = int(max(widths)) + 2 * pad
        h = int(kpx * 1.5) + int(npx * 0.22) + npx + \
            (int(npx * 0.16) + len(label_lines) * T.line_pitch(lpx, "body") if label_lines else 0) + \
            int(npx * 0.22)
        return Fit(kind=kind, width=min(int(width), max_w), height=int(min(h, max_h)),
                   pad=pad, display_px=npx, body_px=lpx, kicker_px=kpx, kicker=kicker,
                   num_str=num_str, unit=unit, label="\n".join(label_lines), rule=True,
                   emphasis=[])

    if kind == "barchart":
        title = str(c.get("title") or "").strip()
        bars_in = c.get("bars") or []
        if not bars_in:
            return None
        kicker = _kicker_for(element, ctx.language)
        kpx = max(11, int(T.TYPE_RATIO["meta"] * ref * 0.95))
        bars = []
        vmax = max(float(b.get("value") or 0) for b in bars_in) or 1.0
        for b in bars_in[:6]:
            v = float(b.get("value") or 0)
            bars.append({"label": str(b.get("label") or ""), "value": v,
                         "frac": max(0.02, v / vmax), "dec": 0 if abs(v - round(v)) < 1e-9 else 1})
        lpx = max(13, int(T.TYPE_RATIO["body"] * ref * 0.88))
        tpx = max(15, int(T.TYPE_RATIO["title"] * ref * 0.55))
        bar_h = max(18, int(ref * 0.038))
        gap = max(10, int(ref * 0.024))
        label_w = max((T.text_width(ctx.book.get("semibold", lpx), b["label"]) for b in bars),
                      default=0)
        width = int(max_w)
        h = int(kpx * 1.5) + (int(tpx * 1.4) if title else 0) + int(ref * 0.02)
        h += len(bars) * (bar_h + gap)
        return Fit(kind=kind, width=min(width, max_w), height=int(min(h, max_h)),
                   pad=pad, title_px=tpx, body_px=lpx, kicker_px=kpx, kicker=kicker,
                   lines=[title] if title else [], bars=bars,
                   label=title, unit=str(c.get("unit") or ""),
                   rule=False, align="left")

    if kind == "compare":
        left = c.get("left") or {}
        right = c.get("right") or {}
        if not left and not right:
            return None
        kicker = _kicker_for(element, ctx.language)
        kpx = max(11, int(T.TYPE_RATIO["meta"] * ref * 0.95))
        tpx = max(15, int(T.TYPE_RATIO["title"] * ref * 0.58 * tier))
        lpx = max(13, int(T.TYPE_RATIO["body"] * ref * 0.9))
        width = int(max_w)
        rows = max(len(left.get("items") or []), len(right.get("items") or []))
        h = int(kpx * 1.5) + int(tpx * 2.2) + rows * T.line_pitch(lpx, "body")
        return Fit(kind=kind, width=min(width, max_w), height=int(min(h, max_h)),
                   pad=pad, title_px=tpx, body_px=lpx, kicker_px=kpx, kicker=kicker,
                   compare={"left": left, "right": right}, rule=False)

    if kind == "quote":
        text = str(c.get("text") or "").strip()
        if not text:
            return None
        attribution = str(c.get("attribution") or "").strip()
        kicker = _kicker_for(element, ctx.language)
        kpx = max(11, int(T.TYPE_RATIO["meta"] * ref * 0.95))
        apx = max(13, int(T.TYPE_RATIO["body"] * ref * 0.9))
        start = int(T.TYPE_RATIO["title"] * ref * 1.05 * tier)
        got = _fitting_font(ctx, text, "display", max_w - 2 * pad - int(start * 0.6),
                            start, 3)
        if not got:
            return None
        _, tpx, lines = got
        pitch = T.line_pitch(tpx, "title")
        width = min(max_w, int(max(T.text_width(ctx.book.get("display", tpx), l) for l in lines)
                               + int(tpx * 0.75)) + 2 * pad)
        h = int(kpx * 1.5) + len(lines) * pitch + (int(tpx * 0.7) if attribution else 0)
        return Fit(kind=kind, width=width, height=int(min(h, max_h)), pad=pad,
                   title_px=tpx, kicker_px=kpx, body_px=apx, pitch=pitch,
                   lines=lines, kicker=kicker, quote={"attribution": attribution},
                   rule=False, align="left")

    return None


# ------------------------------------------------------------------------ paint
def paint_element(element: dict, ctx: Ctx, fit: Fit, t: float, dur: float,
                  *, enter: float = 0.0, exit_at: float | None = None) -> Image.Image:
    """Paint the element at local time t (seconds since its window start)."""
    w, h = fit.width, fit.height
    if w <= 0 or h <= 0:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    pal = ctx.palette
    kicker_color = T.alpha(pal.accent, 0.98)
    ink = pal.ink
    dim = pal.ink_dim
    out = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(out)

    def local(p: float) -> float:
        return T.clamp(p)

    if fit.kind in ("keyword", "card", "quote"):
        x0 = fit.pad * SS
        y = fit.pad * SS
        # kicker -------------------------------------------------------------
        if fit.kicker:
            kp = T.ease("out_cubic", T.progress(t, 0.04, 0.26))
            _draw_kicker(d, x0, y, fit.kicker, ctx, fit.kicker_px * SS,
                         T.alpha(kicker_color, 0.95 * kp))
            y += int(fit.kicker_px * SS * 1.5) + int(fit.pitch * SS * 0.30)
        # display lines ------------------------------------------------------
        dfont = ctx.book.get("display", fit.display_px * SS or fit.title_px * SS)
        line_h = (fit.pitch or T.line_pitch(fit.display_px or fit.title_px)) * SS
        spans = [str(s) for s in (fit.emphasis or [])]
        for i, line in enumerate(fit.lines):
            p = T.ease("out_expo", T.progress(t, -0.08 + i * 0.075, 0.36))
            if p <= 0:
                continue
            dx = x0
            if fit.align == "center":
                dx = int((w * SS - T.text_width(dfont, line)) / 2)
            dy = int((1 - p) * line_h * 0.22)
            _draw_rich(d, dx, y + dy, line, dfont, T.alpha(ink, p),
                       T.alpha(pal.accent, p), spans)
            y += int(line_h)
        # body ---------------------------------------------------------------
        if fit.body_lines:
            bfont = ctx.book.get("semibold", fit.body_px * SS)
            y += int((fit.display_px or fit.title_px) * SS * 0.28)
            for ln in fit.body_lines:
                p = T.ease("out_cubic", T.progress(t, 0.22, 0.34))
                if p > 0:
                    d.text((x0, y), ln, font=bfont, fill=T.alpha(dim, 0.95 * p))
                y += int(T.line_pitch(fit.body_px, "body") * SS)
        # accent rule --------------------------------------------------------
        if fit.rule:
            p = T.ease("out_expo", T.progress(t, 0.30, 0.30))
            rw = int((fit.display_px or fit.title_px) * SS * 1.15 * p)
            ry = h * SS - fit.pad * SS - 3 * SS
            if rw > 0:
                d.rectangle([x0, ry, x0 + rw, ry + 3 * SS], fill=T.alpha(pal.accent, 0.95))
        # quote mark ---------------------------------------------------------
        if fit.kind == "quote":
            qf = ctx.book.get("latin", int((fit.title_px or 40) * SS * 1.9), "bold")
            p = T.ease("out_cubic", T.progress(t, 0.0, 0.3))
            d.text((x0 - int(4 * SS), -int((fit.title_px or 40) * SS * 0.62)),
                   "\u201C", font=qf, fill=T.alpha(pal.accent, 0.9 * p))
            if fit.quote.get("attribution"):
                af = ctx.book.get("semibold", fit.body_px * SS)
                ap = T.ease("out_cubic", T.progress(t, 0.24, 0.3))
                d.text((x0 + int(fit.title_px * SS * 0.62),
                        h * SS - fit.pad * SS - int(fit.body_px * SS * 1.6)),
                       "— " + fit.quote["attribution"], font=af, fill=T.alpha(dim, 0.9 * ap))

    elif fit.kind == "list":
        x0 = fit.pad * SS
        y = fit.pad * SS
        if fit.kicker:
            kp = T.ease("out_cubic", T.progress(t, 0.02, 0.24))
            _draw_kicker(d, x0, y, fit.kicker, ctx, fit.kicker_px * SS,
                         T.alpha(kicker_color, 0.95 * kp))
            y += int(fit.kicker_px * SS * 1.5) + int(fit.title_px * SS * 0.42)
        for i, row in enumerate(fit.rows):
            at = float(row.get("at_s") or 0.0)
            p = T.ease("out_quint", T.progress(t, at - 0.07, 0.34))
            if p <= 0:
                continue
            mpx = int(fit.title_px * SS * 0.62)
            mx = x0 + int((1 - p) * -6 * SS)
            my = y + int(fit.title_px * SS * 0.10)
            d.ellipse([mx, my, mx + mpx, my + mpx], fill=T.alpha(pal.accent, 0.95 * p))
            mf = ctx.book.get("latin", int(mpx * 0.62), "bold")
            mw = T.text_width(mf, row["marker"])
            d.text((mx + (mpx - mw) / 2, my + mpx * 0.10), row["marker"], font=mf,
                   fill=T.alpha((12, 14, 20, 255), p))
            rf = ctx.book.get("semibold", fit.title_px * SS)
            ty = y
            for ln in row["lines"]:
                lx = x0 + int(mpx * 1.55)
                d.text((lx + int((1 - p) * 10 * SS), ty), ln, font=rf,
                       fill=T.alpha(ink, 0.97 * p))
                ty += int(fit.pitch * SS)
            y = ty + int(fit.title_px * SS * 0.42)

    elif fit.kind == "stat":
        x0 = fit.pad * SS
        y = fit.pad * SS
        if fit.kicker:
            kp = T.ease("out_cubic", T.progress(t, 0.02, 0.24))
            _draw_kicker(d, x0, y, fit.kicker, ctx, fit.kicker_px * SS,
                         T.alpha(kicker_color, 0.95 * kp))
            y += int(fit.kicker_px * SS * 1.5) + int(fit.display_px * SS * 0.26)
        if fit.num_str and fit.display_px:
            npx = fit.display_px * SS
            nf = ctx.book.get("latin", npx, "bold")
            p = T.ease("out_expo", T.progress(t, -0.04, 0.60))
            shown = fit.num_str
            if p < 0.999 and fit.num_str and any(ch.isdigit() for ch in fit.num_str):
                target = float(fit.num_str.replace(",", ""))
                dec = len(fit.num_str.split(".")[1]) if "." in fit.num_str else 0
                cur = target * p
                shown = _fmt_number(cur, dec)
                # keep the final integer width stable
                shown = shown.rjust(len(fit.num_str), " ")
            elif p >= 0.999:
                shown = fit.num_str
            d.text((x0, y), shown, font=nf, fill=T.alpha(ink, 1.0))
            nw = T.text_width(nf, fit.num_str)
            if fit.unit:
                uf = ctx.book.get("latin", int(npx * 0.34), "bold")
                d.text((x0 + nw + int(npx * 0.06), y + int(npx * 0.58)), fit.unit,
                       font=uf, fill=T.alpha(pal.accent, 0.95))
            y += int(npx * 1.06)
        elif fit.lines:
            # non-numeric stat
            nf = ctx.book.get("display", fit.display_px * SS)
            p = T.ease("out_expo", T.progress(t, -0.04, 0.46))
            for ln in fit.lines:
                d.text((x0, y), ln, font=nf, fill=T.alpha(ink, p))
                y += int(fit.pitch * SS)
        if fit.label:
            lf = ctx.book.get("semibold", fit.body_px * SS)
            lp = T.ease("out_cubic", T.progress(t, 0.30, 0.34))
            for ln in fit.label.split("\n"):
                d.text((x0, y), ln, font=lf, fill=T.alpha(dim, 0.95 * lp))
                y += int(T.line_pitch(fit.body_px, "body") * SS)
        if fit.rule:
            p = T.ease("out_expo", T.progress(t, 0.34, 0.3))
            ry = h * SS - fit.pad * SS - 3 * SS
            rw = int((fit.display_px or 40) * SS * 0.85 * p)
            if rw > 0:
                d.rectangle([x0, ry, x0 + rw, ry + 3 * SS], fill=T.alpha(pal.accent, 0.95))

    elif fit.kind == "barchart":
        x0 = fit.pad * SS
        y = fit.pad * SS
        if fit.kicker:
            kp = T.ease("out_cubic", T.progress(t, 0.02, 0.24))
            _draw_kicker(d, x0, y, fit.kicker, ctx, fit.kicker_px * SS,
                         T.alpha(kicker_color, 0.95 * kp))
            y += int(fit.kicker_px * SS * 1.5) + int(fit.title_px * SS * 0.3)
        if fit.lines:
            tf = ctx.book.get("display", fit.title_px * SS)
            p = T.ease("out_cubic", T.progress(t, 0.05, 0.3))
            d.text((x0, y), fit.lines[0], font=tf, fill=T.alpha(ink, 0.96 * p))
            y += int(fit.title_px * SS * 1.45)
        lpx = fit.body_px * SS
        lf = ctx.book.get("semibold", lpx)
        vf = ctx.book.get("latin", int(lpx * 1.05), "bold")
        label_w = max((T.text_width(lf, b["label"]) for b in fit.bars), default=0)
        bar_x = x0 + int(label_w) + int(lpx * 0.9)
        bar_w = (w * SS - fit.pad * SS) - bar_x - int(lpx * 3.4)
        bar_h = max(14, int(ctx.ref_h * 0.038 * SS))
        for i, b in enumerate(fit.bars):
            p = T.ease("out_expo", T.progress(t, 0.14 + i * 0.09, 0.6))
            by = y + i * int((bar_h + int(ctx.ref_h * 0.024 * SS)))
            d.text((x0, by - int(lpx * 0.12)), b["label"], font=lf,
                   fill=T.alpha(dim, 0.95 * min(1, p * 1.6)))
            d.rounded_rectangle([bar_x, by, bar_x + bar_w, by + bar_h],
                                radius=bar_h // 2, fill=T.alpha((255, 255, 255, 22)))
            rw = int(bar_w * b["frac"] * p)
            if rw > 2:
                d.rounded_rectangle([bar_x, by, bar_x + rw, by + bar_h],
                                    radius=bar_h // 2, fill=T.alpha(pal.accent, 0.92))
            vtxt = _fmt_number(b["value"] * p, b["dec"])
            vt = _fmt_number(b["value"], b["dec"])
            d.text((bar_x + bar_w + int(lpx * 0.55), by - int(lpx * 0.12)),
                   vt if p > 0.98 else vtxt, font=vf, fill=T.alpha(ink, 0.95))
        _ = (x0, bar_x, bar_w)

    elif fit.kind == "compare":
        left, right = fit.compare.get("left") or {}, fit.compare.get("right") or {}
        x0 = fit.pad * SS
        y = fit.pad * SS
        if fit.kicker:
            kp = T.ease("out_cubic", T.progress(t, 0.02, 0.24))
            _draw_kicker(d, x0, y, fit.kicker, ctx, fit.kicker_px * SS,
                         T.alpha(kicker_color, 0.95 * kp))
            y += int(fit.kicker_px * SS * 1.5) + int(fit.title_px * SS * 0.55)
        col_w = (w * SS - 2 * fit.pad * SS - int(fit.title_px * SS * 0.8)) // 2
        hf = ctx.book.get("display", fit.title_px * SS)
        lf = ctx.book.get("semibold", fit.body_px * SS)
        for ci, side in enumerate((left, right)):
            p = T.ease("out_quint", T.progress(t, 0.08 + ci * 0.12, 0.42))
            if p <= 0:
                continue
            cx = x0 + ci * (col_w + int(fit.title_px * SS * 0.8))
            title = str(side.get("title") or "")
            accent_col = pal.accent if ci == 1 else T.mix(pal.accent, (255, 255, 255, 255), 0.55)
            d.text((cx, y + int((1 - p) * 10 * SS)), title, font=hf,
                   fill=T.alpha(accent_col, 0.98))
            ty = y + int(fit.title_px * SS * 1.7)
            for item in (side.get("items") or [])[:4]:
                ip = T.ease("out_cubic", T.progress(t, 0.18 + ci * 0.12 + 0.06, 0.34))
                d.text((cx, ty), "· " + str(item), font=lf, fill=T.alpha(dim, 0.95 * ip))
                ty += int(T.line_pitch(fit.body_px, "body") * SS)
        # divider
        dp = T.ease("out_cubic", T.progress(t, 0.1, 0.4))
        mx = x0 + col_w + int(fit.title_px * SS * 0.4)
        d.rectangle([mx, y, mx + 1 * SS, y + int((h * SS - 2 * fit.pad * SS) * dp)],
                    fill=T.alpha((255, 255, 255, 46)))

    return out.resize((w, h), Image.LANCZOS)
