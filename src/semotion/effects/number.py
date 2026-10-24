"""Number emphasis effect.

A spoken quantity is kept semantically exact:

* the numeric token is located in the anchor text (Chinese or Arabic numerals,
  including 成 / 百分之 / 个 / 倍 ... unit forms);
* the numeral is progressively counted up from 0 and **terminates on the exact
  parsed value** — the unit is never rewritten and no digit is invented;
* the phrase around the numeral is left untouched, so the rendered sentence says
  the same thing as the transcript.
"""
from __future__ import annotations

import re
from typing import Any

from PIL import Image, ImageDraw

from . import base

CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNITS = {"十": 10, "百": 100, "千": 1000}
CN_BIG = {"万": 10000, "亿": 100000000}

# longest first so "分钟" wins over "分"
UNIT_WORDS = ("分钟", "小时", "百分之", "成", "个", "次", "条", "步", "种", "倍",
              "年", "月", "天", "秒", "分", "位", "层", "块", "页", "张", "份")

CN_NUM_CLASS = "零〇一二两三四五六七八九十百千万亿"
_RE_CN_PERCENT = re.compile(r"百分之([" + CN_NUM_CLASS + r"]+)")
_RE_CN_UNIT = re.compile(r"([" + CN_NUM_CLASS + r"]+)(" + "|".join(UNIT_WORDS) + ")")
_RE_CN_BARE = re.compile(r"([" + CN_NUM_CLASS + r"]+)")
_RE_AR_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_RE_AR_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*(" + "|".join(UNIT_WORDS) + ")")
_RE_AR_BARE = re.compile(r"(\d+(?:\.\d+)?)")


def cn_to_number(s: str) -> float | None:
    """Parse a Chinese numeral (accepts 十/百/千/万/亿 compounds)."""
    if not s:
        return None
    total = 0
    section = 0
    number = 0
    big = 0
    for ch in s:
        if ch in CN_DIGITS:
            number = CN_DIGITS[ch]
        elif ch in CN_UNITS:
            unit = CN_UNITS[ch]
            if number == 0:
                number = 1
            section += number * unit
            number = 0
        elif ch in CN_BIG:
            section += number
            number = 0
            sec = section or 1
            if ch == "万":
                big += sec * CN_BIG[ch]
            else:
                big = (big + sec) * CN_BIG[ch]
            section = 0
        else:
            return None
    if big:
        return float(big + section + number)
    return float(section + number)


def _fmt(value: float, decimals: int) -> str:
    if decimals <= 0:
        return str(int(round(value)))
    return f"{value:.{decimals}f}"


_T2S = None


def _fold_traditional(text: str) -> str:
    """1:1 Traditional→Simplified fold that preserves character offsets."""
    global _T2S
    if _T2S is None:
        try:
            import opencc
            _T2S = opencc.OpenCC("t2s")
        except Exception:  # pragma: no cover - optional dependency
            _T2S = False
    if not _T2S or not text:
        return text
    folded = "".join(_T2S.convert(ch) for ch in text)
    return folded if len(folded) == len(text) else text


# Idiomatic quantities that are not written as numeral+unit and must not be
# degraded into a single digit ("一半" is 50%, not "1半").
_IDIOMS = (("一半", 50.0, "%"), ("一半儿", 50.0, "%"))


def parse_numeric(text: str) -> dict[str, Any] | None:
    """Find the first numeric token and describe how to animate it.

    Returns keys: start, end, value, unit, kind, final, decimals, label.
    """
    text = text or ""
    cands: list[dict[str, Any]] = []
    scan = _fold_traditional(text)

    for word, val, unit in _IDIOMS:
        idx = scan.find(word)
        if idx >= 0:
            cands.append({"start": idx, "end": idx + len(word), "value": val, "unit": unit,
                          "kind": "percent" if unit == "%" else "unit",
                          "final": _fmt(val, 0) + unit, "decimals": 0})

    for m in _RE_CN_PERCENT.finditer(scan):
        v = cn_to_number(m.group(1))
        if v is None:
            continue
        cands.append({"start": m.start(), "end": m.end(), "value": v, "unit": "%",
                      "kind": "percent", "final": _fmt(v, 0) + "%", "decimals": 0})
    for m in _RE_AR_PERCENT.finditer(scan):
        v = float(m.group(1))
        dec = len(m.group(1).split(".")[1]) if "." in m.group(1) else 0
        cands.append({"start": m.start(), "end": m.end(), "value": v, "unit": "%",
                      "kind": "percent", "final": _fmt(v, dec) + "%", "decimals": dec})
    for m in _RE_CN_UNIT.finditer(scan):
        v = cn_to_number(m.group(1))
        if v is None:
            continue
        unit = m.group(2)
        if unit == "成":
            pct = v * 10.0
            cands.append({"start": m.start(), "end": m.end(), "value": pct, "unit": "%",
                          "kind": "percent", "final": _fmt(pct, 0) + "%", "decimals": 0})
        else:
            cands.append({"start": m.start(), "end": m.end(), "value": v, "unit": unit,
                          "kind": "unit", "final": _fmt(v, 0) + unit, "decimals": 0})
    for m in _RE_AR_UNIT.finditer(scan):
        v = float(m.group(1))
        dec = len(m.group(1).split(".")[1]) if "." in m.group(1) else 0
        unit = m.group(2)
        if unit == "成":
            v, unit = v * 10.0, "%"
        cands.append({"start": m.start(), "end": m.end(), "value": v, "unit": unit,
                      "kind": "unit", "final": _fmt(v, dec) + unit, "decimals": dec})
    for m in _RE_CN_BARE.finditer(scan):
        v = cn_to_number(m.group(1))
        if v is None:
            continue
        cands.append({"start": m.start(), "end": m.end(), "value": v, "unit": "",
                      "kind": "plain", "final": _fmt(v, 0), "decimals": 0})
    for m in _RE_AR_BARE.finditer(scan):
        v = float(m.group(1))
        dec = len(m.group(1).split(".")[1]) if "." in m.group(1) else 0
        cands.append({"start": m.start(), "end": m.end(), "value": v, "unit": "",
                      "kind": "plain", "final": _fmt(v, dec), "decimals": dec})

    if not cands:
        return None
    # earliest occurrence wins; longer token breaks ties
    cands.sort(key=lambda c: (c["start"], -(c["end"] - c["start"])))
    best = cands[0]
    best["label"] = scan[best["start"]:best["end"]]
    best["value"] = float(best["value"])
    return best


def _wrap_ranges(text: str, font, max_width: float, max_lines: int) -> list[tuple[str, int, int]]:
    """Wrap keeping each line's [start, end) range in the original string."""
    out: list[tuple[str, int, int]] = []
    start = 0
    cur = ""
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        trial = cur + ch
        if cur and base.text_width(font, trial) > max_width and len(out) < max_lines - 1:
            out.append((cur, start, i))
            start = i
            cur = ch
        else:
            cur = trial
        i += 1
    if cur:
        out.append((cur, start, n))
    return out or [(text, 0, n)]


def measure(element: dict, canvas: dict, *, max_text_width: float | None = None,
            font_dir: str | None = None) -> dict[str, Any]:
    st = element.get("style") or {}
    content = element.get("content") or {}
    font_px = int(st.get("font_size_px") or base.scaled(84, canvas))
    pad = int(st.get("padding_px") or base.scaled(36, canvas))
    radius = int(st.get("radius_px") or base.scaled(28, canvas))
    bg = base.parse_color(st.get("bg"), (12, 16, 28, 184))
    fg = base.parse_color(st.get("color"), (245, 247, 250, 255))
    accent = base.parse_color(st.get("accent"), (242, 178, 51, 255))
    align = st.get("align") or "center"
    font_path = base.resolve_font_path(font_dir, st.get("font_file") or base.FONT_FILE)
    font = base.load_font(font_path, font_px)
    num_px = max(font_px + 10, int(round(font_px * 1.55)))
    num_font = base.load_font(font_path, num_px)

    text = str(content.get("text") or "").strip()
    numeric = content.get("numeric") or parse_numeric(text)
    max_w = float(max_text_width) if max_text_width else float(base.scaled(880, canvas))
    max_w = max(80.0, max_w)

    if not numeric:
        # no quantity found: render the phrase with plain emphasis (still exact)
        lines = base.wrap_text(text, font, max_w, base.MAX_LINES)
        asc, desc = base.text_block_metrics(font)
        step = int(round(font_px * base.LINE_HEIGHT_FACTOR))
        text_h = (len(lines) - 1) * step + asc + desc if lines else 0
        widths = [base.text_width(font, ln) for ln in lines] or [0.0]
        content_w = int(max(widths))
        width = content_w + 2 * pad
        height = text_h + 2 * pad
        return {"kind": "number", "numeric": None, "font_px": font_px, "num_px": font_px,
                "pad": pad, "radius": radius, "bg": bg, "fg": fg, "accent": accent,
                "align": align, "lines": lines, "line_ranges": [],
                "font": font, "num_font": font,
                "line_step": step, "ascent": asc, "descent": desc,
                "width": width, "height": height, "content_w": content_w,
                "content_h": text_h,
                "accent_line": bool(st.get("accent_line", False))}

    final = numeric["final"]
    display = text[:numeric["start"]] + final + text[numeric["end"]:]
    ranges = _wrap_ranges(display, font, max_w, base.MAX_LINES)
    prefix_len = numeric["start"]
    suffix_start = numeric["start"] + len(final)

    asc, desc = base.text_block_metrics(font)
    nasc, ndesc = base.text_block_metrics(num_font)
    ascent = max(asc, nasc)
    descent = max(desc, ndesc)
    step = int(round(max(font_px * base.LINE_HEIGHT_FACTOR, num_px * 1.02)))
    text_h = (len(ranges) - 1) * step + ascent + descent if ranges else 0

    # panel width from the widest possible rendering (final value and a zero)
    widest = max(1, len(final))
    probe_display = text[:numeric["start"]] + ("0" * widest) + text[numeric["end"]:]
    widths = []
    for (_ln, a, b) in ranges:
        widths.append(_line_width(display, a, b, prefix_len, suffix_start, final,
                                  font, num_font, base.text_width))
    for (_ln, a, b) in _wrap_ranges(probe_display, font, max_w, base.MAX_LINES):
        widths.append(_line_width(probe_display, a, b, prefix_len, suffix_start,
                                  "0" * widest, font, num_font, base.text_width))
    content_w = int(max(widths or [0]))
    width = content_w + 2 * pad
    height = text_h + 2 * pad
    return {"kind": "number", "numeric": numeric, "font_px": font_px, "num_px": num_px,
            "pad": pad, "radius": radius, "bg": bg, "fg": fg, "accent": accent,
            "align": align, "display": display, "line_ranges": ranges,
            "prefix_len": prefix_len, "suffix_start": suffix_start,
            "font": font, "num_font": num_font,
            "line_step": step, "ascent": ascent, "descent": descent,
            "width": width, "height": height, "content_w": content_w,
            "content_h": text_h,
            "accent_line": bool(st.get("accent_line", True))}


def _line_width(display: str, a: int, b: int, prefix_len: int, suffix_start: int,
                numstr: str, font, num_font, tw) -> float:
    before, has, after = _split_line(display, a, b, prefix_len, suffix_start)
    w = tw(font, before) + (tw(num_font, numstr) if has else 0.0) + tw(font, after)
    return w


def _split_line(display: str, a: int, b: int, prefix_len: int, suffix_start: int
                ) -> tuple[str, bool, str]:
    """Split a wrapped line into (text-before-numeral, numeral-present, text-after)."""
    has = max(a, prefix_len) < min(b, suffix_start)
    before = display[a:min(b, prefix_len)] if a < prefix_len else ""
    after = display[max(a, suffix_start):b] if b > suffix_start else ""
    return before, has, after


def _numstr(numeric: dict, value: float) -> str:
    dec = int(numeric.get("decimals", 0))
    return _fmt(value, dec) + str(numeric.get("unit") or "")


def draw(element: dict, canvas: dict, spec: dict, t_rel: float, window_s: float,
         *, font_dir: str | None = None) -> Image.Image:
    w, h = int(spec["width"]), int(spec["height"])
    tile = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    base.draw_panel(tile, (w, h), bg=spec["bg"], radius=spec["radius"])
    if spec.get("accent_line"):
        base.draw_round_rect_outline(tile, (w, h), color=(*spec["accent"][:3], 90),
                                     radius=spec["radius"], width=max(1, base.scaled(2, canvas)))
    draw_obj = ImageDraw.Draw(tile)
    pad = spec["pad"]
    numeric = spec.get("numeric")
    lines = spec.get("lines") or []

    if not numeric:
        cy = h / 2.0
        step = spec["line_step"]
        first_baseline = cy - ((len(lines) - 1) * step) / 2.0 + (spec["ascent"] - spec["descent"]) / 2.0
        for i, ln in enumerate(lines):
            base.draw_line(draw_obj, [(ln, spec["font"], spec["fg"])],
                           center_x=w / 2.0, baseline_y=first_baseline + i * step)
        return tile

    # count-up: 0 -> exact target, then hold the exact target
    count_s = min(0.6, max(0.25, window_s * 0.35))
    p = base.out_cubic(min(1.0, max(0.0, t_rel / count_s))) if count_s > 0 else 1.0
    value = numeric["value"] * p
    if t_rel >= count_s:
        value = numeric["value"]
    if abs(value - numeric["value"]) < 1e-9:
        value = numeric["value"]          # terminate on the exact parsed value
    numstr = _numstr(numeric, value)

    prefix_len, suffix_start = spec["prefix_len"], spec["suffix_start"]
    display = spec["display"]
    step = spec["line_step"]
    block_h = (len(spec["line_ranges"]) - 1) * step + spec["ascent"] + spec["descent"]
    first_baseline = pad + (h - 2 * pad - block_h) / 2.0 + spec["ascent"]
    for i, (_ln, a, b) in enumerate(spec["line_ranges"]):
        before, has, after = _split_line(display, a, b, prefix_len, suffix_start)
        segs: list[tuple[str, Any, Any]] = []
        if before:
            segs.append((before, spec["font"], spec["fg"]))
        if has:
            segs.append((numstr, spec["num_font"], spec["accent"]))
        if after:
            segs.append((after, spec["font"], spec["fg"]))
        base.draw_line(draw_obj, segs, center_x=w / 2.0,
                       baseline_y=first_baseline + i * step)
    return tile
