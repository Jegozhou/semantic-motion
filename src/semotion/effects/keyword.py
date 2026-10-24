"""Keyword emphasis: a short phrase anchored to the spoken words.

Entrance is a restrained pop (0.86 -> 1.0, out-cubic) plus fade, the optional
accent rule grows under the phrase, exit is a fade. Everything is painted inside
the IR box; the animation only ever shrinks within it.
"""
from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw

from . import base


def measure(element: dict, canvas: dict, *, max_text_width: float | None = None,
            font_dir: str | None = None) -> dict[str, Any]:
    st = element.get("style") or {}
    content = element.get("content") or {}
    font_px = max(12, int(st.get("font_size_px") or base.scaled(92, canvas)))
    pad = max(8, int(st.get("padding_px") or base.scaled(36, canvas)))
    radius = max(0, int(st.get("radius_px") or base.scaled(28, canvas)))
    bg = base.parse_color(st.get("bg"), (12, 16, 28, 184))
    fg = base.parse_color(st.get("color"), (245, 247, 250, 255))
    accent = base.parse_color(st.get("accent"), (242, 178, 51, 255))
    text = str(content.get("text") or "").strip()
    font_path = base.resolve_font_path(font_dir, st.get("font_file") or base.FONT_FILE)
    font = base.load_font(font_path, font_px)
    max_w = float(max_text_width) if max_text_width else float(base.scaled(880, canvas))
    max_w = max(80.0, max_w)
    lines = base.wrap_text(text, font, max_w, base.MAX_LINES)
    asc, desc = base.text_block_metrics(font)
    step = int(round(font_px * base.LINE_HEIGHT_FACTOR))
    text_h = (len(lines) - 1) * step + asc + desc if lines else 0
    text_w = int(max([base.text_width(font, ln) for ln in lines] or [0]))
    accent_line = bool(st.get("accent_line", True))
    rule_h = base.scaled(6, canvas)
    rule_gap = base.scaled(14, canvas) if accent_line else 0
    if accent_line and text:
        text_h += rule_gap + rule_h
    width = text_w + 2 * pad
    height = text_h + 2 * pad
    return {"kind": "keyword", "lines": lines, "font_px": font_px, "font": font,
            "pad": pad, "radius": radius, "bg": bg, "fg": fg, "accent": accent,
            "line_step": step, "ascent": asc, "descent": desc,
            "text_w": text_w, "text_h": text_h,
            "accent_line": accent_line, "rule_h": rule_h, "rule_gap": rule_gap,
            "width": width, "height": height,
            "content_w": text_w, "content_h": text_h}


def draw(element: dict, canvas: dict, spec: dict, t_rel: float, window_s: float,
         *, font_dir: str | None = None) -> Image.Image:
    w, h = int(spec["width"]), int(spec["height"])
    tile = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    base.draw_panel(tile, (w, h), bg=spec["bg"], radius=spec["radius"])
    if not spec["lines"]:
        return tile
    d = ImageDraw.Draw(tile)
    pad = spec["pad"]
    step = spec["line_step"]
    block_h = (len(spec["lines"]) - 1) * step + spec["ascent"] + spec["descent"]
    first_baseline = pad + (h - 2 * pad - block_h) / 2.0 + spec["ascent"]
    cx = w / 2.0
    for i, ln in enumerate(spec["lines"]):
        base.draw_line(d, [(ln, spec["font"], spec["fg"])], center_x=cx,
                       baseline_y=first_baseline + i * step)
    if spec["accent_line"]:
        # the rule grows in during the first 0.30 s, then holds
        p = base.out_cubic(min(1.0, max(0.0, t_rel / 0.30)))
        rule_w = int(round(spec["text_w"] * p))
        if rule_w > 2:
            y0 = h - pad - spec["rule_h"]
            x0 = int(round(cx - rule_w / 2.0))
            d.rounded_rectangle([x0, y0, x0 + rule_w, y0 + spec["rule_h"]],
                                radius=spec["rule_h"] // 2, fill=tuple(spec["accent"]))
    return tile
