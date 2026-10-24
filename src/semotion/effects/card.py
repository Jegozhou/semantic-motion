"""Information card: a calm, generous panel for one explanatory statement.

Dynamic whitespace comes from generous padding (28-40 px at 1080 wide) and from
the entrance scaling *down* from 0.96, never up past the box, so the card can
never spill over the face-safe area it was laid out in.
"""
from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw

from . import base


def measure(element: dict, canvas: dict, *, max_text_width: float | None = None,
            font_dir: str | None = None) -> dict[str, Any]:
    st = element.get("style") or {}
    content = element.get("content") or {}
    font_px = max(12, int(st.get("font_size_px") or base.scaled(78, canvas)))
    pad = max(10, int(st.get("padding_px") or base.scaled(40, canvas)))
    radius = max(0, int(st.get("radius_px") or base.scaled(30, canvas)))
    bg = base.parse_color(st.get("bg"), (16, 21, 33, 200))
    fg = base.parse_color(st.get("color"), (245, 247, 250, 255))
    accent = base.parse_color(st.get("accent"), (242, 178, 51, 255))
    text = str(content.get("text") or "").strip()
    font_path = base.resolve_font_path(font_dir, st.get("font_file") or base.FONT_FILE)
    font = base.load_font(font_path, font_px)
    accent_bar_w = base.scaled(6, canvas) if st.get("accent_bar", True) else 0
    accent_gap = base.scaled(22, canvas) if accent_bar_w else 0
    max_w = float(max_text_width) if max_text_width else float(base.scaled(880, canvas))
    max_w = max(80.0, max_w) - accent_bar_w - accent_gap
    lines = base.wrap_text(text, font, max_w, base.MAX_LINES)
    asc, desc = base.text_block_metrics(font)
    step = int(round(font_px * base.LINE_HEIGHT_FACTOR))
    text_h = (len(lines) - 1) * step + asc + desc if lines else 0
    text_w = int(max([base.text_width(font, ln) for ln in lines] or [0]))
    width = text_w + 2 * pad + accent_bar_w + accent_gap
    height = text_h + 2 * pad
    return {"kind": "card", "lines": lines, "font_px": font_px, "font": font,
            "pad": pad, "radius": radius, "bg": bg, "fg": fg, "accent": accent,
            "line_step": step, "ascent": asc, "descent": desc,
            "text_w": text_w, "text_h": text_h,
            "accent_bar_w": accent_bar_w, "accent_gap": accent_gap,
            "width": width, "height": height,
            "content_w": text_w + accent_bar_w + accent_gap, "content_h": text_h}


def draw(element: dict, canvas: dict, spec: dict, t_rel: float, window_s: float,
         *, font_dir: str | None = None) -> Image.Image:
    w, h = int(spec["width"]), int(spec["height"])
    tile = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    base.draw_panel(tile, (w, h), bg=spec["bg"], radius=spec["radius"])
    if not spec["lines"]:
        return tile
    d = ImageDraw.Draw(tile)
    pad = spec["pad"]
    bar_w, gap = spec["accent_bar_w"], spec["accent_gap"]
    text_x = pad + bar_w + gap
    step = spec["line_step"]
    asc, desc = spec["ascent"], spec["descent"]
    block_h = (len(spec["lines"]) - 1) * step + asc + desc
    top = pad + (h - 2 * pad - block_h) / 2.0
    first_baseline = top + asc
    for i, ln in enumerate(spec["lines"]):
        d.text((text_x, first_baseline + i * step), ln, font=spec["font"],
               fill=tuple(spec["fg"]), anchor="ls")
    if bar_w:
        # accent rule grows with the entrance (0 -> full over 0.30 s)
        p = base.out_cubic(min(1.0, max(0.0, t_rel / 0.30)))
        bar_h = max(1, int(round(block_h * p)))
        y0 = int(round(top + (block_h - bar_h) / 2.0))
        d.rounded_rectangle([pad, y0, pad + bar_w, y0 + bar_h],
                            radius=bar_w // 2, fill=tuple(spec["accent"]))
    return tile
