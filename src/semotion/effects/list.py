"""Step list: several anchors revealed in spoken order, then one calm exit.

Each row owns its own anchor onset (``content.rows[].at_s`` relative to the
element window), so the list is driven by the words rather than by a guessed
even cadence. Rows fade/rise by 8 px inside the panel; the whole element fades
out over the last 0.20 s of its window (applied by the rasteriser).
"""
from __future__ import annotations

from typing import Any, Sequence

from PIL import Image, ImageDraw

from . import base


def _rows(element: dict) -> list[dict]:
    content = element.get("content") or {}
    rows = content.get("rows")
    if rows:
        out: list[dict] = []
        for i, r in enumerate(rows):
            if isinstance(r, dict):
                out.append({"marker": str(r.get("marker") or i + 1),
                            "text": str(r.get("text") or ""),
                            "at_s": float(r.get("at_s") or 0.0),
                            "text_at_s": float(r.get("text_at_s", r.get("at_s") or 0.0))})
            else:
                out.append({"marker": str(i + 1), "text": str(r), "at_s": 0.0,
                            "text_at_s": 0.0})
        return out
    items = [str(x) for x in (content.get("items") or [])]
    if not items:
        items = [str(content.get("text") or "")]
    onsets = content.get("item_onsets_s") or []
    return [{"marker": str(i + 1), "text": it,
             "at_s": float(onsets[i]) if i < len(onsets) else 0.45 * i,
             "text_at_s": float(onsets[i]) if i < len(onsets) else 0.45 * i}
            for i, it in enumerate(items)]


def measure(element: dict, canvas: dict, *, max_text_width: float | None = None,
            font_dir: str | None = None) -> dict[str, Any]:
    st = element.get("style") or {}
    font_px = max(12, int(st.get("font_size_px") or base.scaled(72, canvas)))
    pad = max(10, int(st.get("padding_px") or base.scaled(36, canvas)))
    radius = max(0, int(st.get("radius_px") or base.scaled(28, canvas)))
    bg = base.parse_color(st.get("bg"), (10, 13, 22, 210))
    fg = base.parse_color(st.get("color"), (245, 247, 250, 255))
    accent = base.parse_color(st.get("accent"), (242, 178, 51, 255))
    font_path = base.resolve_font_path(font_dir, st.get("font_file") or base.FONT_FILE)
    font = base.load_font(font_path, font_px)

    marker_d = int(round(font_px * 0.70))
    marker_gap = int(round(font_px * 0.38))
    row_gap = int(round(font_px * 0.42))
    line_factor = float(st.get("line_height") or 1.35)

    max_w = float(max_text_width) if max_text_width else float(base.scaled(880, canvas))
    # a step list is a prominent element: never a narrow column in the middle
    min_w = int(st.get("min_width_px") or round(0.60 * float(canvas.get("width") or 1080)))
    max_w = max(max_w, float(min_w))
    text_max = max(60.0, max_w - 2 * pad - marker_d - marker_gap)

    rows = _rows(element)
    # prefer single-line steps: fit the type inside the 56-68 px band before wrapping
    min_font = max(10, int(round(56 * base.canvas_scale(canvas))))
    chosen = font_px
    for size in range(font_px, min_font - 1, -2):
        f = base.load_font(font_path, size)
        md = int(round(size * 0.70))
        mg = int(round(size * 0.38))
        tmax = max(60.0, max_w - 2 * pad - md - mg)
        if all(len(base.wrap_text(r["text"], f, tmax, base.MAX_LINES)) <= 1 for r in rows):
            chosen = size
            break
    else:
        chosen = min_font
    if chosen != font_px:
        font_px = chosen
        font = base.load_font(font_path, font_px)
        marker_d = int(round(font_px * 0.70))
        marker_gap = int(round(font_px * 0.38))
        row_gap = int(round(font_px * 0.42))
        text_max = max(60.0, max_w - 2 * pad - marker_d - marker_gap)
    marker_font = base.load_font(font_path, max(10, int(round(font_px * 0.42))))
    asc, desc = base.text_block_metrics(font)
    step = int(round(font_px * line_factor))
    widest = 0.0
    total_h = 0
    laid: list[dict] = []
    for i, r in enumerate(rows):
        lines = base.wrap_text(r["text"], font, text_max, base.MAX_LINES)
        line_w = max([base.text_width(font, ln) for ln in lines] or [0.0])
        n_lines = max(1, len(lines))
        row_h = max(marker_d, (n_lines - 1) * step + asc + desc)
        laid.append({**r, "lines": lines, "row_h": row_h, "line_w": line_w})
        widest = max(widest, line_w)
        total_h += row_h + (row_gap if i < len(rows) - 1 else 0)

    text_w = int(marker_d + marker_gap + widest)
    width = max(text_w + 2 * pad, int(min_w))
    content_w = width - 2 * pad
    height = total_h + 2 * pad
    return {"kind": "list", "rows": laid, "font_px": font_px, "font": font,
            "marker_font": marker_font, "marker_d": marker_d, "marker_gap": marker_gap,
            "row_gap": row_gap, "line_step": step, "ascent": asc, "descent": desc,
            "pad": pad, "radius": radius, "bg": bg, "fg": fg, "accent": accent,
            "width": width, "height": height, "content_w": content_w,
            "content_h": total_h}


def _ordinal_number(marker: str) -> str:
    m = str(marker).strip()
    return m


def draw(element: dict, canvas: dict, spec: dict, t_rel: float, window_s: float,
         *, font_dir: str | None = None) -> Image.Image:
    w, h = int(spec["width"]), int(spec["height"])
    tile = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    base.draw_panel(tile, (w, h), bg=spec["bg"], radius=spec["radius"])
    if not spec["rows"]:
        return tile
    pad = spec["pad"]
    step = spec["line_step"]
    marker_d = spec["marker_d"]
    gap = spec["marker_gap"]
    y = pad
    for row in spec["rows"]:
        row_h = int(row["row_h"])
        at = float(row.get("at_s") or 0.0)
        t_local = t_rel - at
        if t_local < 0:
            y += row_h + spec["row_gap"]
            continue
        p = base.out_cubic(min(1.0, t_local / 0.20))
        alpha = 0.55 + 0.45 * p          # visible on the reveal frame itself
        rise = int(round((1.0 - p) * 8))
        layer = Image.new("RGBA", (w, row_h + 8), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        # marker badge
        cy = 4 + row_h // 2
        mx0 = pad
        d.ellipse([mx0, cy - marker_d // 2, mx0 + marker_d, cy + marker_d // 2],
                  fill=tuple(spec["accent"]))
        label = _ordinal_number(row.get("marker") or "")
        lb_font = spec["marker_font"]
        d.text((mx0 + marker_d / 2.0, cy + 1), label, font=lb_font,
               fill=(18, 16, 12, 255), anchor="mm")
        # step text: its own anchor can arrive a little later than the marker
        t_text = t_rel - float(row.get("text_at_s") or at)
        if t_text < 0:
            text_alpha = 0.0
        else:
            text_alpha = 0.45 + 0.55 * base.out_cubic(min(1.0, t_text / 0.20))
        text_x = pad + marker_d + gap
        lines = row["lines"]
        n = max(1, len(lines))
        block_h = (n - 1) * step + spec["ascent"] + spec["descent"]
        first_baseline = 4 + (row_h - block_h) / 2.0 + spec["ascent"]
        for i, ln in enumerate(lines):
            d.text((text_x, first_baseline + i * step), ln, font=spec["font"],
                   fill=tuple(spec["fg"]), anchor="ls")
        if text_alpha < 1.0:
            a = layer.getchannel("A").point(lambda v: int(v * text_alpha))
            # keep the badge fully visible: redraw it after scaling text alpha
            layer.putalpha(a)
            d2 = ImageDraw.Draw(layer)
            d2.ellipse([mx0, cy - marker_d // 2, mx0 + marker_d, cy + marker_d // 2],
                       fill=tuple(spec["accent"]))
            d2.text((mx0 + marker_d / 2.0, cy + 1), label, font=lb_font,
                    fill=(18, 16, 12, 255), anchor="mm")
        if alpha < 1.0:
            a2 = layer.getchannel("A").point(lambda v: int(v * alpha))
            layer.putalpha(a2)
        tile.alpha_composite(layer, (0, max(0, y + rise)))
        y += row_h + spec["row_gap"]
    return tile
