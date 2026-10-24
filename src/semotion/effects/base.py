"""Shared primitives for declarative motion effects.

Everything here is deterministic: fonts come from fixed paths, no randomness, no
wall-clock input, no evaluation of element text. Text is only ever passed to
Pillow as a string, so injected markup cannot execute (spec §3.6).

A *spec* is the resolved geometry of one element: panel size, font size, wrapped
lines and colours. ``design.layout`` and ``raster.render_element_png`` both use
the same ``measure()`` so the box recorded in the Motion IR is exactly the box
that gets painted.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont

# 1080x1920 is the reference design canvas. Sizes in the IR are absolute pixels,
# but when an IR is authored for another canvas we scale typography by area.
REF_W, REF_H = 1080, 1920
DEFAULT_FONT_DIR = "assets/fonts"
FONT_FILE = "NotoSansSC-Regular.otf"

MAX_LINES = 2                 # per element (list rows: 2 max as well)
LINE_HEIGHT_FACTOR = 1.30
ELLIPSIS = "…"

_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


# --------------------------------------------------------------------------- fonts
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_font_path(font_dir: str | Path | None = None,
                      font_file: str = FONT_FILE) -> Path:
    """Locate the bundled OFL font. Never falls back to system fonts (licensing)."""
    candidates: list[Path] = []
    if font_dir:
        d = Path(font_dir)
        candidates.append(d / font_file)
        if not d.is_absolute():
            candidates.append(_repo_root() / d / font_file)
    candidates.append(_repo_root() / DEFAULT_FONT_DIR / font_file)
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        f"bundled font not found: {font_file} (looked in "
        + ", ".join(str(c) for c in candidates) + ")")


def load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    key = (str(path), int(size))
    f = _font_cache.get(key)
    if f is None:
        f = ImageFont.truetype(str(path), int(max(8, size)))
        _font_cache[key] = f
    return f


# ------------------------------------------------------------------------- colours
def parse_color(value: Any, default: tuple[int, int, int, int] | None = None
                ) -> tuple[int, int, int, int]:
    """Parse '#RRGGBB', '#RRGGBBAA' or 'rgba(r,g,b,a)' into RGBA 0-255."""
    if value is None:
        if default is None:
            raise ValueError("colour is required")
        return default
    if isinstance(value, (tuple, list)):
        t = [int(v) for v in value]
        if len(t) == 3:
            return (t[0], t[1], t[2], 255)
        if len(t) == 4:
            a = t[3]
            return (t[0], t[1], t[2], a if a > 1 else int(round(a * 255)))
        raise ValueError(f"bad colour: {value!r}")
    s = str(value).strip()
    if s.startswith("#"):
        h = s[1:]
        if len(h) == 6:
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)
        if len(h) == 8:
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16))
        raise ValueError(f"bad hex colour: {value!r}")
    if s.lower().startswith("rgba"):
        inner = s[s.index("(") + 1:s.rindex(")")]
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) != 4:
            raise ValueError(f"bad rgba colour: {value!r}")
        rgb = [int(float(p)) for p in parts[:3]]
        a = float(parts[3])
        return (rgb[0], rgb[1], rgb[2], int(round(a * 255)) if a <= 1 else int(a))
    if s.lower().startswith("rgb"):
        inner = s[s.index("(") + 1:s.rindex(")")]
        parts = [int(float(p.strip())) for p in inner.split(",")]
        return (parts[0], parts[1], parts[2], 255)
    raise ValueError(f"bad colour: {value!r}")


def _srgb_channel(v: int) -> float:
    c = v / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(rgba: Sequence[int]) -> float:
    r, g, b = (_srgb_channel(int(rgba[i])) for i in range(3))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg: Sequence[int], bg: Sequence[int]) -> float:
    l1, l2 = luminance(fg), luminance(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def composite_over(fg: Sequence[int], back: Sequence[int]) -> tuple[int, int, int, int]:
    """Alpha-composite rgba fg over opaque rgb back (for contrast estimation)."""
    a = fg[3] / 255.0
    return (int(round(fg[0] * a + back[0] * (1 - a))),
            int(round(fg[1] * a + back[1] * (1 - a))),
            int(round(fg[2] * a + back[2] * (1 - a))), 255)


# --------------------------------------------------------------------------- easing
def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def smoothstep(x: float) -> float:
    x = _clamp01(x)
    return x * x * (3 - 2 * x)


def linear(x: float) -> float:
    return _clamp01(x)


def out_cubic(x: float) -> float:
    x = _clamp01(x)
    return 1 - (1 - x) ** 3


def in_cubic(x: float) -> float:
    x = _clamp01(x)
    return x ** 3


def out_quad(x: float) -> float:
    x = _clamp01(x)
    return 1 - (1 - x) ** 2


def in_quad(x: float) -> float:
    x = _clamp01(x)
    return x * x


def in_out_cubic(x: float) -> float:
    x = _clamp01(x)
    return 4 * x ** 3 if x < 0.5 else 1 - ((-2 * x + 2) ** 3) / 2


EASINGS = {
    "linear": linear,
    "smoothstep": smoothstep,
    "out_cubic": out_cubic,
    "in_cubic": in_cubic,
    "out_quad": out_quad,
    "in_quad": in_quad,
    "in_out_cubic": in_out_cubic,
}


def ease(name: str | None, x: float) -> float:
    return EASINGS.get(name or "out_cubic", out_cubic)(x)


# --------------------------------------------------------------------------- canvas
def canvas_scale(canvas: dict) -> float:
    """Typography scale from the reference 1080x1920 canvas (geometric mean)."""
    try:
        w = float(canvas.get("width") or REF_W)
        h = float(canvas.get("height") or REF_H)
    except (TypeError, ValueError):
        return 1.0
    if w <= 0 or h <= 0:
        return 1.0
    s = math.sqrt((w * h) / float(REF_W * REF_H))
    return max(0.45, min(2.0, s))


def scaled(size: float, canvas: dict) -> int:
    return int(round(size * canvas_scale(canvas)))


# --------------------------------------------------------------------------- text
def text_width(font: ImageFont.FreeTypeFont, s: str) -> float:
    if not s:
        return 0.0
    return float(font.getlength(s))


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: float,
              max_lines: int = MAX_LINES, *, ellipsis: str = ELLIPSIS) -> list[str]:
    """Greedy wrap with a CJK-friendly character break and a hard line limit.

    ``max_width`` must be positive; if a single glyph is wider than the line we
    still emit it on its own line (never loops forever).
    """
    text = (text or "").strip()
    if not text:
        return []
    max_width = max(1.0, float(max_width))
    lines: list[str] = []
    cur = ""
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur = ""
            if len(lines) >= max_lines:
                break
            continue
        trial = cur + ch
        if cur and text_width(font, trial) > max_width:
            lines.append(cur)
            cur = ch
            if len(lines) >= max_lines:
                break
        else:
            cur = trial
    else:
        if cur:
            lines.append(cur)
    # overflow: fold the remaining characters into the last line, then ellipsize
    consumed = sum(len(x) for x in lines)
    if consumed < len(text):
        rest = text[consumed:]
        last = lines[-1] if lines else ""
        joined = last + rest
        while joined and text_width(font, joined + ellipsis) > max_width:
            joined = joined[:-1]
        lines[-1] = (joined + ellipsis) if joined else ellipsis
    out: list[str] = []
    for ln in lines:
        if not ln:
            continue
        # a single over-long line (e.g. one latin word) is hard-clipped once
        while text_width(font, ln) > max_width and len(ln) > 1:
            ln = ln[:-1]
        out.append(ln)
    return out[:max_lines]


def text_block_metrics(font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    asc, desc = font.getmetrics()
    return int(asc), int(desc)


# --------------------------------------------------------------------------- drawing
def draw_panel(tile: Image.Image, box_wh: tuple[int, int], *, bg: Sequence[int],
               radius: int) -> None:
    """Paint the panel exactly inside the tile (never outside the IR box)."""
    w, h = box_wh
    if bg[3] <= 0:
        return
    d = ImageDraw.Draw(tile)
    r = max(0, min(int(radius), w // 2, h // 2))
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=tuple(bg))


def draw_round_rect_outline(tile: Image.Image, box_wh: tuple[int, int], *,
                            color: Sequence[int], radius: int, width: int = 2) -> None:
    w, h = box_wh
    if color[3] <= 0:
        return
    d = ImageDraw.Draw(tile)
    r = max(0, min(int(radius), w // 2, h // 2))
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=r, outline=tuple(color),
                        width=max(1, int(width)))


def draw_line(draw: ImageDraw.ImageDraw, segments: Sequence[tuple[str, Any, Sequence[int]]],
              *, center_x: float, baseline_y: float, anchor: str = "ls") -> float:
    """Draw a row of (text, font, colour) segments bottom-aligned on one baseline.

    Returns the total width painted. Segment text is drawn verbatim.
    """
    widths = [text_width(f, s) for s, f, _c in segments if s]
    total = float(sum(widths))
    x = center_x - total / 2.0
    for (s, f, c) in segments:
        if not s:
            continue
        draw.text((x, baseline_y), s, font=f, fill=tuple(c), anchor=anchor)
        x += text_width(f, s)
    return total


def multiply_alpha(img: Image.Image, factor: float) -> Image.Image:
    """Scale the alpha channel in place-safe fashion. factor in [0,1]."""
    factor = _clamp01(float(factor))
    if factor >= 0.999:
        return img
    if factor <= 0.001:
        return Image.new("RGBA", img.size, (0, 0, 0, 0))
    a = img.getchannel("A").point(lambda v: int(v * factor))
    out = img.copy()
    out.putalpha(a)
    return out


def scale_tile(img: Image.Image, factor: float) -> Image.Image:
    """Scale about the tile centre; result never exceeds the tile."""
    if abs(factor - 1.0) < 1e-3:
        return img
    w, h = img.size
    nw, nh = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
    nw, nh = min(nw, w), min(nh, h)
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def paste_centered(canvas: Image.Image, tile: Image.Image, box: Sequence[int]) -> None:
    x, y = int(box[0]), int(box[1])
    w, h = int(box[2]), int(box[3])
    if tile.size == (w, h):
        canvas.alpha_composite(tile, (x, y))
        return
    tw, th = tile.size
    canvas.alpha_composite(tile, (x + (w - tw) // 2, y + (h - th) // 2))


# --------------------------------------------------------------------------- geometry
def rect_inter_area(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = (float(v) for v in a[:4])
    bx, by, bw, bh = (float(v) for v in b[:4])
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy


def union_boxes(boxes: Iterable[Sequence[float]]) -> list[int] | None:
    bs = [b for b in boxes if b and len(b) >= 4 and b[2] > 0 and b[3] > 0]
    if not bs:
        return None
    x0 = min(float(b[0]) for b in bs)
    y0 = min(float(b[1]) for b in bs)
    x1 = max(float(b[0]) + float(b[2]) for b in bs)
    y1 = max(float(b[1]) + float(b[3]) for b in bs)
    return [int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0))]


def expand_box(box: Sequence[float], margin: float) -> list[int]:
    return [int(round(box[0] - margin)), int(round(box[1] - margin)),
            int(round(box[2] + 2 * margin)), int(round(box[3] + 2 * margin))]
