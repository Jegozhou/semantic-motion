"""Rasterise one motion element into a per-frame RGBA PNG sequence.

All text and shapes are painted by Pillow — never by an ffmpeg text filter —
because this ffmpeg build has no libass/libfreetype, and because rasterising a
string is inherently safe (spec §3.6): element text is only ever a string.

Frames are named ``f%05d.png`` with the **absolute** frame number so the
renderer can address them with ``-start_number``. Every frame is a full-canvas
RGBA image painted strictly inside the element's IR ``layout.box``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from . import effects
from .effects import base


# First-frame visibility: the rendered onset must land on the IR's start frame.
# Starting from alpha 0 would make the metric sampler miss the first frame(s);
# these floors keep the entrance perceptually soft but immediately measurable.
ENTER_ALPHA_FLOOR = {"pop_up": 0.55, "scale_in": 0.55, "fade_up": 0.55, "fade": 0.35}
ENTER_SCALE_FLOOR = {"pop_up": 0.86, "scale_in": 0.96}


def _enter(eased: float, motion: dict) -> tuple[float, float]:
    """Return (alpha, scale) for the entrance at progress ``eased`` in [0,1]."""
    etype = str((motion.get("enter") or {}).get("type") or "pop_up")
    if etype == "reveal":
        return 1.0, 1.0          # rows reveal individually (see effects/list.py)
    floor = ENTER_ALPHA_FLOOR.get(etype, 0.55)
    alpha = floor + (1.0 - floor) * eased
    if etype in ("fade", "fade_up") and etype not in ENTER_SCALE_FLOOR:
        return alpha, 1.0
    scale_from = float((motion.get("enter") or {}).get("scale_from")
                       or ENTER_SCALE_FLOOR.get(etype, 0.9))
    scale = 1.0 + (scale_from - 1.0) * (1.0 - eased)
    return alpha, scale


def _exit_alpha(t_rel: float, window_s: float, motion: dict) -> float:
    ex = motion.get("exit") or {}
    dur = float(ex.get("dur_s") or 0.18)
    if dur <= 0:
        return 1.0
    start = max(0.0, window_s - dur)
    if t_rel <= start:
        return 1.0
    p = base.ease(str(ex.get("ease") or "in_quad"), (t_rel - start) / dur)
    return max(0.0, 1.0 - p)


def element_spec(element: dict, canvas: dict, *, font_dir: str = base.DEFAULT_FONT_DIR) -> dict:
    lay = element.get("layout") or {}
    box = lay.get("box") or [0, 0, int(canvas.get("width")), int(canvas.get("height"))]
    pad = int((element.get("style") or {}).get("padding_px") or 0)
    max_w = max(60.0, float(box[2]) - 2 * pad)
    return effects.measure(element, canvas, max_text_width=max_w, font_dir=font_dir)


def render_element_png(element: dict, canvas: dict, out_dir: str | Path, *,
                       font_dir: str = base.DEFAULT_FONT_DIR) -> list[Path]:
    """Write RGBA frames for ``element.window`` into ``out_dir``.

    Returns the written paths (absolute frame numbers in the file name). An
    element with no drawable text returns an empty list; the renderer records it
    as an empty overlay rather than inventing content.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("f*.png"):
        stale.unlink()

    win = element.get("window") or {}
    sf, ef = int(win.get("start_frame")), int(win.get("end_frame"))
    if ef <= sf:
        return []
    fps = float((canvas or {}).get("fps") or 25.0)
    W = int(canvas.get("width"))
    H = int(canvas.get("height"))
    lay = element.get("layout") or {}
    box = lay.get("box")
    if not box or len(box) < 4:
        return []
    bx, by, bw, bh = (int(v) for v in box[:4])
    if bw <= 0 or bh <= 0 or bx < 0 or by < 0 or bx + bw > W or by + bh > H:
        raise ValueError(f"{element.get('id')}: layout.box {box} is not a valid canvas box")

    content = element.get("content") or {}
    if not str(content.get("text") or "").strip():
        return []

    window_s = ef / fps - sf / fps
    motion = element.get("motion") or {}
    enter_dur = float((motion.get("enter") or {}).get("dur_s") or 0.24)
    spec = element_spec(element, canvas, font_dir=font_dir)

    paths: list[Path] = []
    for frame in range(sf, ef):
        t_rel = frame / fps - sf / fps
        # entrance
        raw = min(1.0, max(0.0, t_rel / enter_dur)) if enter_dur > 0 else 1.0
        eased = base.ease(str((motion.get("enter") or {}).get("ease") or "out_cubic"), raw)
        alpha_in, scale = _enter(eased, motion)
        alpha = alpha_in * _exit_alpha(t_rel, window_s, motion)

        tile = effects.draw(element, canvas, spec, t_rel, window_s, font_dir=font_dir)
        if scale < 0.999:
            tile = base.scale_tile(tile, scale)
        if tile.size[0] > bw or tile.size[1] > bh:
            # never paint outside the IR box
            tile = tile.crop((0, 0, min(tile.size[0], bw), min(tile.size[1], bh)))
        if alpha < 0.999:
            tile = base.multiply_alpha(tile, alpha)

        frame_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        base.paste_centered(frame_img, tile, (bx, by, bw, bh))
        path = out_dir / f"f{frame:05d}.png"
        frame_img.save(path, format="PNG", optimize=False)
        paths.append(path)
    return paths
