"""Design tokens and drawing primitives for the v2 motion renderer.

Everything here is deterministic and dependency-light (Pillow + numpy). The v2
visual language is *editorial*: type first, one accent colour sampled from the
footage, and backdrops that read as cinematography (a directional scrim) rather
than pasted panels. See docs/DESIGN.md for the rationale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
FONT_DIR = REPO_ROOT / "assets" / "fonts"

FONT_FILES = {
    "display": "NotoSansCJKsc-Bold.otf",
    "semibold": "NotoSansCJKsc-Medium.otf",
    "regular": "NotoSansSC-Regular.otf",
    "latin": "InterVariable.ttf",
}
# vertical weight axis for the Inter variable font
LATIN_WEIGHT = {"regular": 450, "medium": 560, "semibold": 640, "bold": 720}

# Type scale comes from the design policy (policies/*.json) so it can be tuned
# without editing code. The numbers below are the import-time default only.
from .policy import DEFAULT as _POLICY  # noqa: E402  (policy has no package imports)

TYPE_RATIO = dict(_POLICY.get("type.ratio", {}) or {})
LINE_FACTOR = dict(_POLICY.get("type.line_factor", {}) or {})
MIN_REF_HEIGHT = int(_POLICY.get("type.reference.min_height", 640))
REF_WIDTH_FACTOR = float(_POLICY.get("type.reference.width_factor", 0.62))
MAX_LINES = dict(_POLICY.get("type.max_lines", {}) or {})


def reference_height(width: int, height: int) -> int:
    """Optical reference for type sizes: portrait uses H, wide uses H*factor."""
    return int(max(MIN_REF_HEIGHT, min(height, width * REF_WIDTH_FACTOR)))


def type_px(kind: str, width: int, height: int, tier: float = 1.0) -> int:
    ref = reference_height(width, height)
    return max(12, int(round(TYPE_RATIO[kind] * ref * tier)))


# --------------------------------------------------------------------- easing
def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else hi if v > hi else v


def ease_out_cubic(t: float) -> float:
    t = clamp(t)
    return 1 - (1 - t) ** 3


def ease_out_quint(t: float) -> float:
    t = clamp(t)
    return 1 - (1 - t) ** 5


def ease_out_expo(t: float) -> float:
    t = clamp(t)
    return 1.0 if t >= 1 else 1 - 2 ** (-10 * t)


def ease_in_quad(t: float) -> float:
    t = clamp(t)
    return t * t


def ease_in_out_cubic(t: float) -> float:
    t = clamp(t)
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_standard(t: float) -> float:
    """A calm 'editorial' curve: fast start, long settle (expo-out blended)."""
    return 0.35 * ease_out_quint(t) + 0.65 * ease_out_expo(t)


EASES = {
    "linear": lambda t: clamp(t),
    "out_cubic": ease_out_cubic,
    "out_quint": ease_out_quint,
    "out_expo": ease_out_expo,
    "in_quad": ease_in_quad,
    "in_out_cubic": ease_in_out_cubic,
    "standard": ease_standard,
}


def ease(name: str, t: float) -> float:
    return EASES.get(name, ease_standard)(t)


def spring(t: float, damping: float = 0.72, frequency: float = 1.6) -> float:
    """Damped spring 0..1, settles at 1; overshoot stays under ~2 %."""
    t = clamp(t)
    if t >= 1:
        return 1.0
    w = math.pi * 2 * frequency
    return 1 - math.exp(-damping * 9 * t) * math.cos(w * t * 0.9) * (1 - t) ** 0.2


def progress(t: float, start: float, dur: float) -> float:
    if dur <= 0:
        return 1.0 if t >= start else 0.0
    return clamp((t - start) / dur)


# --------------------------------------------------------------------- colours
def rgba(value) -> tuple[int, int, int, int]:
    if isinstance(value, (tuple, list)):
        t = list(value)
        if len(t) == 3:
            return (int(t[0]), int(t[1]), int(t[2]), 255)
        a = t[3]
        return (int(t[0]), int(t[1]), int(t[2]), int(round(a * 255)) if a <= 1 else int(a))
    s = str(value).strip()
    if s.startswith("#"):
        h = s[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) == 6:
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)
        if len(h) == 8:
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16))
        raise ValueError(f"bad hex: {value!r}")
    if s.lower().startswith("rgba"):
        inner = s[s.index("(") + 1:s.rindex(")")]
        parts = [float(p) for p in inner.split(",")]
        while len(parts) < 4:
            parts.append(1.0)
        return (int(parts[0]), int(parts[1]), int(parts[2]), int(round(parts[3] * 255)))
    raise ValueError(f"bad colour: {value!r}")


def alpha(color, a: float) -> tuple[int, int, int, int]:
    c = rgba(color)
    return (c[0], c[1], c[2], int(round(clamp(a) * 255)))


def rel_luminance(color) -> float:
    c = rgba(color)
    def ch(v):
        v = v / 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * ch(c[0]) + 0.7152 * ch(c[1]) + 0.0722 * ch(c[2])


def contrast_ratio(a, b) -> float:
    la, lb = rel_luminance(a), rel_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def mix(a, b, t: float):
    ca, cb = rgba(a), rgba(b)
    t = clamp(t)
    return tuple(int(round(ca[i] + (cb[i] - ca[i]) * t)) for i in range(4))


def readable_on(bg, light=(247, 248, 250, 255), dark=(14, 18, 26, 255)):
    return light if contrast_ratio(light, bg) >= contrast_ratio(dark, bg) else dark


# ----------------------------------------------------------------------- fonts
class FontBook:
    """Cached font access. Sizes are passed in *output* pixels."""

    def __init__(self, font_dir: Path = FONT_DIR):
        self.dir = Path(font_dir)
        self._cache: dict[tuple[str, int, str | None], ImageFont.FreeTypeFont] = {}

    def path(self, kind: str) -> Path:
        p = self.dir / FONT_FILES[kind]
        if not p.exists():
            # graceful fallback so a missing font never breaks a render
            p = self.dir / FONT_FILES["regular"]
        return p

    def get(self, kind: str, size: int, weight: str | None = None) -> ImageFont.FreeTypeFont:
        size = max(6, int(round(size)))
        key = (kind, size, weight if kind == "latin" else None)
        f = self._cache.get(key)
        if f is not None:
            return f
        f = ImageFont.truetype(str(self.path(kind)), size)
        if kind == "latin" and weight:
            try:
                f.set_variation_by_axes([LATIN_WEIGHT.get(weight, 520)])
            except Exception:
                pass
        self._cache[key] = f
        return f


# ---------------------------------------------------------------- text metrics
def text_width(font: ImageFont.FreeTypeFont, s: str, tracking: float = 0.0) -> float:
    if not s:
        return 0.0
    if tracking:
        return float(sum(font.getlength(ch) for ch in s) + tracking * (len(s) - 1))
    return float(font.getlength(s))


CJK_START_FORBIDDEN = set("，。、；：！？）》」』】…·%,.;:!?)]}")
# particles and conjunctions must not open a line: they attach to what came before
CJK_BAD_LINE_START = CJK_START_FORBIDDEN | set("的地得着了过吗呢吧啊呀哦嘛和与及或而且")
CJK_BAD_LINE_END = set("（《「『【([{")
CJK_END_FORBIDDEN = set("（《「『【([{")


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF) or (0x3400 <= o <= 0x4DBF) or (0xF900 <= o <= 0xFAFF)


_JIEBA = None


def _segment_words(text: str) -> list[str] | None:
    """Chinese word segmentation, if jieba is available (it is a soft dependency).

    Breaking a line inside a word (我/们, 结/束) is the difference between a
    designed caption and a wrapped one, so words — not characters — are the wrap
    unit when we can get them.
    """
    global _JIEBA
    if _JIEBA is None:
        try:
            import jieba
            jieba.setLogLevel(60)
            _JIEBA = jieba
        except Exception:
            _JIEBA = False
    if not _JIEBA:
        return None
    words = [w for w in _JIEBA.cut(text) if w]
    return words or None


def _tokens(text: str) -> list[str]:
    """Split into wrap tokens: words when jieba is present, else CJK chars."""
    has_cjk = any(_is_cjk(ch) for ch in text)
    if has_cjk:
        words = _segment_words(text)
        if words:
            out: list[str] = []
            for w in words:
                if w.isspace():
                    if out:
                        out[-1] += " "
                    continue
                # keep punctuation as its own token so it can never start a line
                buf = ""
                for ch in w:
                    if ch in CJK_START_FORBIDDEN or ch in CJK_END_FORBIDDEN:
                        if buf:
                            out.append(buf); buf = ""
                        out.append(ch)
                    else:
                        buf += ch
                if buf:
                    out.append(buf)
            return out
    out = []
    buf = ""
    for ch in text:
        if ch.isspace():
            if buf:
                out.append(buf)
                buf = ""
            out.append(" ")
            continue
        if _is_cjk(ch) or ch in CJK_START_FORBIDDEN or ch in CJK_END_FORBIDDEN:
            if buf:
                out.append(buf)
                buf = ""
            out.append(ch)
        else:
            buf += ch
    if buf:
        out.append(buf)
    # merge spaces into the previous token to keep them visible
    merged: list[str] = []
    for t in out:
        if t == " " and merged:
            merged[-1] = merged[-1] + " "
        else:
            merged.append(t)
    return merged


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_w: float,
              max_lines: int = 2, tracking: float = 0.0,
              balance: bool = True) -> list[str]:
    """Greedy wrap that respects CJK punctuation rules and never splits Latin words.

    With two lines the result is rebalanced so the second line is not a lonely
    1-3 character orphan: that is the difference between "designed" and "wrapped".
    """
    lines = _wrap_greedy(text, font, max_w, max_lines, tracking)
    if balance and len(lines) == 2 and max_lines >= 2:
        best = None
        tokens = _tokens(text)
        n = len(tokens)
        for k in range(1, n):
            a = "".join(tokens[:k]).strip()
            b = "".join(tokens[k:]).strip()
            if not a or not b or b[0] in CJK_BAD_LINE_START or a[-1] in CJK_BAD_LINE_END:
                continue
            wa = text_width(font, a, tracking)
            wb = text_width(font, b, tracking)
            if wa > max_w * 1.005 or wb > max_w * 1.005:
                continue
            # balance first; only a genuinely tiny line is penalised
            score = abs(wa - wb) + max(0.0, 0.26 * max_w - min(wa, wb)) * 2.0
            if best is None or score < best[0]:
                best = (score, [a, b])
        if best:
            return best[1]
    return lines


def _wrap_greedy(text: str, font: ImageFont.FreeTypeFont, max_w: float,
                 max_lines: int = 2, tracking: float = 0.0) -> list[str]:
    lines: list[str] = []
    cur = ""
    for tok in _tokens(text):
        if text_width(font, tok, tracking) > max_w and len(tok) > 1:
            # a single word wider than the line: split it, but never lose a glyph
            parts = [tok[i:i + 1] for i in range(len(tok))]
            for piece in parts:
                cand = cur + piece
                if text_width(font, cand.rstrip(), tracking) <= max_w or not cur:
                    cur = cand
                else:
                    lines.append(cur.strip()); cur = piece
            continue
        cand = cur + tok
        if text_width(font, cand.rstrip(), tracking) <= max_w or not cur:
            cur = cand
            continue
        # push the line, but avoid starting the next line with forbidden punctuation
        if tok and tok[0] in CJK_BAD_LINE_START and len(cur) > 1:
            cur = cur + tok
            lines.append(cur.strip())
            cur = ""
            continue
        lines.append(cur.strip())
        cur = tok.lstrip() if tok.strip() else ""
        if len(lines) >= max_lines:
            break
    if cur.strip():
        lines.append(cur.strip())
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    if lines and text_width(font, lines[-1], tracking) > max_w * 1.02:
        # hard truncate with an ellipsis rather than overflowing the box
        s = lines[-1]
        while s and text_width(font, s + "…", tracking) > max_w:
            s = s[:-1]
        lines[-1] = s + "…"
    return lines


def line_pitch(size: float, kind: str = "display") -> int:
    return int(round(size * LINE_FACTOR.get(kind, 1.3)))


# ------------------------------------------------------------------- primitives
def rounded_mask(size: tuple[int, int], radius: int, ss: int = 1) -> Image.Image:
    w, h = size
    m = Image.new("L", (max(1, w * ss), max(1, h * ss)), 0)
    r = max(0, int(radius * ss))
    ImageDraw.Draw(m).rounded_rectangle([0, 0, max(0, w * ss - 1), max(0, h * ss - 1)],
                                        radius=r, fill=255)
    return m if ss == 1 else m.resize((w, h), Image.LANCZOS)


def linear_gradient(size: tuple[int, int], stops: Sequence[tuple[float, tuple]],
                    direction: str = "h") -> Image.Image:
    """Multi-stop gradient; stops are (position 0..1, rgba)."""
    w, h = max(1, size[0]), max(1, size[1])
    n = w if direction == "h" else h
    xs = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out = np.zeros((n, 4), dtype=np.float32)
    stops = sorted(stops, key=lambda s: s[0])
    for i in range(len(stops) - 1):
        p0, c0 = stops[i]
        p1, c1 = stops[i + 1]
        c0 = rgba(c0)
        c1 = rgba(c1)
        mask = (xs >= p0) & (xs <= p1)
        t = np.clip((xs[mask] - p0) / max(1e-6, p1 - p0), 0, 1)[:, None]
        for c in range(4):
            out[mask, c] = c0[c] + t[:, 0] * (c1[c] - c0[c])
    out[:, 3] = np.clip(out[:, 3], 0, 255)
    out[:, :3] = np.clip(out[:, :3], 0, 255)
    line = out.astype(np.uint8)
    if direction == "h":
        arr = np.repeat(line.reshape(1, w, 4), h, axis=0)      # (h, w, 4)
    else:
        arr = np.repeat(line.reshape(h, 1, 4), w, axis=1)      # (h, w, 4)
    return Image.fromarray(arr, "RGBA")


def soft_shadow(size: tuple[int, int], radius: int, blur: int,
                color=(0, 0, 0, 120), spread: int = 0) -> Image.Image:
    w, h = size
    pad = int(blur * 3)
    canvas = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    d.rounded_rectangle([pad - spread, pad - spread, pad + w + spread, pad + h + spread],
                        radius=radius + spread, fill=rgba(color))
    return canvas.filter(ImageFilter.GaussianBlur(blur))


def blur_crop(base: Image.Image, box: Sequence[int], blur: float,
              tint=(10, 14, 22), tint_amount: float = 0.55) -> Image.Image:
    x, y, w, h = [int(v) for v in box]
    x = max(0, min(x, base.width - 1)); y = max(0, min(y, base.height - 1))
    w = max(1, min(w, base.width - x)); h = max(1, min(h, base.height - y))
    crop = base.crop((x, y, x + w, y + h)).convert("RGB")
    if blur > 0:
        crop = crop.filter(ImageFilter.GaussianBlur(blur))
    if tint_amount > 0:
        crop = Image.blend(crop, Image.new("RGB", (w, h), tuple(tint[:3])), clamp(tint_amount))
    return crop.convert("RGBA")


def draw_tracked(draw: ImageDraw.ImageDraw, xy: tuple[float, float], s: str,
                 font: ImageFont.FreeTypeFont, fill, tracking: float = 0.0) -> float:
    x, y = xy
    for ch in s:
        draw.text((x, y), ch, font=font, fill=fill)
        x += font.getlength(ch) + tracking
    return x


# --------------------------------------------------------------------- palette
@dataclass
class Palette:
    accent: tuple = (242, 178, 51, 255)
    accent_soft: tuple = (242, 178, 51, 60)
    ink: tuple = (247, 248, 250, 255)
    ink_dim: tuple = (203, 210, 222, 235)
    scrim: tuple = (7, 10, 16)
    panel: tuple = (12, 16, 24)
    surface_is_dark: bool = True
    ladder: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"accent": "#%02X%02X%02X" % self.accent[:3],
                "ink": "#%02X%02X%02X" % self.ink[:3],
                "scrim": "#%02X%02X%02X" % self.scrim[:3],
                "surface_is_dark": self.surface_is_dark}


def _rgb_to_hsv(arr: np.ndarray):
    r, g, b = arr[..., 0] / 255.0, arr[..., 1] / 255.0, arr[..., 2] / 255.0
    mx = np.max(arr, axis=-1) / 255.0
    mn = np.min(arr, axis=-1) / 255.0
    diff = mx - mn + 1e-9
    h = np.zeros_like(mx)
    m = (mx == r)
    h[m] = (60 * ((g - b) / diff) % 360)[m]
    m = (mx == g)
    h[m] = (60 * ((b - r) / diff) + 120)[m]
    m = (mx == b)
    h[m] = (60 * ((r - g) / diff) + 240)[m]
    s = diff / (mx + 1e-9)
    return h, s, mx


def extract_palette(frames: Iterable[np.ndarray], *, fallback_accent="#E8A33D") -> Palette:
    """Pick a single accent hue from the footage (deterministic, no randomness).

    Strategy: take saturated mid-tone pixels, histogram their hue, prefer warm
    hues when they are within 70 % of the peak (editorial default), then clamp
    saturation/value into a tasteful range so the accent never looks neon.
    """
    hues, sats = [], []
    for fr in frames:
        a = np.asarray(fr)
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        a = a[..., :3]
        if a.shape[0] > 240:
            step = max(1, a.shape[0] // 240)
            a = a[::step, ::step]
        h, s, v = _rgb_to_hsv(a.astype(np.float32))
        m = (s > 0.18) & (v > 0.18) & (v < 0.97)
        if m.sum() < 50:
            continue
        hues.append(h[m])
        sats.append(s[m])
    if not hues:
        return Palette(accent=rgba(fallback_accent))
    hh = np.concatenate(hues)
    hist, edges = np.histogram(hh, bins=36, range=(0, 360))
    # light smoothing over neighbouring bins so a single noisy bin cannot win
    k = np.array([0.15, 0.7, 0.15])
    smooth = np.convolve(hist.astype(np.float32), k, mode="same")
    order = np.argsort(smooth)[::-1]
    best = int(order[0])
    for idx in order[:5]:
        center = (edges[idx] + edges[idx + 1]) / 2
        if 15 <= center <= 65:  # warm brand-friendly accents first
            best = int(idx)
            break
    hue = float((edges[best] + edges[best + 1]) / 2)
    sat = float(np.clip(np.median(np.concatenate(sats)), 0.35, 0.78))
    val = float(np.clip(0.92, 0.0, 1.0))
    # HSV -> RGB
    c = val * sat
    x = c * (1 - abs((hue / 60) % 2 - 1))
    m = val - c
    if hue < 60:
        rr, gg, bb = c, x, 0
    elif hue < 120:
        rr, gg, bb = x, c, 0
    elif hue < 180:
        rr, gg, bb = 0, c, x
    elif hue < 240:
        rr, gg, bb = 0, x, c
    elif hue < 300:
        rr, gg, bb = x, 0, c
    else:
        rr, gg, bb = c, 0, x
    acc = tuple(int(round((v + m) * 255)) for v in (rr, gg, bb)) + (255,)
    # keep the accent bright enough to sit on a dark scrim
    if rel_luminance(acc) < 0.18:
        acc = mix(acc, (255, 255, 255, 255), 0.25)
    return Palette(accent=acc, accent_soft=alpha(acc, 0.24))
