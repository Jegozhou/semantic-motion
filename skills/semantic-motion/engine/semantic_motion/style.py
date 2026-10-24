"""Design tokens. All sizes are in 'u' = min(W, H) / 100 so designs scale across resolutions."""
from __future__ import annotations

import colorsys
from copy import deepcopy

import numpy as np

ACCENTS = [  # curated, tested for legibility on dark glass and on mid-tone footage
    ("amber", "#FFC83D"),
    ("mint", "#3EE6A8"),
    ("sky", "#4CC3FF"),
    ("coral", "#FF7759"),
    ("violet", "#A98BFF"),
]

PLATFORMS = {
    # safe rectangles (x0, y0, x1, y1) as fractions of the frame
    "generic": {"landscape": (0.05, 0.07, 0.95, 0.93), "portrait": (0.06, 0.08, 0.94, 0.88), "square": (0.06, 0.06, 0.94, 0.94)},
    "youtube": {"landscape": (0.05, 0.07, 0.95, 0.89), "portrait": (0.06, 0.10, 0.86, 0.80)},
    "bilibili": {"landscape": (0.05, 0.08, 0.95, 0.88), "portrait": (0.06, 0.10, 0.86, 0.80)},
    "douyin": {"portrait": (0.06, 0.12, 0.84, 0.78), "landscape": (0.05, 0.08, 0.95, 0.88)},
    "tiktok": {"portrait": (0.06, 0.12, 0.84, 0.78), "landscape": (0.05, 0.08, 0.95, 0.88)},
    "reels": {"portrait": (0.06, 0.12, 0.86, 0.80), "landscape": (0.05, 0.08, 0.95, 0.88)},
    "shorts": {"portrait": (0.06, 0.10, 0.86, 0.80), "landscape": (0.05, 0.08, 0.95, 0.88)},
    "xiaohongshu": {"portrait": (0.06, 0.10, 0.90, 0.82), "landscape": (0.05, 0.08, 0.95, 0.90)},
    "wechat": {"portrait": (0.06, 0.10, 0.90, 0.82), "landscape": (0.05, 0.08, 0.95, 0.90)},
}

BASE = {
    "name": "balanced",
    "colors": {"text": "#FFFFFF", "text2": "#FFFFFFC4", "muted": "#FFFFFF8A", "plate": "#0B0D12", "plate_alpha": 0.50,
               "stroke": "#FFFFFF24", "shadow": "#000000", "shadow_alpha": 0.46, "positive": "#3EE6A8", "negative": "#FF6B5E"},
    "type": {  # sizes in u
        "keyword": 7.0, "keyword_portrait": 8.0, "keyword_sub": 2.7, "stat": 12.0, "stat_suffix": 5.6, "stat_label": 2.9,
        "card_title": 3.0, "item": 3.3, "item_sub": 2.3, "title": 6.2, "kicker": 2.2, "quote": 4.6, "byline": 2.4,
        "name": 3.8, "role": 2.5, "caption": 4.3, "caption_portrait": 5.0, "chart_label": 2.4, "chart_value": 2.8,
        "callout": 3.0,
    },
    "weights": {"keyword": 800, "stat": 800, "title": 800, "item": 500, "card_title": 700, "label": 500, "caption": 600,
                "quote": 600, "name": 700, "role": 400},
    "line_height": 1.22,
    "tracking": {"cjk": 0.02, "latin": 0.0, "kicker": 0.18},
    "radius": 2.0,
    "pad": 2.4,
    "blur": 2.6,            # backdrop blur sigma (u)
    "shadow_blur": 1.0,     # text shadow sigma (u)
    "motion": {
        "enter": 0.46, "exit": 0.30, "land_frac": 0.42, "stagger": 0.032, "stagger_cap": 0.36,
        "travel": 1.1, "blur_in": 0.7, "scale_in": 0.965,
        "ease_in_curve": [0.22, 1.0, 0.36, 1.0],    # entrance: fast-out, soft landing
        "ease_out_curve": [0.55, 0.0, 0.75, 0.2],   # exit: accelerate away
        "ease_move": [0.65, 0.0, 0.35, 1.0],
        "overshoot": 0.0,
        "tail": 0.40,           # hold after the semantic span ends
        "min_gap": 0.10,        # gap between consecutive primary elements
    },
    "density": {"max_primary": 1, "max_total": 2, "target_per_min": 5.0, "max_per_min": 9.0, "clean_ratio": 0.40},
    "reading": {"cjk_cps": 6.5, "latin_wps": 3.2, "base": 0.7},
    "max_hold": {"keyword": 4.8, "stat": 5.5, "title": 3.8, "lower_third": 4.8, "quote": 8.0, "callout": 4.5,
                 "list": 14.0, "compare": 12.0, "chart": 12.0, "flow": 14.0, "custom": 10.0},
    "camera": {"punch": 1.10, "push": 1.06, "punch_dur": 0.22, "release_dur": 0.35},
}

STYLES = {
    "balanced": {},
    "minimal": {
        "name": "minimal",
        "type": {"keyword": 5.8, "keyword_portrait": 6.8, "stat": 10.0, "title": 5.2},
        "motion": {"enter": 0.50, "exit": 0.32, "travel": 0.7, "blur_in": 0.5, "stagger": 0.02},
        "density": {"max_primary": 1, "max_total": 2, "target_per_min": 3.0, "max_per_min": 6.0, "clean_ratio": 0.55},
        "camera": {"punch": 1.06, "push": 1.04},
    },
    "bold": {
        "name": "bold",
        "type": {"keyword": 8.2, "keyword_portrait": 9.4, "stat": 14.0, "title": 7.2},
        "motion": {"enter": 0.36, "exit": 0.24, "travel": 1.8, "blur_in": 1.0, "stagger": 0.04, "overshoot": 0.10,
                   "ease_in_curve": [0.16, 1.0, 0.3, 1.0]},
        "density": {"max_primary": 1, "max_total": 2, "target_per_min": 8.0, "max_per_min": 14.0, "clean_ratio": 0.25},
        "camera": {"punch": 1.14, "push": 1.08},
    },
}


def _merge(a: dict, b: dict) -> dict:
    out = deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def hex_to_rgba(h: str, alpha: float | None = None) -> tuple[float, float, float, float]:
    h = h.lstrip("#")
    if len(h) == 6:
        r, g, b, a = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255
    elif len(h) == 8:
        r, g, b, a = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16)
    else:
        raise ValueError(f"bad color {h}")
    return r / 255, g / 255, b / 255, (a / 255 if alpha is None else alpha)


def rel_luminance(rgb: tuple[float, float, float]) -> float:
    def ch(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgb[:3]
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast_ratio(l1: float, l2: float) -> float:
    a, b = max(l1, l2), min(l1, l2)
    return (a + 0.05) / (b + 0.05)


def pick_accent(shot_rgb: np.ndarray | None) -> tuple[str, str]:
    """Choose the curated accent whose hue is furthest from the footage's dominant saturated hues."""
    if shot_rgb is None or shot_rgb.size == 0:
        return ACCENTS[0]
    px = shot_rgb.reshape(-1, 3)
    hues, weights = [], []
    for r, g, b in px[:: max(1, len(px) // 4000)]:
        h, l, s = colorsys.rgb_to_hls(float(r), float(g), float(b))
        if s > 0.25 and 0.15 < l < 0.85:
            # ignore skin tones (hue ~ 0.02-0.11)
            if 0.0 <= h <= 0.11 and s < 0.7:
                continue
            hues.append(h)
            weights.append(s)
    best, best_score = ACCENTS[0], -1.0
    for i, (name, hx) in enumerate(ACCENTS):
        r, g, b, _ = hex_to_rgba(hx)
        ha = colorsys.rgb_to_hls(r, g, b)[0]
        if hues:
            d = np.abs(np.array(hues) - ha)
            d = np.minimum(d, 1 - d)
            # weighted fraction of footage colour mass that is close (< 0.08) to this accent
            clash = float(np.sum(np.array(weights) * (d < 0.08)) / (np.sum(weights) + 1e-6))
        else:
            clash = 0.0
        score = 1.0 - clash - 0.04 * i  # preference order as tie-break
        if score > best_score:
            best, best_score = (name, hx), score
    return best


def resolve_style(name: str = "balanced", accent: str = "auto", shot_rgb=None, overrides: dict | None = None) -> dict:
    st = _merge(BASE, STYLES.get(name, {}))
    if name not in STYLES:
        st["name"] = "balanced"
    if accent and accent != "auto":
        st["colors"]["accent"] = accent if accent.startswith("#") else dict(ACCENTS).get(accent, ACCENTS[0][1])
        st["colors"]["accent_name"] = accent
    else:
        an, ah = pick_accent(shot_rgb)
        st["colors"]["accent"] = ah
        st["colors"]["accent_name"] = an
    if overrides:
        st = _merge(st, overrides)
    return st


def safe_rect(platform: str, orientation: str) -> tuple[float, float, float, float]:
    if platform in (None, "", "auto"):
        platform = "douyin" if orientation == "portrait" else "generic"
    p = PLATFORMS.get(platform, PLATFORMS["generic"])
    return p.get(orientation) or p.get("landscape") or PLATFORMS["generic"]["landscape"]
