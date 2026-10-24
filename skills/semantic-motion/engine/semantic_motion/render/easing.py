"""Deterministic easing functions (pure functions of t in [0, 1])."""
from __future__ import annotations

import math
from functools import lru_cache


@lru_cache(maxsize=64)
def _bezier_solver(x1: float, y1: float, x2: float, y2: float):
    def bx(t):
        return 3 * (1 - t) ** 2 * t * x1 + 3 * (1 - t) * t ** 2 * x2 + t ** 3

    def by(t):
        return 3 * (1 - t) ** 2 * t * y1 + 3 * (1 - t) * t ** 2 * y2 + t ** 3

    def dbx(t):
        return 3 * (1 - t) ** 2 * x1 + 6 * (1 - t) * t * (x2 - x1) + 3 * t ** 2 * (1 - x2)

    def solve(x: float) -> float:
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        t = x
        for _ in range(8):
            d = dbx(t)
            if abs(d) < 1e-6:
                break
            t2 = t - (bx(t) - x) / d
            if t2 < 0 or t2 > 1:
                break
            t = t2
        if abs(bx(t) - x) > 1e-4:  # bisection fallback
            lo, hi = 0.0, 1.0
            for _ in range(40):
                t = (lo + hi) / 2
                if bx(t) < x:
                    lo = t
                else:
                    hi = t
        return by(t)

    return solve


def bezier(curve, x: float) -> float:
    return _bezier_solver(*[float(c) for c in curve])(x)


def spring(t: float, damping: float = 0.62, freq: float = 1.6) -> float:
    """Damped spring settle from 0 -> 1 over normalized t (t=1 ~ settled)."""
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    w = 2 * math.pi * freq
    z = damping
    wd = w * math.sqrt(max(1e-6, 1 - z * z))
    tt = t * 1.0
    v = 1 - math.exp(-z * w * tt) * (math.cos(wd * tt) + (z * w / wd) * math.sin(wd * tt))
    # normalize so that v(1) == 1 exactly
    end = 1 - math.exp(-z * w) * (math.cos(wd) + (z * w / wd) * math.sin(wd))
    return v / end if end else v


NAMED = {
    "linear": (0.0, 0.0, 1.0, 1.0),
    "out": (0.22, 1.0, 0.36, 1.0),
    "in": (0.55, 0.0, 0.75, 0.2),
    "inout": (0.65, 0.0, 0.35, 1.0),
    "outback": (0.34, 1.45, 0.64, 1.0),
    "outexpo": (0.16, 1.0, 0.3, 1.0),
    "outcubic": (0.33, 1.0, 0.68, 1.0),
    "smooth": (0.45, 0.0, 0.55, 1.0),
}


def ease(name_or_curve, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    if isinstance(name_or_curve, (list, tuple)):
        return bezier(tuple(name_or_curve), x)
    if name_or_curve == "spring":
        return spring(x)
    return bezier(NAMED.get(name_or_curve, NAMED["out"]), x)


def prog(t: float, t0: float, dur: float) -> float:
    if dur <= 0:
        return 1.0 if t >= t0 else 0.0
    return 0.0 if t <= t0 else (1.0 if t >= t0 + dur else (t - t0) / dur)
