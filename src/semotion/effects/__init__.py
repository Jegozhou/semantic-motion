"""Declarative motion effect primitives (whitelisted by element type).

``effect_for(element)`` -> module implementing ``measure()`` and ``draw()``.
No module here evaluates element text; it is only rasterised.
"""
from __future__ import annotations

from typing import Any

from . import base, card, keyword, list, number

MODULES = {"keyword": keyword, "number": number, "list": list, "card": card}


def effect_for(element: dict):
    etype = str(element.get("type") or "")
    mod = MODULES.get(etype)
    if mod is None:
        raise ValueError(f"unsupported effect type: {etype!r}")
    return mod


def measure(element: dict, canvas: dict, *, max_text_width: float | None = None,
            font_dir: str | None = None) -> dict[str, Any]:
    return effect_for(element).measure(element, canvas,
                                      max_text_width=max_text_width, font_dir=font_dir)


def draw(element: dict, canvas: dict, spec: dict, t_rel: float, window_s: float,
         *, font_dir: str | None = None):
    return effect_for(element).draw(element, canvas, spec, t_rel, window_s,
                                     font_dir=font_dir)


__all__ = ["base", "card", "keyword", "list", "number", "MODULES", "effect_for",
           "measure", "draw"]
