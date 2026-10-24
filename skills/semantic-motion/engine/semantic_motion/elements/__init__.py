"""Element library. Each element type implements `variants()` (candidate natural sizes for the
layout solver) and `draw()` (pure function of time -> deterministic frames)."""
import importlib as _il

from .base import REGISTRY, RenderCtx, get_element_class  # noqa: F401

MODULES = ("el_keyword", "el_stat", "el_list", "el_compare", "el_chart", "el_flow", "el_quote", "el_title",
           "el_lower_third", "el_callout", "el_custom", "el_captions")


def load_all() -> dict:
    for m in MODULES:
        _il.import_module(f"{__name__}.{m}")
    return REGISTRY


load_all()
