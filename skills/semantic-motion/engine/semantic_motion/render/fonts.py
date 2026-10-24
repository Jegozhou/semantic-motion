"""Font discovery with CJK-first fallback. Deterministic given the same installed fonts."""
from __future__ import annotations

import os
from functools import lru_cache

import skia

CJK_FAMILIES = ["PingFang SC", "Noto Sans SC", "Noto Sans CJK SC", "Source Han Sans SC", "Source Han Sans CN",
                "HarmonyOS Sans SC", "Microsoft YaHei", "Hiragino Sans GB", "WenQuanYi Micro Hei", "Droid Sans Fallback"]
LATIN_FAMILIES = ["Inter", "SF Pro Display", "Helvetica Neue", "Noto Sans", "Roboto", "Arial", "Liberation Sans", "DejaVu Sans"]
EMOJI_FAMILIES = ["Apple Color Emoji", "Noto Color Emoji", "Segoe UI Emoji"]


class FontBook:
    def __init__(self, family: str | None = None, font_dir: str | None = None):
        self.mgr = skia.FontMgr()
        self.files: dict[int, skia.Typeface] = {}
        names = set()
        for i in range(self.mgr.countFamilies()):
            names.add(self.mgr.getFamilyName(i))
        self.available = names
        prefs = ([family] if family and family != "auto" else []) + CJK_FAMILIES
        self.family = next((f for f in prefs if f in names), None)
        self.latin_family = next((f for f in LATIN_FAMILIES if f in names), None)
        self.emoji_family = next((f for f in EMOJI_FAMILIES if f in names), None)
        self._font_dir = font_dir or os.environ.get("SM_FONT_DIR")
        if self.family is None:
            self.family = self.latin_family or "sans-serif"
        self._cache: dict = {}

    def describe(self) -> dict:
        return {"primary": self.family, "latin": self.latin_family, "emoji": self.emoji_family}

    def typeface(self, weight: int = 400, family: str | None = None) -> skia.Typeface:
        fam = family or self.family
        key = (fam, int(weight))
        tf = self._cache.get(key)
        if tf is None:
            tf = self.mgr.matchFamilyStyle(fam, skia.FontStyle(int(weight), skia.FontStyle.kNormal_Width, skia.FontStyle.kUpright_Slant))
            if tf is None:
                tf = skia.Typeface.MakeDefault()
            self._cache[key] = tf
        return tf

    def font(self, size: float, weight: int = 400, family: str | None = None) -> skia.Font:
        f = skia.Font(self.typeface(weight, family), float(size))
        f.setSubpixel(True)
        f.setEdging(skia.Font.Edging.kSubpixelAntiAlias if False else skia.Font.Edging.kAntiAlias)
        f.setHinting(skia.FontHinting.kNone)
        f.setLinearMetrics(True)
        return f

    def has_glyph(self, font: skia.Font, ch: str) -> bool:
        return font.unicharToGlyph(ord(ch)) != 0

    @lru_cache(maxsize=4096)
    def fallback_typeface(self, ch: str, weight: int) -> skia.Typeface | None:
        if self.emoji_family:
            tf = self.typeface(weight, self.emoji_family)
            if tf.unicharToGlyph(ord(ch)) != 0:
                return tf
        tf = self.mgr.matchFamilyStyleCharacter(self.family, skia.FontStyle(int(weight), 5, skia.FontStyle.kUpright_Slant),
                                                ["zh-Hans", "en"], ord(ch))
        return tf
