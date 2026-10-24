"""Text utilities: normalization for anchor matching, CJK detection, Chinese numerals."""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache

_PUNCT_RE = re.compile(r"[\s\u3000-\u303f\uff00-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65"
                       r"!-/:-@\[-`{-~\u2010-\u2027\u2030-\u205e\u00a0-\u00bf]+")
SENT_END = set("。！？!?；;…")
CLAUSE_END = set("，,、：:")
NO_LINE_START = set("，。！？、；：”’）》】」』〉…—·,.!?;:)]}%‰")
NO_LINE_END = set("“‘（《【「『〈([{$¥￥#")


def is_cjk(ch: str) -> bool:
    if not ch:
        return False
    o = ord(ch[0])
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0x20000 <= o <= 0x2A6DF or 0xF900 <= o <= 0xFAFF
            or 0x3040 <= o <= 0x30FF or 0xAC00 <= o <= 0xD7AF)


def cjk_ratio(s: str) -> float:
    chars = [c for c in s if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if is_cjk(c)) / len(chars)


@lru_cache(maxsize=1)
def _t2s():
    try:
        import opencc  # type: ignore

        try:
            return opencc.OpenCC("t2s")
        except Exception:
            return opencc.OpenCC("t2s.json")
    except Exception:
        return None


def to_simplified(s: str) -> str:
    cc = _t2s()
    if cc is None:
        return s
    try:
        return cc.convert(s)
    except Exception:
        return s


def norm(s: str) -> str:
    """Normalization used for phrase anchoring: NFKC, lower-case, simplified, no punctuation/space."""
    s = unicodedata.normalize("NFKC", s).lower()
    s = to_simplified(s)
    return _PUNCT_RE.sub("", s)


def norm_with_map(s: str) -> tuple[str, list[int]]:
    """Normalize while keeping, for each normalized char, the index of its source char."""
    out, idx = [], []
    for i, ch in enumerate(s):
        n = norm(ch)
        for c in n:
            out.append(c)
            idx.append(i)
    return "".join(out), idx


def find_phrase(haystack: str, needle: str, start_at: int = 0, fuzzy: bool = True) -> tuple[int, int, float] | None:
    """Find needle in normalized haystack. Returns (start, end, score); fuzzy fallback >= 0.72."""
    if not needle:
        return None
    k = haystack.find(needle, start_at)
    if k >= 0:
        return k, k + len(needle), 1.0
    k = haystack.find(needle)
    if k >= 0:
        return k, k + len(needle), 0.99
    if not fuzzy:
        return None
    # fuzzy sliding window (handles ASR homophone errors like 龙下/龙虾)
    n = len(needle)
    best = (0.0, -1, -1)
    for L in {max(1, n - 1), n, n + 1}:
        for i in range(0, max(1, len(haystack) - L + 1)):
            r = SequenceMatcher(None, haystack[i:i + L], needle, autojunk=False).ratio()
            if r > best[0]:
                best = (r, i, i + L)
    if best[0] >= 0.72:
        return best[1], best[2], best[0]
    return None


# ----------------------------------------------------------------------------------------------
# Chinese numerals
_CN_DIG = {"零": 0, "〇": 0, "一": 1, "壹": 1, "二": 2, "两": 2, "贰": 2, "三": 3, "叁": 3, "四": 4, "肆": 4,
           "五": 5, "伍": 5, "六": 6, "陆": 6, "七": 7, "柒": 7, "八": 8, "捌": 8, "九": 9, "玖": 9}
_CN_UNIT = {"十": 10, "拾": 10, "百": 100, "佰": 100, "千": 1000, "仟": 1000}
_CN_BIG = {"万": 10 ** 4, "亿": 10 ** 8, "兆": 10 ** 12}
CN_NUM_CHARS = "".join(_CN_DIG) + "".join(_CN_UNIT) + "".join(_CN_BIG) + "点"


def cn_to_number(s: str) -> float | None:
    """Parse Chinese numerals (incl. mixed digits like '3万', '一点五', '百分之三十')."""
    s = s.strip()
    if not s:
        return None
    pct = False
    if s.startswith("百分之"):
        pct = True
        s = s[3:]
    try:
        v = float(s.replace(",", ""))
        return v
    except ValueError:
        pass
    if "点" in s:
        a, b = s.split("点", 1)
        ia = cn_to_number(a) if a else 0
        if ia is None:
            return None
        frac = ""
        for ch in b:
            if ch in _CN_DIG:
                frac += str(_CN_DIG[ch])
            elif ch.isdigit():
                frac += ch
            else:
                return None
        return float(f"{int(ia)}.{frac or 0}")
    total, section, num = 0.0, 0.0, None
    for ch in s:
        if ch.isdigit():
            num = (num or 0) * 10 + int(ch)
        elif ch in _CN_DIG:
            num = _CN_DIG[ch] if num is None or num == 0 else num * 10 + _CN_DIG[ch]
        elif ch in _CN_UNIT:
            section += (num if num is not None else 1) * _CN_UNIT[ch]
            num = None
        elif ch in _CN_BIG:
            section += num or 0
            total += (section or 1) * _CN_BIG[ch]
            section, num = 0.0, None
        else:
            return None
    total += section + (num or 0)
    return total


_NUM_RE = re.compile(
    r"(百分之[零〇一二两三四五六七八九十百千点\d.]+)|"
    r"(\d+(?:[.,]\d+)*\s*(?:%|％|倍|万|亿|千|百|k|K|M|B|x|X)?)|"
    r"([零〇一二两三四五六七八九十百千万亿点]{2,}(?:倍|万|亿)?|[一二两三四五六七八九十][十百千万亿倍])")


def find_numbers(text: str) -> list[dict]:
    """Find numeric expressions with parsed values (best-effort, for heuristics and the brief)."""
    out = []
    for m in _NUM_RE.finditer(text):
        raw = m.group(0)
        val, suffix = None, ""
        if m.group(1):
            val = cn_to_number(raw)
            suffix = "%"
        elif m.group(2):
            r = raw.strip()
            for suf in ("%", "％", "倍", "万", "亿", "千", "百", "k", "K", "M", "B", "x", "X"):
                if r.endswith(suf) and len(r) > len(suf):
                    suffix = suf
                    r = r[: -len(suf)].strip()
                    break
            val = cn_to_number(r)
            mult = {"万": 1e4, "亿": 1e8, "千": 1e3, "百": 100, "k": 1e3, "K": 1e3, "M": 1e6, "B": 1e9}.get(suffix)
            if mult and val is not None:
                val, suffix = val * mult, ""
        else:
            r = raw.strip()
            if r.endswith("倍"):
                suffix, r = "倍", r[:-1]
            val = cn_to_number(r)
        if val is None:
            continue
        if raw in ("一个", "一下", "一些", "一样", "一点") or (val in (1, 2) and not suffix and len(raw) <= 2 and not raw.isdigit()):
            continue
        out.append({"raw": raw, "value": val, "suffix": "%" if suffix == "％" else suffix, "span": [m.start(), m.end()]})
    return out


def fmt_number(v: float, decimals: int | None = None) -> str:
    if decimals is None:
        decimals = 0 if abs(v - round(v)) < 1e-9 else (1 if abs(v * 10 - round(v * 10)) < 1e-9 else 2)
    s = f"{v:,.{decimals}f}"
    return s


def visual_len(s: str) -> float:
    """Approximate visual width in 'em': CJK = 1, Latin/digits ~ 0.55."""
    w = 0.0
    for ch in s:
        if is_cjk(ch) or ch in "，。！？、；：（）《》【】“”‘’":
            w += 1.0
        elif ch.isspace():
            w += 0.3
        elif ch.isupper() or ch in "MW%@":
            w += 0.68
        else:
            w += 0.56
    return w


# ----------------------------------------------------------------------------------------------
# Spoken-number variants for anchoring ("百分之四十" / "forty percent" vs. ASR "40%")
_EN_SMALL = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
_EN_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_EN_SCALE = {"hundred": 100, "thousand": 1000, "million": 10 ** 6, "billion": 10 ** 9}
_EN_NUM_RE = re.compile(r"\b(?:(?:%s)(?:[\s-]+(?:and[\s-]+)?)?)+(?:\s*(?:percent|per\s+cent))?" % "|".join(
    sorted(list(_EN_SMALL) + list(_EN_TENS) + list(_EN_SCALE), key=len, reverse=True)), re.I)


def en_words_to_number(s: str) -> float | None:
    toks = [t for t in re.split(r"[\s-]+", s.lower()) if t and t != "and"]
    if not toks or toks[0] in _EN_SCALE:
        return None
    total, cur = 0, 0
    for t in toks:
        if t in _EN_SMALL:
            cur += _EN_SMALL[t]
        elif t in _EN_TENS:
            cur += _EN_TENS[t]
        elif t == "hundred":
            cur = (cur or 1) * 100
        elif t in _EN_SCALE:
            total += (cur or 1) * _EN_SCALE[t]
            cur = 0
        else:
            return None
    return float(total + cur)


def _digit_forms(v: float, zh: bool) -> list[str]:
    iv = int(v) if abs(v - round(v)) < 1e-9 else None
    forms = [str(iv) if iv is not None else f"{v:g}"]
    if zh and iv is not None and iv >= 10000:
        w, r = divmod(iv, 10000)
        forms.append(f"{iv / 10000:g}万")
        if r and r % 1000 == 0:
            forms.append(f"{w}万{r // 1000}千")
            forms.append(f"{w}万{r // 1000}")
    if not zh and iv is not None:
        for scale, word in ((10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")):
            if iv >= scale and iv % (scale // 10) == 0:
                forms.append(f"{iv / scale:g} {word}")
                if scale == 1000:
                    forms.append(f"{iv // 1000}k")
                break
    return forms


def numeric_variants(phrase: str, limit: int = 8) -> list[str]:
    """Normalized alternatives of `phrase` with spoken numbers written as digits (ASR usually writes digits)."""
    spans: list[tuple[int, int, list[str]]] = []
    for nm in find_numbers(phrase):
        raw = nm["raw"]
        if any(ch in CN_NUM_CHARS for ch in raw if not ch.isdigit()) and not raw.isdigit():
            if raw.strip().isascii():
                continue
            spans.append((nm["span"][0], nm["span"][1], _digit_forms(nm["value"], True)))
    for m in _EN_NUM_RE.finditer(phrase):
        txt = re.sub(r"\s*(percent|per\s+cent)$", "", m.group(0).strip(), flags=re.I)
        v = en_words_to_number(txt)
        if v is None:
            continue
        spans.append((m.start(), m.end(), _digit_forms(v, False)))
    if not spans:
        return []
    spans.sort()
    outs = [""]
    pos = 0
    for a, b, forms in spans:
        if a < pos:
            continue
        outs = [o + phrase[pos:a] + f for o in outs for f in forms][:limit]
        pos = b
    outs = [norm(o + phrase[pos:]) for o in outs]
    base = norm(phrase)
    res = []
    for o in outs:
        if o and o != base and o not in res:
            res.append(o)
    return res
