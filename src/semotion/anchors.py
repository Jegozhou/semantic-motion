"""Semantic Motion — semantic anchors + acoustic onset refinement.

Contract: ``semotion.anchors/1`` (docs/INTERFACES.md §3.3).

Two-stage design
----------------
1. :func:`classify` turns ASR output into *semantic* anchors with a deterministic
   Chinese rule engine (ordinals, numbers/percentages, contrast, conclusion,
   comparison, process, card). **Times come only from ASR word/segment spans** —
   a rule never invents a timestamp; a match that cannot be mapped onto real
   word ids is dropped instead of guessed. Every anchor carries
   ``source="ASR_ALIGNED"`` and the real ``word_ids`` it was built from.
2. :func:`refine_onsets` snaps each anchor's ``start_s`` to the *acoustic*
   onset measured from the 10 ms RMS envelope of the audio (silence-exit with an
   adaptive noise floor; energy-climb fallback for continuous speech). This is
   the piece that removes the systematic ASR word-start bias (measured at
   ≈ +120 ms median on the SYNTH_KNOWN fixture) and it is what makes a
   ``P90 <= 100 ms`` trigger claim possible at all. Every anchor records what
   happened in ``refine`` (``method``/``delta_ms``/``accepted``/``window_ms``),
   so a rejected refinement is visible, never silent.

Honesty rules
-------------
* ASR text is untrusted data: it is only pattern-matched, never executed and
  never used to reach the network.
* ``confidence`` is the *rule strength*, not a probability calibrated against
  human labels. Anchors below 0.6 must not drive precise motion
  (INTERFACES §3.3) and ``semantic.decide`` is expected to downgrade them.
* The transcript's own truth class stays ``MACHINE_WEAK``: :func:`timing_errors`
  compares against an *external* SYNTH_KNOWN/HUMAN truth file, never against
  ASR output. ``source="ASR_ALIGNED"`` on the anchors is deliberate — it says
  "semantics from ASR", not "verified truth".
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from . import asr as asr_mod
from . import media

__all__ = [
    "SCHEMA",
    "classify",
    "refine_onsets",
    "load_truth",
    "timing_errors",
    "prefix_matches",
    "unit_onset_anchors",
    "anchors_cache_path",
    "envelope",
    "find_onset",
    "validate",
    "load",
]

SCHEMA = "semotion.anchors/1"
HOP_S = 0.01          # 10 ms energy frames
SEARCH_S = 0.25       # +/-250 ms onset search window
AUDIO_RATE = 16000

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- rules
# Each entry: (compiled regex, kind, confidence, rule-id, note).
# Order matters: the first rule that claims a character range wins for that
# range, so the specific patterns (numbers, ordinals) come before the generic
# discourse markers.

_ORDINAL = re.compile(r"第[一二三四五六七八九十百零〇\d]{1,3}(?:个|步|点|条|项|章|部分|层|种|位)?")
_ORDINAL_GENERIC = re.compile(r"首先|其次|再次|接着|接下来")
_ORDINAL_FINAL = re.compile(r"最后|最终")
_ENUM = re.compile(r"(?:一是|二是|三是|四是|五是)")

_PERCENT_CN = re.compile(r"百分之[零一二两三四五六七八九十百千\d.]+")
_PERCENT_SYM = re.compile(r"\d+(?:\.\d+)?\s*[%％]")
_ARABIC = re.compile(
    r"\d+(?:\.\d+)?(?:个|条|次|倍|成|种|点|步|人|年|月|日|天|秒|分钟|小时|块|元|岁|页|分|"
    r"万|千|百|斤|米|公里|毫米|厘米|度|项|台|件)?")
_CN_NUM = re.compile(
    r"[一二两三四五六七八九十百千万零〇]{1,4}"
    r"(?:成|个|条|次|倍|种|点|步|人|年|月|日|天|秒|分钟|小时|块|元|岁|页|分|斤|米|公里|"
    r"毫米|厘米|度|项|台|件)")
_HALF = re.compile(r"一半")

# Words that *look* like numbers but carry no data in spoken Chinese. Emitting a
# counting animation for "一遍" or "一个" would be a false claim about the
# content, so they are dropped (they would also be low-value P1 triggers).
_FILLERS = {
    "一个", "一些", "一下", "一样", "一起", "一点", "一直", "一定", "一般", "一切",
    "一边", "一面", "一次", "一遍", "一号", "一下", "一点", "一时",
}

_COMPARISON = re.compile(
    r"相比|对比|比起|不如|胜过|优于|劣于|超过|高于|低于|更好|更差|更快|更慢|更强|更弱|"
    r"前者|后者|而不是")
_PROCESS = re.compile(r"第一步|第二步|第三步|流程图|流程|步骤|依次|顺序|先.{1,8}再|先.{1,8}然后")
_CONCLUSION = re.compile(
    r"所以|因此|于是|总之|综上|总的来说|归根到底|结论|说白了|换句话说|可见|这说明|"
    r"先说结论|核心结论")
_CONTRAST = re.compile(
    r"其实|但是|不过|然而|反而|却是|关键是|重点是|核心是|真正|本质上|说实话|"
    r"问题在于|难就难在|不是.{1,6}而是")
_EMPHASIS = re.compile(
    r"最难|最容易|最怕|最重要|最关键|痛点|误区|秘密|真相|秘诀|底线|原则|"
    r"核心|关键|重点|本质")
# A card is only interesting when the clause *enumerates* something: "三个问题"
# or "两个方面", not the filler "一个" ("聊一个真实的痛点" is not a list).
_CARD_COUNT = re.compile(
    r"(?:[二两三四五六七八九十\d]{1,4}(?:个|条|种|项|步|层)"
    r"|[一二两三四五六七八九十\d]{1,4}(?:方面|问题|指标|步骤|要点|原因|阶段))")

# (regex, kind, confidence, rule id) in evaluation order.
_RULES: list[tuple[re.Pattern[str], str, float, str]] = [
    (_PERCENT_SYM, "number", 0.92, "percent_symbol"),
    (_PERCENT_CN, "number", 0.92, "percent_chinese"),
    (_ORDINAL, "list_item", 0.90, "ordinal"),
    (_ORDINAL_GENERIC, "list_item", 0.82, "ordinal_generic"),
    (_ENUM, "list_item", 0.80, "enumeration"),
    (_ARABIC, "number", 0.88, "arabic_number"),
    (_CN_NUM, "number", 0.86, "chinese_number"),
    (_HALF, "number", 0.84, "fraction_half"),
    (_CONTRAST, "keyword", 0.80, "contrast_marker"),
    (_CONCLUSION, "conclusion", 0.80, "conclusion_marker"),
    (_COMPARISON, "comparison", 0.76, "comparison_marker"),
    (_PROCESS, "process", 0.74, "process_marker"),
    (_EMPHASIS, "keyword", 0.78, "emphasis_word"),
]

# Chinese ASR frequently returns Traditional characters (observed on fx_b:
# "做口波動畫" for "做口播动画", "70%的問題"). Rules are written in Simplified
# Chinese, so matching is done on a 1:1 character-normalised copy of the segment
# text; spans stay valid because every entry maps one character to one character.
# The anchor ``text`` is always the *original* transcript substring (never a
# rewritten one), so the IR "text must exist in the transcript" invariant holds.
_TRAD_TO_SIMP: dict[str, str] = {
    "個": "个", "們": "们", "說": "说", "論": "论", "這": "这", "對": "对", "時": "时",
    "間": "间", "樣": "样", "齊": "齐", "畫": "画", "動": "动", "佔": "占", "擋": "挡",
    "積": "积", "標": "标", "準": "准", "題": "题", "於": "于", "後": "后", "點": "点",
    "種": "种", "條": "条", "邊": "边", "進": "进", "過": "过", "為": "为", "與": "与",
    "從": "从", "會": "会", "現": "现", "場": "场", "幾": "几", "實": "实", "應": "应",
    "該": "该", "麼": "么", "頭": "头", "體": "体", "單": "单", "線": "线", "處": "处",
    "給": "给", "張": "张", "產": "产", "業": "业", "數": "数", "據": "据", "語": "语",
    "詞": "词", "讀": "读", "寫": "写", "聽": "听", "聲": "声", "圖": "图", "機": "机",
    "網": "网", "絡": "络", "電": "电", "腦": "脑", "軟": "软", "顯": "显", "訊": "讯",
    "號": "号", "碼": "码", "開": "开", "關": "关", "長": "长", "見": "见", "覺": "觉",
    "觀": "观", "認": "认", "識": "识", "變": "变", "還": "还", "車": "车", "東": "东",
    "樂": "乐", "學": "学", "愛": "爱", "買": "买", "賣": "卖", "貴": "贵", "價": "价",
    "錢": "钱", "質": "质", "塊": "块", "歲": "岁", "頁": "页", "萬": "万", "億": "亿",
    "兩": "两", "兒": "儿", "總": "总", "結": "结", "誤": "误", "區": "区", "級": "级",
    "構": "构", "組": "组", "簡": "简", "複": "复", "雜": "杂", "難": "难", "項": "项",
    "細": "细", "漢": "汉", "華": "华", "專": "专", "轉": "转", "換": "换", "銜": "衔",
    "離": "离", "節": "节", "閱": "阅", "覽": "览", "頻": "频",
    # second batch: forms that actually appeared in faster-whisper output on the
    # calibration fixtures (順/鍵/捨/驟/終/夠 ...) plus other common ones.
    "順": "顺", "鍵": "键", "捨": "舍", "驟": "骤", "終": "终", "夠": "够", "準": "准",
    "備": "备", "門": "门", "問": "问", "資": "资", "費": "费", "課": "课", "練": "练",
    "習": "习", "慣": "惯", "態": "态", "勢": "势", "導": "导", "週": "周", "圍": "围",
    "議": "议", "討": "讨", "講": "讲", "許": "许", "訂": "订", "記": "记", "試": "试",
    "詳": "详", "誠": "诚", "誰": "谁", "調": "调", "談": "谈", "請": "请", "謝": "谢",
    "譜": "谱", "讚": "赞", "貝": "贝", "負": "负", "財": "财", "貨": "货", "販": "贩",
    "責": "责", "貸": "贷", "貼": "贴", "賈": "贾", "賓": "宾", "賴": "赖", "賺": "赚",
    "購": "购", "賽": "赛", "贈": "赠", "趕": "赶", "趨": "趋", "踐": "践", "軍": "军",
    "較": "较", "載": "载", "輸": "输", "辦": "办", "農": "农", "連": "连", "遊": "游",
    "運": "运", "達": "达", "違": "违", "遠": "远", "適": "适", "選": "选", "遺": "遗",
    "郵": "邮", "鄉": "乡", "釋": "释", "銀": "银", "鋼": "钢", "錄": "录", "鐘": "钟",
    "鐵": "铁", "閉": "闭", "閑": "闲", "陽": "阳", "陰": "阴", "陣": "阵", "階": "阶",
    "際": "际", "陸": "陆", "隨": "随", "險": "险", "隱": "隐", "隻": "只", "雖": "虽",
    "雙": "双", "雞": "鸡", "霧": "雾", "靈": "灵", "靜": "静", "響": "响", "頂": "顶",
    "須": "须", "預": "预", "頓": "顿", "領": "领", "頸": "颈", "顆": "颗", "顏": "颜",
    "願": "愿", "類": "类", "顧": "顾", "風": "风", "飛": "飞", "飯": "饭", "飲": "饮",
    "養": "养", "餘": "余", "館": "馆", "馬": "马", "驗": "验", "驚": "惊", "髮": "发",
    "魚": "鱼", "鮮": "鲜", "鳥": "鸟", "麗": "丽", "麥": "麦", "黃": "黄", "齒": "齿",
    "龍": "龙", "龜": "龟", "橋": "桥", "樹": "树", "權": "权", "歡": "欢", "歲": "岁",
    "歷": "历", "厲": "厉", "壓": "压", "廳": "厅", "彈": "弹", "強": "强", "徑": "径",
    "徹": "彻", "懷": "怀", "戰": "战", "戲": "戏", "戶": "户", "拋": "抛", "揮": "挥",
    "損": "损", "擺": "摆", "攝": "摄", "敗": "败", "敵": "敌", "斷": "断", "於": "于",
    "昇": "升", "書": "书", "爭": "争", "牆": "墙", "獨": "独", "獎": "奖", "環": "环",
    "畢": "毕", "畫": "画", "異": "异", "當": "当", "疊": "叠", "穩": "稳", "窮": "穷",
    "競": "竞", "筆": "笔", "節": "节", "範": "范", "築": "筑", "簽": "签", "籃": "篮",
    "籌": "筹", "籤": "签", "紅": "红", "純": "纯", "紙": "纸", "級": "级", "素": "素",
    "緊": "紧", "綜": "综", "綠": "绿", "緒": "绪", "編": "编", "緩": "缓", "縣": "县",
    "繼": "继", "續": "续", "纖": "纤", "缺": "缺", "罰": "罚", "罷": "罢", "羅": "罗",
    "義": "义", "聖": "圣", "聞": "闻", "聯": "联", "聰": "聪", "職": "职", "聽": "听",
    "肅": "肃", "脅": "胁", "脫": "脱", "腦": "脑", "臉": "脸", "臟": "脏", "臺": "台",
    "與": "与", "舉": "举", "舊": "旧", "艙": "舱", "藝": "艺", "蘇": "苏", "蘋": "苹",
    "處": "处", "號": "号", "蟲": "虫", "衝": "冲", "裝": "装", "製": "制", "複": "复",
    "見": "见", "規": "规", "視": "视", "親": "亲", "觀": "观", "計": "计", "訓": "训",
    "託": "托", "訪": "访", "設": "设", "評": "评", "註": "注", "証": "证", "詩": "诗",
    "話": "话", "誌": "志", "誤": "误", "諸": "诸", "謀": "谋", "謂": "谓", "謎": "谜",
    "謠": "谣", "譯": "译", "護": "护", "讀": "读", "贖": "赎", "跡": "迹", "躍": "跃",
    "軀": "躯", "輛": "辆", "輪": "轮", "轟": "轰", "辭": "辞", "辯": "辩", "邊": "边",
    "邏": "逻", "鑰": "钥", "閃": "闪", "閱": "阅", "闆": "板", "陝": "陕", "隊": "队",
    "隻": "只", "難": "难", "雲": "云", "電": "电", "霧": "雾", "響": "响", "頻": "频",
    "顆": "颗", "題": "题", "額": "额", "顏": "颜", "願": "愿", "類": "类", "顧": "顾",
    "顯": "显", "飛": "飞", "餐": "餐", "館": "馆", "駕": "驾", "騎": "骑", "驗": "验",
    "體": "体", "鬍": "胡", "麗": "丽", "黨": "党", "齊": "齐", "齒": "齿", "龍": "龙",
}

# Clause boundary for card anchors: punctuation, not spaces.
_CLAUSE = re.compile(r"[^，。、；！？,.!?;]+")
_CLAUSE_PUNCT = "，。、；！？,.!?;"

_KIND_PRIORITY = {
    "number": 0, "list_item": 1, "comparison": 2, "process": 3,
    "conclusion": 4, "keyword": 5, "card": 6,
}

_PUNCT = "，。、！？,.!?;；:：\"'“”‘’（）()[]【】—…· \t\n"


# --------------------------------------------------------------------------- utils

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cache_root() -> Path:
    return Path(os.environ.get("SEMOTION_CACHE_ROOT") or (PROJECT_ROOT / "artifacts" / "cache"))


def transcript_sha256(transcript: dict[str, Any]) -> str:
    """Stable digest of a transcript document (provenance included: it is part
    of the stored document, so the same file hashes the same)."""
    payload = json.dumps(transcript, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def anchors_cache_path(transcript: dict[str, Any]) -> Path:
    return cache_root() / "anchors" / f"{transcript_sha256(transcript)}.json"


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _to_simplified(text: str) -> str:
    """1:1 Traditional -> Simplified normalisation used for rule matching only."""
    if not text:
        return ""
    return "".join(_TRAD_TO_SIMP.get(ch, ch) for ch in text)


def _norm_text(text: str) -> str:
    return re.sub(r"[\s" + re.escape(_PUNCT) + r"]+", "", text or "")


def _strip_punct(text: str) -> str:
    return (text or "").strip().strip(_PUNCT).strip()


def _overlap_ratio(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(min(len(sa), len(sb)))


# ------------------------------------------------------------------ classification

def _segment_words(transcript: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for word in transcript.get("words") or []:
        out.setdefault(word["segment_id"], []).append(word)
    for words in out.values():
        words.sort(key=lambda w: (float(w["start_s"]), int(w.get("char_start", 0))))
    return out


def _words_for_span(words: list[dict[str, Any]], c0: int, c1: int) -> list[dict[str, Any]]:
    """Words whose char range overlaps ``[c0, c1)``."""
    return [w for w in words
            if int(w.get("char_start", 0)) < c1 and int(w.get("char_end", 0)) > c0]


def _make_anchor(seg: dict[str, Any], words: list[dict[str, Any]], kind: str,
                 text: str, confidence: float, rule: str, *,
                 span: tuple[int, int] | None = None,
                 source: str = "ASR_ALIGNED") -> dict[str, Any] | None:
    """Build one anchor; times are the union of the matched words' spans."""
    if not words:
        return None
    text = _strip_punct(text) if text else "".join(w["text"] for w in words)
    if not text:
        return None
    start = min(float(w["start_s"]) for w in words)
    end = max(float(w["end_s"]) for w in words)
    if not (0.0 <= start < end):
        return None
    anchor = {
        "id": "",  # assigned after ordering
        "kind": kind,
        "text": text,
        "start_s": round(start, 6),
        "end_s": round(end, 6),
        "confidence": round(float(confidence), 4),
        "source": source,
        "segment_id": seg["id"],
        "word_ids": [w["id"] for w in words],
        "rule": rule,
        "char_span": [int(span[0]) if span else 0, int(span[1]) if span else 0],
    }
    return anchor



# --------------------------------------------------------------------- English
# Chinese-first product, but ASR and the anchor layer must not silently produce
# nothing for English footage (the public-domain real-face fixture is English).
# These rules are intentionally simple and explainable; they never invent times.
EN_STOPWORDS = frozenset("""
a an the and or but if then than that this these those there here
is are was were be been being am do does did done doing have has had having
i you he she it we they me him her us them my your his its our their
of to in on at for from by with about into over after before under above
as so because while when where which who whom whose what how why
not no nor very just also too only still even much many more most some any
can could will would shall should may might must ought
said says say get got go going goes come came make made take took
""".split())

EN_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "million": 1_000_000, "billion": 1_000_000_000, "half": 0.5, "dozen": 12,
}
EN_ORDINALS = ("first", "second", "third", "fourth", "fifth", "finally", "lastly",
               "next", "then")
EN_CONCLUSION = ("so", "therefore", "because", "thus", "hence", "that's why",
                 "the point is", "in short")
_RE_EN_NUMBER = re.compile(r"\b(\d+(?:\.\d+)?)\s*(%|percent)?\b|\b(" +
                           "|".join(sorted(EN_NUMBER_WORDS, key=len, reverse=True)) + r")\b",
                           re.IGNORECASE)


def _en_words_in_span(words: list[dict[str, Any]], c0: int, c1: int) -> list[dict[str, Any]]:
    """Words whose recorded character span overlaps [c0, c1)."""
    out = []
    for w in words:
        ws, we = w.get("char_start"), w.get("char_end")
        if ws is None or we is None:
            continue
        if ws < c1 and we > c0:
            out.append(w)
    return out


def _classify_english(segments: list[dict[str, Any]],
                      per_segment_words: dict[str, list[dict[str, Any]]]
                      ) -> list[dict[str, Any]]:
    """Minimal English rule pass: numbers, ordinal steps, content keywords, cards."""
    cands: list[dict[str, Any]] = []
    ordinal_seen = 0
    for seg in segments:
        words = per_segment_words.get(seg.get("id", ""), [])
        if not words:
            continue
        lower = (seg.get("text") or "").strip()
        if not lower:
            continue
        low = lower.lower()

        # numbers / percentages / spelled-out quantities
        for m in _RE_EN_NUMBER.finditer(lower):
            span_words = _en_words_in_span(words, m.start(), m.end())
            anchor = _make_anchor(seg, span_words, "number", m.group(0), 0.86, "en_number",
                                  span=(m.start(), m.end()))
            if anchor:
                cands.append(anchor)

        # ordinal steps
        for ord_word in EN_ORDINALS:
            for m in re.finditer(r"\b" + re.escape(ord_word) + r"\b", low):
                span_words = _en_words_in_span(words, m.start(), m.end())
                anchor = _make_anchor(seg, span_words, "list_item", m.group(0), 0.84,
                                      "en_ordinal", span=(m.start(), m.end()))
                if anchor:
                    cands.append(anchor)
                    ordinal_seen += 1
                break

        # conclusion markers on a whole sentence
        if any(marker in low for marker in EN_CONCLUSION) and len(words) >= 5:
            anchor = _make_anchor(seg, words, "card", lower, 0.62, "en_conclusion",
                                  span=(0, len(lower)))
            if anchor:
                cands.append(anchor)

        # content keywords: long, high-confidence, non-stopword tokens
        scored = []
        for w in words:
            token = re.sub(r"[^A-Za-z']", "", str(w.get("text", ""))).lower()
            prob = float(w.get("probability") or 0.0)
            if len(token) < 5 or token in EN_STOPWORDS or prob < 0.75:
                continue
            scored.append((prob + 0.02 * len(token), token, w))
        scored.sort(key=lambda x: -x[0])
        for _, token, w in scored[:2]:
            anchor = _make_anchor(seg, [w], "keyword", str(w.get("text", "")).strip(),
                                  0.74, "en_content_word")
            if anchor:
                cands.append(anchor)

    # a run of >=2 ordinals is a step list; a single one is just a keyword
    if ordinal_seen < 2:
        for c in cands:
            if c["kind"] == "list_item":
                c["kind"] = "keyword"
    return cands


def classify(transcript: dict[str, Any], *, use_cache: bool = True,
             write_cache: bool = True) -> dict[str, Any]:
    """Classify a transcript into semantic anchors (``semotion.anchors/1``).

    Returns the anchors document; a cache hit at
    ``artifacts/cache/anchors/<transcript_sha256>.json`` is returned verbatim.
    """
    if use_cache and write_cache:
        cached = anchors_cache_path(transcript)
        if cached.exists():
            try:
                doc = json.loads(cached.read_text(encoding="utf-8"))
                if (isinstance(doc, dict) and doc.get("schema") == SCHEMA
                        and doc.get("transcript_sha256") == transcript_sha256(transcript)):
                    return doc
            except Exception:
                pass

    segments = transcript.get("segments") or []
    per_segment_words = _segment_words(transcript)
    candidates: list[dict[str, Any]] = []
    final_markers: list[dict[str, Any]] = []
    ordinal_count = 0

    for seg in segments:
        original = seg.get("text") or ""
        text = _to_simplified(original)     # rules live in Simplified Chinese
        words = per_segment_words.get(seg["id"], [])
        if not text or not words:
            continue
        claimed: list[tuple[int, int]] = []

        def _is_claimed(c0: int, c1: int) -> bool:
            return any(c0 < b and c1 > a for a, b in claimed)

        for regex, kind, conf, rule in _RULES:
            for match in regex.finditer(text):
                c0, c1 = match.start(), match.end()
                raw = match.group(0)
                if rule != "percent_symbol":
                    raw = raw.rstrip()
                if _is_claimed(c0, c1):
                    continue
                if kind == "number" and _norm_text(raw) in _FILLERS:
                    continue
                if rule == "chinese_number" and _norm_text(raw) in _FILLERS:
                    continue
                matched_words = _words_for_span(words, c0, c1)
                anchor = _make_anchor(seg, matched_words, kind, original[c0:c1], conf,
                                      rule, span=(c0, c1))
                if anchor is None:
                    continue
                if rule in ("ordinal", "ordinal_generic", "enumeration"):
                    ordinal_count += 1
                claimed.append((c0, c1))
                candidates.append(anchor)

        # 最后/最终 is only an ordinal when the clip actually enumerates
        for match in _ORDINAL_FINAL.finditer(text):
            c0, c1 = match.start(), match.end()
            if _is_claimed(c0, c1):
                continue
            matched_words = _words_for_span(words, c0, c1)
            anchor = _make_anchor(seg, matched_words, "list_item", match.group(0), 0.72,
                                  "ordinal_final", span=(c0, c1))
            if anchor:
                final_markers.append(anchor)
                claimed.append((c0, c1))

        # Card: a *clause* that enumerates or concludes. Cards are container
        # anchors, so they may overlap a specific anchor; they are the lowest
        # priority and low confidence by design. Segment-level cards are
        # deliberately not emitted: whisper can return one 25 s segment for a
        # whole clip, and a card covering everything is useless for layout.
        for clause in _CLAUSE.finditer(text):
            c0, c1 = clause.start(), clause.end()
            clause_text = _norm_text(clause.group(0))
            if len(clause_text) < 6:
                continue
            if not (_CARD_COUNT.search(clause_text) or _CONCLUSION.search(clause_text)):
                continue
            clause_words = _words_for_span(words, c0, c1)
            if not clause_words:
                continue
            duration = max(float(w["end_s"]) for w in clause_words) - min(
                float(w["start_s"]) for w in clause_words)
            if duration > 8.0:      # a card cannot plausibly stay up that long
                continue
            card = _make_anchor(seg, clause_words, "card", original[c0:c1], 0.62,
                                "card_clause", span=(c0, c1))
            if card:
                candidates.append(card)

    if not candidates or str(transcript.get("language") or "").lower().startswith("en"):
        candidates = _classify_english(segments, per_segment_words) or candidates

    for marker in final_markers:
        if ordinal_count >= 2:
            candidates.append(marker)
        else:
            anchor = dict(marker)
            anchor.update({"kind": "conclusion", "confidence": 0.78,
                           "rule": "conclusion_marker"})
            candidates.append(anchor)

    anchors = _dedupe(candidates)
    anchors = _dedupe(_merge_keywords(anchors, per_segment_words))
    anchors.sort(key=lambda a: (float(a["start_s"]), _KIND_PRIORITY.get(a["kind"], 9),
                                float(a["end_s"])))
    for i, anchor in enumerate(anchors):
        anchor["id"] = f"a{i:02d}"
        anchor.pop("char_span", None)

    source = transcript.get("source") or {}
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "source_sha256": source.get("sha256"),
        "transcript_sha256": transcript_sha256(transcript),
        "duration_s": round(float(source.get("duration_s") or 0.0), 6),
        "language": transcript.get("language"),
        "anchors": anchors,
        "provenance": {
            "created_utc": _now(),
            "method": "asr_word_spans+rule_classifier",
            "n_anchors": len(anchors),
            "kinds": _kind_counts(anchors),
            "truth_class": "MACHINE_WEAK",
            "note": ("anchor times are ASR word/segment spans (MACHINE_WEAK); "
                     "run refine_onsets() to snap them to the audio envelope"),
            "refine": None,
        },
    }
    if write_cache:
        _write_json(anchors_cache_path(transcript), doc)
    return doc


def _kind_counts(anchors: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for anchor in anchors:
        counts[anchor["kind"]] = counts.get(anchor["kind"], 0) + 1
    return dict(sorted(counts.items()))


def _merge_keywords(anchors: list[dict[str, Any]],
                    per_segment_words: dict[str, list[dict[str, Any]]], *,
                    max_chars: int = 8, max_gap_s: float = 0.15) -> list[dict[str, Any]]:
    """Extend keyword anchors into short phrases (Lead/G1 requirement).

    A keyword anchor such as ``其实`` is often a function word that the director
    layer would drop as filler. Within one segment we therefore extend the
    anchor to the **right** across adjacent words while

    * the gap to the next word is < 150 ms (no pause between them),
    * the merged phrase is <= 8 characters (punctuation not counted), and
    * no clause punctuation sits between the words (a comma ends the phrase).

    ``start_s`` stays the first word's start, ``end_s`` becomes the last merged
    word's end, and ``confidence`` is the minimum of the anchors folded in.
    Only ``keyword`` anchors are merged: ``number``/``list_item``/``card`` keep
    their exact spans because their *duration* is part of the effect design.

    Forward-only is deliberate: in Chinese the modifier precedes the head noun,
    so extending left tends to produce strings like "的痛点" or "套流程".
    """
    out: list[dict[str, Any]] = []
    for anchor in anchors:
        if anchor.get("kind") != "keyword":
            out.append(anchor)
            continue
        words = per_segment_words.get(anchor.get("segment_id"), [])
        ids = set(anchor.get("word_ids") or [])
        positions = [i for i, w in enumerate(words) if w["id"] in ids]
        if not positions:
            out.append(anchor)
            continue
        first, last = min(positions), max(positions)
        phrase = "".join(w["text"] for w in words[first:last + 1])
        end_s = float(words[last]["end_s"])
        j = last
        while j + 1 < len(words):
            nxt = words[j + 1]
            if any(ch in _CLAUSE_PUNCT for ch in words[j]["text"]):
                break                                  # a comma ends the phrase
            if float(nxt["start_s"]) - end_s >= max_gap_s:
                break                                  # there is a pause
            candidate = phrase + nxt["text"]
            if len(_strip_punct(candidate)) > max_chars:
                break
            phrase = candidate
            end_s = float(nxt["end_s"])
            j += 1
        if j == last:
            out.append(anchor)
            continue
        merged = dict(anchor)
        merged["text"] = _strip_punct(phrase)
        merged["end_s"] = round(end_s, 6)
        merged["word_ids"] = [w["id"] for w in words[first:j + 1]]
        merged["rule"] = f"{anchor['rule']}+phrase_merge"
        merged["merged_words"] = j + 1 - first
        # confidence is the min over every keyword anchor folded into the span
        folded = [float(a["confidence"]) for a in anchors
                  if a is not anchor and a.get("kind") == "keyword"
                  and set(a.get("word_ids") or []) <= set(merged["word_ids"])]
        merged["confidence"] = round(min([float(anchor["confidence"])] + folded), 4)
        out.append(merged)
    return out


def _dedupe(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop lower-confidence duplicates of the same kind with heavy word overlap."""
    ordered = sorted(candidates, key=lambda a: (
        _KIND_PRIORITY.get(a["kind"], 9), -float(a["confidence"]),
        float(a["start_s"]), -float(a["end_s"])))
    kept: list[dict[str, Any]] = []
    for anchor in ordered:
        duplicate = False
        for other in kept:
            if other["kind"] != anchor["kind"]:
                continue
            if _overlap_ratio(other["word_ids"], anchor["word_ids"]) >= 0.5:
                duplicate = True
                break
        if not duplicate:
            kept.append(anchor)
    return kept


# ---------------------------------------------------------------------- audio env

def _wav_path(video: Path) -> Path:
    sha = media.sha256_file(video)
    return cache_root() / "audio" / f"{sha}_16k_mono.wav"


def _load_pcm(video: Path) -> tuple[np.ndarray, int]:
    wav = _wav_path(video)
    if not wav.exists():
        media.extract_audio(video, wav, rate=AUDIO_RATE, channels=1)
    with wave.open(str(wav), "rb") as w:
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, rate


def envelope(video: Path, *, hop_s: float = HOP_S) -> dict[str, Any]:
    """10 ms RMS envelope of the video's audio (mono 16 kHz).

    Returns ``{"env": np.ndarray, "hop_s", "rate", "peak", "noise", "duration_s"}``.
    ``peak`` is the 99th percentile (robust to a single click) and ``noise`` the
    20th percentile, used as the adaptive floor for onset detection.
    """
    samples, rate = _load_pcm(Path(video))
    hop = max(1, int(round(hop_s * rate)))
    n = len(samples) // hop
    if n <= 0:
        raise media.MediaError("E_ASR_FAILED", f"audio too short for envelope: {video}")
    env = np.sqrt(np.square(samples[: n * hop].reshape(n, hop)).mean(axis=1)).astype(np.float64)
    return {
        "env": env,
        "hop_s": hop / float(rate),
        "rate": rate,
        "peak": float(np.percentile(env, 99)),
        "noise": float(np.percentile(env, 20)),
        "duration_s": len(samples) / float(rate),
    }


def find_onset(env_doc: dict[str, Any], t_start: float, *, search_s: float = SEARCH_S,
               quiet_factor: float = 2.5, quiet_rel_peak: float = 0.02,
               min_climb_rel_peak: float = 0.05) -> dict[str, Any]:
    """Find the acoustic onset near ``t_start`` in an envelope.

    Two detectors, in order:

    1. **silence exit** — the last quiet frame (<= adaptive floor) near the
       target followed by a frame above the floor. This is what speech onsets
       after a pause look like: it lands within ~10 ms (one hop) of the
       assembly truth on the synthetic fixture.
    2. **energy climb** — the largest positive jump over an 80 ms lookback,
       for speech that never goes quiet (mid-sentence anchors). Accepted only
       when the climb is significant relative to the clip's peak.

    Returns a dict with ``onset_s``, ``method``, ``delta_ms``, ``accepted``,
    ``climb`` and the search window actually used.
    """
    env = env_doc["env"]
    hop = float(env_doc["hop_s"])
    peak = max(float(env_doc["peak"]), 1e-9)
    floor = max(float(env_doc["noise"]) * quiet_factor, peak * quiet_rel_peak)
    n = len(env)
    lo = max(0.0, float(t_start) - float(search_s))
    hi = max(lo, float(t_start) + float(search_s))
    i0 = int(math.floor(lo / hop))
    i1 = int(math.ceil(hi / hop))
    i0 = max(0, min(n - 1, i0))
    i1 = max(i0 + 1, min(n, i1))
    window = env[i0:i1]
    result: dict[str, Any] = {
        "method": "none",
        "delta_ms": 0,
        "accepted": False,
        "window_ms": int(round(search_s * 1000)),   # per-side window, contract says 250
        "search_window_s": [round(i0 * hop, 6), round(min(n, i1) * hop, 6)],
        "quiet_threshold": round(floor, 6),
        "peak": round(peak, 6),
        "climb": 0.0,
        "onset_s": None,
    }
    if window.size == 0:
        return result

    quiet = window <= floor
    # run-length encode quiet stretches; an onset is the frame after a quiet run
    candidates: list[int] = []
    run = 0
    for i, is_quiet in enumerate(quiet):
        if is_quiet:
            run += 1
            continue
        if run >= 3 and env[i0 + i] > peak * quiet_rel_peak:
            candidates.append(i0 + i)
        run = 0
    # NOTE: deliberately no "quiet at the window edge" shortcut. A window that
    # happens to start on one quiet 10 ms frame would otherwise snap to the next
    # frame and make the result depend on window alignment rather than on audio.
    # A candidate needs a real quiet run (>= 3 frames = 30 ms) inside the window.

    # Forward extension: when the +/-250 ms window holds no onset but *ends*
    # in silence, ASR has put the start inside a pause that is still going at
    # the window edge. The true onset can then only be later, so look forward
    # once instead of refusing to move. Two real cases on the fixtures:
    #  * the first word of a clip (whisper pins it to t=0, audio starts at 0.5 s)
    #  * a word at a segment boundary (whisper pins it to the segment start,
    #    which can be ~250 ms before the speech actually resumes)
    ends_quiet = bool(window.size >= 3 and (window[-3:] <= floor).all())
    if not candidates and ends_quiet:
        # The frames right after the window may already be on the onset ramp, so
        # "after a 30 ms quiet run" is the wrong test here: look for the first
        # frame that is clearly speech and that is sustained for at least 2 of
        # the next 3 frames (rejects a click). The threshold is deliberately low
        # (0.35x the adaptive floor, i.e. ~1 % of peak): Chinese onsets are often
        # soft fricatives ("三", "四", "是") whose energy is a few percent of the
        # vowel peak, and a 5 %-of-peak gate delayed one calibration onset by
        # 103 ms. A frame is only accepted after >= 30 ms of verified silence, so
        # this does not open the door to noise.
        strong = max(floor * 0.35, peak * 0.01)
        fwd = min(n, i1 + int(round(1.0 / hop)))
        for j in range(i1, fwd):
            value = float(env[j])
            if value < strong:
                continue
            follow = env[j + 1:j + 4]
            if sum(1 for v in follow if v > strong) < 2:
                continue
            onset = j * hop
            result.update({
                "method": "silence_forward",
                "onset_s": round(onset, 6),
                "delta_ms": int(round((onset - float(t_start)) * 1000.0)),
                "climb": round(value, 6),
                "accepted": True,
                "note": ("window ended in silence with no onset inside; "
                         "onset found ahead of the window"),
            })
            return result
        # still nothing: the pause never resolves within 1 s -> refuse to guess
        result["rejected_reason"] = "silence_after_window_no_onset"
        return result

    if candidates:
        target = float(t_start)
        best = min(candidates, key=lambda idx: (abs(idx * hop - target), idx))
        onset = best * hop
        result.update({
            "method": "silence_exit",
            "onset_s": round(onset, 6),
            "delta_ms": int(round((onset - target) * 1000.0)),
            "climb": round(float(env[best]) - float(max(env[max(0, best - 3):best + 1].min(), 0.0)), 6),
            "accepted": True,
        })
        return result

    # fallback: strongest climb out of a 80 ms local floor, searched strictly
    # inside the window (a lookback crossing the left edge would compare against
    # unrelated speech and invent a 250 ms-early "onset").
    lookback = 8
    climb_best, climb_idx = 0.0, None
    for i in range(i0 + lookback, i1):
        base = float(env[max(i0, i - lookback):i].min())
        climb = float(env[i]) - base
        if climb > climb_best:
            climb_best, climb_idx = climb, i
    result["climb"] = round(climb_best, 6)
    if climb_idx is not None and climb_best >= peak * min_climb_rel_peak:
        onset = climb_idx * hop
        result.update({
            "method": "energy_rise",
            "onset_s": round(onset, 6),
            "delta_ms": int(round((onset - float(t_start)) * 1000.0)),
            "accepted": True,
        })
    return result


# ------------------------------------------------------------------ onset refine

def refine_onsets(video: Path, anchors: dict[str, Any], *, out_json: Path | None = None,
                  search_s: float = SEARCH_S, hop_s: float = HOP_S,
                  write_cache: bool = True) -> dict[str, Any]:
    """Snap every anchor's ``start_s`` to the measured acoustic onset.

    Every anchor gets a ``refine`` block (Lead/verify contract)::

        {"method": "energy_onset", "delta_ms": -184, "accepted": true,
         "window_ms": 250, "detector": "silence_exit", "onset_s": 6.336, ...}

    ``window_ms`` is the per-side search radius (250 = +/-250 ms); the concrete
    interval actually searched is in ``search_window_s``.

    ``delta_ms`` is always ``onset - asr_start`` (negative = the audio starts
    earlier than ASR claimed). Rejected refinements keep the ASR time and say
    ``accepted: false`` with the detector that failed. Anchors are re-sorted and
    re-numbered afterwards, because a 250 ms snap can otherwise invert two
    adjacent anchors' order.
    """
    video = Path(video)
    doc = json.loads(json.dumps(anchors))           # deep copy, no aliasing
    anchors_list = doc.get("anchors") or []
    if not anchors_list:
        return doc

    env_doc = envelope(video, hop_s=hop_s)
    refined = accepted = 0
    for anchor in anchors_list:
        start = float(anchor["start_s"])
        end = float(anchor["end_s"])
        found = find_onset(env_doc, start, search_s=search_s)
        detail = {
            "method": "energy_onset",
            "delta_ms": int(found["delta_ms"]),
            "accepted": False,
            "window_ms": int(found["window_ms"]),
            "detector": found["method"],
            "onset_s": found["onset_s"],
            "search_window_s": found["search_window_s"],
            "quiet_threshold": found["quiet_threshold"],
            "climb": found["climb"],
            "asr_start_s": round(start, 6),
        }
        if found["accepted"] and found["onset_s"] is not None:
            onset = float(found["onset_s"])
            # never let the snap collapse or invert the anchor window
            if onset < end - 0.03:
                anchor["start_s"] = round(onset, 6)
                detail["accepted"] = True
                accepted += 1
            else:
                detail["rejected_reason"] = "onset_not_before_end"
        else:
            detail["rejected_reason"] = "no_significant_onset"
        anchor["refine"] = detail
        refined += 1

    anchors_list.sort(key=lambda a: (float(a["start_s"]), _KIND_PRIORITY.get(a["kind"], 9)))
    for i, anchor in enumerate(anchors_list):
        anchor["id"] = f"a{i:02d}"

    prov = doc.setdefault("provenance", {})
    prov["refine"] = {
        "method": "energy_onset",
        "window_ms": int(round(search_s * 1000)),   # per-side window, contract says 250
        "hop_ms": int(round(env_doc["hop_s"] * 1000)),
        "energy": "rms",
        "video": str(video.resolve()),
        "video_sha256": media.sha256_file(video),
        "n_anchors": refined,
        "n_accepted": accepted,
        "n_rejected": refined - accepted,
    }
    prov["n_anchors"] = len(anchors_list)
    prov["kinds"] = _kind_counts(anchors_list)

    if write_cache:
        sha = doc.get("transcript_sha256")
        if sha:
            # Separate file from classify(): a refined document must never be
            # returned for a classify() cache hit (its times are no longer pure
            # ASR word spans).
            _write_json(cache_root() / "anchors" / f"{sha}_onset.json", doc)
    if out_json is not None:
        _write_json(Path(out_json), doc)
    return doc


# ---------------------------------------------------------------- truth / metrics

def load_truth(ground_truth_json: str | Path) -> list[dict[str, Any]]:
    """Load SYNTH_KNOWN unit truth as a flat list of events.

    Accepts a fixture ``ground_truth.json`` (``units[]``) or a bare list. Each
    event keeps ``unit_id``, ``text``, ``start_s``/``end_s``, frames, ``role``
    and ``truth``. The times come from the audio assembly, never from ASR.
    """
    doc = json.loads(Path(ground_truth_json).read_text(encoding="utf-8"))
    if isinstance(doc, list):
        units = doc
        kind = "LIST"
    else:
        units = doc.get("units") or doc.get("events") or []
        kind = doc.get("kind") or doc.get("truth") or "UNKNOWN"
    out: list[dict[str, Any]] = []
    for i, unit in enumerate(units):
        if "start_s" not in unit or "end_s" not in unit:
            continue
        out.append({
            "unit_id": unit.get("unit_id") or unit.get("id") or f"u{i:02d}",
            "text": unit.get("text", ""),
            "start_s": round(float(unit["start_s"]), 6),
            "end_s": round(float(unit["end_s"]), 6),
            "start_frame": unit.get("start_frame"),
            "end_frame": unit.get("end_frame"),
            "role": unit.get("role", "plain"),
            "truth": unit.get("truth") or kind,
        })
    return out


def _match_units(anchors: Iterable[dict[str, Any]], units: Sequence[dict[str, Any]],
                 tol_s: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Match each truth unit to the nearest anchor whose text sits at its start.

    Matching is by *text containment*, not by time: an anchor only counts for a
    unit when the normalised anchor text appears at the beginning of the unit
    text (unit text starts with the anchor text). Anchors that start mid-unit
    have no independent truth (assembly truth is unit-level), so they are not
    scored — pretending otherwise would be self-referential.
    """
    pairs: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for unit in units:
        unit_norm = _norm_text(_to_simplified(unit["text"]))
        best = None
        for anchor in anchors:
            anchor_norm = _norm_text(_to_simplified(anchor["text"]))
            if not anchor_norm:
                continue
            at_start = unit_norm.startswith(anchor_norm)
            if not at_start:
                continue
            delta = abs(float(anchor["start_s"]) - float(unit["start_s"]))
            if delta > tol_s:
                continue
            if best is None or delta < best[0]:
                best = (delta, anchor)
        if best is None:
            unmatched.append(unit)
            continue
        delta, anchor = best
        pairs.append({
            "unit_id": unit["unit_id"],
            "unit_text": unit["text"],
            "unit_start_s": float(unit["start_s"]),
            "anchor_id": anchor.get("id"),
            "anchor_kind": anchor.get("kind"),
            "anchor_text": anchor.get("text"),
            "anchor_start_s": float(anchor["start_s"]),
            "error_ms": int(round(abs(float(anchor["start_s"]) - float(unit["start_s"])) * 1000.0)),
            "signed_error_ms": int(round((float(anchor["start_s"]) - float(unit["start_s"])) * 1000.0)),
        })
    return pairs, unmatched


def prefix_matches(anchors: dict[str, Any] | Sequence[dict[str, Any]],
                   units: Sequence[dict[str, Any]], *,
                   tol_s: float = 0.9) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """All (truth unit, anchor) pairs where the anchor text is a *unit prefix*.

    Exposed for ``verify``/calibration scripts: on unit-level assembly truth the
    only independently checkable anchor onset is one whose text starts its unit,
    so this is the event set that a timing metric may legitimately use.
    Returns ``(pairs, unmatched_units)``; see :func:`timing_errors` for the
    error distribution built on top of it.
    """
    anchor_list = anchors.get("anchors", []) if isinstance(anchors, dict) else list(anchors)
    return _match_units(anchor_list, units, tol_s)


def _pct(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def timing_errors(anchors: dict[str, Any] | Sequence[dict[str, Any]],
                  units: Sequence[dict[str, Any]], *, tol_s: float = 0.9) -> dict[str, Any]:
    """Timing error distribution of anchors against independent truth units.

    Returns ``{"samples", "errors_ms", "p50_ms", "p90_ms", "max_ms",
    "matched", "unmatched_units", "n_units"}``. Designed to be fed with a
    SYNTH_KNOWN/HUMAN truth file; with ASR output as "truth" the numbers are
    meaningless and the label must say so.
    """
    anchor_list = anchors.get("anchors", []) if isinstance(anchors, dict) else list(anchors)
    pairs, unmatched = _match_units(anchor_list, units, tol_s)
    errors = [p["error_ms"] for p in pairs]
    signed = [p["signed_error_ms"] for p in pairs]
    return {
        "samples": len(errors),
        "n_units": len(units),
        "errors_ms": errors,
        "signed_errors_ms": signed,
        "p50_ms": int(round(_pct(errors, 50))),
        "p90_ms": int(round(_pct(errors, 90))),
        "max_ms": int(max(errors)) if errors else None,
        "mean_signed_ms": int(round(float(np.mean(signed)))) if signed else None,
        "matched": pairs,
        "unmatched_units": [u["unit_id"] for u in unmatched],
    }


def unit_onset_anchors(transcript: dict[str, Any], units: Sequence[dict[str, Any]], *,
                       tol_s: float = 0.6) -> dict[str, Any]:
    """Build *calibration* anchors at each truth unit's ASR word onset.

    This exists only to measure :func:`refine_onsets` against assembly truth
    with more than the handful of units that happen to start with an anchor.
    For every unit we take the ASR word whose start is nearest the unit onset
    (within ``tol_s``) and emit an anchor covering that word. ``source`` is
    ``SYNTH_KNOWN`` because the *target time* is the assembly truth; the ASR
    word is only used to know where the current estimate is.
    """
    words = transcript.get("words") or []
    anchors: list[dict[str, Any]] = []
    for unit in units:
        target = float(unit["start_s"])
        candidates = [w for w in words
                      if abs(float(w["start_s"]) - target) <= tol_s
                      and float(w["end_s"]) > target]
        if not candidates:
            continue
        word = min(candidates, key=lambda w: (abs(float(w["start_s"]) - target),
                                              float(w["start_s"])))
        anchors.append({
            "id": f"c{len(anchors):02d}",
            "kind": "keyword",
            "text": word["text"],
            "start_s": round(float(word["start_s"]), 6),
            "end_s": round(float(word["end_s"]), 6),
            "confidence": 1.0,
            "source": "SYNTH_KNOWN",
            "segment_id": word["segment_id"],
            "word_ids": [word["id"]],
            "rule": "calibration_unit_onset",
        })
    return {
        "schema": SCHEMA,
        "source_sha256": (transcript.get("source") or {}).get("sha256"),
        "transcript_sha256": transcript_sha256(transcript),
        "duration_s": round(float((transcript.get("source") or {}).get("duration_s") or 0.0), 6),
        "anchors": anchors,
        "provenance": {
            "created_utc": _now(),
            "method": "calibration:asr_word_nearest_to_synth_truth",
            "n_anchors": len(anchors),
            "truth_class": "SYNTH_KNOWN",
        },
    }


# ---------------------------------------------------------------------- helpers

def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(doc: dict[str, Any]) -> list[str]:
    """Return schema problems (empty list == valid). Mirrors INTERFACES §3.3."""
    problems: list[str] = []
    if doc.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}")
    kinds = {"keyword", "number", "list_item", "card", "comparison", "process", "conclusion"}
    sources = {"ASR_ALIGNED", "SYNTH_KNOWN", "HUMAN", "MACHINE_WEAK"}
    seen: set[str] = set()
    prev_start = -1.0
    for i, anchor in enumerate(doc.get("anchors") or []):
        aid = anchor.get("id")
        if not aid or aid in seen:
            problems.append(f"anchors[{i}].id missing or duplicated: {aid!r}")
        seen.add(aid)
        if anchor.get("kind") not in kinds:
            problems.append(f"anchors[{i}].kind invalid: {anchor.get('kind')!r}")
        if anchor.get("source") not in sources:
            problems.append(f"anchors[{i}].source invalid: {anchor.get('source')!r}")
        start, end = float(anchor.get("start_s", -1)), float(anchor.get("end_s", -1))
        if not (0.0 <= start < end):
            problems.append(f"anchors[{i}] bad interval {start}-{end}")
        if start < prev_start - 1e-6:
            problems.append(f"anchors[{i}] not ordered by start_s")
        prev_start = start
        conf = float(anchor.get("confidence", -1))
        if not (0.0 <= conf <= 1.0):
            problems.append(f"anchors[{i}].confidence outside [0,1]: {conf}")
        if not anchor.get("word_ids"):
            problems.append(f"anchors[{i}].word_ids empty")
        if not anchor.get("segment_id"):
            problems.append(f"anchors[{i}].segment_id missing")
        if not _strip_punct(anchor.get("text") or ""):
            problems.append(f"anchors[{i}].text empty")
        refine = anchor.get("refine")
        if refine is not None:
            for key in ("method", "delta_ms", "accepted", "window_ms"):
                if key not in refine:
                    problems.append(f"anchors[{i}].refine.{key} missing")
            if refine.get("method") != "energy_onset":
                problems.append(f"anchors[{i}].refine.method must be energy_onset")
    if not doc.get("duration_s"):
        problems.append("duration_s missing (semantic.decide needs it)")
    return problems
