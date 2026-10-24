"""Model-driven direction: let a language model choose *what* deserves motion.

The deterministic rules in :mod:`semotion.pipeline` are a safety net, not the
product. Which sentence is the point, which three lines form a list, whether a
number matters — those are editorial judgements. This module gives that decision
to a model (or to the host agent) through a small, auditable interface:

1. :func:`build_brief` turns measured facts (aligned transcript, perception,
   anchors, policy envelope) into a compact JSON brief. It contains **no
   timestamps the model may choose** — only anchor ids whose times were measured.
2. :func:`direct` validates the model's JSON against the brief: unknown anchors,
   invented digits, overlapping claims and out-of-envelope holds are rejected,
   and the rejection reason is recorded instead of silently rendered.
3. Any OpenAI-compatible chat endpoint works. With no endpoint configured, the
   brief is still written so the **host agent** (Claude Code) can author the
   decisions itself — that is the native Claude path and needs no API key.

The model never writes geometry. Placement stays with the deterministic,
silhouette-aware solver; the model only decides meaning and grouping.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Sequence

from . import policy as policy_mod

SCHEMA = "semotion.director/1"
BRIEF_SCHEMA = "semotion.director-brief/1"
DECISIONS_SCHEMA = "semotion.llm-decisions/1"

EFFECTS = {"keyword", "card", "list", "stat", "barchart", "compare", "quote"}
EFFECT_ALIASES = {"number": "stat", "data": "stat", "list_item": "list",
                  "contrast": "compare", "chart": "barchart"}
MAX_HOLD_S = 2.0
MIN_HOLD_S = 0.0

# The model is told these; validate_decisions enforces the subset that protects truth.
RULES = [
    "Only reference anchor ids from candidates[]; never invent timestamps.",
    "0–4 elements per 10 seconds; plain narration gets nothing.",
    "One idea per element. 8–16 Chinese characters (2 lines max).",
    "Numbers must appear verbatim in the anchor text (digits or Chinese numerals).",
    "A list needs >=2 anchors, spoken in order; use effect 'list' with anchor_ids in order.",
    "Never animate filler ('大家好', '然后', '对').",
    "Prefer: numbers, enumerations, contrasts, conclusions, superlas.",
    "Do not restate the obvious; do not add three cards in a row.",
    "Return strict JSON only, matching output_schema.",
]


# --------------------------------------------------------------------- helpers
def _sentences(alignment: dict) -> list[dict]:
    segs = alignment.get("segments") or []
    chars = alignment.get("chars") or []
    by_seg: dict[str, list[dict]] = {}
    for c in chars:
        by_seg.setdefault(str(c.get("segment_id") or "_"), []).append(c)
    out = []
    for i, seg in enumerate(segs):
        sid = str(seg.get("id"))
        cs = by_seg.get(sid) or []
        # the ASR segment text is the sentence; chars are only used for timing
        text = str(seg.get("text") or "").strip() or \
            "".join(c.get("text") or "" for c in cs).strip()
        out.append({"index": i, "segment_id": sid, "text": text,
                    "start_s": float(seg.get("start_s") or 0.0),
                    "end_s": float(seg.get("end_s") or 0.0)})
    return out


def _features(text: str) -> dict:
    import re as _re
    has_number = bool(_re.search(r"\d", text)) or bool(
        _re.search(r"[零〇一二两三四五六七八九十百千万]+\s*(成|折|%|％)", text))
    enum = any(text.startswith(m) for m in
               ("第一", "第二", "第三", "首先", "其次", "然后", "最后", "一是", "二是",
                "first", "second", "third", "finally"))
    contrast = any(k in text for k in ("但是", "其实", "不是", "而是", "所以", "因为",
                                       "反而", "however", "but", "because"))
    conclusion = any(k in text for k in ("就是", "意味着", "说明", "结论", "等于"))
    questions = "?" in text or "？" in text
    digits = _re.findall(r"\d+(?:\.\d+)?%?", text)
    return {"has_number": has_number, "digits": digits, "enumeration": enum,
            "contrast": contrast, "conclusion": conclusion, "question": questions,
            "chars": len(text)}


def _context(sentences: list[dict], t: float, window: int = 1) -> list[str]:
    if not sentences:
        return []
    idx = 0
    for i, s in enumerate(sentences):
        if s["start_s"] <= t <= s["end_s"] + 1e-6:
            idx = i
            break
    lo = max(0, idx - window)
    hi = min(len(sentences), idx + window + 1)
    return [s["text"] for s in sentences[lo:hi]]


def _anchor_conf(anchor: dict, char_conf: dict[str, list[float]]) -> float | None:
    """Mean acoustic confidence of the characters behind an anchor (CTC or DP)."""
    vals: list[float] = []
    for wid in anchor.get("word_ids") or []:
        vals.extend(char_conf.get(str(wid), []))
    return round(sum(vals) / len(vals), 3) if vals else None


def _person_side(perception: dict | None, t0: float, t1: float, W: int) -> str:
    """Coarse, cheap description of where the speaker is — context for the model."""
    if not perception:
        return "unknown"
    frames = [f for f in (perception.get("frames") or [])
              if t0 - 0.2 <= float(f.get("t_s") or 0) <= t1 + 0.2]
    boxes = [f.get("person_bbox") for f in frames if f.get("person_bbox")]
    if not boxes:
        return "unknown"
    cx = sum(b[0] + b[2] / 2 for b in boxes) / len(boxes) / max(1, W)
    if cx < 0.4:
        return "left"
    if cx > 0.6:
        return "right"
    return "centre"


# ----------------------------------------------------------------------- brief
def build_brief(*, source: dict, canvas: dict, transcript: dict, alignment: dict,
                anchors: Sequence[dict], perception: dict | None = None,
                policy: policy_mod.Policy | None = None,
                style: str = "smart_balanced") -> dict:
    """Assemble the director brief. Facts only; no free-form timestamps."""
    pol = policy or policy_mod.DEFAULT
    W = int(canvas.get("width") or source.get("width") or 1080)
    sentences = _sentences(alignment)
    char_conf: dict[str, list[float]] = {}
    for c in (alignment.get("chars") or []):
        tid = str(c.get("token_id") or "")
        if tid:
            char_conf.setdefault(tid, []).append(float(c.get("conf") or 0.0))
    anc_out = []
    for a in anchors:
        text = str(a.get("text") or "")
        t0 = float(a.get("start_s") or 0.0)
        t1 = float(a.get("end_s") or t0)
        anc_out.append({
            "id": a.get("id"), "text": text,
            "start_s": round(t0, 3), "end_s": round(t1, 3),
            "duration_s": round(t1 - t0, 2),
            "features": _features(text),
            "context": _context(sentences, t0),
            "speaker_side": _person_side(perception, t0, t1, W),
            "kind_hint": a.get("kind"),
            "timing_confidence": _anchor_conf(a, char_conf),
        })
    reg = (perception or {}).get("regions") or {}
    return {
        "schema": BRIEF_SCHEMA,
        "style": style,
        "language": (transcript or {}).get("language") or "zh",
        "video": {"width": W, "height": int(canvas.get("height") or source.get("height") or 1920),
                  "fps": float(canvas.get("fps") or 25.0),
                  "duration_s": float(source.get("duration_s") or 0.0)},
        "policy": {
            "name": pol.name, "version": pol.version,
            "budget": pol.budget,
            "captions": {k: pol.captions.get(k) for k in
                         ("max_lines", "target_chars_cjk", "max_chars_cjk",
                          "max_chars_latin", "min_display_s")},
            "max_element_types": sorted(EFFECTS),
        },
        "transcript": {
            "sentences": sentences,
            "text": "".join((transcript.get("segments") or [{}])[0].get("text", "")[:0]) or
                    " ".join(s["text"] for s in sentences),
        },
        "scene": {
            "existing_text": (reg.get("existing_text") or [])[:12],
            "quiet_zone_count": len(reg.get("quiet_zones") or []),
            "person_union": reg.get("person_union"),
        },
        "candidates": anc_out,
        "rules": RULES,
        "output_schema": {
            "schema": DECISIONS_SCHEMA,
            "decision_model": "<your model id>",
            "prompt_summary": "<one sentence: why these choices for this clip>",
            "decisions": [{
                "anchor_id": "<first anchor id>",
                "anchor_ids": ["<one or more ids, in speech order>"],
                "effect": "<keyword|card|list|stat|barchart|compare|quote>",
                "emphasis": "high|medium|low",
                "hold_s": "<0.0-1.2>",
                "confidence": "<0-1>",
                "reason": "<one short sentence in the video's language>",
                "params": {"kicker": "<optional>", "emphasis": ["<substring of the anchor text>"],
                           "label": "<optional>", "unit": "<optional>",
                           "bars": [{"label": "<s>", "value": "<number>"}],
                           "left": {"title": "<s>", "items": ["<s>"]},
                           "right": {"title": "<s>", "items": ["<s>"]},
                           "attribution": "<optional>"}
            }],
            "skipped": [{"anchor_id": "<id>", "reason": "<why no motion>"}],
        },
    }


# ------------------------------------------------------------------- validation
class DecisionError(ValueError):
    pass


def _digits(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", text)


def validate_decisions(doc: dict, brief: dict) -> dict:
    """Reject decisions that would put words or numbers in the speaker's mouth."""
    if not isinstance(doc, dict):
        raise DecisionError("decisions must be a JSON object")
    if doc.get("schema") not in (DECISIONS_SCHEMA, None):
        raise DecisionError(f"unexpected schema {doc.get('schema')!r}")
    anchors = {str(c["id"]): c for c in brief.get("candidates") or []}
    seen: set[str] = set()
    out: list[dict] = []
    for i, d in enumerate(doc.get("decisions") or []):
        if not isinstance(d, dict):
            raise DecisionError(f"decision #{i} is not an object")
        ids = [str(x) for x in (d.get("anchor_ids") or [d.get("anchor_id")] or [])]
        ids = [x for x in ids if x and x != "None"]
        if not ids:
            raise DecisionError(f"decision #{i} has no anchor id")
        missing = [x for x in ids if x not in anchors]
        if missing:
            raise DecisionError(f"decision #{i} references unknown anchor(s) {missing}")
        if len(set(ids)) != len(ids):
            raise DecisionError(f"decision #{i} repeats an anchor")
        if any(x in seen for x in ids):
            raise DecisionError(f"decision #{i} reuses an anchor already claimed")
        seen.update(ids)
        effect = EFFECT_ALIASES.get(str(d.get("effect") or "").lower(),
                                    str(d.get("effect") or "").lower())
        if effect not in EFFECTS:
            raise DecisionError(f"decision #{i} has unknown effect {d.get('effect')!r}")
        if effect == "list" and len(ids) < 2:
            raise DecisionError(f"decision #{i}: a list needs two or more anchors")
        ordered = sorted(ids, key=lambda x: float(anchors[x]["start_s"]))
        if ids != ordered:
            raise DecisionError(f"decision #{i}: anchor_ids must be in speech order")
        hold = float(d.get("hold_s") if d.get("hold_s") is not None
                     else (brief.get("policy", {}).get("budget", {}) or {}).get("hold_s", 0.45))
        if not (MIN_HOLD_S <= hold <= MAX_HOLD_S):
            raise DecisionError(f"decision #{i}: hold_s {hold} outside [0, {MAX_HOLD_S}]")
        params = dict(d.get("params") or {})
        # emphasis substrings must exist in the text of the anchors they apply to
        text = "".join(anchors[x].get("text") or "" for x in ids)
        for span in params.get("emphasis") or []:
            if span and str(span) not in text:
                raise DecisionError(
                    f"decision #{i}: emphasis {span!r} does not occur in the anchor text")
        # never allow digits that the speaker did not say. Only scan fields that
        # can end up on screen (colours, enums and easing names are not claims).
        allowed = set(_digits(text))
        on_screen: list[str] = []
        for key in ("label", "unit", "prefix", "suffix", "display", "value", "text",
                    "title", "attribution", "kicker"):
            if params.get(key) not in (None, ""):
                on_screen.append(str(params[key]))
        for bar in params.get("bars") or []:
            if isinstance(bar, dict):
                on_screen.append(str(bar.get("label") or ""))
                on_screen.append(str(bar.get("value") or ""))
        for side in ("left", "right"):
            block = params.get(side) or {}
            if isinstance(block, dict):
                on_screen.append(str(block.get("title") or ""))
                on_screen.extend(str(x) for x in (block.get("items") or []))
        for blob in on_screen:
            for got in _digits(blob):
                if got not in allowed:
                    raise DecisionError(
                        f"decision #{i}: number {got!r} is not present in the anchor text")
        out.append({**d, "effect": effect, "anchor_ids": ids, "hold_s": hold,
                    "confidence": float(d.get("confidence") or 0.7)})
    skipped = [s for s in (doc.get("skipped") or []) if isinstance(s, dict)]
    return {"schema": DECISIONS_SCHEMA,
            "decision_model": str(doc.get("decision_model") or "unknown"),
            "prompt_summary": str(doc.get("prompt_summary") or ""),
            "decisions": out, "skipped": skipped,
            "validation": {"checked": len(out), "anchors_available": len(anchors)}}


# -------------------------------------------------------------------- providers
class ChatClient:
    """Minimal OpenAI-compatible client (no SDK, no hidden retries)."""

    def __init__(self, base_url: str, api_key: str, model: str, *, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "ChatClient | None":
        base = os.environ.get("SM_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        key = os.environ.get("SM_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        model = os.environ.get("SM_LLM_MODEL") or os.environ.get("OPENAI_MODEL")
        if not (base and key and model):
            return None
        return cls(base, key, model)

    def complete(self, messages: list[dict], *, json_mode: bool = True) -> str:
        body = {"model": self.model, "messages": messages, "temperature": 0.2}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=data,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        last: Exception | None = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode())
                return payload["choices"][0]["message"]["content"]
            except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
                last = exc
                time.sleep(0.8 * (attempt + 1))
        raise DecisionError(f"model call failed: {last}")


def _strip_json(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    lo, hi = t.find("{"), t.rfind("}")
    return t[lo:hi + 1] if lo >= 0 and hi > lo else t


def direct(brief: dict, *, client: ChatClient | None = None,
           transport: Callable[[list[dict]], str] | None = None,
           out_dir: str | Path | None = None) -> dict | None:
    """Ask a model for decisions. Returns validated decisions or None (caller falls back)."""
    messages = [
        {"role": "system",
         "content": ("You are a senior motion-graphics director for talking-head video. "
                     "Choose sparingly; silence is a valid answer. The layout and timing are "
                     "handled downstream, and you may only reference the anchor ids given. "
                     "Never invent timestamps, numbers or names. Reply with JSON only.")},
        {"role": "user", "content": json.dumps(brief, ensure_ascii=False)},
    ]
    try:
        if transport is not None:
            raw = transport(messages)
        else:
            cl = client or ChatClient.from_env()
            if cl is None:
                return None
            raw = cl.complete(messages)
        doc = json.loads(_strip_json(raw))
        validated = validate_decisions(doc, brief)
    except Exception as exc:  # noqa: BLE001 - recorded, never fatal
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / "director_error.json").write_text(
                json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), encoding="utf-8")
        return None
    validated["director"] = {"schema": SCHEMA, "provider": "chat-completions",
                             "model": (client.model if client else
                                       os.environ.get("SM_LLM_MODEL") or "injected")}
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "decisions.generated.json").write_text(
            json.dumps(validated, ensure_ascii=False, indent=2), encoding="utf-8")
    return validated


def write_brief(brief: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(brief, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
