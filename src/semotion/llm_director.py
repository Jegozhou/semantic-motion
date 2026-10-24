"""LLM director — let a language model choose the effect, keep the mechanics here.

Contract: ``docs/REMOTION.md`` §4 (LLM director) and §5 (caption track).

Division of labour (spec §3.3 / §3.5 / §3.6):

* the LLM only answers **what** to show (which anchor, which effect, emphasis,
  reason, confidence, optional hold / params / list rows);
* **when / where / how fast** stay deterministic: the time window is derived from
  the validated anchor onsets by this module, frames are derived later by the IR,
  and the layout is owned by ``design.py``;
* the LLM output is *data*: it is parsed as JSON, checked against a closed schema
  and rejected on the first violation. Nothing here evaluates strings, imports
  model-authored code, or shells out. Any unknown decision field is refused, so a
  model cannot smuggle ``code`` / ``script`` / ``cmd`` payloads into the IR.

Public API (``docs/REMOTION.md`` §4):

    build_prompt(transcript, anchors, vision, canvas) -> str
    validate_decisions(doc, anchors, transcript) -> list[str]    # [] == legal
    to_plan(doc, anchors, canvas) -> dict                        # semotion.plan/1
    caption_cues(transcript, canvas, *, max_chars=22, max_lines=2) -> list[dict]

Plus small helpers used by tests / tooling:

    load_decisions(path) / dump_decisions(doc, path)
    default_decisions_for(anchors)   # deterministic test fixture, NOT an LLM
    default_max_chars(canvas)        # 22 for 9:16, 30 for 16:9

The eight validation rules are the ones in ``docs/REMOTION.md`` §4. A few extra
schema guards are added because this module is the security boundary: the doc
and every decision are closed-key JSON objects, strings must be non-empty, and
``content.text`` / ``content.items`` must be traceable to the transcript or the
anchor text so the director cannot invent facts (INTERFACES.md §3.5 invariant 8).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA_DECISIONS = "semotion.llm-decisions/1"
SCHEMA_PLAN = "semotion.plan/1"
STYLE = "smart_balanced"
FALLBACK_MODEL = "rules-fallback"

#: The only effect types the renderer implements; adding one is a contract change.
EFFECT_WHITELIST = ("keyword", "number", "list", "card")

#: Semantics handed to the LLM in the prompt (and re-used by the fallback).
EFFECT_SEMANTICS: dict[str, str] = {
    "keyword": (
        "短语级强调。适合一句里最想让人记住的 2-8 个字（论点、冲突、关键名词）；"
        "不适合整句复述，不适合口头禅"
    ),
    "number": (
        "精确数字。适合可量化结论（九成 / 70% / 三个）、计数/增长；"
        "数字必须原样可解析，不得改写、不得四舍五入丢失量级"
    ),
    "list": (
        "步骤 / 并列项顺序揭示。适合“第一…第二…第三…”或同一句式列举；"
        "一个 list 元素可合并多个 list_item 锚点（用 anchor_ids 给出顺序）"
    ),
    "card": (
        "信息卡片。适合一句完整的解释、结论或因果（≥6 字）；"
        "空间占用最大，普通叙述不要用"
    ),
}

#: Rhythm budget (REMOTION.md §4 rule 5).
MAX_ELEMENTS_PER_10S = 4
MAX_COVERAGE = 0.58
MAX_HOLD_S = 0.8
DEFAULT_HOLD_S = 0.30

#: Packing gap before the next element (══ rule 3's ``next_start - 0.04``).
PACK_GAP_S = 0.04

#: A derived window shorter than this cannot legally carry an effect (rule 3/4).
VALIDATION_MIN_WINDOW_S = 1e-6
#: Windows shorter than this are dropped by ``to_plan`` as unusable.
PLAN_MIN_WINDOW_S = 0.10

#: INTERFACES.md §3.3: anchors below this confidence must not drive precise motion.
MIN_ANCHOR_CONFIDENCE = 0.60

EMPHASIS_LEVELS = ("low", "medium", "high")

#: Closed key set for one decision; anything else is rejected (safety boundary).
DECISION_ALLOWED_KEYS = frozenset({
    "id", "anchor_id", "anchor_ids", "effect", "emphasis", "reason",
    "hold_s", "confidence", "params", "content", "window", "collision_ok",
})
CONTENT_ALLOWED_KEYS = frozenset({"text", "items", "numeric"})

#: Caption defaults (REMOTION.md §5); 16:9 callers pass ``max_chars=30``.
CAPTION_DEFAULT_MAX_CHARS = 22
CAPTION_LANDSCAPE_MAX_CHARS = 30
CAPTION_PAUSE_BREAK_S = 0.40

_CLAUSE_PUNCT = "，,、；;：:"
_TERMINAL_PUNCT = "。！？.!?"
_CLOSERS = "，。、；！？,.!?;：:）】」』》〉…%％"
_SPACE_CHARS = " \t\u3000\n"
#: Prefix marking a hard-split continuation of the same token (never spaced).
_CONT = "\u0000"

_LATIN_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[’'\-][A-Za-z0-9]+)*")


# --------------------------------------------------------------------------- utils
def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def _anchor_index(anchors: dict | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for anchor in (anchors or {}).get("anchors") or []:
        if isinstance(anchor, dict) and anchor.get("id"):
            out[str(anchor["id"])] = anchor
    return out


def _fps(canvas: dict | None, transcript: dict | None = None) -> float:
    for source in (canvas or {}, (transcript or {}).get("source") or {}):
        value = _as_float(source.get("fps"))
        if value and value > 0:
            return value
    return 25.0


def _duration(anchors: dict | None = None, canvas: dict | None = None,
              transcript: dict | None = None) -> float:
    """Runtime in seconds: anchors > canvas > transcript > last anchor end."""
    for source in (anchors or {}, canvas or {}, (transcript or {}).get("source") or {}):
        value = _as_float(source.get("duration_s"))
        if value and value > 0:
            return value
    ends = [_as_float(a.get("end_s")) for a in (anchors or {}).get("anchors") or []
            if isinstance(a, dict)]
    ends = [v for v in ends if v is not None and v > 0]
    if ends:
        return max(ends)
    words = (transcript or {}).get("words") or []
    wends = [_as_float(w.get("end_s")) for w in words if isinstance(w, dict)]
    wends = [v for v in wends if v is not None and v > 0]
    return max(wends) if wends else 1.0


def _simplified(text: Any) -> str:
    """Display copy in Simplified Chinese (same folding as ``semantic``)."""
    try:
        from .semantic import to_simplified
        return to_simplified(str(text))
    except Exception:  # pragma: no cover - semantic is a sibling module
        return str(text)


def _norm_for_match(text: str) -> str:
    """Fold text for substring checks: simplified, alphanumeric only, lower-case."""
    folded = _simplified(text or "").lower()
    return "".join(ch for ch in folded if ch.isalnum())


def _parse_numeric(text: str) -> dict | None:
    """Locate a numeric token exactly like the number effect will (shared parser)."""
    from .effects.number import parse_numeric  # local import: keeps import graph light
    return parse_numeric(text or "")


def _decision_model(doc: dict) -> str:
    raw = str(doc.get("decision_model") or "").strip() or "unknown"
    if raw.startswith("llm:") or raw.startswith("rules-"):
        return raw
    return f"llm:{raw}"


def _frame(seconds: float, fps: float) -> int:
    return int(round(float(seconds) * fps))


def _transcript_corpus(transcript: dict | None) -> str:
    """Everything the transcript actually said, flattened for fabrication checks."""
    if not transcript:
        return ""
    pieces: list[str] = []
    for segment in transcript.get("segments") or []:
        if isinstance(segment, dict):
            pieces.append(str(segment.get("text") or ""))
            words = [str(w.get("text") or "") for w in transcript.get("words") or []
                     if isinstance(w, dict) and w.get("segment_id") == segment.get("id")]
            pieces.append("".join(words))
    pieces.append("".join(str(w.get("text") or "") for w in transcript.get("words") or []
                         if isinstance(w, dict)))
    return "".join(pieces)


# ----------------------------------------------------------------------- validation
def _record_label(record: dict) -> str:
    return f"decisions[{record['index']}]({record.get('anchor_id')})"


def _collision_allowed(a: dict, b: dict, pairs: set[frozenset[str]]) -> bool:
    if a.get("collision_ok") or b.get("collision_ok"):
        return True
    return frozenset((str(a.get("anchor_id")), str(b.get("anchor_id")))) in pairs


def _parse_collision_pairs(doc: dict, errors: list[str]) -> set[frozenset[str]]:
    pairs: set[frozenset[str]] = set()
    raw = doc.get("collision_ok")
    if raw is None:
        return pairs
    if not isinstance(raw, list):
        errors.append("doc.collision_ok: expected a list of [anchor_id, anchor_id] pairs")
        return pairs
    for i, pair in enumerate(raw):
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not all(isinstance(x, str) and x for x in pair)):
            errors.append(f"doc.collision_ok[{i}]: expected [anchor_id, anchor_id]")
            continue
        pairs.add(frozenset(pair))
    return pairs


def _content_errors(key: str, content: Any, anchor_texts: Sequence[str],
                    transcript: dict | None) -> tuple[list[str], dict]:
    """Validate ``content`` and return (errors, normalized content)."""
    errors: list[str] = []
    if content is None:
        return errors, {}
    if not isinstance(content, dict):
        return [f"{key}.content: expected an object"], {}
    unknown = sorted(set(content) - CONTENT_ALLOWED_KEYS)
    if unknown:
        errors.append(f"{key}.content: unknown field(s) {unknown}; "
                      f"allowed: {sorted(CONTENT_ALLOWED_KEYS)}")
    out: dict[str, Any] = {}
    text = content.get("text")
    if text is not None:
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{key}.content.text: non-empty string required")
        else:
            out["text"] = text
            if transcript is not None:
                corpus = _norm_for_match(_transcript_corpus(transcript))
                needles = [_norm_for_match(t) for t in anchor_texts]
                needle = _norm_for_match(text)
                supported = (needle in corpus
                             or any(needle in n for n in needles if n))
                if needle and not supported:
                    errors.append(
                        f"{key}.content.text={text!r} is not found in the anchors or the "
                        f"transcript (no invented copy)")
    items = content.get("items")
    if items is not None:
        if (not isinstance(items, list) or not items
                or any(not isinstance(x, str) or not x.strip() for x in items)):
            errors.append(f"{key}.content.items: non-empty list of non-empty strings required")
        elif len(items) > 6:
            errors.append(f"{key}.content.items: at most 6 rows, got {len(items)}")
        else:
            out["items"] = list(items)
            if transcript is not None:
                corpus = _norm_for_match(_transcript_corpus(transcript))
                corpus += "".join(_norm_for_match(t) for t in anchor_texts)
                for row in items:
                    needle = _norm_for_match(row)
                    if needle and needle not in corpus:
                        errors.append(
                            f"{key}.content.items: row {row!r} is not found in the "
                            f"transcript (no invented copy)")
    numeric = content.get("numeric")
    if numeric is not None:
        if isinstance(numeric, dict):
            declared = _as_float(numeric.get("value"))
        else:
            declared = _as_float(numeric)
        if declared is None:
            errors.append(f"{key}.content.numeric: expected a number or {{'value': number}}")
        else:
            out["numeric"] = declared
    return errors, out


def _build_records(doc: dict, anchors: dict,
                   transcript: dict | None = None) -> tuple[list[dict], list[str], float]:
    """Normalize + schema-check every decision; return (records, errors, duration)."""
    errors: list[str] = []
    anchors_doc = anchors if isinstance(anchors, dict) else {}
    index = _anchor_index(anchors_doc)
    duration = _duration(anchors_doc, None, transcript)
    decisions = doc.get("decisions")
    if not isinstance(decisions, list):
        return [], ["doc.decisions: expected a list of decision objects"], duration

    records: list[dict] = []
    for i, decision in enumerate(decisions):
        key = f"decisions[{i}]"
        if not isinstance(decision, dict):
            errors.append(f"{key}: expected an object")
            continue
        unknown = sorted(set(decision) - DECISION_ALLOWED_KEYS)
        if unknown:
            errors.append(f"{key}: unknown field(s) {unknown}; allowed: "
                          f"{sorted(DECISION_ALLOWED_KEYS)}")

        anchor_id = decision.get("anchor_id")
        refs: list[str] = []
        if not isinstance(anchor_id, str) or not anchor_id.strip():
            errors.append(f"{key}.anchor_id: non-empty string required")
        else:
            refs = [anchor_id]
        raw_ids = decision.get("anchor_ids")
        if raw_ids is not None:
            if (not isinstance(raw_ids, list) or not raw_ids
                    or any(not isinstance(x, str) or not x.strip() for x in raw_ids)):
                errors.append(f"{key}.anchor_ids: non-empty list of anchor id strings")
            else:
                refs = list(raw_ids)
                if isinstance(anchor_id, str) and anchor_id and anchor_id not in refs:
                    errors.append(f"{key}.anchor_ids: anchor_id {anchor_id!r} is not listed "
                                  f"in anchor_ids")
        if len(set(refs)) != len(refs):
            errors.append(f"{key}.anchor_ids: duplicate anchor ids {refs}")

        missing = [r for r in refs if r not in index]
        for ref in missing:
            errors.append(f"{key}.anchor_id: {ref!r} does not exist in anchors")
        resolved = [index[r] for r in refs if r in index]

        effect = decision.get("effect")
        if effect not in EFFECT_WHITELIST:
            errors.append(f"{key}({anchor_id}).effect: {effect!r} is not in the whitelist "
                          f"{list(EFFECT_WHITELIST)}")

        emphasis = decision.get("emphasis", "medium")
        if emphasis not in EMPHASIS_LEVELS:
            errors.append(f"{key}.emphasis: {emphasis!r} not in {list(EMPHASIS_LEVELS)}")

        reason = decision.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"{key}.reason: non-empty string required")
        elif len(reason) > 300:
            errors.append(f"{key}.reason: at most 300 characters, got {len(reason)}")

        hold = decision.get("hold_s", DEFAULT_HOLD_S)
        hold_value = _as_float(hold)
        if hold_value is None:
            errors.append(f"{key}.hold_s: numeric seconds required, got {hold!r}")
            hold_value = DEFAULT_HOLD_S
        elif hold_value < 0 or hold_value > MAX_HOLD_S + 1e-9:
            errors.append(f"{key}.hold_s: {hold_value} outside [0, {MAX_HOLD_S}] "
                          f"(tail hold may not exceed {MAX_HOLD_S}s)")

        confidence = decision.get("confidence")
        conf_value = _as_float(confidence)
        if conf_value is None or not (0.0 <= conf_value <= 1.0):
            errors.append(f"{key}.confidence: number in [0, 1] required, got {confidence!r}")

        params = decision.get("params", {})
        if not isinstance(params, dict):
            errors.append(f"{key}.params: expected an object")
            params = {}
        else:
            for name, value in params.items():
                if not isinstance(value, (str, int, float, bool)) and value is not None:
                    errors.append(f"{key}.params.{name}: must be a JSON scalar (data only)")
        if params.get("avoid_face") is not True:
            errors.append(f"{key}.params.avoid_face: must be true (face-safe by default; "
                          f"layout may not be trusted to dodge a face by itself)")

        content_errors, content = _content_errors(
            key, decision.get("content"), [str(a.get("text") or "") for a in resolved],
            transcript)
        errors.extend(content_errors)

        for ref, anchor in zip(refs, resolved):
            kind = str(anchor.get("kind") or "plain")
            if kind == "plain":
                errors.append(f"{key}: anchor {ref!r} has kind=plain; plain speech must "
                              f"stay unanimated")
            conf = _as_float(anchor.get("confidence"))
            if conf is not None and conf < MIN_ANCHOR_CONFIDENCE:
                errors.append(f"{key}: anchor {ref!r} confidence {conf:.2f} < "
                              f"{MIN_ANCHOR_CONFIDENCE:.2f}; use a conservative effect or skip")

        # rule 6: numbers must be exact and match any declared value
        if effect == "number" and resolved:
            source_text = str(content.get("text") or resolved[0].get("text") or "")
            parsed = _parse_numeric(source_text)
            if not parsed:
                errors.append(f"{key}: effect=number but no parseable numeric token in "
                              f"{source_text!r}")
            elif content.get("numeric") is not None:
                if abs(float(content["numeric"]) - float(parsed["value"])) > 1e-6:
                    errors.append(
                        f"{key}.content.numeric: {content['numeric']} != parsed "
                        f"{parsed['value']} (numbers must never be rewritten)")

        starts = [_as_float(a.get("start_s")) for a in resolved]
        ends = [_as_float(a.get("end_s")) for a in resolved]
        window_explicit = decision.get("window")
        if window_explicit is not None and not isinstance(window_explicit, dict):
            errors.append(f"{key}.window: expected an object; time windows are derived "
                          f"from anchors, do not send one")
            window_explicit = None
        if window_explicit is not None:
            unknown_win = sorted(set(window_explicit) - {"start_s", "end_s"})
            if unknown_win:
                errors.append(f"{key}.window: unknown field(s) {unknown_win}")

        if not resolved or any(v is None for v in starts) or any(v is None for v in ends):
            continue
        record = {
            "index": i,
            "anchor_id": str(anchor_id) if isinstance(anchor_id, str) else refs[0],
            "anchor_ids": refs,
            "effect": str(effect),
            "emphasis": str(emphasis),
            "reason": str(reason or ""),
            "hold_s": float(hold_value),
            "confidence": float(conf_value or 0.0),
            "params": params,
            "content": content,
            "start": min(float(v) for v in starts if v is not None),
            "anchor_end": max(float(v) for v in ends if v is not None),
            "window_explicit": window_explicit,
            "collision_ok": decision.get("collision_ok") is True,
        }
        if record["start"] < 0 or record["start"] >= duration + 1e-9:
            errors.append(f"{key}: anchor onset {record['start']:.3f}s outside "
                          f"[0, {duration:.3f}s)")
        records.append(record)
    return records, errors, duration


def _packed_windows(order: Sequence[dict], duration: float) -> dict[int, float]:
    """REMOTION.md §4 rule 3: ``end = min(anchor_end + hold, next_start - .04, dur)``."""
    out: dict[int, float] = {}
    for i, record in enumerate(order):
        next_start = float(order[i + 1]["start"]) if i + 1 < len(order) else duration
        out[int(record["index"])] = min(record["anchor_end"] + record["hold_s"],
                                        next_start - PACK_GAP_S,
                                        duration)
    return out


def _coverage(windows: Iterable[tuple[float, float]], duration: float) -> float:
    """Union length / duration (never double counts overlapping windows)."""
    spans = sorted((float(s), float(e)) for s, e in windows if float(e) > float(s))
    if not spans or duration <= 0:
        return 0.0
    total = 0.0
    cur_start, cur_end = spans[0]
    for start, end in spans[1:]:
        if start <= cur_end + 1e-9:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total / duration


def _density_ok(starts: Sequence[float]) -> bool:
    for start in starts:
        count = sum(1 for t in starts if start - 1e-9 <= t < start + 10.0)
        if count > MAX_ELEMENTS_PER_10S:
            return False
    return True


def _shot_cuts(vision: dict | None) -> list[float]:
    shots = [s for s in (vision or {}).get("shots") or []
             if isinstance(s, dict) and _as_float(s.get("end_s")) is not None]
    cuts: list[float] = []
    for shot in shots[1:]:
        start = _as_float(shot.get("start_s"))
        if start is not None:
            cuts.append(start)
    return cuts


def validate_decisions(doc: dict, anchors: dict, transcript: dict | None = None,
                       *, vision: dict | None = None) -> list[str]:
    """Check an LLM decision document. Empty list = legal, otherwise reject.

    Rules 1-8 of ``docs/REMOTION.md`` §4 are enforced here (effect whitelist,
    anchor existence, anchor-derived windows + hold cap, no overlaps with the same
    moment, rhythm budget, exact numbers, no ``kind=plain`` anchors, face-safety
    flag). ``vision`` is optional and only adds the shot-cut guard; the frozen
    contract signature does not require it.
    """
    errors: list[str] = []
    if not isinstance(doc, dict):
        return ["doc: expected a JSON object"]
    if doc.get("schema") != SCHEMA_DECISIONS:
        errors.append(f"doc.schema: expected {SCHEMA_DECISIONS!r}, got "
                      f"{doc.get('schema')!r}")
    model = doc.get("decision_model")
    if not isinstance(model, str) or not model.strip():
        errors.append("doc.decision_model: non-empty string required (frozen into the IR)")
    summary = doc.get("prompt_summary")
    if not isinstance(summary, str) or not summary.strip():
        errors.append("doc.prompt_summary: non-empty string required (frozen into the IR)")

    collision_pairs = _parse_collision_pairs(doc, errors)
    records, record_errors, duration = _build_records(doc, anchors, transcript)
    errors.extend(record_errors)

    order = sorted(records, key=lambda r: (r["start"], r["index"]))
    windows = _packed_windows(order, duration)

    for i, record in enumerate(order):
        key = _record_label(record)
        start, end = float(record["start"]), windows[int(record["index"])]
        if end - start < VALIDATION_MIN_WINDOW_S:
            neighbour = order[i + 1] if i + 1 < len(order) else None
            exempt = neighbour is not None and _collision_allowed(record, neighbour,
                                                                  collision_pairs)
            if exempt:
                continue
            where = (f"next element {neighbour.get('anchor_id')} starts at "
                     f"{neighbour['start']:.3f}s" if neighbour else "duration exhausted")
            errors.append(f"{key}: derived window is empty (anchor start {start:.3f}s, "
                          f"{where}) — overlapping onsets; skip one element")
            continue

        declared = record.get("window_explicit")
        if isinstance(declared, dict):
            for field, expected in (("start_s", start), ("end_s", end)):
                if field not in declared:
                    continue
                value = _as_float(declared[field])
                if value is None:
                    errors.append(f"{key}.window.{field}: numeric seconds required")
                elif abs(value - expected) > 5e-4:
                    errors.append(f"{key}.window.{field}={value:.3f}s != anchor-derived "
                                  f"{expected:.3f}s (the LLM must not invent times)")
                if field == "end_s" and value is not None and value > duration + 1e-6:
                    errors.append(f"{key}.window.end_s={value:.3f}s exceeds canvas duration "
                                  f"{duration:.3f}s")

        if vision:
            for cut in _shot_cuts(vision):
                if start + 1e-9 < cut < end - 1e-9:
                    errors.append(f"{key}: window [{start:.3f}, {end:.3f}] straddles the "
                                  f"shot cut at {cut:.3f}s (no effect across a cut)")

    for i in range(len(order) - 1):
        left, right = order[i], order[i + 1]
        if windows[int(left["index"])] > right["start"] + 1e-9:
            if _collision_allowed(left, right, collision_pairs):
                continue
            errors.append(f"{_record_label(left)} / {_record_label(right)}: overlapping "
                          f"windows ({windows[int(left['index'])]:.3f}s > "
                          f"{right['start']:.3f}s); only one element per moment")

    starts = [float(r["start"]) for r in order]
    for start in starts:
        count = sum(1 for t in starts if start - 1e-9 <= t < start + 10.0)
        if count > MAX_ELEMENTS_PER_10S:
            errors.append(f"rhythm budget: {count} elements start within 10s from "
                          f"{start:.2f}s (max {MAX_ELEMENTS_PER_10S})")
    coverage = _coverage([(float(r["start"]), windows[int(r["index"])]) for r in order],
                         duration)
    if coverage > MAX_COVERAGE + 1e-9:
        errors.append(f"coverage budget: overlays cover {coverage:.1%} of {duration:.2f}s "
                      f"(max {MAX_COVERAGE:.0%})")

    skipped = doc.get("skipped")
    if skipped is not None:
        if not isinstance(skipped, list):
            errors.append("doc.skipped: expected a list")
        else:
            index = _anchor_index(anchors)
            for i, item in enumerate(skipped):
                key = f"doc.skipped[{i}]"
                if not isinstance(item, dict):
                    errors.append(f"{key}: expected an object")
                    continue
                unknown = sorted(set(item) - {"anchor_id", "reason"})
                if unknown:
                    errors.append(f"{key}: unknown field(s) {unknown}")
                aid = item.get("anchor_id")
                if not isinstance(aid, str) or not aid:
                    errors.append(f"{key}.anchor_id: non-empty string required")
                elif aid not in index:
                    errors.append(f"{key}.anchor_id: {aid!r} does not exist in anchors")
                reason = item.get("reason")
                if not isinstance(reason, str) or not reason.strip():
                    errors.append(f"{key}.reason: non-empty string required")
    return errors


# ---------------------------------------------------------------------------- prompt
def build_prompt(transcript: dict, anchors: dict, vision: dict, canvas: dict) -> str:
    """Chinese prompt for the LLM director (data-only output, no timestamps)."""
    transcript = transcript or {}
    anchors = anchors or {}
    vision = vision or {}
    canvas = canvas or {}
    fps = _fps(canvas, transcript)
    duration = _duration(anchors, canvas, transcript)
    width = int(canvas.get("width") or (transcript.get("source") or {}).get("width") or 0)
    height = int(canvas.get("height") or (transcript.get("source") or {}).get("height") or 0)
    if not width or not height:
        width = int((vision.get("source") or {}).get("width") or 0)
        height = int((vision.get("source") or {}).get("height") or 0)
    orientation = "9:16 竖屏" if height >= width else "16:9 横屏"
    language = transcript.get("language") or anchors.get("language") or "zh"

    rows = ["| anchor_id | kind | text | start_s | end_s | confidence |", "|---|---|---|---|---|---|"]
    for anchor in sorted((a for a in anchors.get("anchors") or [] if isinstance(a, dict)),
                         key=lambda a: _as_float(a.get("start_s")) or 0.0):
        kind = str(anchor.get("kind") or "plain")
        text = str(anchor.get("text") or "").replace("|", "\\|")
        mark = "plain(禁止)" if kind == "plain" else kind
        rows.append(f"| {anchor.get('id')} | {mark} | {text} | "
                    f"{_as_float(anchor.get('start_s')) or 0.0:.2f} | "
                    f"{_as_float(anchor.get('end_s')) or 0.0:.2f} | "
                    f"{_as_float(anchor.get('confidence')) or 0.0:.2f} |")

    segment_lines = []
    for segment in transcript.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        segment_lines.append(
            f"- [{segment.get('start_s', 0.0):.2f}-{segment.get('end_s', 0.0):.2f}] "
            f"{segment.get('text') or ''}")
    if not segment_lines:
        segment_lines.append("（无分段转写，仅使用锚点）")

    face_union = None
    tracks = [t for t in vision.get("face_tracks") or [] if isinstance(t, dict)]
    if tracks:
        face_union = tracks[0].get("bbox_union")
    face_line = ("画面中检测到人脸（规模最大的人脸框 "
                 f"{face_union}，共 {len(vision.get('faces') or [])} 次采样）；"
                 "所有元素的 params.avoid_face 必须为 true") if face_union else \
                "本片未检测到人脸；为避免口径不一，params.avoid_face 仍必须为 true"
    cuts = _shot_cuts(vision)
    cut_line = ("、".join(f"{c:.2f}s" for c in cuts) if cuts else "无（单镜头）")
    zones = vision.get("safe_zones") or {}

    schema_block = """{
  "schema": "semotion.llm-decisions/1",
  "decision_model": "<你实际使用的模型名>",
  "prompt_summary": "<20-60 字，概括你这次的选择原则>",
  "decisions": [
    {
      "anchor_id": "a03",
      "anchor_ids": ["a03"],
      "effect": "number",
      "emphasis": "high",
      "reason": "<为什么这个锚点值得加、为什么用这一类>",
      "hold_s": 0.35,
      "confidence": 0.86,
      "params": {"accent": "#F2B233", "enter": "count_up", "avoid_face": true},
      "content": {"text": "<可选，必须是锚点或转写原文；不写则用锚点原文>"}
    }
  ],
  "skipped": [{"anchor_id": "a01", "reason": "<为什么不加>"}]
}"""

    return f"""你是口播视频的动效导演（LLM director）。你的唯一任务：从下面的语义锚点里判断**哪些值得加动效、用哪一类**。时间窗口、坐标、字号、帧号全部由系统按锚点计算，你不负责也不允许输出它们。

## 1. 硬规则（违反任意一条，系统会拒绝整份输出）
1. `effect` 只能取白名单里的四个值：`keyword`、`number`、`list`、`card`；不要发明新动效，不要输出 `none`（想不加就跳过）。
2. 只能引用锚点清单里真实存在的 `anchor_id`；**不要写任何时间字段**（start_s / end_s / window / frames 都不行），时间只能来自锚点。
3. 每个元素 `hold_s` ≤ {MAX_HOLD_S} 秒（尾部停留，用于读完文字）。系统用
   `end = min(锚点结束 + hold_s, 下一个元素起点 - {PACK_GAP_S}, 片长)` 生成窗口，起点就是锚点起点。
4. 节奏预算：**每 10 秒最多 {MAX_ELEMENTS_PER_10S} 个元素**（滑动窗口）；所有窗口并集占片长的**覆盖率 ≤ {MAX_COVERAGE:.0%}**。超预算的元素会被判非法，所以宁可少选。
5. `number` 必须能精确解析出数值（`九成`、`70%`、`三个` 可以；改写数字、四舍五入、换成"很多"都不行）；如果给 `content.numeric`，必须与原文数值完全一致。
6. 不得引用 `kind=plain` 的锚点；锚点 `confidence` < 0.60 时不得驱动精确动效（跳过）。
7. 安全：`params.avoid_face` 必须为 `true`；不得遮挡人脸与字幕安全区（布局由系统处理，你只需声明）。
8. 不得跨镜头切换：元素窗口不能跨越切点 {cut_line}；如果锚点本身横跨切点，换一个锚点或跳过。

## 2. 四类动效（白名单）与适用语义
- **keyword**：{EFFECT_SEMANTICS['keyword']}。
- **number**：{EFFECT_SEMANTICS['number']}。
- **list**：{EFFECT_SEMANTICS['list']}。合并多个步骤时用 `anchor_ids` 按口播顺序全部列出，并用 `content.items` 给出每一行的短语（每行必须来自转写原文的连续词，不要自己造词）。
- **card**：{EFFECT_SEMANTICS['card']}。

**宁可不加**：普通叙述、口头禅、与已选元素重复的表达都不要加；一份 `decisions` 为空的输出也是合法的。平均每 10 秒 1.5-2.5 个元素就比较克制；"每个锚点都加"是最差的答案。

## 3. 输出格式（只输出一个 JSON 对象；不要 Markdown、不要注释、不要代码、不要命令、不要解释文字）
```json
{schema_block}
```
字段要求：`decisions[]` 必填 `anchor_id`、`effect`、`emphasis`（low|medium|high）、`reason`（中文，具体说明语义理由）、`hold_s`（0-0.8）、`confidence`（0-1）、`params.avoid_face=true`；`params` 里还可以给 `accent`（十六进制色）与 `enter`（`fade_up` / `spring_pop` / `count_up` / `stagger_rows` 之一）。`content` 可省略；列表动效建议给 `content.items`。`skipped` 建议列出你主动放弃的锚点及理由（必须引用真实 anchor_id）。

## 4. 素材上下文
- 画布：{width}×{height}（{orientation}），{fps:g} fps，片长 {duration:.2f}s，语言 {language}。
- {face_line}。
- 镜头切点：{cut_line}。
- 安全区（归一化 x,y,w,h）：{json.dumps(zones, ensure_ascii=False)}。

### 语义锚点（时间只用于你理解上下文，禁止写进输出）
{chr(10).join(rows)}

### 转写分段（帮助你判断上下文；同样不可改写事实）
{chr(10).join(segment_lines)}

现在请输出这份 JSON。
"""


# ------------------------------------------------------------------------------ plan
def _content_for_plan(record: dict, index: dict[str, dict]) -> dict:
    effect = record["effect"]
    content = record.get("content") or {}
    first = index.get(record["anchor_ids"][0]) or {}
    raw_text = content.get("text")
    text = (_simplified(raw_text) if isinstance(raw_text, str) and raw_text.strip()
            else _simplified(first.get("text") or ""))
    items = None
    if effect == "list":
        raw_items = content.get("items")
        if isinstance(raw_items, list) and raw_items:
            items = [_simplified(str(row)) for row in raw_items]
        else:
            items = [_simplified(str((index.get(ref) or {}).get("text") or ""))
                     for ref in record["anchor_ids"]]
    return {"text": text, "items": items}


def _pack_for_plan(records: Sequence[dict], duration: float
                   ) -> tuple[list[dict], dict[int, float], list[dict]]:
    """Pack windows, dropping elements whose window is unusable; stable order."""
    dropped: list[dict] = []
    kept = list(records)
    for _ in range(8):
        order = sorted(kept, key=lambda r: (r["start"], r["index"]))
        windows = _packed_windows(order, duration)
        bad = [r for r in order
               if windows[int(r["index"])] - r["start"] < PLAN_MIN_WINDOW_S]
        if not bad:
            return order, windows, dropped
        bad_ids = {int(r["index"]) for r in bad}
        dropped.extend(bad)
        kept = [r for r in kept if int(r["index"]) not in bad_ids]
    order = sorted(kept, key=lambda r: (r["start"], r["index"]))
    return order, _packed_windows(order, duration), dropped


def to_plan(doc: dict, anchors: dict, canvas: dict, *,
            transcript: dict | None = None) -> dict:
    """Turn a validated LLM document into a ``semotion.plan/1``.

    Raises ``ValueError`` when the document is illegal: a rejected decision is
    never repaired silently (REMOTION.md §4). Pass ``transcript`` to also apply
    the "no invented copy" check (optional keyword so the frozen signature from
    the contract keeps working unchanged).
    """
    errors = validate_decisions(doc, anchors, transcript)
    if errors:
        raise ValueError("invalid LLM decisions: " + "; ".join(errors[:8]))

    index = _anchor_index(anchors)
    duration = _duration(anchors, canvas)
    records, _, _ = _build_records(doc, anchors, transcript)
    order, windows, dropped = _pack_for_plan(records, duration)
    model = _decision_model(doc)
    prompt_summary = str(doc.get("prompt_summary") or "").strip()

    decisions: list[dict] = []
    for number, record in enumerate(order, start=1):
        start = float(record["start"])
        end = windows[int(record["index"])]
        decisions.append({
            "id": f"d{number:02d}",
            "anchor_id": record["anchor_ids"][0],
            "anchor_ids": list(record["anchor_ids"]),
            "effect": record["effect"],
            "emphasis": record["emphasis"],
            "reason": record["reason"],
            "confidence": round(float(record["confidence"]), 4),
            "decision_model": model,
            "content": _content_for_plan(record, index),
            "window": {"start_s": round(start, 6), "end_s": round(end, 6)},
        })

    skipped = [{"anchor_id": str(item.get("anchor_id")),
                "reason": str(item.get("reason"))}
               for item in (doc.get("skipped") or []) if isinstance(item, dict)]
    for record in dropped:
        skipped.append({
            "anchor_id": record["anchor_ids"][0],
            "reason": "window_too_short_after_packing: another element starts at the "
                      "same moment",
        })

    coverage = _coverage([(float(r["start"]), windows[int(r["index"])]) for r in order],
                         duration)
    return {
        "schema": SCHEMA_PLAN,
        "style": STYLE,
        "canvas": {"width": canvas.get("width"), "height": canvas.get("height"),
                   "fps": canvas.get("fps") or _fps(canvas)},
        "duration_s": round(duration, 6),
        "decisions": decisions,
        "notes": {
            "policy": "LLM selects the effect; windows/coordinates are derived from the "
                      "validated anchors (no LLM timestamps, no LLM layout)",
            "skipped": skipped,
            "budget": {"max_new_per_10s": MAX_ELEMENTS_PER_10S,
                       "max_coverage": MAX_COVERAGE,
                       "max_hold_s": MAX_HOLD_S,
                       "min_confidence": MIN_ANCHOR_CONFIDENCE},
            "overlay_coverage": round(coverage, 4),
            "collision_ok": doc.get("collision_ok") or [],
        },
        "provenance": {
            "method": ("llm_director: data-only JSON, schema-validated, no code "
                       f"execution; prompt_summary: {prompt_summary}")[:400],
            "decision_model": model,
            "prompt_summary": prompt_summary,
            "decision_schema": SCHEMA_DECISIONS,
            "anchor_count": len((anchors or {}).get("anchors") or []),
            "decided_count": len(decisions),
        },
    }


# --------------------------------------------------------------------------- io/io
def load_decisions(path: str | Path) -> dict:
    """Load a decision document; raises ValueError for non-JSON / non-object input."""
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"decisions file not found: {file_path}") from None
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{file_path}: invalid JSON ({exc})") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"{file_path}: expected a JSON object at the top level")
    return doc


def dump_decisions(doc: dict, path: str | Path) -> Path:
    """Write a decision document as UTF-8 JSON (atomic replace); returns the path."""
    if not isinstance(doc, dict):
        raise ValueError("decisions document must be a JSON object")
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(doc, ensure_ascii=False, indent=2) + "\n"
    tmp = file_path.with_name(file_path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(file_path)
    return file_path


# ------------------------------------------------------------------- rules fallback
def _fallback_candidates(anchors: dict) -> tuple[list[dict], list[dict]]:
    """Role-mapped candidates, merging consecutive list items into one step list."""
    index = _anchor_index(anchors)
    rows = sorted((a for a in index.values() if isinstance(a, dict)),
                  key=lambda a: (_as_float(a.get("start_s")) or 0.0))
    candidates: list[dict] = []
    skipped: list[dict] = []
    i = 0
    while i < len(rows):
        anchor = rows[i]
        kind = str(anchor.get("kind") or "plain")
        confidence = _as_float(anchor.get("confidence")) or 0.0
        if kind == "plain":
            skipped.append({"anchor_id": anchor.get("id"),
                            "reason": "plain_speech_no_motion"})
            i += 1
            continue
        if confidence < MIN_ANCHOR_CONFIDENCE:
            skipped.append({"anchor_id": anchor.get("id"),
                            "reason": f"low_confidence:{confidence:.2f}"})
            i += 1
            continue
        if kind == "list_item":
            group = [anchor]
            j = i + 1
            while j < len(rows):
                nxt = rows[j]
                if (str(nxt.get("kind")) != "list_item"
                        or (_as_float(nxt.get("start_s")) or 0.0)
                        - (_as_float(group[-1].get("end_s")) or 0.0) > 2.6):
                    break
                group.append(nxt)
                j += 1
            candidates.append({
                "refs": [str(a.get("id")) for a in group],
                "texts": [str(a.get("text") or "") for a in group],
                "kind": "list_item",
                "start": _as_float(group[0].get("start_s")) or 0.0,
                "anchor_end": max((_as_float(a.get("end_s")) or 0.0) for a in group),
                "confidence": min((_as_float(a.get("confidence")) or 0.0) for a in group),
            })
            i = j
            continue
        candidates.append({
            "refs": [str(anchor.get("id"))],
            "texts": [str(anchor.get("text") or "")],
            "kind": kind,
            "start": _as_float(anchor.get("start_s")) or 0.0,
            "anchor_end": _as_float(anchor.get("end_s")) or 0.0,
            "confidence": confidence,
        })
        i += 1
    return candidates, skipped


def _fallback_effect(candidate: dict) -> str | None:
    kind = candidate["kind"]
    text = candidate["texts"][0] if candidate["texts"] else ""
    if kind == "list_item":
        return "list"
    if kind == "number":
        return "number" if _parse_numeric(text) else "keyword"
    if kind == "card":
        return "card"
    if kind == "conclusion" and len(text) >= 6:
        return "card"
    return "keyword"


def _fallback_score(candidate: dict, effect: str) -> float:
    base = {"number": 3.0, "list": 3.0, "card": 2.4, "keyword": 1.6}.get(effect, 1.0)
    text = candidate["texts"][0] if candidate["texts"] else ""
    bonus = 0.2 if len(text) >= 4 else 0.0
    return base * (0.5 + float(candidate["confidence"])) + bonus


def default_decisions_for(anchors: dict) -> dict:
    """Deterministic fallback document — **test fixture only, not an LLM**.

    ``decision_model`` is ``rules-fallback`` so a fallback can never be mistaken
    for a model decision; the output still has to pass ``validate_decisions`` and
    ``to_plan`` exactly like a real LLM document (that is what makes the tests
    independent of any external model).
    """
    candidates, skipped = _fallback_candidates(anchors)
    duration = _duration(anchors)

    scored: list[tuple[float, dict, str]] = []
    for candidate in candidates:
        effect = _fallback_effect(candidate)
        if effect is None:
            skipped.append({"anchor_id": candidate["refs"][0],
                            "reason": "no_mappable_effect"})
            continue
        scored.append((_fallback_score(candidate, effect), candidate, effect))
    scored.sort(key=lambda row: (-row[0], row[1]["start"]))

    kept: list[dict] = []
    for _, candidate, effect in scored:
        trial = kept + [candidate]
        order = sorted(trial, key=lambda c: c["start"])
        ends = _pack_spans([(c["start"], c["anchor_end"],
                             _fallback_hold(effect)) for c in order], duration)
        if any(end - c["start"] < PLAN_MIN_WINDOW_S for end, c in zip(ends, order)):
            skipped.append({"anchor_id": candidate["refs"][0],
                            "reason": "overlapping_onset_with_kept_element"})
            continue
        if not _density_ok([c["start"] for c in order]):
            skipped.append({"anchor_id": candidate["refs"][0],
                            "reason": "rhythm_density_10s"})
            continue
        coverage = _coverage([(c["start"], end) for end, c in zip(ends, order)], duration)
        if coverage > MAX_COVERAGE + 1e-9:
            skipped.append({"anchor_id": candidate["refs"][0], "reason": "coverage_budget"})
            continue
        kept.append(candidate)
        candidate["_effect"] = effect

    kept.sort(key=lambda c: c["start"])
    decisions: list[dict] = []
    for candidate in kept:
        effect = str(candidate.get("_effect"))
        decision = {
            "anchor_id": candidate["refs"][0],
            "anchor_ids": list(candidate["refs"]),
            "effect": effect,
            "emphasis": "high" if effect in ("number", "list") else "medium",
            "reason": f"fallback_rule:{candidate['kind']}",
            "hold_s": _fallback_hold(effect),
            "confidence": round(float(candidate["confidence"]), 4),
            "params": {"avoid_face": True, "accent": "#F2B233"},
        }
        if effect == "list":
            decision["content"] = {
                "items": [_simplified(text) for text in candidate["texts"]]}
        decisions.append(decision)

    skipped.sort(key=lambda s: (str(s.get("anchor_id")), str(s.get("reason"))))
    return {
        "schema": SCHEMA_DECISIONS,
        "decision_model": FALLBACK_MODEL,
        "prompt_summary": ("deterministic rules fallback for tests: role→effect map, "
                           "packed anchor windows, rhythm budget"),
        # constant timestamp keeps the fixture byte-identical across runs
        "created_utc": "1970-01-01T00:00:00Z",
        "decisions": decisions,
        "skipped": skipped,
    }


def _fallback_hold(effect: str) -> float:
    return {"number": 0.35, "list": 0.45, "card": 0.45}.get(effect, DEFAULT_HOLD_S)


def _pack_spans(spans: Sequence[tuple[float, float, float]], duration: float) -> list[float]:
    ends: list[float] = []
    for i, (start, anchor_end, hold) in enumerate(spans):
        next_start = spans[i + 1][0] if i + 1 < len(spans) else duration
        ends.append(min(anchor_end + hold, next_start - PACK_GAP_S, duration))
    return ends


# -------------------------------------------------------------------------- captions
def default_max_chars(canvas: dict) -> int:
    """Line budget by orientation: 22 units for 9:16, 30 for 16:9 (REMOTION.md §5)."""
    width = _as_float((canvas or {}).get("width")) or 0.0
    height = _as_float((canvas or {}).get("height")) or 0.0
    return CAPTION_LANDSCAPE_MAX_CHARS if width > height else CAPTION_DEFAULT_MAX_CHARS


def _caption_words(transcript: dict) -> list[dict]:
    words: list[dict] = []
    for word in (transcript or {}).get("words") or []:
        if not isinstance(word, dict):
            continue
        text = str(word.get("text") or "")
        start = _as_float(word.get("start_s"))
        end = _as_float(word.get("end_s"))
        if not text.strip() or start is None or end is None:
            continue
        if end < start:
            end = start
        words.append({"id": word.get("id"), "segment_id": word.get("segment_id"),
                      "text": text, "start_s": float(start), "end_s": float(end)})
    words.sort(key=lambda w: (w["start_s"], w.get("id") or ""))
    return words


def _caption_chunks(words: Sequence[dict]) -> list[list[dict]]:
    """Split the word stream like a subtitle editor: punctuation, pause, segment."""
    chunks: list[list[dict]] = []
    current: list[dict] = []
    for i, word in enumerate(words):
        current.append(word)
        text = word["text"].rstrip()
        last = text[-1] if text else ""
        should_break = bool(last) and (last in _TERMINAL_PUNCT or last in _CLAUSE_PUNCT)
        if not should_break and i + 1 < len(words):
            nxt = words[i + 1]
            if nxt["start_s"] - word["end_s"] >= CAPTION_PAUSE_BREAK_S:
                should_break = True
            elif str(nxt.get("segment_id")) != str(word.get("segment_id")):
                should_break = True
        if should_break:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def _tokens(text: str) -> list[str]:
    """Split into wrap units: one CJK/other character each, Latin runs as words."""
    tokens: list[str] = []
    i = 0
    while i < len(text):
        match = _LATIN_TOKEN.match(text, i)
        if match:
            tokens.append(match.group(0))
            i = match.end()
            continue
        if text[i] in _SPACE_CHARS:
            while i < len(text) and text[i] in _SPACE_CHARS:
                i += 1
            tokens.append(" ")
            continue
        tokens.append(text[i])
        i += 1
    return tokens


def _needs_space(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return left[-1].isascii() and left[-1].isalnum() and right[0].isascii() and right[0].isalnum()


def _join(tokens: Sequence[str]) -> str:
    out = ""
    for token in tokens:
        continued = token.startswith(_CONT)
        if continued:
            token = token[1:]
        if token == " ":
            if out and not out.endswith(" "):
                out += " "
            continue
        if out and not continued and _needs_space(out, token):
            out += " "
        out += token
    return out.strip()


def _screen_tokens(words: Sequence[dict], max_chars: int) -> list[str]:
    tokens: list[str] = []
    chunks: list[str] = []
    for word in words:
        chunks.extend(_tokens(str(word.get("text") or "")))
    for token in chunks:
        if token == " ":
            tokens.append(" ")
        elif display_width(token) > max_chars + 1e-9:
            pieces = _split_by_width(token, max_chars)
            tokens.append(pieces[0])
            tokens.extend(_CONT + piece for piece in pieces[1:])
        else:
            tokens.append(token)
    return tokens


#: Trailing characters a burned caption line must not end with (subtitle habit).
_CAPTION_TAIL_PUNCT = "，,。.、；;：:！!？?…·-— "

#: Width in ``max_chars`` units: CJK / full-width counts 1, Latin / ASCII 0.5.
_ASCII_WIDTH = 0.5

_T2S_CONVERTER = None
_T2S_UNAVAILABLE = False
_T2S_CHAR_CACHE: dict[str, str] = {}


def _t2s():
    """Phrase-level Traditional→Simplified converter; None when OpenCC is absent."""
    global _T2S_CONVERTER, _T2S_UNAVAILABLE
    if _T2S_CONVERTER is not None or _T2S_UNAVAILABLE:
        return _T2S_CONVERTER
    try:
        from opencc import OpenCC
        _T2S_CONVERTER = OpenCC("t2s")
    except Exception:  # pragma: no cover - optional dependency
        _T2S_UNAVAILABLE = True
        _T2S_CONVERTER = None
    return _T2S_CONVERTER


def _fold_char(converter, char: str) -> str:
    """Fold one character and keep it exactly one character (offsets are fixed)."""
    cached = _T2S_CHAR_CACHE.get(char)
    if cached is not None:
        return cached
    try:
        piece = converter.convert(char)
    except Exception:  # pragma: no cover - converter failure
        piece = char
    if len(piece) != 1:
        piece = char
    _T2S_CHAR_CACHE[char] = piece
    return piece


def _fold_display_text(text: str) -> tuple[str, str | None]:
    """Fold caption copy to Simplified Chinese **without changing its length**.

    Word offsets and character spans are load-bearing (highlight timing, cue
    alignment), so OpenCC's phrase conversion is used only when the result has the
    same length; otherwise every character is folded on its own and a character
    that would gain or lose characters is kept as-is. Returns ``(display, raw)``
    where ``raw`` is the original copy when anything changed, else ``None``.
    """
    raw = str(text or "")
    if not raw:
        return raw, None
    converter = _t2s()
    if converter is None:
        return raw, None
    folded = converter.convert(raw)
    if len(folded) != len(raw):
        folded = "".join(_fold_char(converter, ch) for ch in raw)
    if folded == raw:
        return raw, None
    return folded, raw


def _caption_display_text(text: str) -> str:
    """Display form of a caption line: Simplified Chinese, no trailing punctuation.

    Whisper occasionally emits Traditional characters for Mandarin and ASR copy
    carries sentence punctuation; burned captions should read like subtitles, not
    like a transcript. The transcript/anchor data itself is never rewritten.
    """
    folded, _ = _fold_display_text(text)
    out = folded.strip()
    while out and out[-1] in _CAPTION_TAIL_PUNCT:
        out = out[:-1].rstrip()
    return out


def _char_width(char: str) -> float:
    return _ASCII_WIDTH if char.isascii() else 1.0


def display_width(text: str) -> float:
    """Caption line width in ``max_chars`` units (CJK 1, Latin/ASCII 0.5).

    The budget is a display width, not a raw character count: 22 units of 9:16
    copy is about 22 Chinese characters or about 44 Latin letters. English words
    are never cut to fill a line; the narrower Latin advance is simply counted.
    """
    return sum(_char_width(ch) for ch in str(text or ""))


def _split_by_width(text: str, max_chars: int) -> list[str]:
    """Hard-split one over-long token into pieces that each fit ``max_chars``."""
    pieces: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if current and display_width(candidate) > max_chars + 1e-9:
            pieces.append(current)
            current = char
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _wrap_screen(words: Sequence[dict], max_chars: int, max_lines: int) -> list[str] | None:
    tokens = _screen_tokens(words, max_chars)
    if not tokens:
        return None
    lines: list[str] = []
    current: list[str] = []
    for token in tokens:
        if not current and token == " ":
            continue
        candidate = current + [token]
        if display_width(_join(candidate)) <= max_chars + 1e-9 or not current:
            current = candidate
            continue
        if token in _CLOSERS and len(current) >= 2:
            # never start a line with closing punctuation: take the previous unit along
            moved = current.pop()
            line = _join(current)
            if line:
                lines.append(line)
            current = [moved, token]
            if display_width(_join(current)) > max_chars + 1e-9:
                lines.append(_join([moved]))
                current = [token]
            continue
        lines.append(_join(current))
        current = [token]
    if current:
        lines.append(_join(current))
    lines = [line for line in lines if line]
    if not lines or len(lines) > max_lines:
        return None
    if any(display_width(line) > max_chars + 1e-9 for line in lines):
        return None
    return lines


def _split_oversized_word(word: dict, max_chars: int) -> list[dict]:
    """A single ASR token longer than one line is chunked, splitting its time too."""
    text = str(word.get("text") or "")
    pieces = _split_by_width(text, max_chars)
    if len(pieces) <= 1:
        return [word]
    start, end = float(word["start_s"]), float(word["end_s"])
    span = max(end - start, 1e-6)
    total = display_width(text) or 1.0
    out: list[dict] = []
    offset = 0.0
    for piece in pieces:
        t0 = start + span * (offset / total)
        offset += display_width(piece)
        t1 = start + span * (offset / total)
        out.append({"id": word.get("id"), "segment_id": word.get("segment_id"),
                    "text": piece, "start_s": round(t0, 6),
                    "end_s": round(max(t1, t0 + 1e-6), 6)})
    return out


def _split_screens(words: Sequence[dict], max_chars: int, max_lines: int) -> list[list[dict]]:
    screens: list[list[dict]] = []
    current: list[dict] = []
    for word in words:
        if current and _wrap_screen(current + [word], max_chars, max_lines) is not None:
            current.append(word)
            continue
        if current:
            screens.append(current)
            current = []
        if _wrap_screen([word], max_chars, max_lines) is not None:
            current = [word]
        else:  # pathological: one ASR token spanning more than max_lines lines
            screens.extend([[piece] for piece in _split_oversized_word(word, max_chars)])
    if current:
        screens.append(current)
    return screens


def caption_cues(transcript: dict, canvas: dict, *, max_chars: int = 22,
                 max_lines: int = 2) -> list[dict]:
    """Build the caption track (REMOTION.md §5) from word-level transcript times.

    Each cue carries ``startFrame/endFrame/lines/words``; ``words`` keep per-word
    frames for karaoke-style highlighting. Cues are monotonic and never overlap,
    and a cue's first frame is the frame of its first word.

    Display copy: Traditional characters are folded to Simplified **1:1** (OpenCC
    phrase conversion when it preserves length, per-character otherwise), so word
    character offsets stay valid; a word that was rewritten also keeps its
    ``raw_text``. Lines are then stripped of trailing punctuation (``，,。.、；;：:``
    and friends) and never start with closing punctuation; a line that ends up
    empty is dropped, and a cue with no line left is skipped.

    Line width is a display width in ``max_chars`` units: CJK / full-width = 1,
    Latin / ASCII = 0.5, so 22 units of 9:16 copy is ~22 Chinese characters or
    ~44 Latin letters. English words are never split (only a pathological single
    token longer than a whole line is hard-split, with its time split too).
    Position (bottom safe zone) and the face-collision nudge belong to the
    renderer/layout stage.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    if max_lines < 1:
        raise ValueError("max_lines must be >= 1")
    words = _caption_words(transcript)
    if not words:
        return []
    fps = _fps(canvas, transcript)
    duration = _duration(None, canvas, transcript)
    duration_frames = max(1, _frame(duration, fps)) if duration > 0 else None

    screens: list[list[dict]] = []
    for chunk in _caption_chunks(words):
        screens.extend(_split_screens(chunk, max_chars, max_lines))
    if not screens:
        return []

    cues: list[dict] = []
    previous_end = 0
    for i, screen in enumerate(screens):
        lines = _wrap_screen(screen, max_chars, max_lines)
        if not lines:  # pragma: no cover - guarded by _split_screens
            continue
        # subtitles read as Simplified Chinese without trailing punctuation
        lines = [x for x in (_caption_display_text(line) for line in lines) if x]
        if not lines:
            continue
        start = max(_frame(screen[0]["start_s"], fps), previous_end)
        if duration_frames is not None and start >= duration_frames:
            continue
        end = max(_frame(screen[-1]["end_s"], fps), start + 1)
        if duration_frames is not None:
            end = min(end, duration_frames)
        if end <= start:
            continue
        words_out: list[dict] = []
        word_end = start
        for word in screen:
            word_start = max(_frame(word["start_s"], fps), word_end, start)
            word_stop = max(_frame(word["end_s"], fps), word_start + 1)
            word_stop = min(word_stop, end)
            if word_start >= end:
                break
            folded, raw = _fold_display_text(str(word.get("text") or ""))
            entry = {"text": folded, "startFrame": word_start,
                     "endFrame": word_stop}
            if raw is not None:
                # keep the original copy when the fold rewrote anything, so the
                # preview/QA can trace every displayed word back to the transcript
                entry["raw_text"] = raw
            words_out.append(entry)
            word_end = word_stop
        cues.append({"startFrame": start, "endFrame": end, "lines": lines,
                     "words": words_out})
        previous_end = end
    return cues
