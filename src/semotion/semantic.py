"""Semantic Motion — the "smart balanced" director.

Turns semantic anchors into explicit editing decisions. This is deliberately a
rule engine, not a language model: every decision must be explainable, and the
engine must be able to decide *not* to add motion (plain speech often gets none).

Design goals (spec §3.3):
* only important information triggers an effect (keywords, numbers, steps,
  contrast, causal, conclusion);
* restrained rhythm — never two competing overlays, never a wall of text;
* deterministic output for identical anchors (the IR freezes the result).
"""
from __future__ import annotations

from typing import Any

_T2S = None


def to_simplified(text: str) -> str:
    """Display text in Simplified Chinese; Thai/English pass through unchanged.

    Whisper sometimes emits Traditional characters for Mandarin. Overlay copy must
    be Simplified for the product's audience, so the displayed string is folded
    here while the original transcript/anchor text stays untouched in the IR.
    """
    global _T2S
    if _T2S is None:
        try:
            import opencc
            _T2S = opencc.OpenCC("t2s")
        except Exception:  # pragma: no cover - optional dependency
            _T2S = False
    if not text or not _T2S:
        return text
    return _T2S.convert(str(text))

STYLE_SMART_BALANCED = "smart_balanced"
# Calibration policy: animate *every* non-plain anchor (no rhythm budget) so that
# timing can be measured on as many independent events as the fixtures provide.
# It is a measurement instrument, not the product's default taste.
STYLE_CALIBRATION = "calibration"

# Rhythm budget for Smart Balanced.
MAX_NEW_ELEMENTS_PER_10S = 4.0     # new overlays per 10 seconds
MIN_GAP_BETWEEN_ELEMENTS_S = 0.9   # breathing room between overlay windows
MAX_OVERLAY_COVERAGE = 0.58        # fraction of runtime that may carry an overlay
MIN_CONFIDENCE = 0.60              # below this, do not animate precisely
LIST_MERGE_GAP_S = 2.6             # consecutive list anchors merge into one list

# Camera cuts: an overlay that straddles a cut reads as a mistake, so windows are
# kept inside one shot. `CUT_GUARD_S` is the exclusion band around each cut.
CUT_GUARD_S = 0.30
MIN_ELEMENT_AFTER_GUARD_S = 0.35

# Information value of a semantic role. Numbers and steps outrank loose keywords
# when the rhythm budget is scarce (Smart Balanced: important information first).
ROLE_PRIORITY = {
    "number": 3.0,
    "list_item": 3.0,
    "card": 2.6,
    "conclusion": 2.2,
    "comparison": 2.0,
    "process": 2.0,
    "keyword": 1.6,
}

# Discourse fillers that carry no emphasis value even if a transcript labels them
# as keywords. Kept short and explainable on purpose.
STOPWORDS = {
    "其实", "这个", "那个", "就是", "然后", "所以", "但是", "因为", "如果",
    "我们", "你们", "他们", "一个", "一下", "现在", "今天", "大家", "什么",
}


def _conf(a: dict) -> float:
    try:
        return float(a.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _apply_shot_policy(decisions: list[dict], vision: dict, skipped: list[dict]) -> list[dict]:
    """Keep every overlay inside a single shot, away from the cut itself.

    Without shot information (no vision, or a single shot) this is a no-op.
    """
    shots = [s for s in (vision or {}).get("shots", []) if s.get("end_s")]
    if len(shots) < 2:
        return decisions
    cuts = [float(s["start_s"]) for s in shots[1:]]
    out: list[dict] = []
    for d in decisions:
        ws, we = float(d["window"]["start_s"]), float(d["window"]["end_s"])
        near = [c for c in cuts if ws - CUT_GUARD_S < c < we + CUT_GUARD_S]
        if not near:
            out.append(d)
            continue
        nxt = min(c for c in cuts if c >= ws - CUT_GUARD_S)
        trimmed_end = nxt - CUT_GUARD_S
        if trimmed_end - ws >= MIN_ELEMENT_AFTER_GUARD_S:
            d = {**d, "window": {**d["window"], "end_s": round(trimmed_end, 6)}}
            d["notes"] = f"trimmed_before_cut_at_{nxt:.2f}s"
            out.append(d)
            continue
        # the anchor itself sits on the cut: start after the cut instead
        shifted_start = nxt + CUT_GUARD_S
        if we - shifted_start >= MIN_ELEMENT_AFTER_GUARD_S:
            d = {**d, "window": {"start_s": round(shifted_start, 6), "end_s": round(we, 6)}}
            d["notes"] = f"shifted_after_cut_at_{nxt:.2f}s"
            out.append(d)
            continue
        skipped.append({"anchor_id": d["anchor_id"], "reason": f"crosses_cut_at_{nxt:.2f}s"})
    return out


def _norm(text: str) -> str:
    return "".join(ch for ch in str(text) if ch.isalnum())


_ORDINAL_PREFIX = ("第一个", "第一", "第二个", "第二", "第三个", "第三", "第四个", "第四",
                   "首先", "其次", "然后", "最后", "接着", "另外")


def _transcript_slice(transcript: dict | None, t0: float, t1: float) -> str:
    """Spoken text between two times, from the ASR word stream."""
    if not transcript:
        return ""
    words = transcript.get("words") or []
    picked = [w for w in words if t0 - 0.02 <= float(w.get("start_s", 0.0)) < t1 - 0.02]
    return "".join(str(w.get("text", "")) for w in picked)


def _compose_list_items(group: list[dict], all_items: list[dict],
                        transcript: dict | None = None) -> list[str]:
    """Attach each step's spoken content to its ordinal marker.

    Anchors come from ASR words, so "第一" is often its own anchor followed by the
    step's phrase anchors. Showing only "① 第一" is semantically thin; the list row
    should read "① 字幕出现的时间". We join the following non-list anchors that sit
    inside the same step window and strip a leading ordinal from the joined text.
    """
    starts = [float(g.get("start_s", 0.0)) for g in group]
    out: list[str] = []
    for idx, g in enumerate(group):
        start = starts[idx]
        end = starts[idx + 1] if idx + 1 < len(starts) else float(g.get("end_s", start)) + LIST_MERGE_GAP_S
        if idx + 1 >= len(starts):
            nxt_anchor = [float(it.get("start_s", 0.0)) for it in all_items
                          if not _is_list(it) and float(it.get("start_s", 0.0)) > start + 0.05]
            if nxt_anchor:
                end = min(end, min(nxt_anchor) - 0.05)
        follows = [it for it in all_items
                   if not _is_list(it) and start - 1e-6 <= float(it.get("start_s", 0.0)) < end]
        follows.sort(key=lambda x: float(x.get("start_s", 0.0)))
        pieces: list[str] = []
        for it in follows:
            t = str(it.get("text", "")).strip("，,。.、；;：: ")
            if t and not any(t in p or p in t for p in pieces):
                pieces.append(t)
        spoken = _transcript_slice(transcript, start, end).strip("，,。.、；;：: ")
        joined = spoken or "".join(pieces)
        for pre in sorted(_ORDINAL_PREFIX, key=len, reverse=True):
            if joined.startswith(pre):
                joined = joined[len(pre):].strip("，,。.、；;：: ")
                break
        joined = joined.strip("，,。.、；;：: ")
        out.append(joined if joined else str(g.get("text", "")))
    return out


def _is_list(a: dict) -> bool:
    return a.get("kind") == "list_item"


def _decision_id(i: int) -> str:
    return f"d{i:02d}"


def decide(anchors: dict, vision: dict, canvas: dict, *, style: str = STYLE_SMART_BALANCED,
           transcript: dict | None = None) -> dict:
    """Produce a plan (`semotion.plan/1`) from anchors.

    ``vision`` is accepted for future shot-aware policy (e.g. do not animate across
    a camera cut) and for face-density awareness; it does not invent coordinates.
    """
    items: list[dict] = sorted(anchors.get("anchors", []), key=lambda a: a.get("start_s", 0.0))
    raw_items = list(items)          # before stopword/dedupe filtering, for list text
    calibration = style == STYLE_CALIBRATION
    duration = float(anchors.get("duration_s") or
                     (canvas.get("duration_s") if canvas.get("duration_s") else
                      max([a.get("end_s", 0.0) for a in items] or [0.0])))
    if duration <= 0:
        duration = max([a.get("end_s", 0.0) for a in items] or [1.0])

    decisions: list[dict] = []
    skipped: list[dict] = []
    recent_starts: list[float] = []

    def _score(a: dict) -> float:
        conf = _conf(a)
        base = ROLE_PRIORITY.get(a.get("kind", ""), 0.5)
        text = str(a.get("text", ""))
        length_bonus = 0.15 if len(text) >= 4 else 0.0
        return base * (0.5 + conf) + length_bonus

    if calibration:
        return _calibration_plan(items, canvas, duration)

    # Pre-filter: drop filler words; keep at most one loose keyword per ASR segment
    # so a single sentence cannot produce a stack of one-word overlays.
    by_segment: dict[str, list[dict]] = {}
    pruned: list[dict] = []
    for a in items:
        text = str(a.get("text", ""))
        if a.get("kind") in ("keyword", "comparison", "process", "conclusion") and \
                _norm(text) in STOPWORDS and len(_norm(text)) <= 4:
            skipped.append({"anchor_id": a.get("id"), "reason": "filler_word"})
            continue
        if a.get("kind") == "keyword":
            by_segment.setdefault(str(a.get("segment_id")), []).append(a)
        else:
            pruned.append(a)
    for seg, group in by_segment.items():
        group.sort(key=_score, reverse=True)
        pruned.append(group[0])
        for extra in group[1:]:
            skipped.append({"anchor_id": extra.get("id"), "reason": "segment_keyword_dedupe"})
    items = sorted(pruned, key=lambda a: a.get("start_s", 0.0))

    def _budget_ok(start: float) -> tuple[bool, str]:
        if len(recent_starts) >= 2 and start - recent_starts[-2] < MIN_GAP_BETWEEN_ELEMENTS_S:
            return False, "rhythm_gap"
        window_start = start - 10.0
        recent = [t for t in recent_starts if t >= window_start]
        if len(recent) >= MAX_NEW_ELEMENTS_PER_10S:
            return False, "rhythm_density"
        return True, ""

    i = 0
    while i < len(items):
        a = items[i]
        kind = a.get("kind", "plain")
        conf = _conf(a)

        if kind == "plain":
            skipped.append({"anchor_id": a.get("id"), "reason": "plain_speech_no_motion"})
            i += 1
            continue

        if conf < MIN_CONFIDENCE:
            skipped.append({"anchor_id": a.get("id"), "reason": f"low_confidence:{conf:.2f}"})
            i += 1
            continue

        start = float(a.get("start_s", 0.0))
        end = float(a.get("end_s", start + 1.0))

        # --- steps / list: merge a run of consecutive list items into one element
        if _is_list(a):
            group = [a]
            j = i + 1
            while j < len(items) and _is_list(items[j]) and \
                    float(items[j].get("start_s", 0.0)) - float(group[-1].get("end_s", 0.0)) <= LIST_MERGE_GAP_S:
                group.append(items[j])
                j += 1
            g_start = float(group[0].get("start_s", start))
            g_end = float(group[-1].get("end_s", end)) + 0.45  # hold after last item
            ok, why = _budget_ok(g_start)
            if not ok:
                skipped.append({"anchor_id": a.get("id"), "reason": why})
                i = j
                continue
            composed = _compose_list_items(group, raw_items, transcript)
            decisions.append({
                "id": _decision_id(len(decisions) + 1),
                "anchor_id": a.get("id"),
                "anchor_ids": [g.get("id") for g in group],
                "effect": "list",
                "emphasis": "high" if len(group) >= 3 else "medium",
                "reason": f"sequence_of_{len(group)}_steps",
                "confidence": round(min(_conf(g) for g in group), 4),
                "decision_model": "rules-v1-smart-balanced",
                "content": {"text": to_simplified(a.get("text", "")),
                            "items": [to_simplified(t) for t in composed],
                            "items_raw": [g.get("text", "") for g in group]},
                "window": {"start_s": round(g_start, 6), "end_s": round(g_end, 6)},
            })
            recent_starts.append(g_start)
            i = j
            continue

        # --- numbers
        if kind == "number":
            ok, why = _budget_ok(start)
            if not ok:
                skipped.append({"anchor_id": a.get("id"), "reason": why})
                i += 1
                continue
            decisions.append({
                "id": _decision_id(len(decisions) + 1),
                "anchor_id": a.get("id"),
                "anchor_ids": [a.get("id")],
                "effect": "number",
                "emphasis": "high",
                "reason": "quantified_claim",
                "confidence": conf,
                "decision_model": "rules-v1-smart-balanced",
                "content": {"text": to_simplified(a.get("text", ""))},
                "window": {"start_s": round(start, 6), "end_s": round(end + 0.35, 6)},
            })
            recent_starts.append(start)
            i += 1
            continue

        # --- cards: longer explanatory statements
        text = str(a.get("text", ""))
        if kind == "card" or (kind == "conclusion" and len(text) >= 6):
            ok, why = _budget_ok(start)
            if not ok:
                skipped.append({"anchor_id": a.get("id"), "reason": why})
                i += 1
                continue
            decisions.append({
                "id": _decision_id(len(decisions) + 1),
                "anchor_id": a.get("id"),
                "anchor_ids": [a.get("id")],
                "effect": "card",
                "emphasis": "medium",
                "reason": "explanatory_statement",
                "confidence": conf,
                "decision_model": "rules-v1-smart-balanced",
                "content": {"text": to_simplified(text)},
                "window": {"start_s": round(start, 6), "end_s": round(end + 0.5, 6)},
            })
            recent_starts.append(start)
            i += 1
            continue

        # --- keywords: the default emphasis element
        if kind in ("keyword", "comparison", "process", "conclusion"):
            ok, why = _budget_ok(start)
            if not ok:
                skipped.append({"anchor_id": a.get("id"), "reason": why})
                i += 1
                continue
            decisions.append({
                "id": _decision_id(len(decisions) + 1),
                "anchor_id": a.get("id"),
                "anchor_ids": [a.get("id")],
                "effect": "keyword",
                "emphasis": "high" if kind == "keyword" else "medium",
                "reason": f"semantic_role:{kind}",
                "confidence": conf,
                "decision_model": "rules-v1-smart-balanced",
                "content": {"text": to_simplified(text)},
                "window": {"start_s": round(start, 6), "end_s": round(end + 0.30, 6)},
            })
            recent_starts.append(start)
            i += 1
            continue

        skipped.append({"anchor_id": a.get("id"), "reason": f"unhandled_kind:{kind}"})
        i += 1

    # --- redundancy guard: a card that merely restates a shorter element adds
    # noise, and overlapping overlays collide. Keep one element per time span,
    # preferring the higher-information one (numbers/steps over prose cards).
    def _keep_score(d: dict) -> float:
        eff = d.get("effect")
        base = {"number": 3.0, "list": 3.0, "card": 2.4, "keyword": 1.6}.get(eff, 1.0)
        text = str(d.get("content", {}).get("text", ""))
        return base * (0.5 + float(d.get("confidence", 0.0))) + (0.2 if len(text) >= 4 else 0.0)

    decisions.sort(key=lambda d: (d["window"]["start_s"], _keep_score(d) * -1))
    kept: list[dict] = []
    for d in decisions:
        conflict = None
        for k in kept:
            if d["window"]["start_s"] < k["window"]["end_s"] and \
                    k["window"]["start_s"] < d["window"]["end_s"]:
                contained = (str(d["content"].get("text", "")) in str(k["content"].get("text", ""))
                             or str(k["content"].get("text", "")) in str(d["content"].get("text", "")))
                if contained:
                    k_text, d_text = str(k["content"].get("text", "")), str(d["content"].get("text", ""))
                    richer_is_k = len(k_text) >= len(d_text)
                    if not richer_is_k:
                        kept.remove(k)
                        skipped.append({"anchor_id": k["anchor_id"],
                                        "reason": f"superseded_by_{d['id']}_richer_text"})
                        continue
                    conflict = (k, "contains " + k["id"])
                else:
                    # small overlap: trim the earlier element instead of dropping it,
                    # so a card that leads into a list still gets its moment.
                    overlap = float(k["window"]["end_s"]) - float(d["window"]["start_s"])
                    if 0 < overlap <= 0.7 and (float(d["window"]["start_s"])
                                               - float(k["window"]["start_s"])) >= 0.6:
                        k["window"]["end_s"] = round(float(d["window"]["start_s"]) - 0.04, 6)
                        k["notes"] = f"trimmed_before_{d['id']}"
                        kept.append(d)
                        continue
                if _keep_score(k) >= _keep_score(d):
                    conflict = (k, f"temporal_conflict_with_{k['id']}")
                else:
                    kept.remove(k)
                    skipped.append({"anchor_id": k["anchor_id"],
                                    "reason": f"superseded_by_{d['id']}"})
                    continue
                break
        if conflict is None:
            kept.append(d)
        else:
            skipped.append({"anchor_id": d["anchor_id"], "reason": conflict[1]})
    decisions = kept

    # --- camera-cut policy (G3): no overlay may straddle a shot change
    decisions = _apply_shot_policy(decisions, vision, skipped)

    # --- global coverage guard: drop the least important elements if over budget
    overlay = sum(max(0.0, d["window"]["end_s"] - d["window"]["start_s"]) for d in decisions)
    if duration > 0 and overlay / duration > MAX_OVERLAY_COVERAGE:
        keep: list[dict] = []
        covered = 0.0
        for d in decisions:
            w = max(0.0, d["window"]["end_s"] - d["window"]["start_s"])
            if (covered + w) / duration > MAX_OVERLAY_COVERAGE:
                skipped.append({"anchor_id": d["anchor_id"], "reason": "coverage_budget"})
                continue
            keep.append(d)
            covered += w
        decisions = keep

    plan = {
        "schema": "semotion.plan/1",
        "style": style,
        "canvas": {"width": canvas.get("width"), "height": canvas.get("height"),
                   "fps": canvas.get("fps")},
        "duration_s": round(duration, 6),
        "decisions": decisions,
        "notes": {
            "policy": "P1 effects only; plain speech intentionally gets zero motion; "
                      "rhythm budget enforced",
            "skipped": skipped,
            "budget": {"max_new_per_10s": MAX_NEW_ELEMENTS_PER_10S,
                       "min_gap_s": MIN_GAP_BETWEEN_ELEMENTS_S,
                       "max_coverage": MAX_OVERLAY_COVERAGE,
                       "min_confidence": MIN_CONFIDENCE},
            "overlay_coverage": round(overlay / duration, 4) if duration else 0.0,
        },
        "provenance": {
            "method": "rule_engine (no LLM): semantic roles + rhythm budget + confidence gate",
            "decision_model": "rules-v1-smart-balanced",
            "anchor_count": len(items),
            "decided_count": len(decisions),
        },
    }
    return plan


def _calibration_plan(items: list[dict], canvas: dict, duration: float) -> dict:
    """One element per non-plain anchor, windows tiled so they never overlap."""
    rows = [a for a in items if a.get("kind") not in (None, "", "plain")
            and _conf(a) >= MIN_CONFIDENCE]
    rows.sort(key=lambda a: float(a.get("start_s", 0.0)))
    decisions: list[dict] = []
    for i, a in enumerate(rows):
        start = float(a.get("start_s", 0.0))
        nxt = float(rows[i + 1].get("start_s", duration)) if i + 1 < len(rows) else duration
        end = min(float(a.get("end_s", start + 1.0)) + 0.45, nxt - 0.05, duration - 0.05)
        if end - start < 0.30:
            continue
        kind = a.get("kind")
        effect = {"number": "number", "list_item": "list", "card": "card",
                  "conclusion": "card"}.get(kind, "keyword")
        decisions.append({
            "id": f"cal{i + 1:02d}",
            "anchor_id": a.get("id"),
            "anchor_ids": [a.get("id")],
            "effect": effect,
            "emphasis": "high",
            "reason": "calibration_all_anchors",
            "confidence": _conf(a),
            "decision_model": "rules-v1-calibration",
            "content": {"text": to_simplified(a.get("text", "")), "items": None},
            "window": {"start_s": round(start, 6), "end_s": round(end, 6)},
        })
    return {
        "schema": "semotion.plan/1",
        "style": STYLE_CALIBRATION,
        "canvas": {"width": canvas.get("width"), "height": canvas.get("height"),
                   "fps": canvas.get("fps")},
        "duration_s": round(duration, 6),
        "decisions": decisions,
        "notes": {
            "policy": "calibration: every non-plain anchor animated, windows tiled; "
                      "used ONLY to measure timing, never presented as product output",
            "skipped": [],
            "budget": None,
            "overlay_coverage": round(sum(
                d["window"]["end_s"] - d["window"]["start_s"] for d in decisions) / max(duration, 1e-6), 4),
        },
        "provenance": {"method": "rule_engine (calibration)", "decision_model": "rules-v1-calibration",
                       "anchor_count": len(items), "decided_count": len(decisions)},
    }


def summarize(plan: dict) -> dict[str, Any]:
    """Compact human-readable summary used in logs and reports."""
    by_effect: dict[str, int] = {}
    for d in plan.get("decisions", []):
        by_effect[d["effect"]] = by_effect.get(d["effect"], 0) + 1
    return {
        "decisions": len(plan.get("decisions", [])),
        "by_effect": by_effect,
        "skipped": len(plan.get("notes", {}).get("skipped", [])),
        "coverage": plan.get("notes", {}).get("overlay_coverage"),
    }
