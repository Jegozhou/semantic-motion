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

STYLE_SMART_BALANCED = "smart_balanced"

# Rhythm budget for Smart Balanced.
MAX_NEW_ELEMENTS_PER_10S = 4.0     # new overlays per 10 seconds
MIN_GAP_BETWEEN_ELEMENTS_S = 0.9   # breathing room between overlay windows
MAX_OVERLAY_COVERAGE = 0.58        # fraction of runtime that may carry an overlay
MIN_CONFIDENCE = 0.60              # below this, do not animate precisely
LIST_MERGE_GAP_S = 2.0             # consecutive list anchors merge into one list

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


def _norm(text: str) -> str:
    return "".join(ch for ch in str(text) if ch.isalnum())


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
            decisions.append({
                "id": _decision_id(len(decisions) + 1),
                "anchor_id": a.get("id"),
                "anchor_ids": [g.get("id") for g in group],
                "effect": "list",
                "emphasis": "high" if len(group) >= 3 else "medium",
                "reason": f"sequence_of_{len(group)}_steps",
                "confidence": round(min(_conf(g) for g in group), 4),
                "decision_model": "rules-v1-smart-balanced",
                "content": {"text": a.get("text", ""),
                            "items": [g.get("text", "") for g in group]},
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
                "content": {"text": a.get("text", "")},
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
                "content": {"text": text},
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
                "content": {"text": text},
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
                    conflict = (k, "contains " + k["id"] if _keep_score(k) >= _keep_score(d)
                                else k["id"] + " contained")
                elif _keep_score(k) >= _keep_score(d):
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
