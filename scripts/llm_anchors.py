#!/usr/bin/env python3
"""Build a `semotion.anchors/1` document from LLM-chosen word spans.

Real Chinese ASR often returns character-level words, so the rule classifier emits
truncated phrases ("其实还远远没有结"). Letting the LLM choose *which words to group*
and *what semantic role they play* fixes that without letting it invent timestamps:
a span names transcript word ids, and every timestamp is copied from those words.

    python scripts/llm_anchors.py --transcript t.json --spans spans.json --out anchors.json

spans.json:
    [{"id": "a00", "kind": "keyword", "word_ids": ["w5", "w6"],
      "role": "梦想"}]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

KINDS = ("keyword", "number", "list_item", "card", "comparison", "process", "conclusion")
PUNCT = "，,。.、；;：:！!？? \n"


def build(transcript: dict, spans: list[dict]) -> dict:
    words = {w["id"]: w for w in transcript.get("words") or []}
    anchors = []
    for i, span in enumerate(spans):
        kind = str(span.get("kind") or "keyword")
        if kind not in KINDS:
            raise SystemExit(f"span {i}: unknown kind {kind!r}")
        ids = [str(x) for x in span.get("word_ids") or []]
        missing = [x for x in ids if x not in words]
        if missing:
            raise SystemExit(f"span {i}: unknown word ids {missing}")
        if not ids:
            raise SystemExit(f"span {i}: empty word_ids")
        picked = [words[x] for x in ids]
        text = "".join(str(w.get("text") or "") for w in picked).strip(PUNCT)
        start = min(float(w["start_s"]) for w in picked)
        end = max(float(w["end_s"]) for w in picked)
        anchors.append({
            "id": span.get("id") or f"a{i:02d}",
            "kind": kind,
            "text": text,
            "start_s": round(start, 6),
            "end_s": round(end, 6),
            "confidence": float(span.get("confidence") or 0.85),
            "source": "LLM_SPAN",
            "segment_id": picked[0].get("segment_id"),
            "word_ids": ids,
            "rule": "llm_word_span",
            "refine": {"method": "llm_word_span", "delta_ms": 0, "accepted": True},
        })
    return {
        "schema": "semotion.anchors/1",
        "source_sha256": (transcript.get("source") or {}).get("sha256"),
        "transcript_sha256": None,
        "duration_s": float((transcript.get("source") or {}).get("duration_s") or 0.0),
        "language": transcript.get("language"),
        "anchors": anchors,
        "provenance": {
            "method": "llm_word_spans (the LLM chooses spans, transcript supplies every timestamp)",
            "n_anchors": len(anchors),
            "truth_class": "MACHINE_WEAK",
            "note": "no timestamp is invented: start/end are the union of the named words",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--spans", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tr = json.loads(Path(a.transcript).read_text(encoding="utf-8"))
    spans = json.loads(Path(a.spans).read_text(encoding="utf-8"))
    doc = build(tr, spans if isinstance(spans, list) else spans.get("spans") or [])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for an in doc["anchors"]:
        print(f"  {an['id']} {an['kind']:<9} {an['text']}  {an['start_s']:.2f}-{an['end_s']:.2f}s")
    print(f"{len(doc['anchors'])} anchors -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
