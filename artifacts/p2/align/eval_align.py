#!/usr/bin/env python
"""Evidence: score ``semotion.align`` against SYNTH_KNOWN fixtures.

For every fixture with a ``ground_truth.json`` and a *cached* faster-whisper
transcript (``artifacts/cache/asr``), this script:

* loads the existing ASR transcript through the real pipeline (cache hit, no
  model call), then computes the old baselines from it
  (``word``: nearest ASR word start; ``naive_char``: every word evenly split
  over its characters, which is how captions were built before),
* runs the new envelope-DP aligner,
* scores predicted character onsets against per-unit assembly truth (nearest
  onset, the same rule as ``tests/eval_timing.py``),
* aggregates the distributions (pooled n) and writes JSON.

Usage::

    PYTHONPATH=src .venv/bin/python artifacts/p2/align/eval_align.py \
        [--out artifacts/p2/align/fixtures_score.json] [--params '{"w_dur": 1.2}']
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from semotion import align, asr  # noqa: E402

FIXTURES = [
    "fx_a_zh_portrait", "fx_b_zh_landscape", "fx_c_zh_cuts",
    "fx_d_zh_portrait", "fx_e_zh_landscape", "fx_face_pan",
    "fx_face_zh_portrait", "fx_t1_zh_phrase", "fx_t2_zh_phrase",
    "fx_t3_zh_phrase",
]


def stats(values_ms: list[float]) -> dict:
    values = np.asarray([abs(v) for v in values_ms], dtype=float)
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "p50_ms": round(float(np.percentile(values, 50)), 2),
        "p90_ms": round(float(np.percentile(values, 90)), 2),
        "p95_ms": round(float(np.percentile(values, 95)), 2),
        "max_ms": round(float(values.max()), 2),
        "mean_ms": round(float(values.mean()), 2),
        "bias_ms": round(float(np.mean(values_ms)), 2),
    }


def nearest_errors(onsets: list[float], units: list[dict]) -> list[float]:
    """Signed (predicted - truth) ms of the nearest onset per unit."""
    out: list[float] = []
    for unit in units:
        t = float(unit["start_s"])
        if not onsets:
            continue
        nearest = min(onsets, key=lambda o: abs(o - t))
        out.append((nearest - t) * 1000.0)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "artifacts/p2/align/fixtures_score.json"))
    ap.add_argument("--clips", default=str(ROOT / "tests/fixtures/clips"))
    ap.add_argument("--params", default=None, help="JSON object overriding DEFAULT_PARAMS")
    ap.add_argument("--fixtures", nargs="*", default=FIXTURES)
    args = ap.parse_args(argv)
    params = json.loads(args.params) if args.params else None

    clips = Path(args.clips)
    rows: list[dict] = []
    pooled_new: list[float] = []
    pooled_word: list[float] = []
    pooled_naive: list[float] = []
    skipped: list[dict] = []
    t_all = time.time()

    for fid in args.fixtures:
        gt_path = clips / fid / "ground_truth.json"
        video = clips / fid / f"{fid}.mp4"
        if not gt_path.exists() or not video.exists():
            skipped.append({"fixture": fid, "reason": "missing fixture"})
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        cached = asr.cache_path(gt["sha256"], "medium", "cpu")
        if not cached.exists():
            skipped.append({"fixture": fid, "reason": "no cached transcript"})
            continue
        transcript = asr.transcribe(video, ROOT / "artifacts/p2/align/work" / f"{fid}.transcript.json")
        t0 = time.time()
        doc = align.align_video(video, transcript, params=params)
        elapsed = time.time() - t0
        score = align.score_unit_onsets(doc, gt)
        old_word = align.score_word_onsets(transcript, gt, mode="word")
        old_naive = align.score_word_onsets(transcript, gt, mode="naive_char")
        pooled_new.extend([r["error_ms"] for r in score["rows"]])
        word_onsets = sorted(float(w["start_s"]) for w in transcript["words"])
        naive_onsets = sorted(align._naive_char_onsets(transcript))
        pooled_word.extend(nearest_errors(word_onsets, gt["units"]))
        pooled_naive.extend(nearest_errors(naive_onsets, gt["units"]))
        rows.append({
            "fixture": fid,
            "duration_s": gt["duration_s"],
            "n_units": len(gt["units"]),
            "n_scored": score["n_scored"],
            "n_chars": len(doc["chars"]),
            "n_syllables": len(doc["syllables"]),
            "chunks": doc["quality"]["n_chunks"],
            "feasible_chunks": doc["quality"]["feasible_chunks"],
            "median_syllable_ms": doc["quality"]["syllable_ms"]["median"],
            "monotonicity_violations": doc["quality"]["monotonicity_violations"],
            "onset_percentile_mean": doc["quality"]["onset_envelope"]["onset_percentile_mean"],
            "point_biserial_r": doc["quality"]["onset_envelope"]["point_biserial_r"],
            "low_conf": doc["quality"]["low_confidence_syllables"],
            "elapsed_s": round(elapsed, 3),
            "new": {k: v for k, v in score.items() if k != "rows"},
            "old_word": old_word,
            "old_naive_char": old_naive,
            "rows": score["rows"],
            "warnings": doc["warnings"],
        })
        print(f"[{fid}] n={score['n_scored']} new P90={score['p90_ms']}ms "
              f"old_word P90={old_word['p90_ms']}ms  old_naive P90={old_naive['p90_ms']}ms "
              f"median_syll={doc['quality']['syllable_ms']['median']}ms "
              f"R={doc['quality']['onset_envelope']['point_biserial_r']} ({elapsed:.2f}s)")

    report = {
        "schema": "semotion.align.evidence/1",
        "kind": "SYNTH_KNOWN dev fixtures (per-unit TTS assembly truth)",
        "params": params,
        "n_fixtures": len(rows),
        "n_units": int(sum(r["n_units"] for r in rows)),
        "n_scored": int(sum(r["n_scored"] for r in rows)),
        "pooled_new": stats(pooled_new),
        "pooled_old_word": stats(pooled_word),
        "pooled_old_naive_char": stats(pooled_naive),
        "elapsed_s": round(time.time() - t_all, 2),
        "skipped": skipped,
        "fixtures": rows,
        "note": ("Nearest predicted character onset per truth unit (same rule as "
                 "tests/eval_timing.py). SYNTH_KNOWN is concatenated per-unit TTS, "
                 "not human speech; see docs/ALIGNMENT.md for coverage limits."),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {k: report[k] for k in ("n_fixtures", "n_units", "n_scored",
                                      "pooled_new", "pooled_old_word",
                                      "pooled_old_naive_char", "elapsed_s")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
