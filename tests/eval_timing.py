#!/usr/bin/env python
"""Reproduce the speech/vision timing accuracy table (dev set).

Runs the real pipeline — ASR (faster-whisper, cached) -> anchor classification
-> energy-onset refinement — on every SYNTH_KNOWN fixture that has a
``ground_truth.json`` and reports timing error distributions against that
assembly truth (never against ASR output).

Usage
-----
    PYTHONPATH=src .venv/bin/python tests/eval_timing.py \
        [--fixtures fx_a_zh_portrait fx_b_zh_landscape ...] \
        [--out artifacts/anchors/timing_report.json]

Two independent measures are reported per fixture:

``word_onset``
    For every truth unit we build a calibration anchor on the ASR word whose
    start is nearest the unit onset (``anchors.unit_onset_anchors``), measure
    its error before/after refinement. This measures the onset snapper itself
    and covers (almost) every unit.
``anchor_onset``
    The anchors that production classification emitted, scored only when their
    text sits at the *start* of a truth unit (assembly truth is unit-level, so
    a mid-unit anchor has no independent target). This measures the anchors the
    renderer will actually use.

Truth used: SYNTH_KNOWN (per-unit TTS concatenation, onsets measured by
ffprobe during assembly; ASR is never used to define a target). This is a dev
calibration set, not real-human speech: spec §5 requires >= 50 independent
events for a P90 acceptance claim, so the totals are reported as exploratory
until enough fixtures/labels exist.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from semotion import anchors as A  # noqa: E402
from semotion import asr  # noqa: E402

# Timing-calibration fixtures: every 5-9 character unit starts with the phrase
# the rules must emit, so each unit yields an independently checkable event.
CALIBRATION_FIXTURES = ["fx_t1_zh_phrase", "fx_t2_zh_phrase", "fx_t3_zh_phrase"]

DEFAULT_FIXTURES = [
    "fx_a_zh_portrait",
    "fx_b_zh_landscape",
    "fx_c_zh_cuts",
    "fx_d_zh_portrait",
    "fx_e_zh_landscape",
    *CALIBRATION_FIXTURES,
]


def stats(errors_ms: list[int]) -> dict:
    if not errors_ms:
        return {"n": 0}
    values = np.asarray(errors_ms, dtype=float)
    return {
        "n": int(values.size),
        "p50_ms": int(round(np.percentile(values, 50))),
        "p90_ms": int(round(np.percentile(values, 90))),
        "max_ms": int(round(values.max())),
        "mean_ms": int(round(values.mean())),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", nargs="*", default=DEFAULT_FIXTURES)
    ap.add_argument("--clips", default=str(ROOT / "tests/fixtures/clips"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/anchors/timing_report.json"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts/anchors"))
    args = ap.parse_args(argv)

    clips = Path(args.clips)
    art = Path(args.artifacts)
    art.mkdir(parents=True, exist_ok=True)
    report: dict = {"per_fixture": {}, "totals": {}}
    word_before: list[int] = []
    word_after: list[int] = []
    anchor_before: list[int] = []
    anchor_after: list[int] = []
    units_total = 0
    calibration: dict = {}

    for fid in args.fixtures:
        clip = clips / fid / f"{fid}.mp4"
        gt_path = clips / fid / "ground_truth.json"
        if not clip.exists() or not gt_path.exists():
            print(f"[skip] {fid}: missing clip or ground truth")
            continue
        transcript = asr.transcribe(clip, art / f"{fid}.transcript.json")
        truth = A.load_truth(gt_path)
        units_total += len(truth)
        doc = A.classify(transcript, use_cache=False)
        refined = A.refine_onsets(clip, doc, out_json=art / f"{fid}.anchors.json")
        cal = A.unit_onset_anchors(transcript, truth)
        cal_refined = A.refine_onsets(clip, cal, out_json=art / f"{fid}.calibration.json")

        all_pairs, _unmatched = A.prefix_matches(refined, truth)     # after refinement
        before_pairs, _ = A.prefix_matches(doc, truth)               # raw ASR word spans
        ab = A.timing_errors(doc, truth)
        aa = A.timing_errors(refined, truth)
        wb = A.timing_errors(cal, truth)
        wa = A.timing_errors(cal_refined, truth)
        word_before += wb["errors_ms"]
        word_after += wa["errors_ms"]
        anchor_before += ab["errors_ms"]
        anchor_after += aa["errors_ms"]

        report["per_fixture"][fid] = {
            "units": len(truth),
            "segments": len(transcript["segments"]),
            "words": len(transcript["words"]),
            "anchors": len(refined["anchors"]),
            "kinds": refined["provenance"]["kinds"],
            "refine_accepted": refined["provenance"]["refine"]["n_accepted"],
            "refine_rejected": refined["provenance"]["refine"]["n_rejected"],
            "word_onset": {
                "before": stats(wb["errors_ms"]),
                "after": stats(wa["errors_ms"]),
                "unmatched_units": len(wb["unmatched_units"]),
            },
            "anchor_onset": {
                "samples": ab["samples"],
                "before": stats(ab["errors_ms"]),
                "after": stats(aa["errors_ms"]),
                "unmatched_units": ab["unmatched_units"],
                "per_unit_after_ms": {p["unit_id"]: p["error_ms"] for p in aa["matched"]},
                "per_unit_before_ms": {p["unit_id"]: p["error_ms"] for p in ab["matched"]},
            },
            "refine_deltas_ms": [a["refine"]["delta_ms"] for a in refined["anchors"]],
            "prefix_events": {
                "anchors_with_prefix_text": len(all_pairs),
                "units_with_prefix_anchor": len({p["unit_id"] for p in all_pairs}),
                "pairs": [
                    {"unit_id": p["unit_id"], "unit_text": p["unit_text"],
                     "anchor_id": p["anchor_id"], "anchor_kind": p["anchor_kind"],
                     "anchor_text": p["anchor_text"],
                     "error_ms_before": p["error_ms"]}
                    for p in all_pairs
                ],
            },
        }
        if fid in CALIBRATION_FIXTURES:
            calibration[fid] = {
                "units": len(truth),
                "anchors": len(refined["anchors"]),
                "events": len({p["unit_id"] for p in all_pairs}),
                "anchors_with_prefix_text": len(all_pairs),
                "errors_before_ms": [p["error_ms"] for p in before_pairs],
                "errors_after_ms": [p["error_ms"] for p in all_pairs],
                "pairs": report["per_fixture"][fid]["prefix_events"]["pairs"],
            }
        print(f"{fid}: units={len(truth)} anchors={len(refined['anchors'])} "
              f"anchor n={aa['samples']} before p50/p90/max="
              f"{ab['p50_ms']}/{ab['p90_ms']}/{ab['max_ms']} after="
              f"{aa['p50_ms']}/{aa['p90_ms']}/{aa['max_ms']}")

    report["totals"] = {
        "units_total": units_total,
        "word_onset": {
            "before": stats(word_before), "after": stats(word_after),
            "errors_before_ms": word_before, "errors_after_ms": word_after,
        },
        "anchor_onset": {
            "before": stats(anchor_before), "after": stats(anchor_after),
            "errors_before_ms": anchor_before, "errors_after_ms": anchor_after,
        },
        "truth_class": "SYNTH_KNOWN (per-unit TTS assembly, ffprobe-measured; ASR not used)",
        "split": "dev",
        "thresholds": {"p90_target_ms": 100, "min_events_for_acceptance": 50},
        "verdict": ("PASS (exploratory)" if anchor_after and stats(anchor_after)["p90_ms"] <= 100
                    and len(anchor_after) >= 50 else "EXPLORATORY (<50 events)"),
        "note": ("Dev calibration on synthetic TTS, not real human speech; "
                 "P90 acceptance per spec §5 needs >= 50 independent events "
                 "and at least one independently labelled (HUMAN) set."),
    }
    if calibration:
        calib_report = {
            "schema": "semotion.timing-calibration/1",
            "created_utc": report["per_fixture"] and A._now(),
            "truth_class": "SYNTH_KNOWN",
            "split": "calibration",
            "purpose": ("Minimum-event timing calibration: every unit starts with a phrase the "
                        "anchor rules emit, so anchor->unit onset comparison is independent of ASR."),
            "match_rule": ("anchor text (punctuation stripped, Traditional->Simplified folded) is a "
                           "prefix of the unit text AND |anchor.start - unit.start| <= 0.9 s; "
                           "one nearest anchor per unit counts as an event"),
            "fixtures": calibration,
            "totals": {
                "units": sum(c["units"] for c in calibration.values()),
                "events": sum(c["events"] for c in calibration.values()),
                "errors_before_ms": [e for c in calibration.values() for e in c["errors_before_ms"]],
                "errors_after_ms": [e for c in calibration.values() for e in c["errors_after_ms"]],
                "before": stats([e for c in calibration.values() for e in c["errors_before_ms"]]),
                "after": stats([e for c in calibration.values() for e in c["errors_after_ms"]]),
                "min_events_required": 50,
                "verdict": ("PASS" if sum(c["events"] for c in calibration.values()) >= 50
                            else "FAIL"),
            },
            "note": ("Calibration material only: synthetic TTS with ffprobe-measured assembly "
                     "truth. It says nothing about real-human speech or aesthetics."),
        }
        calib_path = art / "calibration_report.json"
        calib_path.write_text(json.dumps(calib_report, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
        print(f"calibration events: {calib_report['totals']['events']} -> {calib_path}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["totals"]["anchor_onset"], ensure_ascii=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
