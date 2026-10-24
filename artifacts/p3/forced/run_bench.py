#!/usr/bin/env python
"""Benchmark: envelope-DP (:mod:`semotion.align`) vs CTC forced alignment
(:mod:`semotion.forced`).

Writes ``artifacts/p3/forced/evidence.json`` and ``timeline.png``.

Truth classes (never mixed):
  * SYNTH_KNOWN  — per-unit TTS concatenated on a measured timeline
                   (``tests/fixtures/clips/*/ground_truth.json``).  The unit
                   start is the splice point; it is valid truth for *onset
                   placement*, not for human speech.
  * REAL_UNVERIFIED — ``inputs/zhengqinwen_25s.mp4`` + the cached Whisper
                   transcript.  There are no human labels here: we measure
                   posterior confidence, monotonicity and agreement with the
                   envelope aligner, never "accuracy".

Run:
    PYTHONPATH=src .venv/bin/python artifacts/p3/forced/run_bench.py
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from semotion import align, asr, forced  # noqa: E402

os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("SEMOTION_FORCED_OFFLINE", "1")

CLIPS = ROOT / "tests/fixtures/clips"
REAL_VIDEO = ROOT / "inputs/zhengqinwen_25s.mp4"
REAL_TRANSCRIPT = ROOT / "artifacts/dl/run/transcript.json"
METHODS = ("energy", "ctc")


# ------------------------------------------------------------------ helpers


def _ms(values):
    return [float(v) for v in values]


def _summary(abs_values):
    if not abs_values:
        return {"n": 0, "p50_ms": None, "p90_ms": None, "max_ms": None, "mean_ms": None}
    arr = np.asarray(abs_values, dtype=float)
    return {
        "n": int(arr.size),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "max_ms": round(float(arr.max()), 2),
        "mean_ms": round(float(arr.mean()), 2),
    }


def _fixture_transcript(gt: dict) -> dict | None:
    cache = asr.cache_path(gt["sha256"], "medium", "cpu")
    if not cache.exists():
        return None
    return json.loads(cache.read_text(encoding="utf-8"))


def _time_call(fn):
    t0 = time.perf_counter()
    doc = fn()
    return doc, round(time.perf_counter() - t0, 2)


def _file_size(path: Path) -> int:
    if not path.exists():
        return 0
    return int(path.stat().st_size)


# --------------------------------------------------------------- synthetic set


def run_synthetic() -> dict:
    fixtures = []
    pooled = {m: [] for m in METHODS}
    failures = []
    for gt_path in sorted(CLIPS.glob("*/ground_truth.json")):
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        name = gt_path.parent.name
        video = gt_path.parent / f"{name}.mp4"
        transcript = _fixture_transcript(gt)
        row = {
            "fixture": name,
            "kind": gt.get("kind"),
            "truth": "SYNTH_KNOWN",
            "n_units": len(gt.get("units") or []),
            "duration_s": gt.get("duration_s"),
            "asr_transcript_cached": transcript is not None,
        }
        if transcript is None or not video.exists():
            row["status"] = "SKIPPED (no cached ASR transcript)"
            fixtures.append(row)
            continue
        try:
            energy_doc, energy_s = _time_call(
                lambda: align.align_video(video, transcript, language="zh"))
            ctc_doc, ctc_s = _time_call(
                lambda: forced.align_video(video, transcript, language="zh",
                                           params={"max_chunk_s": 20.0}))
            scored = {}
            for method, doc in (("energy", energy_doc), ("ctc", ctc_doc)):
                sc = align.score_unit_onsets(doc, gt)
                scored[method] = {k: v for k, v in sc.items() if k != "rows"}
                scored[method]["runtime_s"] = energy_s if method == "energy" else ctc_s
                scored[method]["median_syllable_ms"] = (doc["quality"]["syllable_ms"]["median"])
                scored[method]["monotonicity_violations"] = (
                    doc["quality"]["monotonicity_violations"])
                pooled[method].extend(abs(float(r["error_ms"])) for r in sc["rows"])
            row.update({
                "status": "OK",
                "energy": scored["energy"],
                "ctc": scored["ctc"],
                "ctc_posterior": {
                    "mean_conf": ctc_doc["quality"]["posterior"]["mean_conf"],
                    "low_conf_rate": ctc_doc["quality"]["posterior"]["low_conf_rate"],
                    "mean_logp": ctc_doc["quality"]["posterior"]["mean_logp"],
                    "alignment_score_per_frame":
                        ctc_doc["quality"]["alignment_score_per_frame"],
                },
                "ctc_coverage": ctc_doc["quality"]["coverage"],
                "ctc_n_chunks": ctc_doc["quality"]["n_chunks"],
            })
            if name == "fx_a_zh_portrait":
                row["_docs"] = {"energy": energy_doc, "ctc": ctc_doc}
        except Exception as exc:  # pragma: no cover - evidence must not lie
            row.update({"status": f"FAILED: {type(exc).__name__}: {exc}"})
            failures.append({"fixture": name, "error": traceback.format_exc(limit=3)})
        fixtures.append(row)

    aggregate = {}
    for method in METHODS:
        values = pooled[method]
        aggregate[method] = _summary(values)
        aggregate[method]["n_fixtures"] = sum(
            1 for f in fixtures if f.get("status") == "OK" and method in f)
    ranked = [f for f in fixtures if f.get("status") == "OK"]
    wins = {"ctc_better": 0, "energy_better": 0, "tie": 0}
    for f in ranked:
        e = f["energy"]["p50_ms"]
        c = f["ctc"]["p50_ms"]
        if e is None or c is None:
            continue
        if c < e - 0.5:
            wins["ctc_better"] += 1
        elif e < c - 0.5:
            wins["energy_better"] += 1
        else:
            wins["tie"] += 1
    return {
        "truth_class": "SYNTH_KNOWN",
        "metric": "unit-start error: nearest predicted character onset vs the "
                  "TTS assembly boundary (same rule as align.score_unit_onsets)",
        "n_fixtures_scored": len(ranked),
        "n_units_scored": len(pooled["energy"]),
        "aggregate": aggregate,
        "per_fixture_p50_wins": wins,
        "fixtures": fixtures,
        "failures": failures,
    }


# --------------------------------------------------------------- real speech


def _char_pairs(energy_doc: dict, ctc_doc: dict) -> tuple[list[tuple[dict, dict]], int]:
    """Pair characters from both documents in order (same tokenizer, same text)."""
    a = energy_doc.get("chars") or []
    b = ctc_doc.get("chars") or []
    pairs = []
    unmatched = 0
    for i in range(min(len(a), len(b))):
        if a[i].get("text") == b[i].get("text"):
            pairs.append((a[i], b[i]))
        else:
            unmatched += 1
    unmatched += abs(len(a) - len(b))
    return pairs, unmatched


def _agreement(pairs):
    if not pairs:
        return {"n": 0}
    d_start = np.array([abs(float(c["start_s"]) - float(e["start_s"])) * 1e3
                        for e, c in pairs])
    d_end = np.array([abs(float(c["end_s"]) - float(e["end_s"])) * 1e3
                      for e, c in pairs])
    conf = np.array([float(c["conf"]) for _e, c in pairs])
    out = {
        "n": int(d_start.size),
        "onset": {
            "median_abs_ms": round(float(np.median(d_start)), 1),
            "p90_abs_ms": round(float(np.percentile(d_start, 90)), 1),
            "within_50ms_rate": round(float((d_start <= 50).mean()), 4),
            "within_100ms_rate": round(float((d_start <= 100).mean()), 4),
        },
        "offset": {
            "median_abs_ms": round(float(np.median(d_end)), 1),
            "p90_abs_ms": round(float(np.percentile(d_end, 90)), 1),
            "within_50ms_rate": round(float((d_end <= 50).mean()), 4),
        },
    }
    if d_start.size >= 10:
        med = float(np.median(conf))
        hi = d_start[conf >= med]
        lo = d_start[conf < med]
        out["confidence_vs_agreement"] = {
            "confidence_split": round(med, 3),
            "high_conf_median_abs_ms": round(float(np.median(hi)), 1) if hi.size else None,
            "low_conf_median_abs_ms": round(float(np.median(lo)), 1) if lo.size else None,
            "spearman_conf_vs_abs_delta": round(float(np.corrcoef(
                np.argsort(np.argsort(conf)), np.argsort(np.argsort(d_start)))[0, 1]), 3),
        }
    return out


def _whisper_third_opinion(transcript: dict, energy_doc: dict, ctc_doc: dict) -> dict:
    """Compare both aligners against Whisper's *word* times.

    Whisper word times come from cross-attention, i.e. a different timing
    mechanism from a CTC posterior path, so they are a useful (still
    MACHINE_WEAK) third opinion on where a word starts.  Whisper is not ground
    truth, and it produced the transcript text both aligners consume.
    """
    def word_times(doc):
        out = {}
        for w in transcript.get("words") or []:
            sid = str(w.get("segment_id"))
            cs, ce = int(w["char_start"]), int(w["char_end"])
            chars = [x for x in doc.get("chars") or []
                     if x["segment_id"] == sid
                     and cs <= int(x["char_start"]) < ce]
            if chars:
                out[w["id"]] = (float(min(float(x["start_s"]) for x in chars)),
                                float(w["start_s"]),
                                float(max(float(x["end_s"]) for x in chars)),
                                float(w["end_s"]))
        return out

    we, wc = word_times(energy_doc), word_times(ctc_doc)
    common = sorted(set(we) & set(wc))
    if not common:
        return {"n": 0}
    out = {"n": int(len(common)), "reference": "Whisper word times (MACHINE_WEAK)"}
    for name, table in (("energy", we), ("ctc", wc)):
        d_start = np.array([abs(table[k][0] - table[k][1]) * 1e3 for k in common])
        d_end = np.array([abs(table[k][2] - table[k][3]) * 1e3 for k in common])
        out[name] = {
            "word_start_median_abs_ms": round(float(np.median(d_start)), 1),
            "word_start_p90_abs_ms": round(float(np.percentile(d_start, 90)), 1),
            "word_start_within_100ms_rate": round(float((d_start <= 100).mean()), 4),
            "word_end_median_abs_ms": round(float(np.median(d_end)), 1),
        }
    # does the envelope aligner stretch characters to padded segment bounds?
    by_seg: dict[str, list] = {}
    for a, b in zip(energy_doc.get("chars") or [], ctc_doc.get("chars") or []):
        if a.get("text") != b.get("text"):
            continue
        by_seg.setdefault(a["segment_id"], []).append(
            (float(b["start_s"]) - float(a["start_s"])) * 1e3)
    first_half, second_half = [], []
    for deltas in by_seg.values():
        n = len(deltas)
        for i, value in enumerate(deltas):
            (first_half if i < n / 2 else second_half).append(value)
    out["signed_delta_ctc_minus_energy_within_segment"] = {
        "first_half_median_ms": round(float(np.median(first_half)), 1) if first_half else None,
        "second_half_median_ms": round(float(np.median(second_half)), 1) if second_half else None,
        "interpretation": "energy spreads characters to the Whisper segment bounds; "
                          "CTC follows the acoustics, so it runs early in the second "
                          "half of padded segments",
    }
    return out


def _degenerate(doc: dict, transcript: dict) -> dict:
    chars = doc.get("chars") or []
    stride = float(doc["method"]["model_info"]["frame_stride_s"]) \
        if doc.get("method", {}).get("model_info") else 0.02
    repeated = []
    for seg in transcript.get("segments") or []:
        text = str(seg.get("text") or "")
        i = 0
        while i < len(text):
            j = i + 1
            while j < len(text) and text[j] == text[i]:
                j += 1
            if j - i >= 2:
                repeated.append({"segment_id": seg.get("id"), "text": text[i:j],
                                 "count": j - i})
            i = j
    # do repeated characters keep distinct, ordered spans?  Look them up by
    # (segment, char offset) so a third occurrence elsewhere is not confused.
    by_offset = {(c.get("segment_id"), int(c.get("char_start"))): c for c in chars}
    runs = []
    for rep in repeated:
        seg = next((s for s in transcript.get("segments") or []
                    if str(s.get("id")) == rep["segment_id"]), None)
        text = str((seg or {}).get("text") or "")
        start = text.find(rep["text"])
        spans = []
        for k in range(rep["count"]):
            c = by_offset.get((rep["segment_id"], start + k))
            if c is not None:
                spans.append([float(c["start_s"]), float(c["end_s"]), float(c["conf"])])
        distinct = (len({(round(a, 6), round(b, 6)) for a, b, _c in spans}) == len(spans))
        ordered = all(spans[i][1] <= spans[i + 1][0] + 1e-6 for i in range(len(spans) - 1))
        runs.append({**rep, "aligned_spans": spans, "distinct_spans": distinct,
                     "strictly_ordered": ordered})
    short = [c for c in chars if float(c["end_s"]) - float(c["start_s"]) <= 2 * stride + 1e-9]
    long = [c for c in chars if float(c["end_s"]) - float(c["start_s"]) > 1.0]
    q = doc.get("quality") or {}
    return {
        "n_chars": len(chars),
        "monotonicity_violations": q.get("monotonicity_violations"),
        "oov_symbols": q.get("coverage", {}).get("oov"),
        "coverage": q.get("coverage", {}).get("coverage"),
        "approx_tokens": q.get("coverage", {}).get("approx_tokens"),
        "repeated_character_runs_in_text": repeated,
        "repeated_runs_aligned": runs,
        "chars_at_or_below_2_frames": len(short),
        "chars_longer_than_1s": len(long),
        "max_char_duration_s": round(max((float(c["end_s"]) - float(c["start_s"]))
                                          for c in chars), 3) if chars else None,
        "min_char_duration_s": round(min((float(c["end_s"]) - float(c["start_s"]))
                                          for c in chars), 3) if chars else None,
        "degenerate_targets_without_frames": q.get("degenerate_targets"),
        "chunk_feasible": q.get("chunk_feasible"),
        "note": ("character CTC: the phoneme-level degenerate case is a run of "
                 "identical characters; distinct spans are checked above"),
    }


def _positive_control(doc: dict) -> dict:
    """How much posterior mass the correct transcript gets, for reference."""
    q = doc.get("quality") or {}
    return {
        "mean_conf": q.get("posterior", {}).get("mean_conf"),
        "median_conf": q.get("posterior", {}).get("median_conf"),
        "alignment_score_per_frame": q.get("alignment_score_per_frame"),
        "n_chunks": q.get("n_chunks"),
    }


def run_real() -> dict:
    gt = {"truth": "REAL_UNVERIFIED"}
    transcript = json.loads(REAL_TRANSCRIPT.read_text(encoding="utf-8"))
    energy_doc, energy_s = _time_call(
        lambda: align.align_video(REAL_VIDEO, transcript, language="zh"))
    ctc_doc, ctc_s = _time_call(
        lambda: forced.align_video(REAL_VIDEO, transcript, language="zh",
                                   params={"max_chunk_s": 20.0}))
    pairs, unmatched = _char_pairs(energy_doc, ctc_doc)
    agreement = _agreement(pairs)
    third = _whisper_third_opinion(transcript, energy_doc, ctc_doc)

    # negative control: same audio, same symbol multiset, wrong order
    model = forced.load_model(None, language="zh", allow_download=False)
    samples, rate = align.load_audio(REAL_VIDEO)
    t0 = time.perf_counter()
    lp, stride = model.log_probs(samples, rate)
    infer_s = round(time.perf_counter() - t0, 2)
    base_doc = forced.align_posteriors(
        transcript, lp, model.vocab, blank=model.blank, frame_stride_s=model.frame_stride_s,
        language="zh", model=model, params={"max_chunk_s": 0, "ctc_engine": "numpy"})
    shuffled = json.loads(json.dumps(transcript, ensure_ascii=False))
    rng = np.random.default_rng(20261025)
    for seg in shuffled.get("segments") or []:
        chars = list(str(seg.get("text") or ""))
        rng.shuffle(chars)
        seg["text"] = "".join(chars)
    shuf_doc = forced.align_posteriors(
        shuffled, lp, model.vocab, blank=model.blank, frame_stride_s=model.frame_stride_s,
        language="zh", model=model, params={"max_chunk_s": 0, "ctc_engine": "numpy"})
    pos = _positive_control(base_doc)
    neg = _positive_control(shuf_doc)

    return {
        "truth_class": "REAL_UNVERIFIED",
        "source": {
            "video": str(REAL_VIDEO.relative_to(ROOT)),
            "video_bytes": _file_size(REAL_VIDEO),
            "duration_s": 25.0,
            "transcript": str(REAL_TRANSCRIPT.relative_to(ROOT)),
            "transcript_model": (transcript.get("model") or {}).get("name"),
            "n_segments": len(transcript.get("segments") or []),
        },
        "energy": {
            "runtime_s": energy_s,
            "n_chars": len(energy_doc.get("chars") or []),
            "monotonicity_violations": energy_doc["quality"]["monotonicity_violations"],
            "median_syllable_ms": energy_doc["quality"]["syllable_ms"]["median"],
        },
        "ctc": {
            "runtime_s": ctc_s,
            "n_chars": len(ctc_doc.get("chars") or []),
            "monotonicity_violations": ctc_doc["quality"]["monotonicity_violations"],
            "median_syllable_ms": ctc_doc["quality"]["syllable_ms"]["median"],
            "posterior": ctc_doc["quality"]["posterior"],
            "coverage": ctc_doc["quality"]["coverage"],
            "n_chunks": ctc_doc["quality"]["n_chunks"],
            "chunk_feasible": ctc_doc["quality"]["chunk_feasible"],
            "alignment_score_per_frame": ctc_doc["quality"]["alignment_score_per_frame"],
            "full_clip_inference_s": infer_s,
        },
        "boundary_agreement_energy_vs_ctc": {
            "matched_chars": len(pairs),
            "unmatched_chars": unmatched,
            **agreement,
            "caveat": "agreement is not accuracy: neither method has human labels here",
        },
        "third_opinion_whisper_word_times": third,
        "degenerate_cases": _degenerate(ctc_doc, transcript),
        "negative_control": {
            "design": "same audio posteriors, character order shuffled within each "
                      "segment (same symbol multiset), single 25 s pass",
            "positive": pos,
            "shuffled": neg,
            "score_delta": (
                round((neg["alignment_score_per_frame"] or 0)
                      - (pos["alignment_score_per_frame"] or 0), 4)),
            "mean_conf_delta": round((neg["mean_conf"] or 0) - (pos["mean_conf"] or 0), 4),
            "verdict": "shuffled targets score materially worse" if (
                (neg["alignment_score_per_frame"] or -99) <
                (pos["alignment_score_per_frame"] or -99) - 0.2) else "NO CLEAR DROP",
        },
        "_docs": {"energy": energy_doc, "ctc": ctc_doc,
                  "posterior_ctc": base_doc, "posterior_shuffled": shuf_doc},
    }


# ------------------------------------------------------------------ timeline


def _font(size: int):
    from PIL import ImageFont
    for path in ("/System/Library/Fonts/Helvetica.ttc",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_timeline(real: dict, synth: dict, fx_docs: dict, path: Path) -> None:
    from PIL import Image, ImageDraw
    W, H = 1600, 1050
    img = Image.new("RGB", (W, H), (250, 250, 248))
    d = ImageDraw.Draw(img)
    f_title, f, f_small = _font(26), _font(17), _font(14)

    # ---------------- panel A: synthetic truth (SYNTH_KNOWN)
    fx = next((x for x in synth["fixtures"]
               if x["fixture"] == "fx_a_zh_portrait" and x.get("status") == "OK"), None)
    d.text((40, 22), "Panel A  fx_a_zh_portrait  (SYNTH_KNOWN: TTS unit splice points)",
           fill=(20, 20, 20), font=f_title)
    if fx:
        gt = json.loads((CLIPS / "fx_a_zh_portrait/ground_truth.json").read_text())
        duration = float(gt["duration_s"])
        x0, x1, y0 = 90, 1560, 90
        sx = (x1 - x0) / duration
        d.text((40, 60), "unit bars (truth), then predicted character onsets",
               fill=(90, 90, 90), font=f_small)
        for unit in gt["units"]:
            a = x0 + float(unit["start_s"]) * sx
            b = x0 + float(unit["end_s"]) * sx
            d.rectangle([a, y0, b, y0 + 16], fill=(214, 214, 210), outline=(160, 160, 156))
        energy_doc = fx_docs.get("energy")
        ctc_doc = fx_docs.get("ctc")

        def ticks(doc, y, color):
            if doc is None:
                return
            for c in doc.get("chars") or []:
                if c.get("kind") != "han":
                    continue
                x = x0 + float(c["start_s"]) * sx
                d.line([x, y, x, y + 26], fill=color, width=2)

        ticks(energy_doc, y0 + 26, (40, 90, 200))
        ticks(ctc_doc, y0 + 58, (220, 110, 20))
        d.text((40, y0 + 26), "energy", fill=(40, 90, 200), font=f_small)
        d.text((40, y0 + 58), "ctc", fill=(220, 110, 20), font=f_small)

        # error scatter vs unit index
        ey = 290
        d.text((40, ey - 52), "unit-start error (ms): blue = energy DP, orange = CTC "
                              f"(n={fx['ctc']['n_scored']} units)",
               fill=(60, 60, 60), font=f)
        for method, color in (("energy", (40, 90, 200)), ("ctc", (220, 110, 20))):
            rows = align.score_unit_onsets(
                energy_doc if method == "energy" else ctc_doc, gt)["rows"]
            for row in rows:
                x = x0 + float(row["truth_start_s"]) * sx
                yerr = float(row["error_ms"])
                y = ey + 90 - max(-60.0, min(60.0, yerr * 2.0))
                d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color)
        d.line([x0, ey + 90, x1, ey + 90], fill=(150, 150, 150), width=1)
        for label, value in (("+50 ms", 50), ("0", 0), ("-50 ms", -50)):
            y = ey + 90 - value * 2.0
            d.text((x0 - 82, y - 8), label, fill=(120, 120, 120), font=f_small)
        d.text((40, ey + 200),
               f"aggregate over {synth['n_fixtures_scored']} fixtures / "
               f"{synth['n_units_scored']} units: "
               f"energy P50 {synth['aggregate']['energy']['p50_ms']} ms, "
               f"P90 {synth['aggregate']['energy']['p90_ms']} ms | "
               f"CTC P50 {synth['aggregate']['ctc']['p50_ms']} ms, "
               f"P90 {synth['aggregate']['ctc']['p90_ms']} ms",
               fill=(20, 20, 20), font=f)

    # ---------------- panel B: real speech (REAL_UNVERIFIED)
    by = 560
    d.text((40, by), "Panel B  zhengqinwen_25s  (REAL_UNVERIFIED: no human labels)",
           fill=(20, 20, 20), font=f_title)
    duration = float(real["source"]["duration_s"])
    x0, x1 = 90, 1560
    sx = (x1 - x0) / duration
    transcript = json.loads(REAL_TRANSCRIPT.read_text(encoding="utf-8"))
    d.text((40, by + 34), "Whisper segments (grey), energy onsets (blue), CTC onsets "
                          "(orange; bar height = posterior confidence)",
           fill=(90, 90, 90), font=f_small)
    for seg in transcript.get("segments") or []:
        a = x0 + float(seg["start_s"]) * sx
        b = x0 + float(seg["end_s"]) * sx
        d.rectangle([a, by + 60, b, by + 74], fill=(222, 222, 218), outline=(180, 180, 176))
    energy_doc = real["_docs"]["energy"]
    ctc_doc = real["_docs"]["ctc"]
    for c in energy_doc.get("chars") or []:
        if c.get("kind") != "han":
            continue
        x = x0 + float(c["start_s"]) * sx
        d.line([x, by + 78, x, by + 112], fill=(40, 90, 200), width=2)
    for c in ctc_doc.get("chars") or []:
        if c.get("kind") != "han":
            continue
        x = x0 + float(c["start_s"]) * sx
        h = 14 + 26 * float(c.get("conf") or 0.0)
        d.line([x, by + 120, x, by + 120 + h], fill=(220, 110, 20), width=2)

    pairs, _unmatched = _char_pairs(energy_doc, ctc_doc)
    ay = by + 320
    d.text((40, ay - 40), "onset disagreement |energy - ctc| per character (ms); "
                          f"median {real['boundary_agreement_energy_vs_ctc'].get('onset', {}).get('median_abs_ms')} ms, "
                          f"within 50 ms {100 * (real['boundary_agreement_energy_vs_ctc'].get('onset', {}).get('within_50ms_rate') or 0):.0f}%",
           fill=(60, 60, 60), font=f)
    d.line([x0, ay + 90, x1, ay + 90], fill=(150, 150, 150), width=1)
    d.text((x0 - 70, ay + 82), "0", fill=(120, 120, 120), font=f_small)
    d.text((x0 - 86, ay + 90 - 100 * 0.55 - 8), "100 ms", fill=(120, 120, 120), font=f_small)
    for e, c in pairs:
        x = x0 + float(c["start_s"]) * sx
        delta = abs(float(e["start_s"]) - float(c["start_s"])) * 1e3
        y = ay + 90 - min(90.0, delta * 0.55)
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(150, 80, 150))

    nc = real["negative_control"]
    d.text((40, H - 70),
           "Negative control (same posteriors, shuffled characters): "
           f"score/frame {nc['positive']['alignment_score_per_frame']} -> "
           f"{nc['shuffled']['alignment_score_per_frame']}  |  "
           f"mean conf {nc['positive']['mean_conf']} -> {nc['shuffled']['mean_conf']}",
           fill=(20, 20, 20), font=f_small)
    d.text((40, H - 42),
           "Timeline PNG generated by artifacts/p3/forced/run_bench.py; "
           "SYNTH_KNOWN truth is TTS assembly order, REAL_UNVERIFIED has no truth.",
           fill=(120, 120, 120), font=f_small)
    img.save(path)


# ---------------------------------------------------------------------- main


def environment() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "machine": os.uname().machine if hasattr(os, "uname") else None,
        "numpy": np.__version__,
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["torch_threads"] = torch.get_num_threads()
    except Exception as exc:  # pragma: no cover
        env["torch"] = f"missing ({exc})"
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except Exception as exc:  # pragma: no cover
        env["transformers"] = f"missing ({exc})"
    try:
        import torchaudio
        env["torchaudio"] = torchaudio.__version__
    except Exception as exc:  # pragma: no cover
        env["torchaudio"] = f"missing ({exc})"
    return env


def main() -> int:
    if not forced.available("zh"):
        print("zh CTC model is not cached; refusing to invent numbers", file=sys.stderr)
        return 2
    started = datetime.now(timezone.utc)
    load_t0 = time.perf_counter()
    model = forced.load_model(None, language="zh", allow_download=False)
    load_s = round(time.perf_counter() - load_t0, 2)

    print("[1/3] synthetic fixtures ...", flush=True)
    synth = run_synthetic()
    print("[2/3] real clip ...", flush=True)
    real = run_real()
    print("[3/3] artefacts ...", flush=True)

    info = forced.model_info("zh")
    evidence = {
        "schema": "semotion.forced-evidence/1",
        "created_utc": started.isoformat().replace("+00:00", "Z"),
        "truth_classes": {
            "SYNTH_KNOWN": "TTS units concatenated on a measured timeline; truth is "
                           "the splice boundary. Valid for onset placement only.",
            "REAL_UNVERIFIED": "real human audio with an ASR transcript; no human "
                               "labels. Confidence/agreement/monotonicity only.",
        },
        "environment": environment(),
        "engine_paths": {
            "dp": "torchaudio.functional.forced_align" if forced._use_torchaudio()
                  else "numpy ctc_viterbi (torchaudio unavailable)",
            "acoustic_model": forced.ENGINE_HF,
            "mms_fa_evaluated": forced.mms_fa_probe(),
        },
        "model": {
            "id": model.model_id,
            "engine": model.engine,
            "device": model.device,
            "n_params": model.n_params,
            "vocab_size": len(model.vocab),
            "frame_stride_ms": round(model.frame_stride_s * 1e3, 2),
            "blank_id": model.blank,
            "cold_load_s": load_s,
            "cache": {
                "path": info["cache_dir"],
                "model_bytes": info["model_cache_bytes"],
                "hf_home_bytes": info["hf_home_bytes"],
            },
            "offline_load_env": info["offline_env"],
        },
        "energy_aligner": {
            "method": align.METHOD,
            "truth_class": "MACHINE_WEAK",
            "note": "deterministic envelope DP; no model, no download",
        },
        "synthetic": synth,
        "real": {k: v for k, v in real.items() if not k.startswith("_")},
        "disk_footprint": {
            "zh_model_bytes": info["model_cache_bytes"],
            "hf_home_bytes": info["hf_home_bytes"],
            "torch_cache_bytes": forced._dir_size(ROOT / ".cache/torch"),
            "uv_cache_bytes": forced._dir_size(ROOT / ".cache/uv"),
            "note": "model weights are symlinked into the HF hub cache; sizes are "
                    "reported per cache root without double counting",
        },
        "caveats": [
            "SYNTH_KNOWN truth is the TTS splice point, not a human syllable label; "
            "on that material the envelope DP is near-exact by construction because "
            "every unit starts at a spliced energy jump.",
            "The real clip has no human labels. 'Agreement with the energy aligner' is "
            "not accuracy; neither method is certified here.",
            "The zh engine is a character-CTC model (XLSR-53), i.e. syllable "
            "resolution, not phoneme resolution. MMS_FA is English-phoneme only and "
            "cannot align Mandarin.",
            "Confidence is the mean CTC posterior of the chosen state over the "
            "symbol's frames; it is not calibrated against human judgements.",
            "Runtime is CPU-only on this machine and excludes the one-off download.",
        ],
    }
    # documents are bulky: keep them in their own files, not inside evidence.json
    fx_row = next((x for x in synth["fixtures"] if x["fixture"] == "fx_a_zh_portrait"), None)
    fx_docs = (fx_row or {}).get("_docs") or {}
    for row in synth["fixtures"]:
        row.pop("_docs", None)
    for name, doc in (("fx_a_energy.json", fx_docs.get("energy")),
                      ("fx_a_ctc.json", fx_docs.get("ctc")),
                      ("real_energy.json", real["_docs"]["energy"]),
                      ("real_ctc.json", real["_docs"]["ctc"])):
        if doc:
            (OUT / name).write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    out = OUT / "evidence.json"
    out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    draw_timeline(real, synth, fx_docs, OUT / "timeline.png")
    (OUT / "real_docs.json").write_text(
        json.dumps({"energy": real["_docs"]["energy"], "ctc": real["_docs"]["ctc"]},
                   ensure_ascii=False) + "\n", encoding="utf-8")

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    print(f"peak RSS {peak_mb:.0f} MB")
    print(json.dumps({
        "synthetic": evidence["synthetic"]["aggregate"],
        "real_agreement": evidence["real"]["boundary_agreement_energy_vs_ctc"],
        "negative": evidence["real"]["negative_control"],
    }, ensure_ascii=False, indent=1))
    print(f"wrote {out} and {OUT / 'timeline.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
