#!/usr/bin/env python
"""Evidence builder for the P2 alignment work (artifacts/p2/align/).

Produces, deterministically:

* ``fx_a_zh_portrait.alignment.json`` + caption JSON/SRT/ASS for the SYNTH_KNOWN
  fixture (the only clip with character-level-ish truth: per-unit TTS assembly),
* ``zhengqinwen_25s.alignment.json`` + caption JSON/SRT/ASS for the real
  Mandarin talking-head clip,
* ``zhengqinwen_25s.timeline.png`` — waveform + onset envelope with predicted
  character onsets, ASR word starts and segment bounds overlaid (human check),
* ``fx_a_zh_portrait.timeline.png`` — same picture against the unit truth,
* ``fixtures_score.json`` — pooled character-onset error over 10 SYNTH_KNOWN
  fixtures (n=147 units) plus the old pipeline baselines,
* ``negative_controls.json`` — shuffled envelope, uniform baseline,
  attack-refinement ablation, and the real-speech evidence contrast,
* ``old_pipeline_fx_a.json`` — what the existing ASR/word pipeline reports for
  the fixture, for the before/after table,
* ``caption_samples.txt`` — cue listing for human inspection,
* ``evidence.json`` — one summary document with every measured number.

Usage::

    PYTHONPATH=src .venv/bin/python artifacts/p2/align/evidence.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from semotion import align, asr, captions  # noqa: E402

OUT = ROOT / "artifacts" / "p2" / "align"
WORK = OUT / "work"
CLIPS = ROOT / "tests" / "fixtures" / "clips"
FX_A = CLIPS / "fx_a_zh_portrait" / "fx_a_zh_portrait.mp4"
FX_A_GT = CLIPS / "fx_a_zh_portrait" / "ground_truth.json"
REAL = ROOT / "inputs" / "zhengqinwen_25s.mp4"
REAL_TR = ROOT / "artifacts" / "dl" / "run" / "transcript.json"
FIXTURES = [
    "fx_a_zh_portrait", "fx_b_zh_landscape", "fx_c_zh_cuts",
    "fx_d_zh_portrait", "fx_e_zh_landscape", "fx_face_pan",
    "fx_face_zh_portrait", "fx_t1_zh_phrase", "fx_t2_zh_phrase",
    "fx_t3_zh_phrase",
]

COL_BG = (16, 18, 26)
COL_GRID = (38, 42, 56)
COL_WAVE = (120, 132, 156)
COL_SPEECH = (30, 46, 48)
COL_ENV = (72, 196, 150)
COL_ENV_FILL = (26, 74, 62)
COL_CHAR = (255, 196, 64)
COL_CHAR_DARK = (150, 108, 18)
COL_WORD = (255, 122, 96)
COL_SEG = (96, 144, 255)
COL_TEXT = (226, 232, 240)
COL_DIM = (140, 150, 168)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in ("/System/Library/Fonts/Hiragino Sans GB.ttc",
                      "/System/Library/Fonts/STHeiti Medium.ttc",
                      "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:  # pragma: no cover - platform font quirk
                continue
    return ImageFont.load_default()


def _fmt(t: float) -> str:
    return f"{t:6.2f}s"


def timeline_png(samples: np.ndarray, doc: dict, *, title: str,
                 units: list[dict] | None, out_path: Path,
                 caption_doc: dict | None = None, sr: int = 16000) -> Path:
    """Waveform + onset envelope + character onsets + caption cues."""
    features = align.audio_features(samples)
    duration = len(samples) / sr
    width, height = 2400, 1200
    img = Image.new("RGB", (width, height), COL_BG)
    d = ImageDraw.Draw(img)
    f_title = _font(28)
    f_small = _font(17)
    f_label = _font(16)
    left, right = 70, width - 30

    def x_of(t: float) -> int:
        return int(left + (right - left) * min(max(t / max(duration, 1e-6), 0.0), 1.0))

    d.text((left, 14), title, font=f_title, fill=COL_TEXT)
    d.text((left, 50), "green = predicted character onset (semotion.align)   "
                       "orange = Whisper word start (old pipeline)   "
                       "blue dashes = ASR segment bounds", font=f_small, fill=COL_DIM)
    if units:
        d.text((left, 74), "grey diamonds = SYNTH_KNOWN unit truth (assembly)",
               font=f_small, fill=COL_DIM)

    # ---------------------------------------------------------------- waveform
    top, bottom = 108, 478
    mid = (top + bottom) // 2
    d.rectangle([left, top, right, bottom], outline=COL_GRID)
    for start, end in features.regions:
        d.rectangle([x_of(start), top, x_of(end), bottom], fill=COL_SPEECH)
    n_px = right - left
    step = max(1, samples.size // n_px)
    amp = float(np.percentile(np.abs(samples), 99.0)) if samples.size else 1.0
    amp = max(amp, 1e-6)
    half = (bottom - top) * 0.46
    for px in range(n_px):
        chunk = samples[px * step:(px + 1) * step]
        if chunk.size == 0:
            continue
        lo = float(np.clip(chunk.min() / amp, -1.0, 1.0))
        hi = float(np.clip(chunk.max() / amp, -1.0, 1.0))
        y0 = int(mid - hi * half)
        y1 = int(mid - lo * half)
        d.line([(left + px, y0), (left + px, y1)], fill=COL_WAVE)

    # segment bounds
    for seg in doc.get("segments") or []:
        x = x_of(float(seg["start_s"]))
        for y in range(top, bottom, 12):
            d.line([(x, y), (x, y + 5)], fill=COL_SEG, width=2)
        d.text((x + 3, top + 2), str(seg.get("id")), font=f_label, fill=COL_SEG)

    # old ASR word starts (from the transcript used as input)
    transcript_path = doc.get("_transcript_path")
    if transcript_path and Path(transcript_path).exists():
        tr = json.loads(Path(transcript_path).read_text(encoding="utf-8"))
        for word in tr.get("words") or []:
            x = x_of(float(word["start_s"]))
            d.line([(x, bottom - 26), (x, bottom - 6)], fill=COL_WORD, width=2)

    # unit truth
    if units:
        for unit in units:
            x = x_of(float(unit["start_s"]))
            d.polygon([(x, bottom - 22), (x + 7, bottom - 14), (x, bottom - 6),
                       (x - 7, bottom - 14)], fill=(210, 214, 224))

    # predicted character onsets (fixture: every char; real: every char)
    chars = sorted(doc.get("chars") or [], key=lambda c: float(c["start_s"]))
    for k, char in enumerate(chars):
        x = x_of(float(char["start_s"]))
        d.line([(x, bottom + 2), (x, bottom + 20)], fill=COL_CHAR, width=2)
        if k % 6 == 0:
            label = str(char.get("text") or "")
            d.text((x - 4, bottom + 24), label, font=f_label, fill=COL_CHAR)

    # ---------------------------------------------------------------- envelope
    etop, ebottom = 648, 980
    emid = ebottom
    d.rectangle([left, etop, right, ebottom], outline=COL_GRID)
    env = features.onset
    peak = float(env.max()) if env.size else 1.0
    points = []
    for px in range(n_px):
        t = duration * px / max(n_px - 1, 1)
        idx = min(int(t / features.hop_s), env.size - 1)
        y = int(ebottom - (env[idx] / max(peak, 1e-6)) * (ebottom - etop - 10))
        points.append((left + px, y))
    d.polygon(points + [(right, ebottom), (left, ebottom)], fill=COL_ENV_FILL)
    d.line(points, fill=COL_ENV, width=2)

    # prominent peaks
    peaks = align._local_peaks(env)
    for p in peaks:
        x = x_of(p * features.hop_s)
        y = int(ebottom - (env[p] / max(peak, 1e-6)) * (ebottom - etop - 10))
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=COL_ENV)

    # character onsets on the envelope
    for char in chars:
        x = x_of(float(char["start_s"]))
        d.line([(x, etop), (x, ebottom)], fill=(70, 64, 40), width=1)
        d.ellipse([x - 3, etop - 3, x + 3, etop + 3], fill=COL_CHAR)

    q = doc.get("quality") or {}
    syl = q.get("syllable_ms") or {}
    oe = q.get("onset_envelope") or {}
    lines = [
        f"n_chars={q.get('n_chars')}  n_chunks={q.get('n_chunks')}  "
        f"monotonicity_violations={q.get('monotonicity_violations')}",
        f"syllable duration: median={syl.get('median')} ms  p10={syl.get('p10')} ms  "
        f"p90={syl.get('p90')} ms  min={syl.get('min')} ms",
        f"onset near prominent envelope peak: median="
        f"{(oe.get('nearest_peak_ms') or {}).get('median_ms')} ms, "
        f"within50ms={(oe.get('nearest_peak_ms') or {}).get('within_50ms_rate')}  "
        f"point-biserial r={oe.get('point_biserial_r')}",
    ]
    for i, line in enumerate(lines):
        d.text((left, 536 + i * 25), line, font=f_small, fill=COL_TEXT)

    # time axis
    d.line([(left, ebottom + 2), (right, ebottom + 2)], fill=COL_GRID, width=1)
    for t in np.arange(0.0, duration + 1e-6, 1.0):
        x = x_of(float(t))
        d.line([(x, etop), (x, etop + 8)], fill=COL_GRID, width=1)
        d.text((x + 2, ebottom + 6), f"{t:.0f}", font=f_label, fill=COL_DIM)

    # ------------------------------------------------------------- caption strip
    if caption_doc:
        ctop, cbottom = 1030, 1160
        d.text((left, ctop - 30),
               f"caption cues ({caption_doc['source']['fps']:.0f} fps, "
               f"max {caption_doc['quality']['max_line_chars']} chars/line, "
               f"min display {caption_doc['quality']['min_duration_s']:.2f}s)",
               font=f_small, fill=COL_TEXT)
        for i, cue in enumerate(caption_doc.get("cues") or []):
            x0 = x_of(float(cue["start_s"]))
            x1 = x_of(float(cue["end_s"]))
            y0 = ctop + 6 + (i % 2) * 58
            d.rectangle([x0, y0, max(x0 + 3, x1), y0 + 50], outline=COL_SEG,
                        fill=(26, 34, 54))
            label = (cue["lines"][0] + ("…" if len(cue["lines"]) > 1 else ""))
            d.text((x0 + 5, y0 + 5), label, font=f_label, fill=(224, 230, 244))
            d.text((x0 + 5, y0 + 28), f"{cue['start_frame']}-{cue['end_frame']}f",
                   font=f_label, fill=COL_DIM)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


# --------------------------------------------------------------------------- scoring


def _fixture_cases() -> list[tuple[str, dict, dict, np.ndarray]]:
    cases = []
    for fid in FIXTURES:
        gt_path = CLIPS / fid / "ground_truth.json"
        video = CLIPS / fid / f"{fid}.mp4"
        if not gt_path.exists() or not video.exists():
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        cache = asr.cache_path(gt["sha256"], "medium", "cpu")
        if not cache.exists():
            continue
        if not (WORK / f"{fid}.transcript.json").exists():
            asr.transcribe(video, WORK / f"{fid}.transcript.json")
        transcript = json.loads((WORK / f"{fid}.transcript.json").read_text(encoding="utf-8"))
        samples, _ = align.load_audio(video)
        cases.append((fid, gt, transcript, samples))
    return cases


def _stats(values: list[float]) -> dict:
    arr = np.asarray([abs(v) for v in values], dtype=float)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "max_ms": round(float(arr.max()), 2),
        "mean_ms": round(float(arr.mean()), 2),
        "bias_ms": round(float(np.mean(values)), 2),
    }


def _score_all(cases, *, params=None, shuffle_seed: int | None = None) -> dict:
    errors: list[float] = []
    per_fixture: list[dict] = []
    for fid, gt, transcript, samples in cases:
        features = None
        if shuffle_seed is not None:
            features = align.audio_features(samples)
            rng = np.random.default_rng(shuffle_seed)
            shuffled = features.onset.copy()
            rng.shuffle(shuffled)
            features = replace(features, onset=shuffled)
        doc = align.align_transcript(transcript, samples=samples,
                                     params=params, features=features)
        score = align.score_unit_onsets(doc, gt)
        errors.extend(r["error_ms"] for r in score["rows"])
        per_fixture.append({"fixture": fid, "n": score["n_scored"],
                            "p90_ms": score["p90_ms"],
                            "monotonicity_violations":
                                doc["quality"]["monotonicity_violations"]})
    result = _stats(errors)
    result["per_fixture"] = per_fixture
    return result


def _naive_baseline(cases) -> dict:
    errors: list[float] = []
    for _fid, gt, transcript, _samples in cases:
        onsets: list[float] = []
        for seg in transcript.get("segments") or []:
            text = seg.get("text") or ""
            chars = [ch for ch in text if align.is_cjk_char(ch)]
            n = len(chars)
            span = float(seg["end_s"]) - float(seg["start_s"])
            for k in range(n):
                onsets.append(float(seg["start_s"]) + span * k / max(n, 1))
        for unit in gt.get("units") or []:
            t = float(unit["start_s"])
            nearest = min(onsets, key=lambda o: abs(o - t))
            errors.append((nearest - t) * 1000.0)
    return _stats(errors)


def _old_pipeline_scores(cases) -> dict:
    word: list[float] = []
    naive: list[float] = []
    for _fid, gt, transcript, _samples in cases:
        word_onsets = sorted(float(w["start_s"]) for w in transcript.get("words") or [])
        naive_onsets = sorted(align._naive_char_onsets(transcript))
        for unit in gt.get("units") or []:
            t = float(unit["start_s"])
            if word_onsets:
                word.append((min(word_onsets, key=lambda o: abs(o - t)) - t) * 1000.0)
            if naive_onsets:
                naive.append((min(naive_onsets, key=lambda o: abs(o - t)) - t) * 1000.0)
    return {"word": _stats(word), "naive_char": _stats(naive)}


def _percentile_against(reference: np.ndarray, times: list[float], hop_s: float,
                        *, window_s: float = 0.150) -> dict:
    """Mean local percentile of the *reference* envelope at the given times.

    This is the honest way to compare two alignments on real speech: both are
    scored against the same (real) envelope.  With shuffled evidence the DP can
    still land on peaks of the shuffled array, so only this cross-check reveals
    that it is no longer tracking the audio.
    """
    if not times or reference.size == 0:
        return {"n": 0, "mean": None, "median": None}
    w = max(1, int(round(window_s / hop_s)))
    values = []
    for t in times:
        f = min(max(int(round(t / hop_s)), 0), reference.size - 1)
        win = reference[max(0, f - w): f + w + 1]
        values.append(float((win < reference[f]).mean()) if win.size else 0.0)
    arr = np.asarray(values, dtype=float)
    return {"n": int(arr.size), "mean": round(float(arr.mean()), 4),
            "median": round(float(np.median(arr)), 4)}


def _real_contrast(real_samples: np.ndarray, fitted_doc: dict, real_tr: dict) -> dict:
    """Real-speech evidence contrast: fitted vs shuffled envelope, both scored
    against the *real* onset envelope (no truth needed)."""
    raw = align.audio_features(real_samples)
    rng = np.random.default_rng(7)
    shuffled = raw.onset.copy()
    rng.shuffle(shuffled)
    shuffled_doc = align.align_transcript(real_tr, features=replace(raw, onset=shuffled))

    def times(doc):
        return [float(c["start_s"]) for c in doc.get("chars") or []]

    speech = raw.speech
    speech_mean = float(raw.onset[speech].mean()) if speech.any() else 1.0

    def contrast(doc):
        ts = times(doc)
        vals = [float(raw.onset[min(max(int(round(t / raw.hop_s)), 0),
                                     raw.n_frames - 1)]) for t in ts]
        return round(float(np.mean(vals)) / max(speech_mean, 1e-9), 3) if vals else None

    fitted_times = times(fitted_doc)
    shuffled_times = times(shuffled_doc)
    return {
        "fitted": {
            "percentile_against_real_envelope": _percentile_against(raw.onset, fitted_times,
                                                                    raw.hop_s),
            "energy_contrast_against_real_envelope": contrast(fitted_doc),
        },
        "shuffled_evidence": {
            "percentile_against_real_envelope": _percentile_against(raw.onset,
                                                                    shuffled_times,
                                                                    raw.hop_s),
            "energy_contrast_against_real_envelope": contrast(shuffled_doc),
        },
        "explanation": (
            "Both alignments are scored against the original (unshuffled) onset "
            "envelope. When the evidence is shuffled the predicted onsets move off "
            "the real peaks, so their local percentile/contrast drop. This is a "
            "plausibility control, not an error measurement: real speech has no "
            "human-labelled truth in this repo."),
    }


def _caption_metrics(cap: dict) -> dict:
    q = cap.get("quality") or {}
    return {
        "n_cues": len(cap.get("cues") or []),
        "mid_word_cue_starts": q.get("mid_word_cue_starts"),
        "segment_boundary_breaks": q.get("segment_boundary_breaks"),
        "min_duration_s": q.get("min_duration_s"),
        "max_duration_s": q.get("max_duration_s"),
        "max_line_chars": q.get("max_line_chars"),
        "problems": captions.validate(cap),
        "options": cap.get("options"),
        "first_cue": (cap["cues"][0]["display_text"] if cap.get("cues") else None),
    }


# --------------------------------------------------------------------------- main


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "schema": "semotion.align.evidence/1",
        "generated_by": "artifacts/p2/align/evidence.py",
        "note": ("Character-onset accuracy is measured only against SYNTH_KNOWN "
                 "(per-unit TTS assembly) truth. zhengqinwen_25s has no human "
                 "labels: for it we report monotonicity, duration distribution, "
                 "envelope-peak agreement and a visual timeline, not an error."),
    }

    # ---------------------------------------------------------------- fixture fx_a
    gt = json.loads(FX_A_GT.read_text(encoding="utf-8"))
    fx_tr = asr.transcribe(FX_A, WORK / "fx_a_zh_portrait.transcript.json")  # cache hit
    fx_doc = align.align_video(FX_A, fx_tr)
    align.save(fx_doc, OUT / "fx_a_zh_portrait.alignment.json")
    fx_score = align.score_unit_onsets(fx_doc, gt)
    old_word = align.score_word_onsets(fx_tr, gt, mode="word")
    old_naive = align.score_word_onsets(fx_tr, gt, mode="naive_char")
    fx_cap = captions.build_captions(fx_doc)  # task-3b defaults: segment-respecting
    fx_cap_before = captions.build_captions(fx_doc, respect_segments=False,
                                            respect_words=False)
    fx_paths = captions.export(fx_cap, OUT, "fx_a_zh_portrait")
    fx_samples, _ = align.load_audio(FX_A)
    timeline_png(fx_samples, fx_doc, title="fx_a_zh_portrait — aligned character onsets",
                 units=gt["units"], out_path=OUT / "fx_a_zh_portrait.timeline.png",
                 caption_doc=fx_cap)

    summary["fixture_fx_a"] = {
        "n_units": len(gt["units"]),
        "n_scored": fx_score["n_scored"],
        "character_onset_error": {k: fx_score[k] for k in
                                  ("p50_ms", "p90_ms", "max_ms", "mean_ms", "bias_ms")},
        "old_pipeline_word_onset": old_word,
        "old_pipeline_naive_char": old_naive,
        "monotonicity_violations": fx_doc["quality"]["monotonicity_violations"],
        "syllable_ms": fx_doc["quality"]["syllable_ms"],
        "captions": {**_caption_metrics(fx_cap), "files": fx_paths,
                     "before_task_3b": _caption_metrics(fx_cap_before)},
        "caveat": ("n=14 independent units: this is the fixture acceptance number, "
                   "not a robust P90 claim. The pooled number below uses n=147."),
    }

    # what the *old* pipeline says about the fixture's first segment
    first_seg = fx_tr["segments"][0]
    summary["fixture_fx_a"]["old_first_segment"] = {
        "segment_start_s": first_seg["start_s"],
        "first_word_start_s": fx_tr["words"][0]["start_s"],
        "first_word_text": fx_tr["words"][0]["text"],
        "truth_first_unit_start_s": gt["units"][0]["start_s"],
        "new_first_char_start_s": fx_doc["chars"][0]["start_s"],
        "new_first_char_text": fx_doc["chars"][0]["text"],
    }

    # ------------------------------------------------------------ real speech clip
    real_tr = json.loads(REAL_TR.read_text(encoding="utf-8"))
    real_doc = align.align_video(REAL, real_tr)
    real_doc["_transcript_path"] = str(REAL_TR)  # only for the PNG; stripped below
    align.save(real_doc, OUT / "zhengqinwen_25s.alignment.json")
    real_samples, _ = align.load_audio(REAL)
    real_cap = captions.build_captions(real_doc)  # task-3b defaults
    real_cap_before = captions.build_captions(real_doc, respect_segments=False,
                                              respect_words=False)
    real_paths = captions.export(real_cap, OUT, "zhengqinwen_25s")
    timeline_png(real_samples, real_doc,
                 title="zhengqinwen_25s — aligned character onsets over the real audio",
                 units=None, out_path=OUT / "zhengqinwen_25s.timeline.png",
                 caption_doc=real_cap)
    real_doc.pop("_transcript_path", None)
    align.save(real_doc, OUT / "zhengqinwen_25s.alignment.json")

    q = real_doc["quality"]
    syl = q["syllable_ms"]
    oe = q["onset_envelope"]
    summary["real_zhengqinwen_25s"] = {
        "duration_s": real_doc["source"]["duration_s"],
        "n_chars": q["n_chars"],
        "n_syllables": q["n_syllables"],
        "n_chunks": q["n_chunks"],
        "monotonicity_violations": q["monotonicity_violations"],
        "syllable_ms": syl,
        "onset_vs_local_peaks": {
            "nearest_peak_ms": oe["nearest_peak_ms"],
            "point_biserial_r": oe["point_biserial_r"],
            "onset_percentile_mean": oe["onset_percentile_mean"],
            "contrast_ratio": oe["contrast_ratio"],
            "peak_hit_rate": oe["peak_hit_rate"],
        },
        "low_confidence_syllables": q["low_confidence_syllables"],
        "segment_anchor_dev_ms": q["segment_anchor_dev_ms"],
        "captions": {
            **_caption_metrics(real_cap), "files": real_paths,
            "before_task_3b": _caption_metrics(real_cap_before),
            "cue_list": [
                {"start_frame": c["start_frame"], "end_frame": c["end_frame"],
                 "duration_s": c["duration_s"], "lines": c["lines"],
                 "segment_ids": c["segment_ids"]}
                for c in real_cap["cues"]
            ],
        },
        "old_first_segment": {
            "segment_start_s": real_tr["segments"][0]["start_s"],
            "first_word_text": real_tr["words"][0]["text"],
            "first_word_start_s": real_tr["words"][0]["start_s"],
            "second_word_text": real_tr["words"][1]["text"],
            "second_word_start_s": real_tr["words"][1]["start_s"],
            "new_first_char_text": real_doc["chars"][0]["text"],
            "new_first_char_start_s": real_doc["chars"][0]["start_s"],
            "speech_island_start_s": real_doc["segments"][0]["speech_start_s"],
        },
        "caveat": ("No human-labelled real corpus exists for this clip. These are "
                   "plausibility checks (monotonicity, duration distribution, "
                   "envelope-peak agreement), not an accuracy measurement; the "
                   "timeline PNG is for human inspection."),
    }

    # ------------------------------------------------------------- pooled fixtures
    cases = _fixture_cases()
    fitted = _score_all(cases)
    shuffled = _score_all(cases, shuffle_seed=7)
    refined = _score_all(cases, params={"refine_onsets": 1})
    uniform = _naive_baseline(cases)
    old = _old_pipeline_scores(cases)
    summary["pooled_fixtures"] = {
        "n_fixtures": len(cases),
        "n_units": sum(len(c[1]["units"]) for c in cases),
        "n_scored": fitted["n"],
        "new_character_onset": fitted,
        "old_pipeline_word_onset": old["word"],
        "old_pipeline_naive_char": old["naive_char"],
        "uniform_segment_split_baseline": uniform,
        "negative_control_shuffled_envelope": shuffled,
        "ablation_attack_refinement": refined,
        "caveat": ("SYNTH_KNOWN = per-unit TTS concatenated on a measured timeline. "
                   "Pauses between units make this easier than continuous human "
                   "speech; the n=147 pooled P90 is a dev-set number."),
    }

    # --------------------------------------------------------- real negative control
    summary["negative_control_real_speech"] = _real_contrast(real_samples, real_doc, real_tr)

    # ---------------------------------------------------------- old pipeline detail
    old_detail = {
        "fixture": "fx_a_zh_portrait",
        "old_transcript_first_segment": {
            "start_s": first_seg["start_s"], "end_s": first_seg["end_s"],
            "text": first_seg["text"],
            "first_word": fx_tr["words"][0],
        },
        "old_per_unit": [],
        "new_per_unit": [],
    }
    for unit in gt["units"]:
        t = float(unit["start_s"])
        word_start = min((float(w["start_s"]) for w in fx_tr["words"]),
                         key=lambda o: abs(o - t))
        naive_onsets = align._naive_char_onsets(fx_tr)
        naive_start = min(naive_onsets, key=lambda o: abs(o - t))
        old_detail["old_per_unit"].append({
            "unit_id": unit["unit_id"], "text": unit["text"],
            "truth_start_s": round(t, 6),
            "nearest_word_start_s": round(word_start, 6),
            "word_error_ms": round((word_start - t) * 1000.0, 2),
            "nearest_naive_char_ms": round((naive_start - t) * 1000.0, 2),
        })
    for row in fx_score["rows"]:
        old_detail["new_per_unit"].append(row)
    (OUT / "old_pipeline_fx_a.json").write_text(
        json.dumps(old_detail, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------------------- caption samples
    def cue_block(cap: dict, title: str) -> list[str]:
        lines = [f"== {title} ==",
                 f"cue  start_frame  end_frame  frames  duration  lines"]
        for cue in cap["cues"]:
            lines.append(
                f"{cue['id']:>4}  {cue['start_frame']:>10}  {cue['end_frame']:>9}  "
                f"{cue['end_frame'] - cue['start_frame']:>6}  "
                f"{cue['duration_s']:>8.2f}  {' / '.join(cue['lines'])}")
        lines.append("")
        return lines

    samples = []
    samples.append("caption before/after (task-3b; respect_segments + word guard)")
    for name, before, after in (("fx_a_zh_portrait", fx_cap_before, fx_cap),
                                ("zhengqinwen_25s", real_cap_before, real_cap)):
        samples.append(
            f"{name}: before {len(before['cues'])} cues / "
            f"mid-word starts {before['quality']['mid_word_cue_starts']} / "
            f"min display {before['quality']['min_duration_s']:.2f}s  ->  "
            f"after {len(after['cues'])} cues / "
            f"mid-word starts {after['quality']['mid_word_cue_starts']} / "
            f"min display {after['quality']['min_duration_s']:.2f}s")
    samples.append("")
    samples += cue_block(fx_cap, "fx_a_zh_portrait (CJK, 25 fps, after)")
    samples += cue_block(real_cap, "zhengqinwen_25s (CJK, 25 fps, after)")
    (OUT / "caption_samples.txt").write_text("\n".join(samples), encoding="utf-8")

    summary["captions_before_after"] = {
        "fx_a_zh_portrait": {"before": _caption_metrics(fx_cap_before),
                             "after": _caption_metrics(fx_cap)},
        "zhengqinwen_25s": {"before": _caption_metrics(real_cap_before),
                            "after": _caption_metrics(real_cap)},
        "note": ("before = task-3 behaviour (respect_segments=False, respect_words=False); "
                 "after = task-3b defaults (segment-respecting cues + ASR word guard)"),
    }

    # ------------------------------------------------------------------ summary IO
    summary["caption_rules"] = {
        "cjk_max_chars_per_line": captions.DEFAULT_STYLE["cjk"]["max_chars_per_line"],
        "cjk_target_chars_per_line": captions.DEFAULT_STYLE["cjk"]["target_chars_per_line"],
        "cjk_max_lines": captions.DEFAULT_STYLE["cjk"]["max_lines"],
        "latin_max_chars_per_line": captions.DEFAULT_STYLE["latin"]["max_chars_per_line"],
        "min_display_s": captions.DEFAULT_STYLE["min_display_s"],
        "merge_gap_s": captions.DEFAULT_STYLE["merge_gap_s"],
        "font_cjk": captions.DEFAULT_STYLE["font_cjk"],
        "font_latin": captions.DEFAULT_STYLE["font_latin"],
        "respect_segments_default": "on when the alignment carries ASR segment ids",
        "respect_words_default": True,
    }
    (OUT / "evidence.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # also re-run the standalone fixture scorer so its JSON matches this run
    subprocess.run([sys.executable, str(Path(__file__).parent / "eval_align.py"),
                    "--out", str(OUT / "fixtures_score.json")],
                   check=True, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
                   capture_output=True, text=True, cwd=str(ROOT))

    print(json.dumps({
        "fx_a_p90_ms": summary["fixture_fx_a"]["character_onset_error"]["p90_ms"],
        "fx_a_n": summary["fixture_fx_a"]["n_scored"],
        "old_word_p90_ms": summary["fixture_fx_a"]["old_pipeline_word_onset"]["p90_ms"],
        "pooled_n": summary["pooled_fixtures"]["n_scored"],
        "pooled_new_p90_ms": summary["pooled_fixtures"]["new_character_onset"]["p90_ms"],
        "pooled_old_p90_ms": summary["pooled_fixtures"]["old_pipeline_word_onset"]["p90_ms"],
        "shuffled_p90_ms": summary["pooled_fixtures"]["negative_control_shuffled_envelope"]["p90_ms"],
        "refined_p90_ms": summary["pooled_fixtures"]["ablation_attack_refinement"]["p90_ms"],
        "real_median_syllable_ms": syl["median"],
        "real_monotonicity": q["monotonicity_violations"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
