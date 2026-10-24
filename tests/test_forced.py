"""Tests for :mod:`semotion.forced` — acoustic CTC forced alignment.

Run with::

    PYTHONPATH=src .venv/bin/python -m pytest tests/test_forced.py -q

What is verified (and how, without needing a 1.2 GB download):

* **Schema compatibility** — the document passes ``align.validate`` (the shared
  ``semotion.alignment/1`` validator): monotonic character and word times,
  ``token_id`` links, confidence bounds, ``quality`` block, ``method.name ==
  "ctc-forced"``.
* **CTC forced alignment itself** — synthetic posteriors with known emission
  frames, run through the NumPy Viterbi: exact recovery at frame resolution,
  repeated characters keep *distinct* spans (this is the ``merge_tokens`` trap),
  and a target sequence longer than the audio is reported infeasible instead of
  silently producing garbage.
* **Phone/character → source mapping** — Han characters map 1:1, a numeral
  expands to its spoken Mandarin characters (``len(reading)`` is asserted equal
  to ``align.number_syllables``), a Latin word in Chinese text has no acoustic
  target and is interpolated (``approx``), and an out-of-vocabulary transcript
  is refused rather than faked.
* **Chunking** — a 1.4 s posterior matrix cut into short chunks yields the same
  onset times as the single-pass alignment (±1 frame), stays monotonic, and
  reports the chunk count.
* **Negative control** — the same posteriors with a shuffled transcript must
  score materially worse per frame and lose confidence.  A method that ignored
  the target sequence would pass the alignment tests but fail this one.

Heavy tests (the real XLSR-53 zh model, the real fixture) skip automatically
when the weights are not cached; they never download.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from semotion import align, asr, forced  # noqa: E402

STRIDE = 0.02
VOCAB = {"<pad>": 0, "大": 1, "家": 2, "好": 3, "谢": 4, "今": 5, "天": 6,
         "A": 7, "I": 8, "二": 9, "十": 10, "五": 11, "号": 12, "镜": 13, "头": 14}
BLANK = 0


# --------------------------------------------------------------------- helpers


def synth_posteriors(seq: str, n_frames: int, *, noise: float = 0.0,
                     seed: int = 0, vocab: dict[str, int] | None = None,
                     emit_frames: int = 3) -> np.ndarray:
    """Log-posteriors where each target is emitted over ``emit_frames`` frames.

    Targets are evenly spaced; ``noise`` adds Gaussian jitter to the background
    logits (still normalised), which lets a shuffled order score worse without
    changing the frame count.
    """
    vocab = vocab or VOCAB
    rng = np.random.default_rng(seed)
    base = np.full((n_frames, len(vocab)), -8.0)
    if noise:
        base += rng.normal(0.0, noise, size=base.shape)
    base[:, BLANK] = 2.0
    lp = forced._log_softmax(base)
    slot = n_frames / max(1, len(seq))
    for i, ch in enumerate(seq):
        cell = int(vocab[ch])
        start = int(round(i * slot + slot * 0.3))
        for f in range(start, min(start + emit_frames, n_frames)):
            row = np.full(len(vocab), -8.0)
            row[cell] = 3.0
            lp[f] = forced._log_softmax(row)
    return lp


def transcript_for(text: str, duration: float, *, start: float = 0.0,
                   segments: list | None = None) -> dict:
    segs = segments or [{"id": "s0", "start_s": start, "end_s": start + duration,
                         "text": text}]
    return {
        "schema": "semotion.transcript/1",
        "language": "zh",
        "source": {"path": "/tmp/forced-test.mp4", "duration_s": start + duration,
                   "fps": 25.0, "language": "zh"},
        "segments": segs,
        "words": [],
    }


def run(text: str, n_frames: int, **kwargs) -> dict:
    duration = n_frames * STRIDE
    params = {"ctc_engine": "numpy", **(kwargs.pop("params", {}) or {})}
    return forced.align_posteriors(
        transcript_for(text, duration, **kwargs),
        synth_posteriors(text, n_frames), VOCAB, blank=BLANK,
        frame_stride_s=STRIDE, language="zh", params=params)


# ------------------------------------------------------------------- numerals


@pytest.mark.parametrize("raw, year_hint", [
    ("0", False), ("7", False), ("10", False), ("11", False), ("20", False),
    ("25", False), ("100", False), ("101", False), ("1000", False),
    ("10000", False), ("10500", False), ("2024", False), ("2024", True),
    ("3.5", False), ("90%", False), ("-5", False), ("1,000", False),
    ("12", True), ("0.5", False),
])
def test_mandarin_reading_matches_number_syllables(raw: str, year_hint: bool):
    """``mandarin_reading`` must produce exactly ``number_syllables`` characters.

    The settle point is the transcript tokenizer: if the two disagree, a numeral
    would consume the wrong number of CTC targets and every later character
    would shift.
    """
    reading = forced.mandarin_reading(raw, year_hint=year_hint)
    assert reading is not None
    assert len(reading) == align.number_syllables(raw, year_hint=year_hint)


def test_mandarin_reading_rejects_non_numbers():
    assert forced.mandarin_reading("abc") is None
    assert forced.mandarin_reading("") is None
    assert forced.mandarin_reading("3.5.7") is None


# ------------------------------------------------------------ target mapping


def test_han_char_maps_to_one_ctc_target():
    tokens = align.tokenize_text("大家好", "zh")
    targets, groups, info = forced.build_targets(tokens, VOCAB, language="zh")
    assert [t.text for t in targets] == ["大", "家", "好"]
    assert info["coverage"] == 1.0
    assert all(len(g.target_ids) == 1 for g in groups)
    assert [t.vocab_id for t in targets] == [VOCAB["大"], VOCAB["家"], VOCAB["好"]]


def test_numeral_expands_to_spoken_characters():
    tokens = align.tokenize_text("25号", "zh")
    targets, groups, info = forced.build_targets(tokens, VOCAB, language="zh")
    # 25 -> 二十五 (3 targets); 号 is in the test vocab and must still be aligned
    assert [t.text for t in targets] == ["二", "十", "五", "号"]
    assert info["n_targets"] == 4
    assert info["coverage"] == 1.0


def test_numeral_targets_follow_tokenizer_syllable_count():
    tokens = align.tokenize_text("今天25号", "zh")
    targets, groups, _info = forced.build_targets(tokens, VOCAB, language="zh")
    number = next(g for g in groups if g.token.kind == "number")
    assert len(number.target_ids) == align.number_syllables("25") == 3
    assert [targets[i].text for i in number.target_ids] == ["二", "十", "五"]


def test_latin_word_in_chinese_text_is_approximated_not_faked():
    tokens = align.tokenize_text("AI很强", "zh")
    targets, groups, info = forced.build_targets(tokens, VOCAB, language="zh")
    latin = next(g for g in groups if g.token.kind == "latin")
    assert latin.approx is True
    assert latin.target_ids == []
    assert info["n_approx_tokens"] == 1
    assert "A" not in [t.text for t in targets]


def test_oov_chars_are_reported_not_silently_dropped():
    tokens = align.tokenize_text("大家龘", "zh")
    targets, groups, info = forced.build_targets(tokens, {**VOCAB}, language="zh")
    assert info["n_oov"] == 1
    assert "龘" in info["oov_chars"]
    assert 0.0 < info["coverage"] < 1.0


def test_oov_transcript_is_refused():
    transcript = transcript_for("龘龘龘龘", 2.0)
    with pytest.raises(Exception) as exc:
        forced.align_posteriors(transcript, synth_posteriors("大家", 100), VOCAB,
                                blank=BLANK, frame_stride_s=STRIDE, language="zh")
    assert "E_ALIGN_MODEL_UNSUPPORTED" in str(exc.value)


def test_traditional_characters_are_looked_up_as_simplified():
    """A Traditional transcript against a Simplified vocabulary (OpenCC t2s).

    The fixture set contains Traditional clips; without this lookup a third of
    their characters would be dropped and the rest smeared across the timeline.
    The emitted character text must stay Traditional (it is what captions show).
    """
    pytest.importorskip("opencc")
    tokens = align.tokenize_text("鏡頭", "zh")
    targets, groups, info = forced.build_targets(tokens, VOCAB, language="zh")
    assert [t.text for t in targets] == ["鏡", "頭"]
    assert [t.vocab_id for t in targets] == [VOCAB["镜"], VOCAB["头"]]
    assert info["n_script_converted"] == 2
    assert info["coverage"] == 1.0


# --------------------------------------------------------------------- Viterbi


def test_viterbi_recovers_known_emission_frames():
    seq = "大家好今天"
    lp = synth_posteriors(seq, 100)
    res = forced.ctc_viterbi(lp, [VOCAB[c] for c in seq], blank=BLANK)
    assert res["feasible"]
    # every target starts on the frame where its emission begins (slot*0.3)
    slot = 100 / len(seq)
    for i, start in enumerate(res["starts"]):
        assert abs(start - round(i * slot + slot * 0.3)) <= 1, (i, start)
    assert all(np.isfinite(res["logp"]))


def test_viterbi_keeps_repeated_characters_distinct():
    seq = "谢谢"
    lp = synth_posteriors(seq, 40, emit_frames=2)
    res = forced.ctc_viterbi(lp, [VOCAB[c] for c in seq], blank=BLANK)
    assert res["feasible"]
    assert res["starts"][0] < res["starts"][1]
    assert res["ends"][0] <= res["starts"][1]


def test_viterbi_reports_infeasible_when_audio_too_short():
    res = forced.ctc_viterbi(np.log(np.full((3, len(VOCAB)), 1 / len(VOCAB))),
                             [VOCAB[c] for c in "大家好"],
                             blank=BLANK)
    assert res["feasible"] is False
    assert res["starts"] == []


# ------------------------------------------------------------- schema / timing


def test_document_is_schema_compatible():
    doc = run("大家好今天", 120)
    problems = align.validate(doc)
    assert problems == [], problems
    assert doc["schema"] == align.SCHEMA
    assert doc["method"]["name"] == "ctc-forced"
    assert doc["method"]["truth_class"] == "MACHINE_WEAK"
    assert doc["method"]["model"] and doc["method"]["device"]
    assert doc["quality"]["posterior"]["mean_conf"] is not None
    assert doc["quality"]["monotonicity_violations"] == 0
    chars = doc["chars"]
    assert [c["text"] for c in chars] == list("大家好今天")
    assert all(0.0 <= float(c["conf"]) <= 1.0 for c in chars)
    word_ids = {w["id"] for w in doc["words"]}
    assert all(c["token_id"] in word_ids for c in chars)
    assert doc["segments"][0]["speech_start_s"] is not None
    assert doc["segments"][0]["speech_end_s"] is not None


def test_character_times_are_monotonic_and_inside_media():
    doc = run("大家好今天谢", 120)
    starts = [float(c["start_s"]) for c in doc["chars"]]
    ends = [float(c["end_s"]) for c in doc["chars"]]
    assert starts == sorted(starts)
    assert all(e >= s for s, e in zip(starts, ends))
    duration = float(doc["source"]["duration_s"])
    assert starts[0] >= 0.0
    assert ends[-1] <= duration + 1e-6
    # syllable durations in a plausible band for the synthetic layout
    durations = [e - s for s, e in zip(starts, ends)]
    assert min(durations) >= STRIDE - 1e-9
    assert max(durations) <= 1.0


def test_word_span_covers_its_characters():
    doc = run("大家好", 60)
    for word in doc["words"]:
        own = [c for c in doc["chars"] if c["token_id"] == word["id"]]
        assert own, word
        assert float(word["start_s"]) <= min(float(c["start_s"]) for c in own) + 1e-6
        assert float(word["end_s"]) >= max(float(c["end_s"]) for c in own) - 1e-6


# ------------------------------------------------------------------- chunking


def test_chunked_alignment_matches_single_pass():
    text = "大家好今天谢"
    lp = synth_posteriors(text, 300)
    transcript = transcript_for(text, 300 * STRIDE, segments=[
        {"id": "s0", "start_s": 0.0, "end_s": 3.0, "text": "大家好"},
        {"id": "s1", "start_s": 3.0, "end_s": 6.0, "text": "今天谢"},
    ])
    base = forced.align_posteriors(
        transcript, lp, VOCAB, blank=BLANK,
        frame_stride_s=STRIDE, language="zh", params={"ctc_engine": "numpy"})
    chunked = forced.align_posteriors(
        transcript, lp, VOCAB, blank=BLANK,
        frame_stride_s=STRIDE, language="zh",
        params={"ctc_engine": "numpy", "max_chunk_s": 1.0})
    assert chunked["quality"]["n_chunks"] > 1
    assert len(chunked["chars"]) == len(base["chars"]) == len(text)
    assert align.validate(chunked) == []
    deltas = [abs(float(a["start_s"]) - float(b["start_s"]))
              for a, b in zip(base["chars"], chunked["chars"])]
    assert max(deltas) <= STRIDE * 2 + 1e-9, deltas


def test_pathological_chunk_size_still_aligns_everything():
    text = "大家好今天"
    doc = forced.align_posteriors(
        transcript_for(text, 120 * STRIDE), synth_posteriors(text, 120), VOCAB,
        blank=BLANK, frame_stride_s=STRIDE, language="zh",
        params={"ctc_engine": "numpy", "max_chunk_s": 0.12})
    # chunks too small for their tokens are merged, never dropped
    assert len(doc["chars"]) == len(text)
    assert doc["quality"]["monotonicity_violations"] == 0
    assert align.validate(doc) == []


# ------------------------------------------------------------- negative control


def test_shuffled_transcript_scores_materially_worse():
    """The negative control: same audio, same symbols, wrong order."""
    text = "大家好今天谢"
    lp = synth_posteriors(text, 160, noise=0.5, seed=3)
    good = forced.align_posteriors(
        transcript_for(text, 160 * STRIDE), lp, VOCAB, blank=BLANK,
        frame_stride_s=STRIDE, language="zh", params={"ctc_engine": "numpy"})
    shuffled = "谢天今好家大"
    bad = forced.align_posteriors(
        transcript_for(shuffled, 160 * STRIDE), lp, VOCAB, blank=BLANK,
        frame_stride_s=STRIDE, language="zh", params={"ctc_engine": "numpy"})
    good_score = good["quality"]["alignment_score_per_frame"]
    bad_score = bad["quality"]["alignment_score_per_frame"]
    assert good_score is not None and bad_score is not None
    assert bad_score < good_score - 0.5, (good_score, bad_score)
    good_conf = good["quality"]["posterior"]["mean_conf"]
    bad_conf = bad["quality"]["posterior"]["mean_conf"]
    assert bad_conf < good_conf
    # the shuffled document is still schema-valid: it is wrong, not broken
    assert align.validate(bad) == []


# ------------------------------------------------------------------- engines


def test_torchaudio_and_numpy_engines_agree():
    torchaudio = pytest.importorskip("torchaudio.functional")
    text = "大家好今天"
    lp = synth_posteriors(text, 100)
    ids = [VOCAB[c] for c in text]
    np_res = forced.forced_align_spans(lp, ids, blank=BLANK, engine="numpy")
    ta_res = forced.forced_align_spans(lp, ids, blank=BLANK, engine="torchaudio")
    assert ta_res["feasible"]
    assert np_res["starts"] == pytest.approx(ta_res["starts"], abs=1)
    del torchaudio


# --------------------------------------------------------------- heavy (real)


def _zh_model_cached() -> bool:
    return forced.available("zh")


def test_real_model_alignment_fx_a():
    """Full XLSR-53 zh model on the SYNTH_KNOWN fixture (skipped when not cached)."""
    if not _zh_model_cached():
        pytest.skip("zh CTC model not cached (tests never download weights)")
    gt_path = ROOT / "tests/fixtures/clips/fx_a_zh_portrait/ground_truth.json"
    video = gt_path.parent / "fx_a_zh_portrait.mp4"
    gt = json.loads(gt_path.read_text())
    cache = asr.cache_path(gt["sha256"], "medium", "cpu")
    if not cache.exists():
        pytest.skip("ASR transcript not cached")
    transcript = json.loads(cache.read_text())
    doc = forced.align_video(video, transcript, language="zh",
                             params={"max_chunk_s": 20.0})
    problems = align.validate(doc)
    assert problems == [], problems
    assert doc["method"]["name"] == "ctc-forced"
    assert doc["method"]["model_info"]["id"]
    assert doc["quality"]["monotonicity_violations"] == 0
    char_segments = {c["segment_id"] for c in doc["chars"]}
    assert char_segments == {s["id"] for s in transcript["segments"]}
    assert len(doc["chars"]) >= 60  # one character per syllable-bearing token
    scored = align.score_unit_onsets(doc, gt)
    assert scored["n_scored"] >= len(gt["units"]) - 1
    # CTC is a real acoustic model: it must be within a syllable of the truth
    assert scored["p90_ms"] is not None and scored["p90_ms"] <= 120.0, scored
    assert doc["quality"]["posterior"]["mean_conf"] > 0.4
    assert doc["quality"]["coverage"]["coverage"] > 0.9


def test_model_not_cached_refuses_offline(monkeypatch, tmp_path):
    """A missing cache must raise, not download or invent an alignment."""
    monkeypatch.setenv("SEMOTION_HF_HOME", str(tmp_path / "empty-hf"))
    monkeypatch.setenv("SEMOTION_FORCED_OFFLINE", "1")
    forced.load_model.cache_clear()
    try:
        with pytest.raises(Exception) as exc:
            forced.load_model("definitely/not-cached-model", language="zh",
                              allow_download=False)
        assert "E_ALIGN_MODEL_MISSING" in str(exc.value)
    finally:
        forced.load_model.cache_clear()
