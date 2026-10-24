"""Tests for deterministic character alignment (:mod:`semotion.align`) and
caption grouping/export (:mod:`semotion.captions`).

Run with::

    PYTHONPATH=src .venv/bin/python -m pytest tests/test_align.py -q

What is verified
----------------
* The tokenizer: one Han character = one syllable, Arabic digits spelled out
  (25 -> 二十五 = 3), percent (90% -> 百分之九十 = 5), Latin words one onset in
  Chinese text, punctuation carries no duration, English syllable heuristic.
* A fully synthetic clip with known per-syllable onsets: the envelope DP must
  recover them (P90 <= 50 ms) while a *shuffled onset envelope* — the negative
  control — must be clearly worse.  This is the test that proves the metric can
  fail, i.e. the alignment actually uses the audio evidence.
* The real SYNTH_KNOWN fixture (``fx_a_zh_portrait``) via the cached ASR
  transcript: character-onset P90 <= 80 ms against assembly truth, monotonic
  times, plausible Mandarin syllable durations.  Skipped when the ASR cache is
  absent (tests never download or run the model).
* Caption rules: <= 24 CJK chars per line (<= 42 Latin), <= 2 lines, no line
  ending in punctuation, no punctuation-only line, display >= 0.9 s, frame
  accuracy, gap merging, SRT/ASS export.

These are engineering checks against SYNTH_KNOWN truth, not human labels; see
``docs/ALIGNMENT.md`` for what is and is not measured.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from semotion import align, asr, captions  # noqa: E402

SR = 16000
CLIPS = ROOT / "tests/fixtures/clips"
FX_A = CLIPS / "fx_a_zh_portrait" / "fx_a_zh_portrait.mp4"
FX_A_GT = CLIPS / "fx_a_zh_portrait" / "ground_truth.json"


# =========================================================================== helpers


def _syllable_burst(duration: float = 0.145, *, f0: float = 150.0, sr: int = SR,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """A vowel-like burst with a consonant-ish noise attack (synthetic speech)."""
    n = int(round(duration * sr))
    t = np.arange(n, dtype=np.float64) / sr
    sig = (np.sin(2 * np.pi * f0 * t)
           + 0.5 * np.sin(2 * np.pi * f0 * 3.1 * t)
           + 0.3 * np.sin(2 * np.pi * 880.0 * t))
    attack = np.minimum(1.0, t / 0.018)
    release = np.minimum(1.0, (duration - t) / 0.035)
    env = np.clip(np.minimum(attack, release), 0.0, 1.0)
    out = 0.32 * sig * env
    # short noise burst at the attack so the energy rise has a consonant edge
    burst = max(1, int(0.012 * sr))
    noise = (rng or np.random.default_rng(0)).normal(0.0, 0.08, burst)
    out[:burst] += noise
    return out.astype(np.float32)


def build_synthetic_audio(phrases: list[str], *, sr: int = SR,
                          seed: int = 0) -> tuple[np.ndarray, list[float], list[float]]:
    """Concatenate one burst per Han character at known times."""
    rng = np.random.default_rng(seed)
    chars = [ch for phrase in phrases for ch in phrase if align.is_cjk_char(ch)]
    starts: list[float] = []
    ends: list[float] = []
    cursor = 0.40
    chunks: list[np.ndarray] = [np.zeros(int(cursor * sr), dtype=np.float32)]
    gap = 0.028
    # a longer pause after the first phrase makes the second a separate island
    pause_after = {len([c for c in phrases[0] if align.is_cjk_char(c)]): 0.34}
    for i, _ch in enumerate(chars):
        syllable = _syllable_burst(rng=rng)
        n = syllable.size
        start = cursor
        end = start + n / sr
        chunks.append(syllable)
        starts.append(start)
        ends.append(end)
        pad = 0.0
        if i in pause_after:
            pad = pause_after[i]
        elif i == len(chars) - 1:
            pad = 0.30
        else:
            pad = gap
        pad_n = int(round(pad * sr))
        if pad_n:
            chunks.append(np.zeros(pad_n, dtype=np.float32))
            cursor = end + pad_n / sr
        else:
            cursor = end
    out = np.concatenate(chunks)
    out += rng.normal(0.0, 1e-4, out.size).astype(np.float32)
    return out, starts, ends


def synthetic_transcript(phrases: list[str], starts: list[float],
                         ends: list[float], *, shift_s: float = 0.10) -> dict:
    """A transcript whose *segment bounds are deliberately imperfect* (±100 ms),
    exactly like ASR output: text is trustworthy, word times are not."""
    per_phrase = [[ch for ch in p if align.is_cjk_char(ch)] for p in phrases]
    segments = []
    units = []
    idx = 0
    for p_index, (phrase, chars) in enumerate(zip(phrases, per_phrase)):
        first = starts[idx]
        last = ends[idx + len(chars) - 1]
        seg_start = max(0.0, first - (shift_s if p_index % 2 == 0 else shift_s * 0.6))
        seg_end = last + (shift_s * 0.5)
        segments.append({"id": f"s{p_index}", "start_s": round(seg_start, 6),
                         "end_s": round(seg_end, 6), "text": phrase + ("，" if p_index == 0 else "。"),
                         "avg_logprob": -0.1, "no_speech_prob": 0.01})
        for k, ch in enumerate(chars):
            units.append({"unit_id": f"u{idx + k}", "text": ch,
                          "start_s": round(starts[idx + k], 6),
                          "end_s": round(ends[idx + k], 6)})
        idx += len(chars)
    return {
        "schema": asr.SCHEMA,
        "source": {"path": "/dev/null/synthetic.wav", "sha256": "0" * 64,
                   "duration_s": round(ends[-1] + 0.35, 6), "width": 1080,
                   "height": 1920, "fps": 25.0},
        "language": "zh",
        "model": {"name": "synthetic", "size": "none", "device": "cpu", "version": "test"},
        "segments": segments,
        "words": [],
        "provenance": {"created_utc": "2026-01-01T00:00:00Z", "command": "test",
                       "truth_class": "MACHINE_WEAK"},
    }


PHRASES_A = ["今天我们来测试一下", "字符对齐必须准确可靠", "很好"]
PHRASES_B = ["这个方法在真实语音上", "也会遇到连读和轻声", "继续验证"]


def synthetic_case(phrases: list[str] = PHRASES_A, seed: int = 0):
    samples, starts, ends = build_synthetic_audio(phrases, seed=seed)
    transcript = synthetic_transcript(phrases, starts, ends)
    truth = {"units": [{"unit_id": f"u{i}", "text": "字",
                        "start_s": round(s, 6), "end_s": round(e, 6)}
                       for i, (s, e) in enumerate(zip(starts, ends))]}
    return samples, transcript, truth, starts, ends


# ========================================================================= tokenizer


class TestTokenizer:
    def test_han_one_syllable_each_and_punctuation_no_duration(self):
        tokens = align.tokenize_text("今天，很好！", "zh")
        assert [t.text for t in tokens] == ["今", "天", "很", "好"]
        assert all(t.n_syll == 1 for t in tokens)
        assert tokens[1].trailing_punct == "，"
        assert tokens[-1].trailing_punct == "！"
        assert sum(t.n_syll for t in tokens) == 4

    def test_arabic_digits_are_spelled_out(self):
        assert align.number_syllables("25") == 3          # 二十五
        assert align.number_syllables("100") == 2         # 一百
        assert align.number_syllables("1,000") == 2       # 一千
        assert align.number_syllables("2024") == 4        # 二零二四 (year-like)
        assert align.number_syllables("3.5") == 3         # 三点五
        assert align.number_syllables("90%") == 5         # 百分之九十
        assert align.number_syllables("-5") == 2          # 负五
        tokens = align.tokenize_text("2024年，25%的人用AI", "zh")
        assert [t.kind for t in tokens] == ["number", "han", "number", "han", "han",
                                            "han", "latin"]
        assert [t.n_syll for t in tokens] == [4, 1, 6, 1, 1, 1, 1]
        assert align.tokenize_text("2024年", "zh")[0].n_syll == 4

    def test_latin_word_one_onset_in_chinese(self):
        tokens = align.tokenize_text("用AI工具", "zh")
        assert [t.text for t in tokens] == ["用", "AI", "工", "具"]
        assert tokens[1].kind == "latin"
        assert tokens[1].n_syll == 1

    def test_english_syllable_counts(self):
        assert align.english_syllables("we") == 1
        assert align.english_syllables("align") == 2
        assert align.english_syllables("beautiful") == 3
        assert align.english_syllables("word") == 1
        assert align.english_syllables("rhythm") >= 1
        tokens = align.tokenize_text("We align every word.", "en")
        assert [t.text for t in tokens] == ["We", "align", "every", "word"]
        assert sum(t.n_syll for t in tokens) == 1 + 2 + 3 + 1

    def test_whitespace_and_symbols_are_not_tokens(self):
        tokens = align.tokenize_text("  ——  …… ", "zh")
        assert tokens == []


# ============================================================================== DP


@pytest.fixture(scope="module")
def case():
    return synthetic_case()


class TestSyntheticAlignment:
    def test_output_contract_and_monotonicity(self, case):
        samples, transcript, truth, _, _ = case
        doc = align.align_transcript(transcript, samples=samples)
        assert doc["schema"] == "semotion.alignment/1"
        assert align.validate(doc) == []
        assert doc["quality"]["monotonicity_violations"] == 0
        starts = [float(c["start_s"]) for c in doc["chars"]]
        assert starts == sorted(starts)
        assert doc["method"]["truth_class"] == "MACHINE_WEAK"
        assert len(doc["chars"]) == len(truth["units"]) == 21
        # every character has a confidence in range and a positive duration
        for char in doc["chars"]:
            assert 0.0 <= char["conf"] <= 1.0
            assert float(char["end_s"]) > float(char["start_s"])

    def test_character_onsets_recover_known_times(self, case):
        samples, transcript, truth, _, _ = case
        doc = align.align_transcript(transcript, samples=samples)
        score = align.score_unit_onsets(doc, truth)
        assert score["n_scored"] == len(truth["units"]) == 21
        assert score["p90_ms"] is not None and score["p90_ms"] <= 50.0
        assert abs(score["bias_ms"] or 0.0) <= 30.0

    def test_shuffled_envelope_negative_control(self, case):
        """Shuffling the onset evidence must make the same DP clearly worse."""
        samples, transcript, truth, _, _ = case
        features = align.audio_features(samples)
        fitted = align.align_transcript(transcript, features=features)
        fitted_score = align.score_unit_onsets(fitted, truth)

        rng = np.random.default_rng(7)
        shuffled_onset = features.onset.copy()
        rng.shuffle(shuffled_onset)
        shuffled = replace(features, onset=shuffled_onset)
        shuffled_score = align.score_unit_onsets(
            align.align_transcript(transcript, features=shuffled), truth)

        assert fitted_score["p90_ms"] <= 50.0
        assert shuffled_score["p90_ms"] > fitted_score["p90_ms"] + 25.0, (
            f"shuffled envelope scored {shuffled_score['p90_ms']} ms vs fitted "
            f"{fitted_score['p90_ms']} ms — the metric would not detect bad evidence"
        )

    def test_uniform_baseline_is_worse(self, case):
        """No-audio control: even spacing inside each segment, like a naive split."""
        samples, transcript, truth, _, _ = case
        doc = align.align_transcript(transcript, samples=samples)
        new = align.score_unit_onsets(doc, truth)
        naive_onsets: list[float] = []
        for seg in transcript["segments"]:
            chars = [ch for ch in seg["text"] if align.is_cjk_char(ch)]
            n = len(chars)
            span = float(seg["end_s"]) - float(seg["start_s"])
            for k in range(n):
                naive_onsets.append(float(seg["start_s"]) + span * k / n)
        errors = [abs(min(naive_onsets, key=lambda o: abs(o - u["start_s"])) - u["start_s"]) * 1e3
                  for u in truth["units"]]
        naive_p90 = float(np.percentile(errors, 90))
        assert new["p90_ms"] <= 50.0
        assert naive_p90 > new["p90_ms"] + 20.0

    def test_pause_split_and_refinement(self, case):
        samples, transcript, _, starts, _ = case
        doc = align.align_transcript(transcript, samples=samples)
        # the long pause after the first phrase must not be filled by a character
        gap_start = starts[9] + 0.145
        for char in doc["chars"]:
            a, b = float(char["start_s"]), float(char["end_s"])
            assert not (gap_start + 0.10 < a < gap_start + 0.28), (
                f"character {char['text']} starts inside the pause: {a:.3f}s")
        # a refined onset never lands more than 100 ms before its envelope peak
        for syl in doc["syllables"]:
            assert float(syl["start_s"]) <= float(syl["peak_s"]) + 1e-6
            assert float(syl["peak_s"]) - float(syl["start_s"]) <= 0.101


class TestSpeechFeatures:
    def test_vad_finds_the_phrases(self):
        samples, _, _, starts, ends = synthetic_case()
        features = align.audio_features(samples)
        # one long pause after the first phrase -> two speech islands
        assert len(features.regions) == 2
        for (s, e), (truth_s, truth_e) in zip(features.regions,
                                              [(starts[0], ends[9]),
                                               (starts[10], ends[20])]):
            assert abs(s - truth_s) < 0.12
            assert abs(e - truth_e) < 0.20

    def test_envelope_is_zero_in_silence_and_peaks_on_onsets(self):
        samples, _, _, starts, _ = synthetic_case()
        features = align.audio_features(samples)
        assert features.onset.max() > 0.5
        # a frame 150 ms before the first syllable is silence
        quiet = int(0.15 / features.hop_s)
        assert features.onset[quiet] == 0.0
        # the strongest onset in the first 0.6 s is near the first syllable
        head = features.onset[: int(0.9 / features.hop_s)]
        assert abs(int(np.argmax(head)) * features.hop_s - starts[0]) < 0.15


# ============================================================== real fixture (cache)


def _fx_a_transcript() -> dict | None:
    if not FX_A_GT.exists():
        return None
    gt = json.loads(FX_A_GT.read_text(encoding="utf-8"))
    cache = asr.cache_path(gt["sha256"], "medium", "cpu")
    if not cache.exists() or not FX_A.exists():
        return None
    return json.loads(cache.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fx_a_case():
    transcript = _fx_a_transcript()
    if transcript is None:
        pytest.skip("fx_a transcript not cached (tests never run the ASR model)")
    gt = json.loads(FX_A_GT.read_text(encoding="utf-8"))
    doc = align.align_video(FX_A, transcript)
    return transcript, gt, doc


class TestRealFixture:
    def test_contract_is_valid(self, fx_a_case):
        _, _, doc = fx_a_case
        assert align.validate(doc) == []
        assert doc["quality"]["monotonicity_violations"] == 0

    def test_character_onset_p90_under_80ms(self, fx_a_case):
        _, gt, doc = fx_a_case
        score = align.score_unit_onsets(doc, gt)
        assert score["n_scored"] == len(gt["units"]) == 14  # n=14: exploratory, not robust
        assert score["p90_ms"] <= 80.0, score
        assert score["max_ms"] <= 120.0

    def test_beats_the_existing_word_times(self, fx_a_case):
        transcript, gt, doc = fx_a_case
        new = align.score_unit_onsets(doc, gt)
        old_word = align.score_word_onsets(transcript, gt, mode="word")
        old_naive = align.score_word_onsets(transcript, gt, mode="naive_char")
        assert new["p90_ms"] < old_word["p90_ms"]
        assert new["p90_ms"] < old_naive["p90_ms"]

    def test_syllable_durations_are_plausible_mandarin(self, fx_a_case):
        _, _, doc = fx_a_case
        stats = doc["quality"]["syllable_ms"]
        assert 150.0 <= stats["median"] <= 320.0, stats
        assert stats["min"] >= 40.0

    def test_captions_follow_segments(self, fx_a_case):
        _, _, doc = fx_a_case
        cap = captions.build_captions(doc)
        assert cap["options"]["respect_segments"] is True
        n_segments = len({w["segment_id"] for w in doc["words"]})
        assert len(cap["cues"]) == n_segments
        assert cap["cues"][0]["text"] == "大家好,今天聊一个真实的痛点"
        assert cap["quality"]["mid_word_cue_starts"] == 0
        assert cap["quality"]["min_duration_s"] >= 0.9 - 1e-9
        assert captions.validate(cap) == []

    def test_known_bad_segment_boundary_is_repaired(self, fx_a_case):
        """The old pipeline's first segment claims start 0.64 s; speech starts at 0.50."""
        _, _, doc = fx_a_case
        first = doc["chars"][0]
        assert abs(float(first["start_s"]) - 0.50) <= 0.10
        assert doc["quality"]["onset_envelope"]["point_biserial_r"] is not None
        assert doc["quality"]["onset_envelope"]["point_biserial_r"] > 0.15


# ========================================================================== captions


def fake_alignment(items: list[tuple[str, float, float, str]], *, language: str = "zh",
                   duration_s: float | None = None, fps: float = 25.0,
                   segments: list[str] | None = None,
                   groups: list[str | None] | None = None) -> dict:
    """Build a minimal alignment document from ``(text, start, end, punct)`` items."""
    words = []
    for i, (text, start, end, punct) in enumerate(items):
        seg_id = segments[i] if segments is not None else "s0"
        group = groups[i] if groups is not None else None
        words.append({
            "id": f"w{i}", "segment_id": seg_id, "text": text, "kind": "han",
            "n_syllables": 1, "start_s": round(start, 6), "end_s": round(end, 6),
            "conf": 0.9, "char_start": i, "char_end": i + 1,
            "trailing_punct": punct, "group_id": group, "syllable_ids": [i],
        })
    return {
        "schema": "semotion.alignment/1",
        "source": {"path": "/dev/null", "sha256": "0" * 64,
                   "duration_s": duration_s or (items[-1][2] + 1.0 if items else 1.0),
                   "fps": fps, "width": 1080, "height": 1920, "language": language,
                   "audio_rate": SR},
        "method": {"name": align.METHOD, "truth_class": "MACHINE_WEAK", "params": {}},
        "segments": [],
        "syllables": [],
        "words": words,
        "chars": [],
        "quality": {},
        "warnings": [],
    }


def continuous_items(text: str, *, start: float = 0.5, step: float = 0.12,
                     punct_at: dict[int, str] | None = None,
                     gaps: dict[int, float] | None = None) -> list[tuple[str, float, float, str]]:
    items = []
    t = start
    for i, ch in enumerate(text):
        gap = (gaps or {}).get(i, 0.0)
        t += gap
        items.append((ch, t, t + step, (punct_at or {}).get(i, "")))
        t += step
    return items


class TestCaptions:
    def test_cjk_line_and_duration_rules(self):
        text = "今天我们来测试字幕分行的效果看看是否满足所有的约束条件并且没有行尾标点"
        doc = fake_alignment(continuous_items(text, step=0.10),
                             duration_s=0.5 + len(text) * 0.10 + 1.0)
        cap = captions.build_captions(doc)
        assert captions.validate(cap) == []
        assert cap["quality"]["max_line_chars"] <= 24
        assert cap["quality"]["min_duration_s"] >= 0.90 - 1e-9
        for cue in cap["cues"]:
            assert len(cue["lines"]) <= 2
            for line in cue["lines"]:
                assert 1 <= len(line) <= 24
                assert captions.strip_trailing_punct(line) == line
            assert cue["duration_s"] >= 0.90 - 1e-9
            assert cue["start_s"] == pytest.approx(cue["start_frame"] / 25.0, abs=1e-6)
            assert cue["end_s"] == pytest.approx(cue["end_frame"] / 25.0, abs=1e-6)

    def test_never_ends_line_with_punctuation(self):
        text = "第一句话在这里结束第二句话马上开始继续测试分行规则是否可靠"
        punct = {6: "。", 13: "，" , 20: "。"}
        doc = fake_alignment(continuous_items(text, step=0.20, punct_at=punct),
                             duration_s=0.5 + len(text) * 0.20 + 1.0)
        cap = captions.build_captions(doc)
        assert captions.validate(cap) == []
        for cue in cap["cues"]:
            for line in cue["lines"]:
                assert not re.search(r"[，。、！？；：,.!?;:…]$", line), line

    def test_short_cue_is_extended_or_merged(self):
        items = [("好", 1.0, 1.15, ""), ("的", 2.5, 2.65, ""), ("确", 2.65, 2.80, ""),
                 ("实", 2.80, 2.95, ""), ("如", 2.95, 3.10, ""), ("此", 3.10, 3.25, "")]
        cap = captions.build_captions(fake_alignment(items, duration_s=5.0))
        assert captions.validate(cap) == []
        assert cap["cues"][0]["duration_s"] >= 0.90 - 1e-9
        assert all(c["duration_s"] >= 0.90 - 1e-9 for c in cap["cues"])

    def test_small_gaps_are_merged(self):
        text = "字幕之间的小空隙不应该闪烁"
        items = continuous_items(text, step=0.15, gaps={5: 0.10, 10: 0.05})
        cap = captions.build_captions(fake_alignment(items, duration_s=8.0))
        assert captions.validate(cap) == []
        for a, b in zip(cap["cues"], cap["cues"][1:]):
            gap = float(b["start_s"]) - float(a["end_s"])
            assert gap == 0.0 or gap >= 0.18 - 1 / 25.0, (
                f"flicker gap {gap:.3f}s between {a['id']} and {b['id']}")

    def test_latin_wrap_at_42_chars(self):
        text = ("this caption line should wrap at forty two characters without "
                "cutting any word in half and stay within two lines")
        words = text.split()
        items: list[tuple[str, float, float, str]] = []
        t = 0.4
        for i, word in enumerate(words):
            items.append((word, t, t + 0.28, ""))
            t += 0.30
        cap = captions.build_captions(fake_alignment(items, language="en", duration_s=t + 1.0))
        assert captions.validate(cap) == []
        for cue in cap["cues"]:
            assert len(cue["lines"]) <= 2
            for line in cue["lines"]:
                assert len(line) <= 42, line
                # Latin wrapping must not split a word: every line is whole words
                assert line.replace(" ", "") != "" 
                for word in line.split():
                    assert word in words

    def test_export_json_srt_ass(self, tmp_path):
        text = "导出字幕文件并检查时间轴是否准确"
        doc = fake_alignment(continuous_items(text, step=0.18), duration_s=8.0)
        cap = captions.build_captions(doc)
        paths = captions.export(cap, tmp_path, "demo")
        data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
        assert data["schema"] == "semotion.captions/1"
        srt = Path(paths["srt"]).read_text(encoding="utf-8")
        blocks = [b for b in srt.strip().split("\n\n") if b]
        assert len(blocks) == len(cap["cues"])
        stamps = re.findall(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", srt)
        assert len(stamps) == len(cap["cues"])
        for (s0, _e0), (s1, _e1) in zip(stamps, stamps[1:]):
            assert s1 >= s0
        ass = Path(paths["ass"]).read_text(encoding="utf-8")
        assert "[V4+ Styles]" in ass and "Dialogue:" in ass
        assert ass.count("Dialogue:") == len(cap["cues"])
        assert "\\N" in ass or len(cap["cues"]) >= 1

    def test_validate_rejects_bad_caption(self):
        text = "正常字幕"
        cap = captions.build_captions(fake_alignment(
            continuous_items(text, step=0.4), duration_s=5.0))
        broken = json.loads(json.dumps(cap))
        broken["cues"][0]["lines"] = ["这" * 30]
        broken["cues"][0]["end_frame"] = broken["cues"][0]["start_frame"] + 2
        problems = captions.validate(broken)
        assert any("chars" in p for p in problems)
        assert any("display" in p for p in problems)

    def test_json_is_deterministic(self):
        text = "同一份对齐必须输出同一份字幕并且逐帧一致"
        doc = fake_alignment(continuous_items(text, step=0.13), duration_s=8.0)
        a = captions.build_captions(doc)
        b = captions.build_captions(doc)
        assert json.dumps(a, ensure_ascii=False, sort_keys=True) == \
            json.dumps(b, ensure_ascii=False, sort_keys=True)


class TestCaptionSegmentsAndWords:
    """task-3b: cue breaks follow ASR segments and never split an ASR word."""

    def test_default_respects_segment_boundaries(self):
        items = [("第", 0.50, 0.64, ""), ("一", 0.64, 0.78, ""), ("句", 0.78, 0.92, ""),
                 ("第", 1.02, 1.16, ""), ("二", 1.16, 1.30, ""), ("句", 1.30, 1.44, "")]
        alignment = fake_alignment(items, segments=["s0"] * 3 + ["s1"] * 3)
        cap = captions.build_captions(alignment)
        assert cap["options"]["respect_segments"] is True
        assert len(cap["cues"]) == 2, cap["cues"]
        assert cap["cues"][0]["lines"] == ["第一句"]
        assert cap["cues"][1]["lines"] == ["第二句"]
        for cue in cap["cues"]:
            assert len(cue["segment_ids"]) == 1
        assert cap["quality"]["mid_word_cue_starts"] == 0
        assert cap["quality"]["segment_boundary_breaks"] == 1
        assert captions.validate(cap) == []

    def test_without_respect_segments_a_short_gap_merges(self):
        items = [("第", 0.50, 0.64, ""), ("一", 0.64, 0.78, ""), ("句", 0.78, 0.92, ""),
                 ("第", 1.02, 1.16, ""), ("二", 1.16, 1.30, ""), ("句", 1.30, 1.44, "")]
        alignment = fake_alignment(items, segments=["s0"] * 3 + ["s1"] * 3)
        old = captions.build_captions(alignment, respect_segments=False)
        assert len(old["cues"]) == 1  # 0.10 s gap < merge_gap -> merged across segments
        new = captions.build_captions(alignment)
        assert len(new["cues"]) == 2

    def test_segment_mode_ignores_pauses_inside_a_segment(self):
        # one ASR segment with a 0.5 s pause in the middle -> one cue when
        # segment-respecting, two cues with pause heuristics.
        items = [("前", 0.5, 0.64, ""), ("半", 0.64, 0.78, ""), ("句", 0.78, 0.92, ""),
                 ("后", 1.42, 1.56, ""), ("半", 1.56, 1.70, ""), ("句", 1.70, 1.84, "")]
        alignment = fake_alignment(items)
        new = captions.build_captions(alignment)
        old = captions.build_captions(alignment, respect_segments=False)
        assert len(new["cues"]) == 1
        assert len(old["cues"]) == 2
        assert captions.validate(new) == []

    def test_word_guard_can_fail_and_prevents_mid_word_cue_start(self):
        text = "".join("字" for _ in range(50))
        items = [(ch, 0.5 + 0.1 * i, 0.6 + 0.1 * i, "") for i, ch in enumerate(text)]
        # one 3-character ASR word straddling the *current* cue cap (derived from
        # the policy + canvas, not the old 44-char constant)
        probe = captions.build_captions(fake_alignment(items,
                                                       groups=[f"g{i}" for i in range(50)]))
        boundary = int(probe["style"]["cjk"]["max_cue_chars"])
        groups = [f"g{i}" for i in range(50)]
        for k in (boundary - 2, boundary - 1, boundary):
            groups[k] = "g-long"
        alignment = fake_alignment(items, groups=groups)
        unguarded = captions.build_captions(alignment, respect_words=False)
        guarded = captions.build_captions(alignment)
        assert unguarded["quality"]["mid_word_cue_starts"] >= 1
        assert guarded["quality"]["mid_word_cue_starts"] == 0
        assert captions.validate(guarded) == []

    def test_fonts_match_renderer(self):
        assert captions.DEFAULT_STYLE["font_cjk"] == "Noto Sans SC"
        assert captions.DEFAULT_STYLE["font_latin"] == "Inter"
        items = continuous_items("字幕字体", step=0.4)
        doc = captions.build_captions(fake_alignment(items, duration_s=5.0))
        ass = captions.write_ass(doc, ROOT / "artifacts/p2/align/work_font_check.ass").read_text(
            encoding="utf-8")
        assert "Noto Sans SC" in ass
        (ROOT / "artifacts/p2/align/work_font_check.ass").unlink(missing_ok=True)
        en = captions.build_captions(fake_alignment(
            [("hello", 0.5, 0.8, ""), ("world", 0.8, 1.1, "")], language="en",
            duration_s=3.0))
        en_ass = captions.write_ass(en, ROOT / "artifacts/p2/align/work_font_check_en.ass")
        assert "Inter" in en_ass.read_text(encoding="utf-8")
        en_ass.unlink(missing_ok=True)


# ==================================================================== schema guards


class TestValidation:
    def test_alignment_validate_catches_non_monotonic_characters(self, fx_a_case):
        _, _, doc = fx_a_case
        broken = json.loads(json.dumps(doc))
        broken["chars"][2]["start_s"] = broken["chars"][0]["start_s"] - 0.5
        problems = align.validate(broken)
        assert any("monotonic" in p for p in problems)

    def test_alignment_validate_catches_bad_interval(self, fx_a_case):
        _, _, doc = fx_a_case
        broken = json.loads(json.dumps(doc))
        broken["chars"][1]["end_s"] = broken["chars"][1]["start_s"] - 1.0
        assert any("bad interval" in p for p in align.validate(broken))

    def test_schema_names(self, fx_a_case):
        _, _, doc = fx_a_case
        assert align.SCHEMA == "semotion.alignment/1"
        assert captions.SCHEMA == "semotion.captions/1"
        assert doc["method"]["name"] == align.METHOD


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
