"""Semantic Motion — deterministic character / word forced alignment.

Why this module exists
----------------------
Whisper's Chinese *word* timestamps are not character timestamps. On real
speech they are visibly wrong at syllable level (e.g. a segment starting at
0.00 s whose second reported "word" 运 starts at 0.66 s, while the speakers'
first syllable is long gone), and the pipeline's cruder "energy onset snap"
move an anchor to the next local energy maximum, which can be the next
syllable.  Since every overlay and caption in this project is anchored to a
word, that error is the product's error.

This module ignores Whisper's word times and re-derives them deterministically
from the audio itself:

1. decode to 16 kHz mono PCM (through :mod:`semotion.media`),
2. compute log-mel spectral flux + low-band flux + energy rise -> an *onset
   strength* envelope (numpy only),
3. estimate speech islands with a hysteresis energy VAD,
4. tokenise the transcript text into syllable-bearing units
   (one Han character = one syllable, Arabic digit strings = their spoken
   Mandarin reading, a Latin word = one onset in Chinese text, English words
   get a syllable-count heuristic), punctuation carries no duration,
5. allocate the syllable sequence to speech islands (pauses are hard chunk
   boundaries, punctuation is used to snap the allocation),
6. run a monotonic duration-constrained DP that places every syllable start
   (and the final offset) on the onset envelope, with soft anchors from the
   ASR segment boundaries (±120 ms) and N-syllable-per-island length priors,
7. optionally refine each onset backwards from the envelope peak to the energy
   rise (consonant attack).  It is **off by default**: on the SYNTH_KNOWN
   fixtures it moves onsets ~12 ms early and roughly doubles P90 error
   (7.0 -> 16.4 ms), because a spectral-flux peak is already the syllable
   onset for this material.  ``refine_onsets=1`` keeps it available for
   material with long unvoiced consonants; the evidence records the ablation.

Output contract: ``semotion.alignment/1`` — per-character and per-word times,
syllable-level detail, segment boundaries, the method/parameters used, and a
per-file quality report.  The document is labelled ``MACHINE_WEAK``: it is
aligned to the audio, not certified by a human.

Truth / evaluation
------------------
The only ground truth available offline is SYNTH_KNOWN (per-unit TTS
concatenated on a known timeline, ``tests/fixtures/clips/*/ground_truth.json``).
:func:`score_unit_onsets` scores predicted character onsets against those unit
boundaries.  It is a *dev* set; ``docs/ALIGNMENT.md`` reports n and refuses to
present small-n numbers as robust.  There is no human-labelled real-speech
corpus in this repo, and this module cannot create one.
"""
from __future__ import annotations

import json
import math
import re
import tempfile
import unicodedata
import wave
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import media

__all__ = [
    "SCHEMA",
    "METHOD",
    "AUDIO_RATE",
    "DEFAULT_PARAMS",
    "AudioFeatures",
    "Token",
    "align_transcript",
    "align_video",
    "load_audio",
    "audio_features",
    "speech_regions",
    "tokenize_text",
    "number_syllables",
    "english_syllables",
    "save",
    "load",
    "validate",
    "score_unit_onsets",
    "syllable_duration_stats",
    "onset_peak_report",
    "text_at_time",
    "char_at_time",
]

SCHEMA = "semotion.alignment/1"
METHOD = "envelope-dp/1"
AUDIO_RATE = 16000

DEFAULT_PARAMS: dict[str, float] = {
    # front end
    "frame_s": 0.025,
    "hop_s": 0.010,
    "n_mels": 40,
    "fmin_hz": 80.0,
    "fmax_hz": 7600.0,
    # voice activity detection (energy dB hysteresis)
    "vad_min_speech_s": 0.060,
    "vad_fill_gap_s": 0.090,
    "vad_floor_db": 4.0,
    "vad_high_frac": 0.25,
    # pause / chunking
    "pause_min_s": 0.140,
    # duration model (Mandarin syllable ~150-260 ms; neutral tone shorter)
    "seq_med_syllable_s": 0.205,
    "seq_min_syllable_s": 0.060,
    "seq_max_syllable_s": 1.300,
    # DP weights
    "w_onset": 1.00,
    "w_dur": 1.20,
    "w_anchor": 0.70,
    "anchor_tol_s": 0.120,  # trust ASR segment bounds to ~±120 ms
    "anchor_scale": 40.0,
    # onset refinement
    "refine_onsets": 0,
    "refine_max_lead_s": 0.120,
    "refine_min_lead_s": 0.025,
    "refine_energy_drop_db": 16.0,
    # reporting
    "low_conf": 0.35,
}

_EPS = 1e-12

# --------------------------------------------------------------------------- audio


@dataclass
class AudioFeatures:
    """Frame-wise audio analysis used by the aligner (also used by evidence)."""

    samples: np.ndarray
    sample_rate: int
    hop_s: float
    frame_s: float
    energy_db: np.ndarray
    onset: np.ndarray
    speech: np.ndarray
    regions: list[tuple[float, float]]
    noise_db: float
    speech_db: float
    threshold_db: float
    params: dict[str, float] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.energy_db.size)

    @property
    def duration_s(self) -> float:
        return len(self.samples) / float(self.sample_rate)

    def frame_time(self, frame: int) -> float:
        return float(frame) * self.hop_s

    @property
    def speech_fraction(self) -> float:
        return float(self.speech.mean()) if self.n_frames else 0.0


def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise media.MediaError("E_ALIGN_FAILED", f"expected 16-bit PCM wav: {path}")
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    if not raw:
        raise media.MediaError("E_ALIGN_FAILED", f"empty audio decoded from {path}")
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, int(rate)


def load_audio(source: str | Path, *, rate: int = AUDIO_RATE) -> tuple[np.ndarray, int]:
    """Decode any ffmpeg-readable media (or a wav) to mono float32 at ``rate``."""
    source = Path(source)
    if not source.exists():
        raise media.MediaError("E_INPUT_MISSING", f"input not found: {source}")
    with tempfile.TemporaryDirectory(prefix="semotion-align-") as tmp:
        wav = media.extract_audio(source, Path(tmp) / "audio.wav", rate=rate, channels=1)
        samples, got = _read_wav_mono(wav)
    if got != rate:  # pragma: no cover - ffmpeg contract guard
        raise media.MediaError("E_ALIGN_FAILED", f"expected {rate} Hz, decoded {got} Hz")
    if samples.size < rate // 10:
        raise media.MediaError("E_ALIGN_FAILED", f"audio too short: {samples.size} samples")
    return samples, got


@lru_cache(maxsize=8)
def _mel_filterbank(n_mels: int, n_fft: int, sample_rate: int,
                    fmin: float, fmax: float) -> np.ndarray:
    """Triangular mel filterbank (n_mels, n_fft//2 + 1) with numpy only."""
    def hz2mel(f: float) -> float:
        return 2595.0 * math.log10(1.0 + f / 700.0)

    def mel2hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * np.array([mel2hz(m) for m in mels]) / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, centre, right = int(bins[m - 1]), int(bins[m]), int(bins[m + 1])
        centre = max(centre, left + 1)
        right = max(right, centre + 1)
        for k in range(left, min(centre, n_fft // 2 + 1)):
            fb[m - 1, k] = (k - left) / float(centre - left)
        for k in range(centre, min(right, n_fft // 2 + 1)):
            fb[m - 1, k] = (right - k) / float(right - centre)
    return fb


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) index runs of True values."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _robust_norm(values: np.ndarray, low: float = 5.0, high: float = 95.0) -> np.ndarray:
    """Scale a non-negative feature to roughly [0, 1] using robust percentiles."""
    values = np.asarray(values, dtype=np.float64)
    lo = float(np.percentile(values, low))
    hi = float(np.percentile(values, high))
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 2.0)


def audio_features(samples: Sequence[float], *, sample_rate: int = AUDIO_RATE,
                   params: dict[str, Any] | None = None) -> AudioFeatures:
    """Compute log-mel flux, energy and a hysteresis VAD.

    The onset strength is a weighted sum of three complementary cues, each
    robustly normalised: full-band log-mel positive flux (consonant/vowel
    change), low-band (<800 Hz) flux (syllable nuclei in Mandarin) and frame
    energy rise (stop bursts).  Pure log energy alone is ambiguous on merged
    syllables, which is exactly why the old refinement failed.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    hop_s = float(p["hop_s"])
    frame_s = float(p["frame_s"])
    n_fft = int(round(frame_s * sample_rate))
    hop = int(round(hop_s * sample_rate))
    if n_fft < 32 or hop < 1:
        raise media.MediaError("E_ALIGN_FAILED", f"bad frame/hop: {frame_s}s/{hop_s}s")

    x = np.asarray(samples, dtype=np.float32).flatten()
    if x.size <= n_fft:
        raise media.MediaError("E_ALIGN_FAILED", "audio shorter than one analysis frame")
    x = x - float(x.mean())

    padded = np.pad(x, (n_fft // 2, n_fft // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]
    window = np.hanning(n_fft).astype(np.float32)
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2  # (T, F)
    del frames

    energy_db = 10.0 * np.log10(power.sum(axis=1) + _EPS)
    # 5-frame (~50 ms) smoothing for VAD / refinement only
    kernel = np.ones(5, dtype=np.float64) / 5.0
    energy_sm = np.convolve(energy_db, kernel, mode="same")

    fb = _mel_filterbank(int(p["n_mels"]), n_fft, sample_rate,
                         float(p["fmin_hz"]), float(p["fmax_hz"]))
    mel = np.log(power @ fb.T + 1e-10)  # (T, n_mels)
    dmel = np.diff(mel, axis=0, prepend=mel[:1])
    flux = np.maximum(dmel, 0.0).sum(axis=1)

    low_bins = max(4, int(round(800.0 / (sample_rate / 2.0) * (n_fft // 2 + 1))))
    low = np.log(power[:, :low_bins].sum(axis=1) + 1e-10)
    low_flux = np.maximum(np.diff(low, prepend=low[:1]), 0.0)
    d_energy = np.maximum(np.diff(energy_sm, prepend=energy_sm[:1]), 0.0)

    # VAD first, so normalisation is not diluted by silence
    noise_db = float(np.percentile(energy_sm, 10))
    speech_db = float(np.percentile(energy_sm, 92))
    dynamic = max(speech_db - noise_db, 1.0)
    hi_thr = noise_db + max(8.0, float(p["vad_high_frac"]) * dynamic)
    lo_thr = noise_db + max(float(p["vad_floor_db"]), 0.10 * dynamic)

    core = energy_sm > hi_thr
    loose = energy_sm > lo_thr
    speak = np.zeros(energy_db.size, dtype=bool)
    for start, end in _runs(loose):
        if core[start:end].any():
            speak[start:end] = True

    fill = max(1, int(round(float(p["vad_fill_gap_s"]) / hop_s)))
    min_speech = max(1, int(round(float(p["vad_min_speech_s"]) / hop_s)))
    run_list = _runs(speak)
    merged: list[tuple[int, int]] = []
    for start, end in run_list:
        if merged and start - merged[-1][1] <= fill:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    merged = [(s, e) for s, e in merged if e - s >= min_speech]
    speak = np.zeros(energy_db.size, dtype=bool)
    for s, e in merged:
        speak[s:e] = True

    mask = speak
    if mask.sum() >= 10:
        flux_n = _robust_norm(flux[mask])
        low_n = _robust_norm(low_flux[mask])
        dE_n = _robust_norm(d_energy[mask])
        onset = np.zeros_like(flux, dtype=np.float64)
        onset[mask] = 0.62 * flux_n + 0.22 * low_n + 0.16 * dE_n
    else:  # degenerate file (no detected speech): fall back to all frames
        onset = 0.62 * _robust_norm(flux) + 0.22 * _robust_norm(low_flux) \
            + 0.16 * _robust_norm(d_energy)
        mask = np.ones_like(mask)

    # light smoothing (keeps peaks, removes single-frame jitter)
    onset = np.convolve(onset, np.array([0.25, 0.5, 0.25]), mode="same")
    onset = np.clip(onset, 0.0, None)
    onset[~mask] = 0.0

    regions = [(float(s) * hop_s, float(e) * hop_s) for s, e in _runs(mask)]
    return AudioFeatures(
        samples=x, sample_rate=sample_rate, hop_s=hop_s, frame_s=frame_s,
        energy_db=energy_sm, onset=onset, speech=mask, regions=regions,
        noise_db=noise_db, speech_db=speech_db, threshold_db=float(hi_thr), params=p,
    )


def speech_regions(samples: Sequence[float], **kwargs: Any) -> list[tuple[float, float]]:
    """Convenience: speech islands ``[(start_s, end_s), ...]`` for raw samples."""
    return audio_features(samples, **kwargs).regions


# ----------------------------------------------------------------------- tokenise

_CJK_RANGES = (
    (0x3400, 0x4DBF),    # CJK Ext A
    (0x4E00, 0x9FFF),    # CJK Unified
    (0xF900, 0xFAFF),    # CJK Compatibility
    (0x3040, 0x30FF),    # Hiragana / Katakana (spoken as syllables too)
    (0xAC00, 0xD7AF),    # Hangul syllables
)

_DIGITS = "零一二三四五六七八九"


def is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def _is_punct(ch: str) -> bool:
    if ch.isspace():
        return True
    cat = unicodedata.category(ch)
    # P* = punctuation, S* = symbols (·…~ are separators in captions)
    return cat.startswith("P") or ch in "…—·~～"


def _int_to_chinese(n: int) -> str:
    """Cardinal Mandarin reading of a non-negative integer (for syllable counts)."""
    if n == 0:
        return "零"
    groups: list[int] = []
    while n > 0:
        groups.append(n % 10000)
        n //= 10000
    big = ["", "万", "亿", "万亿"]
    out = ""
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            if out and not out.endswith("零"):
                out += "零"
            continue
        if out and g < 1000 and not out.endswith("零"):
            out += "零"
        four = ""
        zero = False
        for pos, unit in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
            d = (g // pos) % 10
            if d == 0:
                if four and (g % pos) != 0:
                    zero = True
            else:
                if zero:
                    four += "零"
                    zero = False
                four += _DIGITS[d] + unit
        out += four + big[i]
    out = out.rstrip("零") or "零"
    # 一十 -> 十 (10..19) is the normal spoken form
    if out.startswith("一十"):
        out = out[1:]
    return out


def number_syllables(text: str, *, year_hint: bool = False) -> int:
    """How many Mandarin syllables an Arabic numeral string is spoken with.

    Rules (documented in docs/ALIGNMENT.md, all heuristic):
    * 4-digit 1900..2100 (and any digit string followed by 年/月/日/号) is read
      digit by digit (二零二四 = 4), which is how Chinese TTS/people read years;
    * other integers use the cardinal reading (25 -> 二十五 = 3, 100 -> 一百 = 2);
    * decimals: integer + 点 + one syllable per fractional digit (3.5 -> 3);
    * ``%`` adds 百分之 (3), a leading minus adds 负 (1).
    """
    raw = unicodedata.normalize("NFKC", str(text)).strip()
    raw = raw.replace(",", "").replace(" ", "").replace("，", "")
    if not raw:
        return 1
    pct = raw.endswith("%")
    if pct:
        raw = raw[:-1]
    neg = raw.startswith("-") or raw.startswith("−")
    if neg:
        raw = raw[1:]
    if not re.fullmatch(r"\d+(\.\d+)?", raw or ""):
        return 1
    if "." in raw:
        int_part, frac = raw.split(".", 1)
        count = number_syllables(int_part) + 1 + len(frac)  # 点 + digits
    else:
        int_part = raw
        if year_hint or (len(int_part) == 4 and 1900 <= int(int_part) <= 2100):
            count = len(int_part)
        else:
            count = len(_int_to_chinese(int(int_part)))
    if neg:
        count += 1
    if pct:
        count += 3  # 百分之
    return max(1, int(count))


def english_syllables(word: str) -> int:
    """Vowel-group syllable count heuristic for English words."""
    letters = re.sub(r"[^a-z]", "", unicodedata.normalize("NFKC", word).lower())
    if not letters:
        return 1
    groups = re.findall(r"[aeiouy]+", letters)
    n = len(groups)
    if n == 0:
        return 1
    if letters.endswith("e") and not letters.endswith(("le", "ee", "ye", "oe")) and n > 1:
        n -= 1
    if letters.endswith("es") and n > 1 and not letters.endswith(("ses", "xes", "zes", "ces", "ches", "shes")):
        n -= 1
    return max(1, n)


@dataclass
class Token:
    """One lexical unit carrying one or more syllable onsets."""

    kind: str          # "han" | "number" | "latin"
    text: str
    char_start: int    # offset in the segment text
    char_end: int
    n_syll: int
    trailing_punct: str = ""
    segment_id: str = ""
    index: int = 0
    #: ASR word this token belongs to (``<segment>:<word id>``), or None when the
    #: transcript's word spans do not cover it / the token crosses a word edge.
    group_id: str | None = None


def tokenize_text(text: str, language: str = "zh") -> list[Token]:
    """Split transcript text into syllable-bearing tokens; punctuation is kept
    as ``trailing_punct`` (it never consumes time)."""
    cjk_mode = str(language).lower().startswith(("zh", "ja", "ko"))
    text = str(text or "")
    tokens: list[Token] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        # --- signed Arabic number (with optional separators / decimal / percent)
        if ch in "-−" and i + 1 < n and text[i + 1].isdigit() and (
                not tokens or tokens[-1].char_end < i):
            start = i
            i += 1
            while i < n and text[i].isdigit():
                i += 1
            raw = text[start:i]
            tokens.append(Token("number", raw, start, i, number_syllables(raw)))
            continue
        # --- Arabic number (with optional group separators / decimal / percent)
        if ch.isdigit():
            start = i
            seen_dot = False
            while i < n and (text[i].isdigit() or (text[i] in ".," and i + 1 < n and text[i + 1].isdigit())):
                if text[i] in ".," and text[i + 1].isdigit():
                    # "1,000" is a separator; "3.5" is a decimal point. Only one
                    # dot is a decimal; commas are always separators.
                    if text[i] == "." and not seen_dot:
                        seen_dot = True
                    elif text[i] == "." and seen_dot:
                        break
                i += 1
            pct = False
            if i < n and text[i] == "%":
                pct = True
                i += 1
            raw = text[start:i]
            nxt = text[i:].lstrip()
            year_hint = bool(nxt) and nxt[0] in "年月日号"
            tokens.append(Token("number", raw, start, i,
                                number_syllables(raw, year_hint=year_hint)))
            if pct:
                tokens[-1].text = raw
            continue
        # --- CJK character: one syllable
        if is_cjk_char(ch):
            tokens.append(Token("han", ch, i, i + 1, 1))
            i += 1
            continue
        # --- Latin / Greek word: one onset in zh mode, syllable-weighted in en
        if ch.isalpha() and not is_cjk_char(ch):
            start = i
            while i < n:
                c = text[i]
                if c.isalpha() and not is_cjk_char(c):
                    i += 1
                elif c in "'’-" and i + 1 < n and text[i + 1].isalpha() \
                        and not is_cjk_char(text[i + 1]) and i > start:
                    i += 1
                else:
                    break
            word = text[start:i]
            n_syll = 1 if cjk_mode else english_syllables(word)
            tokens.append(Token("latin", word, start, i, n_syll))
            continue
        # --- punctuation / symbols: no duration, attach to the previous token
        if _is_punct(ch) or not ch.isalnum():
            if tokens:
                tokens[-1].trailing_punct += ch
            i += 1
            continue
        i += 1
    for idx, tok in enumerate(tokens):
        tok.index = idx
    return tokens


# ------------------------------------------------------------------------ scoring


def _pct(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def score_unit_onsets(alignment: dict[str, Any], ground_truth: dict[str, Any], *,
                      max_match_s: float = 0.6) -> dict[str, Any]:
    """Score predicted character onsets against SYNTH_KNOWN unit boundaries.

    For every truth unit the *nearest* predicted character onset is used (the
    same rule as ``tests/eval_timing.py``), so a mis-transcribed character does
    not disqualify the unit — this measures onset placement, not text identity.
    ``skipped`` counts units with no predicted onset inside ``max_match_s``.
    """
    chars = sorted(alignment.get("chars") or [], key=lambda c: float(c["start_s"]))
    onsets = [float(c["start_s"]) for c in chars]
    units = ground_truth.get("units") or []
    errors_ms: list[float] = []
    rows: list[dict[str, Any]] = []
    skipped = 0
    for unit in units:
        t0 = float(unit["start_s"])
        if not onsets:
            skipped += 1
            continue
        idx = int(np.argmin([abs(o - t0) for o in onsets]))
        err = (onsets[idx] - t0) * 1000.0
        if abs(err) > max_match_s * 1000.0:
            skipped += 1
            continue
        errors_ms.append(err)
        rows.append({
            "unit_id": unit.get("unit_id"), "text": unit.get("text"),
            "truth_start_s": round(t0, 6),
            "predicted_start_s": round(onsets[idx], 6),
            "predicted_char": chars[idx].get("text"),
            "error_ms": round(err, 2),
        })
    abs_values = [abs(v) for v in errors_ms]
    return {
        "n_units": len(units),
        "n_scored": len(errors_ms),
        "n_skipped": skipped,
        "p50_ms": round(_pct(abs_values, 50), 2) if abs_values else None,
        "p90_ms": round(_pct(abs_values, 90), 2) if abs_values else None,
        "max_ms": round(max(abs_values), 2) if abs_values else None,
        "mean_ms": round(float(np.mean(abs_values)), 2) if abs_values else None,
        "bias_ms": round(float(np.mean(errors_ms)), 2) if errors_ms else None,
        "rows": rows,
    }


def _naive_char_onsets(transcript: dict[str, Any]) -> list[float]:
    """Old-pipeline-style character onsets: split each ASR word evenly.

    Whisper only returns word times; captions built from them spread the word's
    duration uniformly over its characters.  This reproduces that behaviour so
    the evidence can compare like with like.
    """
    out: list[float] = []
    for word in transcript.get("words") or []:
        text = (word.get("text") or "").strip()
        letters = [c for c in text if not _is_punct(c)]
        n = max(1, len(letters))
        start, end = float(word["start_s"]), float(word["end_s"])
        for k in range(n):
            out.append(start + (end - start) * k / n)
    return out


def score_word_onsets(transcript: dict[str, Any], ground_truth: dict[str, Any], *,
                      mode: str = "word") -> dict[str, Any]:
    """Baseline score for the *existing* ASR pipeline (nearest word / naive char)."""
    if mode == "word":
        onsets = sorted(float(w["start_s"]) for w in transcript.get("words") or [])
    elif mode == "naive_char":
        onsets = sorted(_naive_char_onsets(transcript))
    else:  # pragma: no cover - guard
        raise ValueError(f"unknown mode {mode!r}")
    errors: list[float] = []
    for unit in ground_truth.get("units") or []:
        t0 = float(unit["start_s"])
        if not onsets:
            continue
        err = min(onsets, key=lambda o: abs(o - t0)) - t0
        errors.append(err * 1000.0)
    abs_values = [abs(v) for v in errors]
    return {
        "mode": mode,
        "n": len(abs_values),
        "p50_ms": round(_pct(abs_values, 50), 2) if abs_values else None,
        "p90_ms": round(_pct(abs_values, 90), 2) if abs_values else None,
        "max_ms": round(max(abs_values), 2) if abs_values else None,
        "bias_ms": round(float(np.mean(errors)), 2) if errors else None,
    }


# --------------------------------------------------------------------- DP internals


def _anchor_cost(frames: np.ndarray, target: int, tol_s: float, weight: float,
                 hop_s: float, scale: float) -> np.ndarray:
    dev = np.maximum(0.0, np.abs(frames.astype(np.float64) - float(target)) * hop_s - tol_s)
    return weight * scale * dev * dev


def _uniform_chunk(lo: int, hi: int, n: int) -> list[int]:
    """Evenly spaced boundary frames (fallback when the DP has no feasible path)."""
    if n <= 0:
        return []
    span = max(hi - lo, n)
    return [lo + int(round(span * k / n)) for k in range(n + 1)]


def _dp_chunk(onset: np.ndarray, speech: np.ndarray, lo: int, hi: int, n: int,
              anchors: dict[int, list[tuple[int, float, float]]],
              med_s: float, params: dict[str, Any]) -> tuple[list[int], bool]:
    """Monotonic DP for ``n`` syllable starts + one final offset inside [lo, hi).

    Returns (frames of the ``n+1`` boundaries, feasible flag).  Layer ``i``
    (0 <= i < n) is the start of syllable ``i``; layer ``n`` is the offset of
    the last syllable (no onset evidence, island/segment end anchors only).
    """
    hop_s = float(params["hop_s"])
    w_on = float(params["w_onset"])
    w_dur = float(params["w_dur"])
    scale = float(params["anchor_scale"])
    dmin = max(2, int(round(float(params["seq_min_syllable_s"]) / hop_s)))
    dmax = max(dmin + 1, int(round(float(params["seq_max_syllable_s"]) / hop_s)))
    dmax = min(dmax, max(2, hi - lo - 1))
    span = hi - lo
    if span <= n or n <= 0:
        return _uniform_chunk(lo, hi, n), False

    e = np.asarray(onset[lo:hi], dtype=np.float64)
    allowed = np.asarray(speech[lo:hi], dtype=bool).copy()
    # Never require a boundary outside the island, but allow 1-frame edges.
    allowed[0] = True
    allowed[-1] = True
    inf = 1e18
    frames = np.arange(span, dtype=np.float64)

    def layer_cost(layer: int, base: np.ndarray) -> np.ndarray:
        cost = base
        if layer < n:
            cost = cost - w_on * e
        for target, tol, weight in anchors.get(layer, ()):
            cost = cost + _anchor_cost(frames, target - lo, tol, weight, hop_s, scale)
        cost = np.where(allowed, cost, inf)
        return cost

    back = np.full((n + 1, span), -1, dtype=np.int32)
    cost = layer_cost(0, np.zeros(span, dtype=np.float64))
    dur_cost = np.array([
        w_dur * math.log((d * hop_s) / max(med_s, 1e-6)) ** 2
        for d in range(dmin, dmax + 1)
    ], dtype=np.float64)

    for layer in range(1, n + 1):
        prev = cost
        new = np.full(span, inf, dtype=np.float64)
        new_back = np.full(span, -1, dtype=np.int32)
        for off, d in enumerate(range(dmin, dmax + 1)):
            if d >= span:
                break
            shifted = np.full(span, inf, dtype=np.float64)
            shifted[d:] = prev[:span - d] + dur_cost[off]
            better = shifted < new
            if better.any():
                new[better] = shifted[better]
                new_back[better] = np.arange(span, dtype=np.int32)[better] - d
        new = layer_cost(layer, new)
        # invalidate unreachable
        new[new >= inf / 2] = inf
        cost, back[layer] = new, new_back

    best = int(np.argmin(cost))
    if not np.isfinite(cost[best]) or cost[best] >= inf / 2:
        return _uniform_chunk(lo, hi, n), False
    path = [0] * (n + 1)
    cur = best
    path[n] = cur
    for layer in range(n, 0, -1):
        prev = int(back[layer][cur])
        if prev < 0:
            return _uniform_chunk(lo, hi, n), False
        cur = prev
        path[layer - 1] = cur
    return [lo + int(f) for f in path], True


def _refine_onset(features: AudioFeatures, peak: int, lower_frame: int,
                  params: dict[str, Any]) -> int:
    """Move an onset-envelope peak back to the syllable attack.

    The flux peak sits on the vowel nucleus; a listener hears the syllable start
    at the consonant / energy rise before it.  We take the deepest local energy
    trough in the ``refine_max_lead_s`` window before the peak and return the
    first frame where the smoothed level has risen clearly above it.  If there is
    no clear trough (e.g. a voiced glide straight into the vowel) the peak is
    returned unchanged, so the refinement can never invent an onset.

    This is deliberately conservative: a wrong early onset is worse for overlay
    sync than a peak that is a few tens of ms late.
    """
    hop_s = float(params["hop_s"])
    db = features.energy_db
    max_lead = max(1, int(round(float(params["refine_max_lead_s"]) / hop_s)))
    lo = max(int(lower_frame), 0, int(peak) - max_lead)
    lo = min(lo, int(peak) - 1)
    if int(peak) - lo < 2:
        return int(peak)
    window = db[lo: int(peak) + 1]
    trough = float(window.min())
    peak_db = float(db[peak])
    rise = peak_db - trough
    if rise < 10.0:  # no clear attack inside the window: keep the envelope peak
        return int(peak)
    local = window - trough
    # the *closest* local minimum before the peak is the attack candidate: an
    # earlier basin may belong to the previous syllable
    idx = int(np.argmin(local))
    for j in range(local.size - 2, 0, -1):
        if local[j] <= local[j - 1] and local[j] <= local[j + 1] \
                and local[j] <= trough + 0.35 * rise:
            idx = j
            break
    onset = int(lo + idx)
    # only move when a real consonant-like lead exists; otherwise the flux peak
    # is already the syllable onset and moving it would add error
    if (int(peak) - onset) * hop_s < float(params.get("refine_min_lead_s", 0.025)):
        return int(peak)
    threshold = trough + max(3.0, min(10.0, 0.30 * rise))
    i = onset
    while i < int(peak) and db[i] < threshold:
        i += 1
    return int(min(i, int(peak)))


# ------------------------------------------------------------------------ alignment


@dataclass
class _Syl:
    token_index: int
    segment_id: str
    first_of_token: bool
    last_of_token: bool
    first_of_seg: bool
    last_of_seg: bool


def _segment_list(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    segments = [s for s in (transcript.get("segments") or []) if (s.get("text") or "").strip()]
    segments.sort(key=lambda s: float(s["start_s"]))
    return segments


def _allocate_segment(seg_start: float, seg_end: float, a: int, b: int,
                      islands: Sequence[tuple[float, float]],
                      punct_zero: Sequence[bool]) -> list[tuple[int, int]] | None:
    """Assign one segment's syllable range ``[a, b)`` to the islands it covers.

    The segment's own syllable count is known from the transcript text, so this
    is far tighter than a global duration split: a segment that sits inside one
    speech island sends *all* its syllables there, and only a segment that
    straddles a pause is split (proportionally, snapped to punctuation).

    Returns ``[(island_index, count), ...]`` in time order, or ``None`` when the
    segment overlaps no detected speech (the caller then aligns it inside its
    ASR bounds and flags it).
    """
    n = b - a
    if n <= 0:
        return []
    overlaps: list[tuple[int, float]] = []
    for idx, (start, end) in enumerate(islands):
        ov = min(seg_end, end) - max(seg_start, start)
        if ov >= 0.04:
            overlaps.append((idx, ov))
    if not overlaps:
        return None
    if len(overlaps) == 1:
        return [(overlaps[0][0], n)]
    if len(overlaps) > n:  # cannot give each island a syllable: keep the longest
        overlaps = sorted(overlaps, key=lambda item: -item[1])[:n]
        overlaps.sort()
    total = sum(ov for _, ov in overlaps)
    counts: list[int] = []
    prev = 0
    for k in range(1, len(overlaps)):
        max_allowed = n - (len(overlaps) - k)
        target = int(round(sum(ov for _, ov in overlaps[:k]) / total * n))
        cut = max(prev + 1, min(target, max_allowed))
        candidates = [j for j in range(a + prev + 2, a + max_allowed + 1)
                      if punct_zero[j] and abs((j - a) - cut) <= 2]
        if candidates:
            cut = min(candidates, key=lambda j: abs((j - a) - cut)) - a
        counts.append(cut - prev)
        prev = cut
    counts.append(n - prev)
    return [(idx, count) for (idx, _), count in zip(overlaps, counts) if count > 0]


def _local_peaks(env: np.ndarray, *, min_value: float = 0.20,
                 prominence: float = 0.15, min_sep: int = 4) -> np.ndarray:
    """Prominent local maxima of the onset envelope (numpy only, no scipy).

    A raw local-max test fires on every jitter of the envelope; requiring
    prominence over a ±50 ms window and non-maximum suppression keeps the peaks
    that a listener would call syllable onsets.
    """
    n = int(env.size)
    if n < 3:
        return np.empty(0, dtype=np.int64)
    candidates: list[int] = []
    for i in range(1, n - 1):
        if env[i] < min_value or env[i] < env[i - 1] or env[i] <= env[i + 1]:
            continue
        lo = max(0, i - 5)
        hi = min(n, i + 6)
        if env[i] - float(env[lo:hi].min()) < prominence:
            continue
        candidates.append(i)
    kept: list[int] = []
    for i in sorted(candidates, key=lambda k: -float(env[k])):
        if all(abs(i - j) >= min_sep for j in kept):
            kept.append(i)
    return np.array(sorted(kept), dtype=np.int64)


def _quality(features: AudioFeatures, syllables: list[dict[str, Any]],
             chars: list[dict[str, Any]], segments: list[dict[str, Any]],
             params: dict[str, Any]) -> dict[str, Any]:
    hop_s = features.hop_s
    durations = [float(s["end_s"]) - float(s["start_s"]) for s in syllables]
    starts = [int(round(float(s["start_s"]) / hop_s)) for s in syllables]

    violations = 0
    prev_start = -1.0
    for c in chars:
        start, end = float(c["start_s"]), float(c["end_s"])
        if start + 1e-6 < prev_start or end < start:
            violations += 1
        prev_start = max(prev_start, start)
    del prev_start

    # Evidence quality: where the chosen onsets sit on the envelope.
    percentiles: list[float] = []
    peak_hits = 0
    n_starts = 0
    w = max(1, int(round(0.150 / hop_s)))
    for f in starts:
        if f <= 0 or f >= features.n_frames - 1:
            continue
        win = features.onset[max(0, f - w): f + w + 1]
        if win.size < 3:
            continue
        value = features.onset[f]
        percentiles.append(float((win < value).mean()))
        local = features.onset[max(0, f - 4): f + 5]
        if local.size and value >= float(local.max()) * 0.98:
            peak_hits += 1
        n_starts += 1
    # distance from a chosen frame to the nearest *prominent* envelope peak
    env = features.onset
    peak_frames = _local_peaks(env)

    def _peak_distance(frames: Sequence[int]) -> dict[str, Any]:
        if peak_frames.size == 0 or not frames:
            return {"n": 0}
        values = []
        for f in frames:
            values.append(float(np.min(np.abs(peak_frames - int(f))) * hop_s * 1e3))
        arr = np.asarray(values, dtype=float)
        return {
            "n": int(arr.size),
            "median_ms": round(float(np.median(arr)), 1),
            "p90_ms": round(float(np.percentile(arr, 90)), 1),
            "within_50ms_rate": round(float((arr <= 50.0).mean()), 4),
            "within_80ms_rate": round(float((arr <= 80.0).mean()), 4),
        }

    onset_mean = float(features.onset[starts].mean()) if starts else 0.0
    speech_frames = int(features.speech.sum())
    speech_mean = (float(features.onset[features.speech].mean())
                   if speech_frames else 0.0)
    # point-biserial correlation between predicted onsets and the envelope
    corr = None
    if speech_frames > 10 and starts:
        idx = np.flatnonzero(features.speech)
        ind = np.zeros(idx.size, dtype=np.float64)
        pos = {int(s): i for i, s in enumerate(idx)}
        for f in starts:
            if int(f) in pos:
                ind[pos[int(f)]] = 1.0
        env = features.onset[idx]
        if ind.std() > 0 and env.std() > 0:
            corr = float(np.corrcoef(ind, env)[0, 1])

    med = float(np.median(durations)) if durations else None
    seg_dev_start = [abs(float(s["speech_start_s"]) - float(s["start_s"]))
                     for s in segments if s.get("n_syllables")]
    seg_dev_end = [abs(float(s["speech_end_s"]) - float(s["end_s"]))
                   for s in segments if s.get("n_syllables")]

    low = float(params["low_conf"])
    return {
        "n_chars": len(chars),
        "n_syllables": len(syllables),
        "monotonicity_violations": int(violations),
        "syllable_ms": {
            "n": len(durations),
            "min": round(min(durations) * 1e3, 1) if durations else None,
            "p10": round(_pct(durations, 10) * 1e3, 1) if durations else None,
            "median": round(med * 1e3, 1) if med is not None else None,
            "p90": round(_pct(durations, 90) * 1e3, 1) if durations else None,
            "max": round(max(durations) * 1e3, 1) if durations else None,
        },
        "onset_envelope": {
            "onset_percentile_mean": round(float(np.mean(percentiles)), 4) if percentiles else None,
            "peak_hit_rate": round(peak_hits / n_starts, 4) if n_starts else None,
            "speech_mean_onset": round(speech_mean, 4),
            "onset_mean": round(onset_mean, 4),
            "contrast_ratio": round(onset_mean / speech_mean, 3) if speech_mean > 1e-9 else None,
            "point_biserial_r": round(corr, 4) if corr is not None else None,
            "nearest_peak_ms": _peak_distance(starts),
            "dp_peak_to_peak_ms": _peak_distance(
                [int(round(float(s["peak_s"]) / hop_s)) for s in syllables]),
        },
        "segment_anchor_dev_ms": {
            "start_median": round(_pct(seg_dev_start, 50) * 1e3, 1) if seg_dev_start else None,
            "end_median": round(_pct(seg_dev_end, 50) * 1e3, 1) if seg_dev_end else None,
        },
        "low_confidence_syllables": int(sum(
            1 for s in syllables if float(s["conf"]) < low)),
        "syllable_start_on_speech": round(
            float(np.mean([features.speech[min(features.n_frames - 1,
                                              max(0, int(round(float(s["start_s"]) / hop_s))))]
                           for s in syllables])) if syllables else 0.0, 4),
        "speech_fraction": round(features.speech_fraction, 4),
    }


def align_transcript(transcript: dict[str, Any], *, samples: Sequence[float] | None = None,
                     video: str | Path | None = None, language: str | None = None,
                     params: dict[str, Any] | None = None,
                     fps: float | None = None,
                     features: AudioFeatures | None = None) -> dict[str, Any]:
    """Align ``transcript`` text to the audio in ``samples`` / ``video``.

    Provide audio as ``samples`` (mono float), a ``video`` path, or a
    pre-computed :class:`AudioFeatures`.  The ``features`` route exists so tests
    and evidence can feed a controlled envelope (e.g. a shuffled negative
    control) without changing the aligner; when it is used, the feature's own
    parameters win.  Returns a ``semotion.alignment/1`` document; nothing is
    written to disk here.
    """
    if features is not None:
        if params:
            p = {**features.params, **params}
        else:
            p = dict(features.params)
        samples = features.samples
    else:
        p = {**DEFAULT_PARAMS, **(params or {})}
        if samples is None and video is None:
            raise media.MediaError("E_ALIGN_FAILED", "need audio: pass samples= or video=")
        if samples is None:
            samples, _ = load_audio(video)
        features = audio_features(samples, params=p)
    hop_s = features.hop_s
    duration = features.duration_s
    lang = str(language or transcript.get("language") or "zh")
    cjk_mode = lang.lower().startswith(("zh", "ja", "ko"))

    segments_in = _segment_list(transcript)
    if not segments_in:
        raise media.MediaError("E_ALIGN_FAILED", "transcript has no non-empty segments")

    # ---------------------------------------------------------------- tokens
    tokens: list[Token] = []
    seg_docs: list[dict[str, Any]] = []
    warnings: list[str] = []
    for seg in segments_in:
        # ASR word char offsets in the transcript are relative to the *stripped*
        # segment text, so tokenise the same string to keep group ids aligned.
        seg_text = str(seg.get("text") or "").strip()
        seg_tokens = tokenize_text(seg_text, lang)
        seg_id = str(seg.get("id"))
        for tok in seg_tokens:
            tok.segment_id = seg_id
            tok.index = len(tokens)
            tokens.append(tok)
        n_syll = sum(t.n_syll for t in seg_tokens)
        if not seg_tokens:
            warnings.append(f"segment {seg_id} has no syllable-bearing tokens")
        seg_docs.append({
            "id": seg_id,
            "start_s": round(float(seg["start_s"]), 6),
            "end_s": round(float(seg["end_s"]), 6),
            "text": seg_text,
            "n_tokens": len(seg_tokens),
            "n_syllables": n_syll,
            "speech_start_s": None,
            "speech_end_s": None,
        })
    if not tokens:
        raise media.MediaError("E_ALIGN_FAILED", "transcript text has no alignable tokens")

    # Attach the ASR word grouping to each token.  The caption builder uses it to
    # avoid splitting a cue inside a Whisper word; token *times* are still ours.
    words_by_segment: dict[str, list[dict[str, Any]]] = {}
    for word in transcript.get("words") or []:
        sid = word.get("segment_id")
        if sid is None or word.get("char_start") is None or word.get("char_end") is None:
            continue
        words_by_segment.setdefault(str(sid), []).append(word)
    for seg_words in words_by_segment.values():
        seg_words.sort(key=lambda w: int(w["char_start"]))
    for tok in tokens:
        group = None
        for word in words_by_segment.get(tok.segment_id, ()):
            start, end = int(word["char_start"]), int(word["char_end"])
            if start <= tok.char_start < end:
                if tok.char_end <= end:
                    group = f"{tok.segment_id}:{word.get('id')}"
                break
        tok.group_id = group

    syllables: list[_Syl] = []
    seg_first: dict[str, int] = {}
    seg_last: dict[str, int] = {}
    for tok in tokens:
        seg_first.setdefault(tok.segment_id, tok.index)
        seg_last[tok.segment_id] = tok.index
    for tok in tokens:
        first_of_seg = tok.index == seg_first[tok.segment_id]
        last_of_seg = tok.index == seg_last[tok.segment_id]
        for k in range(tok.n_syll):
            syllables.append(_Syl(
                token_index=tok.index, segment_id=tok.segment_id,
                first_of_token=(k == 0), last_of_token=(k == tok.n_syll - 1),
                first_of_seg=first_of_seg and k == 0,
                last_of_seg=last_of_seg and k == tok.n_syll - 1,
            ))
    n_syll = len(syllables)

    # ------------------------------------------------------------ chunking
    min_pause = float(p["pause_min_s"])
    islands = [(float(s), float(e)) for s, e in features.regions
               if e - s >= float(p["vad_min_speech_s"])]
    # ignore the very end of the file if it is a click / encoder tail
    islands = [(s, min(e, duration)) for s, e in islands if s < duration - 0.02]
    if not islands:
        islands = [(0.0, duration)]
        warnings.append("no speech island detected; aligned uniformly over the file")
    # gaps shorter than pause_min_s are intra-phrase: keep them inside one chunk
    merged_islands: list[tuple[float, float]] = []
    for start, end in islands:
        if merged_islands and start - merged_islands[-1][1] < min_pause:
            merged_islands[-1] = (merged_islands[-1][0], end)
        else:
            merged_islands.append((start, end))
    islands = merged_islands
    # more islands than syllables cannot be allocated one-to-one: merge the
    # closest pairs until every island can hold at least one syllable
    while len(islands) > n_syll and len(islands) > 1:
        gaps = [islands[i + 1][0] - islands[i][1] for i in range(len(islands) - 1)]
        k = int(np.argmin(gaps))
        islands = islands[:k] + [(islands[k][0], islands[k + 1][1])] + islands[k + 2:]
    # punct_zero[i] -> syllable i starts a token whose predecessor ended a phrase
    punct_zero = [False] * n_syll
    prev_token = None
    for i, syl in enumerate(syllables):
        tok = tokens[syl.token_index]
        if syl.first_of_token and prev_token is not None:
            punct_zero[i] = bool(prev_token.trailing_punct.strip())
        if syl.last_of_token:
            prev_token = tok
    # segment-aware allocation: each segment's known syllable count is spread
    # only over the islands that segment actually covers
    seg_ranges: list[tuple[dict[str, Any], int, int]] = []
    cursor = 0
    for seg in seg_docs:
        count = int(seg["n_syllables"])
        if count:
            seg_ranges.append((seg, cursor, cursor + count))
            cursor += count
    if cursor != n_syll:  # pragma: no cover - invariant guard
        raise media.MediaError("E_ALIGN_FAILED", "segment syllable counts do not add up")

    syl_island = [-1] * n_syll
    pseudo_islands: dict[int, tuple[float, float]] = {}
    next_pseudo = len(islands)
    for seg, a, b in seg_ranges:
        alloc = _allocate_segment(float(seg["start_s"]), float(seg["end_s"]),
                                  a, b, islands, punct_zero)
        if alloc is None:
            idx = next_pseudo
            next_pseudo += 1
            pseudo_islands[idx] = (max(0.0, float(seg["start_s"])),
                                   min(duration, float(seg["end_s"])))
            warnings.append(f"segment {seg['id']} lies outside detected speech; "
                            "aligned inside its ASR bounds")
            alloc = [(idx, b - a)]
        pos = a
        for idx, count in alloc:
            for i in range(pos, pos + count):
                syl_island[i] = idx
            pos += count
    if any(idx < 0 for idx in syl_island):  # pragma: no cover - defensive
        for i, idx in enumerate(syl_island):
            if idx < 0:
                syl_island[i] = 0

    def island_bounds(idx: int) -> tuple[float, float]:
        return islands[idx] if idx < len(islands) else pseudo_islands[idx]

    # chunks = maximal runs of syllables sharing one island
    chunks: list[tuple[int, int, int]] = []
    for i, idx in enumerate(syl_island):
        if chunks and chunks[-1][2] == idx and chunks[-1][1] == i:
            chunks[-1] = (chunks[-1][0], i + 1, idx)
        else:
            chunks.append((i, i + 1, idx))

    # --------------------------------------------------------------- DP per chunk
    starts_frame: list[int] = [0] * n_syll
    ends_frame: list[int] = [0] * n_syll
    feasible_flags: list[bool] = []
    for ci, (a, b, island_idx) in enumerate(chunks):
        island_start, island_end = island_bounds(island_idx)
        lo = max(0, int(math.floor((island_start - 3 * hop_s) / hop_s)))
        hi = min(features.n_frames, int(math.ceil((island_end + 3 * hop_s) / hop_s)) + 1)
        n_chunk = b - a
        if hi - lo < 2 or n_chunk <= 0:
            lo, hi = 0, max(2, features.n_frames)
        med = float(np.clip((island_end - island_start) / max(1, n_chunk),
                            float(p["seq_min_syllable_s"]) * 1.1,
                            float(p["seq_max_syllable_s"]) * 0.5))
        # soft blend with the global Mandarin prior (avoid overfitting a short island)
        med = 0.65 * med + 0.35 * float(p["seq_med_syllable_s"])
        anchors: dict[int, list[tuple[int, float, float]]] = {
            0: [(int(round(island_start / hop_s)), 0.060, 0.9)],
            n_chunk: [(max(lo, int(round(island_end / hop_s)) - 1), 0.030, 1.1)],
        }
        seg_by_id = {s["id"]: s for s in seg_docs}
        for j in range(a, b):
            syl = syllables[j]
            seg = seg_by_id[syl.segment_id]
            if syl.first_of_seg:
                anchors.setdefault(j - a, []).append(
                    (int(round(float(seg["start_s"]) / hop_s)), float(p["anchor_tol_s"]), 0.35))
            if syl.last_of_seg and j - a + 1 <= n_chunk:
                anchors.setdefault(j - a + 1, []).append(
                    (int(round(float(seg["end_s"]) / hop_s)), float(p["anchor_tol_s"]), 0.35))
        path, feasible = _dp_chunk(features.onset, features.speech, lo, hi, n_chunk,
                                   anchors, med, p)
        feasible_flags.append(feasible)
        if not feasible:
            warnings.append(f"chunk {ci} ({island_start:.2f}s) had no feasible "
                            f"DP path; used uniform spacing")
        for j in range(n_chunk):
            starts_frame[a + j] = int(path[j])
            ends_frame[a + j] = int(path[j + 1])
        if a < b:
            ends_frame[b - 1] = int(path[n_chunk])

    # ------------------------------------------------------------- refinement
    min_frame_gap = max(2, int(round(float(p["seq_min_syllable_s"]) / hop_s)))
    refine = bool(p.get("refine_onsets", 0))
    peaks_frame = list(starts_frame)  # DP peaks, kept for reporting / ablation
    for i in range(n_syll):
        peak = starts_frame[i]
        lower = max(0, starts_frame[i - 1] + min_frame_gap) if i else 0
        refined = _refine_onset(features, peak, lower, p) if refine else peak
        starts_frame[i] = max(lower, min(refined, peak))
        if i:
            starts_frame[i] = max(starts_frame[i], starts_frame[i - 1] + min_frame_gap)

    # --------------------------------------------------------- syllable records
    p90_env = float(np.percentile(features.onset[features.speech], 90)) \
        if features.speech.any() else 1.0
    syl_docs: list[dict[str, Any]] = []
    for i, syl in enumerate(syllables):
        tok = tokens[syl.token_index]
        s_frame = starts_frame[i]
        if i + 1 < n_syll and syl_island[i + 1] == syl_island[i]:
            e_frame = max(s_frame + min_frame_gap, starts_frame[i + 1])
        else:
            e_frame = max(s_frame + min_frame_gap, ends_frame[i])
        start_s = s_frame * hop_s
        end_s = e_frame * hop_s
        peak_strength = float(features.onset[min(s_frame + 1, features.n_frames - 1)])
        conf_on = float(np.clip(peak_strength / max(p90_env, 1e-6), 0.0, 1.0))
        dur = max(end_s - start_s, 1e-3)
        med = float(p["seq_med_syllable_s"])
        conf_dur = float(math.exp(-0.5 * (math.log(dur / med) / 0.55) ** 2))
        conf = round(float(np.clip(0.65 * conf_on + 0.35 * conf_dur, 0.0, 1.0)), 3)
        syl_docs.append({
            "index": i,
            "token_index": tok.index,
            "segment_id": syl.segment_id,
            "start_s": round(float(start_s), 6),
            "end_s": round(float(end_s), 6),
            "conf": conf,
            "island": int(syl_island[i]),
            "peak_s": round(float(peaks_frame[i]) * hop_s, 6),
        })

    # ----------------------------------------------------------- words & chars
    words: list[dict[str, Any]] = []
    chars: list[dict[str, Any]] = []
    for tok in tokens:
        idxs = [d["index"] for d in syl_docs if d["token_index"] == tok.index]
        if not idxs:
            continue
        first, last = syl_docs[idxs[0]], syl_docs[idxs[-1]]
        start, end = float(first["start_s"]), float(last["end_s"])
        conf = float(np.mean([syl_docs[i]["conf"] for i in idxs]))
        seg_text = next(s["text"] for s in seg_docs if s["id"] == tok.segment_id)
        word = {
            "id": f"w{len(words)}",
            "segment_id": tok.segment_id,
            "text": tok.text,
            "kind": tok.kind,
            "n_syllables": tok.n_syll,
            "start_s": round(start, 6),
            "end_s": round(end, 6),
            "conf": round(conf, 3),
            "char_start": tok.char_start,
            "char_end": tok.char_end,
            "trailing_punct": tok.trailing_punct,
            "group_id": tok.group_id,
            "syllable_ids": idxs,
        }
        words.append(word)
        if tok.kind == "han" and tok.n_syll == 1:
            chars.append({
                "id": f"c{len(chars)}",
                "segment_id": tok.segment_id,
                "token_id": word["id"],
                "text": seg_text[tok.char_start:tok.char_end],
                "kind": "han",
                "start_s": round(start, 6),
                "end_s": round(end, 6),
                "conf": round(conf, 3),
                "char_start": tok.char_start,
                "char_end": tok.char_end,
                "group_id": tok.group_id,
                "approx": False,
            })
            continue
        # multi-syllable token (digits / Latin) or English word: distribute the
        # token span over its printable characters (marked approx -> captions
        # must use the word span, not the letter spans).
        letters = [k for k in range(tok.char_start, tok.char_end)
                   if not _is_punct(seg_text[k])]
        if not letters:
            letters = [tok.char_start]
        n_letter = len(letters)
        for j, ci_ in enumerate(letters):
            cs = start + (end - start) * j / n_letter
            ce = start + (end - start) * (j + 1) / n_letter
            chars.append({
                "id": f"c{len(chars)}",
                "segment_id": tok.segment_id,
                "token_id": word["id"],
                "text": seg_text[ci_],
                "kind": tok.kind,
                "start_s": round(cs, 6),
                "end_s": round(ce, 6),
                "conf": round(conf, 3),
                "char_start": ci_,
                "char_end": ci_ + 1,
                "group_id": tok.group_id,
                "approx": True,
            })

    # ----------------------------------------------------------- segment report
    seg_by_id = {s["id"]: s for s in seg_docs}
    for w in words:
        seg = seg_by_id[w["segment_id"]]
        if seg["speech_start_s"] is None or w["start_s"] < seg["speech_start_s"]:
            seg["speech_start_s"] = w["start_s"]
        if seg["speech_end_s"] is None or w["end_s"] > seg["speech_end_s"]:
            seg["speech_end_s"] = w["end_s"]
    for seg in seg_docs:
        if seg["speech_start_s"] is None:
            seg["speech_start_s"] = seg["start_s"]
            seg["speech_end_s"] = seg["end_s"]

    # next-token punctuation is needed by the caption builder
    quality = _quality(features, syl_docs, chars, seg_docs, p)
    quality["n_words"] = len(words)
    quality["n_chunks"] = len(chunks)
    quality["feasible_chunks"] = int(sum(feasible_flags))
    quality["onset_median_ms"] = None
    if len(chars) > 1:
        onset_med = float(np.median(np.diff([float(c["start_s"]) for c in chars])))
        quality["onset_median_ms"] = round(onset_med * 1e3, 1)

    if quality["monotonicity_violations"]:
        warnings.append(f"{quality['monotonicity_violations']} monotonicity violations")
    if quality["low_confidence_syllables"]:
        warnings.append(f"{quality['low_confidence_syllables']} low-confidence syllables")
    med_ms = quality["syllable_ms"]["median"]
    if cjk_mode and med_ms is not None and not (110.0 <= med_ms <= 340.0):
        warnings.append(f"median syllable {med_ms} ms outside the expected 110-340 ms band")
    if not cjk_mode:
        warnings.append("language is not CJK: syllable counts come from a heuristic")

    source = transcript.get("source") or {}
    doc = {
        "schema": SCHEMA,
        "source": {
            "path": str(Path(video).resolve()) if video is not None else source.get("path"),
            "sha256": source.get("sha256"),
            "duration_s": round(float(duration), 6),
            "fps": float(fps if fps is not None else (source.get("fps") or 25.0)),
            "width": source.get("width"),
            "height": source.get("height"),
            "language": lang,
            "audio_rate": int(features.sample_rate),
        },
        "method": {
            "name": METHOD,
            "truth_class": "MACHINE_WEAK",
            "engine": "numpy (log-mel spectral flux + energy VAD + monotonic DP)",
            "params": {k: p[k] for k in sorted(p)},
            "notes": [
                "Whisper word times are not used; only the transcript text and segment bounds",
                "segment bounds are soft anchors (±120 ms); VAD pauses are hard chunk edges",
                "punctuation carries no duration and never gets a boundary",
                "onset = prominent onset-envelope peak (optional attack refinement, off by default)",
            ],
        },
        "segments": seg_docs,
        "syllables": syl_docs,
        "words": words,
        "chars": chars,
        "quality": quality,
        "warnings": warnings,
    }
    return doc


def align_video(video: str | Path, transcript: dict[str, Any] | str | Path, *,
                language: str | None = None, params: dict[str, Any] | None = None,
                fps: float | None = None) -> dict[str, Any]:
    """Align a transcript document/path against a video (or wav) file."""
    if isinstance(transcript, (str, Path)):
        transcript = json.loads(Path(transcript).read_text(encoding="utf-8"))
    return align_transcript(transcript, video=video, language=language,
                            params=params, fps=fps)


# ---------------------------------------------------------------------- IO helpers


def save(doc: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(doc: dict[str, Any]) -> list[str]:
    """Schema + invariant problems (empty list == valid)."""
    problems: list[str] = []
    if doc.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}")
    src = doc.get("source") or {}
    for key in ("path", "duration_s", "fps", "language"):
        if key not in src:
            problems.append(f"source.{key} missing")
    duration = float(src.get("duration_s") or 0.0)
    method = doc.get("method") or {}
    if method.get("truth_class") != "MACHINE_WEAK":
        problems.append("method.truth_class must be MACHINE_WEAK")
    if not method.get("params"):
        problems.append("method.params missing")
    seg_ids = set()
    for i, seg in enumerate(doc.get("segments") or []):
        sid = seg.get("id")
        if not sid or sid in seg_ids:
            problems.append(f"segments[{i}] id missing/duplicated: {sid!r}")
        seg_ids.add(sid)
        if not (0.0 <= float(seg.get("start_s", -1)) <= float(seg.get("end_s", -1))
                <= duration + 0.5):
            problems.append(f"segments[{i}] bad interval")
    prev_start = -1.0
    word_ids = {w.get("id") for w in doc.get("words") or []}
    for i, c in enumerate(doc.get("chars") or []):
        start, end = float(c.get("start_s", -1)), float(c.get("end_s", -1))
        if not (0.0 <= start <= end <= duration + 0.5):
            problems.append(f"chars[{i}] bad interval {start}-{end}")
        if start + 1e-9 < prev_start:
            problems.append(f"chars[{i}] not monotonic ({start:.3f} < {prev_start:.3f})")
        prev_start = max(prev_start, start)
        if c.get("token_id") not in word_ids:
            problems.append(f"chars[{i}] unknown token_id {c.get('token_id')!r}")
        if not (0.0 <= float(c.get("conf", -1)) <= 1.0):
            problems.append(f"chars[{i}] conf outside [0,1]")
    prev_word = -1.0
    for i, w in enumerate(doc.get("words") or []):
        start, end = float(w.get("start_s", -1)), float(w.get("end_s", -1))
        if not (0.0 <= start < end <= duration + 0.5):
            problems.append(f"words[{i}] bad interval {start}-{end}")
        if start + 1e-9 < prev_word:
            problems.append(f"words[{i}] not monotonic")
        prev_word = max(prev_word, start)
        if int(w.get("n_syllables", 0)) < 1:
            problems.append(f"words[{i}] n_syllables < 1")
        if w.get("segment_id") is not None and w.get("segment_id") not in seg_ids:
            problems.append(f"words[{i}] unknown segment_id")
    q = doc.get("quality") or {}
    if not q:
        problems.append("quality missing")
    elif int(q.get("monotonicity_violations", -1)) != 0:
        problems.append("quality.monotonicity_violations must be 0")
    return problems


# --------------------------------------------------------------- lookup helpers


def char_at_time(doc: dict[str, Any], t: float) -> dict[str, Any] | None:
    """Character covering ``t`` (nearest when ``t`` is in a gap)."""
    chars = doc.get("chars") or []
    if not chars:
        return None
    for c in chars:
        if float(c["start_s"]) <= t < float(c["end_s"]):
            return c
    return min(chars, key=lambda c: min(abs(float(c["start_s"]) - t),
                                        abs(float(c["end_s"]) - t)))


def text_at_time(doc: dict[str, Any], t: float, *, max_chars: int = 12) -> str:
    """Approximate transcript text around ``t`` (for inspection/evidence)."""
    chars = doc.get("chars") or []
    if not chars:
        return ""
    best = min(range(len(chars)), key=lambda i: abs(float(chars[i]["start_s"]) - t))
    lo = max(0, best - max_chars // 2)
    return "".join(str(c["text"]) for c in chars[lo:best + max_chars // 2])


def syllable_duration_stats(doc: dict[str, Any]) -> dict[str, Any]:
    return dict((doc.get("quality") or {}).get("syllable_ms") or {})


def onset_peak_report(doc: dict[str, Any]) -> dict[str, Any]:
    return dict((doc.get("quality") or {}).get("onset_envelope") or {})
