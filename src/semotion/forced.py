"""Semantic Motion — acoustic CTC forced alignment (real aligner).

What this module is
-------------------
:mod:`semotion.align` places character onsets on a spectral-flux envelope with a
duration-constrained DP.  It is deterministic and needs no model, but the times
it produces are *evidence-driven guesses*: on real speech there is no truth to
check them against, only "is the onset on an envelope peak, monotonic, and in a
plausible duration band".  This module adds the missing acoustic model:

1. decode 16 kHz mono PCM (through :mod:`semotion.align` / :mod:`semotion.media`),
2. run a **CTC acoustic model** over the audio -> frame log-posteriors,
3. force the *known transcript* through the posteriors with a monotonic CTC
   Viterbi (:func:`torchaudio.functional.forced_align` when torchaudio is
   installed, an independent NumPy implementation otherwise),
4. map the resulting CTC token spans back onto the :mod:`semotion.align`
   tokenizer's characters / words / segments,
5. emit the same ``semotion.alignment/1`` document shape, with a per-token
   posterior confidence and a method block that says exactly which model ran.

Model choice (and why it is not simply MMS_FA)
----------------------------------------------
* **English**: ``MMS_FA`` (``torchaudio.pipelines.MMS_FA``) is a phone-level
  model.  Its label set is 29 English/IPA phonemes, so it can only align
  *English phoneme* transcripts and needs a grapheme-to-phoneme step that this
  project does not install.  It was therefore evaluated and is **not shipped as
  an alignment path**: :func:`mms_fa_probe` reports what it is, and selecting it
  raises a clear error instead of guessing.  The default English engine is
  ``facebook/wav2vec2-base-960h``, a *letter* CTC model: no G2P, no fabrication,
  and the letter span is the word span.
* **Chinese**: ``MMS_FA`` **cannot** align Chinese — it has no Mandarin phones.
  The default zh engine is
  ``jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`` (XLSR-53, CTC over a
  Chinese **character** vocabulary), with an OpenCC Traditional→Simplified
  lookup so Traditional transcripts still hit the Simplified vocabulary.  That
  is a syllable-resolution acoustic aligner: one Han character = one syllable,
  which is exactly the unit the rest of this project anchors overlays to.  It is a *character* CTC model,
  not a phoneme model, and the docs say so.

Both run offline once the weights are cached (``HF_HOME`` defaults to
``<repo>/.cache/huggingface``); :func:`model_info` reports the cache path, size
and device actually used.

Output contract
---------------
``semotion.alignment/1`` — same keys as :mod:`semotion.align`
(``chars[] words[] segments[] syllables[] quality{} method{}``) plus CTC-specific
quality fields.  ``method.name`` is ``"ctc-forced"``.  The truth class stays
``MACHINE_WEAK``: a CTC posterior is audio evidence, not a human label.

Honesty
-------
* Synthetic fixtures (``ground_truth.json``, kind ``SYNTH_KNOWN``) measure unit
  *start* placement only, and the truth comes from the TTS assembly timeline.
* The real clip has no human labels.  What can be measured there is posterior
  confidence, monotonicity, and agreement with the envelope aligner — never
  "accuracy".
* The negative control (shuffled target sequence) is the proof that the
  posterior score is sensitive to the transcript: a model that ignored the
  targets would score the same.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import align as energy
from . import media

__all__ = [
    "SCHEMA",
    "METHOD",
    "DEFAULT_MODELS",
    "ENGINE_MMS_FA",
    "ENGINE_HF",
    "DEFAULT_PARAMS",
    "ModelHandle",
    "available",
    "model_info",
    "load_model",
    "mms_fa_probe",
    "build_targets",
    "ctc_viterbi",
    "align_transcript",
    "align_video",
    "align_posteriors",
    "save",
    "load",
    "validate",
]

SCHEMA = "semotion.alignment/1"  # same document family as align.py
METHOD = "ctc-forced"            # required by the adapter contract
METHOD_REV = "ctc-forced/1"

#: Model ids.  Both are CTC models that output one symbol per ~20 ms frame.
DEFAULT_MODELS: dict[str, str] = {
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "en": "facebook/wav2vec2-base-960h",
}

ENGINE_HF = "hf-wav2vec2-ctc"
ENGINE_MMS_FA = "torchaudio-mms-fa"

DEFAULT_PARAMS: dict[str, Any] = {
    # audio / VAD (same front end as align.py, so the two aligners see the same
    # speech islands and the benchmark compares like with like)
    "audio_rate": 16000,
    "vad_min_speech_s": 0.060,
    "vad_fill_gap_s": 0.090,
    "vad_floor_db": 4.0,
    "vad_high_frac": 0.25,
    # chunking: wav2vec2 attention is quadratic, so long clips are cut at
    # pauses.  20 s keeps a 16 GB M4 CPU comfortable; 0 disables chunking.
    "max_chunk_s": 20.0,
    "chunk_merge_gap_s": 0.400,
    "chunk_pad_s": 0.060,       # context around a chunk (both sides)
    "chunk_min_targets": 1,
    "min_chunk_s": 0.180,
    # CTC
    "min_token_frames": 1,      # a character may not be shorter than this
    "low_conf": 0.35,
    # reporting
    "confidence_floor_db": 0.0,  # unused, kept for forward-compatible params
}


# --------------------------------------------------------------------- utilities


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def hf_home() -> Path:
    """Writable cache root this module downloads into.

    ``SEMOTION_HF_HOME`` wins; otherwise ``<repo>/.cache/huggingface``, so a run
    never writes to the user's home directory (the agent sandbox forbids it).
    Existing weights are looked up in ``SEMOTION_HF_HOME`` and ``HF_HOME`` too
    (see :func:`_locate_model`), so a shared cache can be reused read-only.
    """
    override = os.environ.get("SEMOTION_HF_HOME")
    if override:
        return Path(override)
    return _repo_root() / ".cache" / "huggingface"


def _cache_candidates() -> list[Path]:
    """Cache roots searched for an already-downloaded model, in priority order."""
    out: list[Path] = []
    for key in ("SEMOTION_HF_HOME", "HF_HOME"):
        value = os.environ.get(key)
        if value:
            out.append(Path(value))
    out.append(_repo_root() / ".cache" / "huggingface")
    seen: set[str] = set()
    uniq: list[Path] = []
    for path in out:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            uniq.append(path)
    return uniq


def _dir_size(path: Path) -> int:
    """Bytes under ``path``, counting each regular file once (hub blobs are
    hard-linked/symlinked into snapshots; following the links would double it)."""
    total = 0
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                st = fp.lstat()
            except OSError:  # pragma: no cover - transient
                continue
            if not fp.is_symlink():
                total += st.st_size
    return total


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=-1, keepdims=True)
    z = x - m
    return z - np.log(np.sum(np.exp(z), axis=-1, keepdims=True))


# --------------------------------------------------------------- CTC alignment


def ctc_viterbi(log_probs: np.ndarray, targets: Sequence[int], *,
                blank: int = 0, min_frames: int = 1) -> dict[str, Any]:
    """Best monotonic CTC path for ``targets`` through ``log_probs`` (NumPy).

    ``log_probs`` is ``(T, V)`` (already log-softmaxed).  Standard CTC state
    graph: ``2N+1`` states, blanks on even indices, targets on odd ones, with the
    repeat-skip transition only where two neighbouring targets differ.

    Returns the frame span of every target, the per-target mean log-posterior of
    the chosen state, and the path log-likelihood.  ``feasible`` is ``False``
    when ``T < 2N-1`` or no path exists; callers must then merge chunks rather
    than pretend.

    This mirrors ``torchaudio.functional.forced_align`` (same DP, same answer);
    it exists so the module and its tests work without torch, and so repeated
    characters keep distinct spans (``merge_tokens`` collapses repeated labels,
    which would merge 谢谢 into one character).
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    if lp.ndim != 2:
        raise ValueError(f"log_probs must be 2-D (T, V), got {lp.shape}")
    t_total, vocab = lp.shape
    tgt = [int(t) for t in targets]
    n = len(tgt)
    if n == 0:
        return {"feasible": False, "reason": "no targets", "n_targets": 0,
                "starts": [], "ends": [], "logp": [], "path_score": float("nan")}
    if not (0 <= blank < vocab):
        raise ValueError(f"blank id {blank} outside vocab [0,{vocab})")
    if any(t < 0 or t >= vocab for t in tgt):
        raise ValueError("target id outside vocab")
    if t_total < 2 * n - 1:
        return {"feasible": False, "reason": "audio too short for target count",
                "n_targets": n, "starts": [], "ends": [], "logp": [],
                "path_score": float("nan")}

    n_states = 2 * n + 1
    neg_inf = -1e30
    # emission[t, s] = log p(label(s) | t)
    labels = np.full(n_states, blank, dtype=np.int64)
    labels[1::2] = np.asarray(tgt, dtype=np.int64)
    emit = lp[:, labels]  # (T, S)

    # predecessor mask: state s can follow s (stay), s-1, and s-2 when s is a
    # target and the two neighbouring targets differ (blank between repeats).
    allow2 = np.zeros(n_states, dtype=bool)
    for i in range(1, n):
        allow2[2 * i + 1] = tgt[i] != tgt[i - 1]

    back = np.zeros((t_total, n_states), dtype=np.int8)
    alpha = np.full(n_states, neg_inf, dtype=np.float64)
    alpha[0] = emit[0, 0]
    if n_states > 1:
        alpha[1] = emit[0, 1]
    for t in range(1, t_total):
        prev = alpha
        stay = prev
        step1 = np.full(n_states, neg_inf, dtype=np.float64)
        step1[1:] = prev[:-1]
        step2 = np.full(n_states, neg_inf, dtype=np.float64)
        if allow2.any():
            step2[allow2] = prev[:-2][allow2[2:]]
        stack = np.stack([stay, step1, step2], axis=0)  # (3, S)
        best = np.argmax(stack, axis=0).astype(np.int8)
        alpha = emit[t] + stack[best, np.arange(n_states)]
        back[t] = best

    end_state = int(np.argmax([alpha[n_states - 1], alpha[n_states - 2]]))
    end_state = n_states - 1 if end_state == 0 else n_states - 2
    if not np.isfinite(alpha[end_state]) or alpha[end_state] <= neg_inf / 2:
        return {"feasible": False, "reason": "no finite path", "n_targets": n,
                "starts": [], "ends": [], "logp": [], "path_score": float("nan")}

    path = np.empty(t_total, dtype=np.int64)
    s = end_state
    for t in range(t_total - 1, -1, -1):
        path[t] = s
        if t:
            s = s - int(back[t, s])
    starts = np.full(n, -1, dtype=np.int64)
    ends = np.full(n, -1, dtype=np.int64)
    for t, state in enumerate(path):
        if state % 2 == 1:
            i = state // 2
            if starts[i] < 0:
                starts[i] = t
            ends[i] = t + 1
    # a target may be assigned zero frames when the path skips it through the
    # blank skip transition; give it a minimal span so output stays usable and
    # flag it.
    degenerate = int(np.sum(starts < 0))
    for i in range(n):
        if starts[i] < 0:
            anchor = ends[i - 1] if i and ends[i - 1] >= 0 else 0
            starts[i] = anchor
            ends[i] = anchor + 1
    ends = np.maximum(ends, starts + int(max(1, min_frames)))
    ends = np.minimum(ends, t_total)
    starts = np.minimum(starts, t_total - 1)

    logp = np.empty(n, dtype=np.float64)
    for i in range(n):
        span = emit[starts[i]:max(ends[i], starts[i] + 1), 2 * i + 1]
        logp[i] = float(np.mean(span)) if span.size else neg_inf

    return {
        "feasible": True,
        "n_targets": n,
        "starts": starts.tolist(),
        "ends": ends.tolist(),
        "logp": logp.tolist(),
        "path_score": float(alpha[end_state]),
        "n_frames": int(t_total),
        "degenerate_targets": degenerate,
    }


def _torchaudio_viterbi(log_probs: np.ndarray, targets: Sequence[int], *,
                        blank: int = 0) -> dict[str, Any] | None:
    """``torchaudio.functional.forced_align`` wrapper (returns ``None`` on failure).

    Handles both torchaudio APIs seen in the wild:

    * >= 2.8 — ``forced_align`` returns per-frame alignments ``(B, T)`` and
      per-frame scores ``(B, T)``; ``merge_tokens(alignments[0], scores[0])``
      gives the spans (blank-separated runs), so repeated characters stay two
      spans and ``TokenSpan.score`` is the span's mean log-posterior.
    * older — ``alignments`` is ``(B, N)`` of per-target start frames and
      ``scores`` is ``(B,)``; spans are derived from consecutive starts.
    """
    try:  # pragma: no cover - exercised only when torchaudio is installed
        import torch
        import torchaudio.functional as taf
    except Exception:
        return None
    try:  # pragma: no cover - heavy dependency path
        lp = torch.as_tensor(np.asarray(log_probs, dtype=np.float32))[None]  # (1,T,V)
        tgt = torch.as_tensor(np.asarray(targets, dtype=np.int64))[None]
        alignments, scores = taf.forced_align(lp, tgt, blank=blank)
        n = len(targets)
        if scores.dim() == 2:  # per-frame API (torchaudio >= 2.8)
            spans = taf.merge_tokens(alignments[0], scores[0], blank=blank)
            if len(spans) != n:
                return None
            return {
                "feasible": True,
                "n_targets": n,
                "starts": [int(s.start) for s in spans],
                "ends": [int(s.end) for s in spans],
                "logp": [float(s.score) for s in spans],
                "path_score": float(scores[0].sum()),
                "n_frames": int(lp.shape[1]),
                "degenerate_targets": 0,
                "n_merged_tokens": len(spans),
            }
        # older API: per-target start frames
        starts = [int(a) for a in alignments[0].tolist()]
        ends = [starts[i + 1] if i + 1 < n else int(lp.shape[1]) for i in range(n)]
        logp = []
        for i in range(n):
            start = starts[i]
            stop = max(ends[i], start + 1)
            seg = lp[0, start:stop, int(targets[i])].numpy()
            logp.append(float(np.mean(seg)) if seg.size else float("-inf"))
        return {
            "feasible": True,
            "n_targets": n,
            "starts": starts,
            "ends": ends,
            "logp": logp,
            "path_score": float(scores.sum()),
            "n_frames": int(lp.shape[1]),
            "degenerate_targets": 0,
            "n_merged_tokens": len(taf.merge_tokens(
                torch.as_tensor(alignments[0]), torch.zeros(len(alignments[0])),
                blank=blank)),
        }
    except Exception:  # pragma: no cover - fall back to NumPy
        return None


_engine_cache: dict[str, bool] = {}


def _use_torchaudio() -> bool:
    if "ok" not in _engine_cache:
        try:
            import torchaudio.functional as taf  # noqa: F401
            _engine_cache["ok"] = True
        except Exception:
            _engine_cache["ok"] = False
    return _engine_cache["ok"]


def forced_align_spans(log_probs: np.ndarray, targets: Sequence[int], *,
                       blank: int = 0, min_frames: int = 1,
                       engine: str = "auto") -> dict[str, Any]:
    """CTC forced alignment with the best available engine.

    ``engine="auto"`` prefers ``torchaudio.functional.forced_align`` and falls
    back to :func:`ctc_viterbi`.  ``engine="numpy"`` is deterministic and needs
    no torch (used by tests and by callers that bring their own posteriors).
    """
    if engine in ("auto", "torchaudio") and _use_torchaudio():
        out = _torchaudio_viterbi(log_probs, targets, blank=blank)
        if out is not None and out.get("feasible"):
            return out
        if engine == "torchaudio":
            raise RuntimeError("torchaudio forced_align failed")
    return ctc_viterbi(log_probs, targets, blank=blank, min_frames=min_frames)


# --------------------------------------------------------------- tokenisation


_DIGITS = "零一二三四五六七八九"
_BIG = ["", "万", "亿", "万亿"]


def _mandarin_int(n: int) -> str:
    """Cardinal Mandarin reading; mirrors align.py's private helper (tested for
    parity in tests/test_forced.py)."""
    if n == 0:
        return "零"
    groups: list[int] = []
    while n > 0:
        groups.append(n % 10000)
        n //= 10000
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
        out += four + _BIG[i]
    out = out.rstrip("零") or "零"
    if out.startswith("一十"):
        out = out[1:]
    return out


def mandarin_reading(text: str, *, year_hint: bool = False) -> str | None:
    """The Mandarin character sequence an Arabic numeral string is spoken as.

    Returns ``None`` when ``text`` is not a numeral.  Rules match
    :func:`semotion.align.number_syllables`: percent -> 百分之…, a leading minus
    -> 负, decimals -> 点 + digits, 4-digit years (or digits before 年/月/日/号)
    digit-by-digit, everything else cardinal.  ``len(reading)`` is asserted equal
    to ``align.number_syllables`` for a wide table in the tests.
    """
    raw = unicodedata.normalize("NFKC", str(text)).strip()
    raw = raw.replace(",", "").replace(" ", "").replace("，", "")
    if not raw:
        return None
    pct = raw.endswith("%")
    if pct:
        raw = raw[:-1]
    neg = raw.startswith("-") or raw.startswith("−")
    if neg:
        raw = raw[1:]
    if not re.fullmatch(r"\d+(\.\d+)?", raw or ""):
        return None
    if "." in raw:
        int_part, frac = raw.split(".", 1)
        body = _mandarin_int(int(int_part)) + "点" + "".join(_DIGITS[int(d)] for d in frac)
    elif year_hint or (len(raw) == 4 and 1900 <= int(raw) <= 2100):
        body = "".join(_DIGITS[int(d)] for d in raw)
    else:
        body = _mandarin_int(int(raw))
    if neg:
        body = "负" + body
    if pct:
        body = "百分之" + body
    return body


@dataclass
class _Target:
    """One CTC target symbol (a phone, a letter or a Chinese character)."""

    text: str
    vocab_id: int
    token_index: int          # index into align.tokenize_text()
    segment_id: str
    char_start: int           # source character span it came from
    char_end: int
    synthetic: bool = False   # a character synthesised from a numeral


@dataclass
class _TokenTargets:
    token: energy.Token
    target_ids: list[int] = field(default_factory=list)
    oov: list[str] = field(default_factory=list)
    approx: bool = False      # no acoustic evidence -> interpolated span


def build_targets(tokens: Sequence[energy.Token], vocab: dict[str, int], *,
                  language: str = "zh") -> tuple[list[_Target], list[_TokenTargets], dict[str, Any]]:
    """Map align.py tokens onto the CTC vocabulary.

    Han characters and English letters map 1:1.  A numeral is expanded to its
    spoken Mandarin characters (each one CTC target).  A Latin word inside
    Chinese text has no honest character-level target for a Chinese model, so it
    gets no acoustic targets and is flagged ``approx`` (its span is interpolated
    between neighbours, exactly like align.py distributes digits and Latin words
    over their letters).

    Out-of-vocabulary symbols are dropped, reported, and the affected token is
    marked ``approx``; the caller fails the document when too much of the
    transcript would be faked.
    """
    cjk = str(language).lower().startswith(("zh", "ja", "ko"))
    targets: list[_Target] = []
    groups: list[_TokenTargets] = []
    oov: list[str] = []
    converted: list[str] = []
    n_source = 0

    for tok in tokens:
        group = _TokenTargets(token=tok)
        ids: list[int] = []
        if tok.kind == "han":
            n_source += 1
            ch = tok.text
            key = _vocab_key(ch, vocab)
            if key != ch:
                converted.append(f"{ch}->{key}")
            if key in vocab:
                ids.append(len(targets))
                targets.append(_Target(ch, vocab[key], tok.index, tok.segment_id,
                                       tok.char_start, tok.char_end))
            else:
                group.oov.append(ch)
                oov.append(ch)
        elif tok.kind == "number":
            reading = mandarin_reading(tok.text)
            if reading is None:
                group.oov.append(tok.text)
                oov.append(tok.text)
            else:
                n_source += len(reading)
                for ch in reading:
                    if ch in vocab:
                        ids.append(len(targets))
                        targets.append(_Target(ch, vocab[ch], tok.index, tok.segment_id,
                                               tok.char_start, tok.char_end, synthetic=True))
                    else:
                        group.oov.append(ch)
                        oov.append(ch)
        elif tok.kind == "latin" and not cjk:
            letters = [c for c in tok.text if c.isalpha()]
            n_source += len(letters)
            for k, ch in enumerate(letters):
                key = ch.upper() if ch.upper() in vocab else ch
                if key in vocab:
                    ids.append(len(targets))
                    targets.append(_Target(key, vocab[key], tok.index, tok.segment_id,
                                           tok.char_start + k, tok.char_start + k + 1))
                else:
                    group.oov.append(ch)
                    oov.append(ch)
        else:
            # Latin word inside Chinese text: the zh CTC model has no honest
            # target for it -> interpolate (never invent acoustics).
            group.approx = True
        group.target_ids = ids
        if ids:
            group.approx = False
        groups.append(group)

    info = {
        "n_source_symbols": n_source,
        "n_targets": len(targets),
        "n_oov": len(oov),
        "oov_chars": sorted(set(oov))[:40],
        "n_script_converted": len(converted),
        "script_converted": converted[:40],
        "coverage": round((n_source - len(oov)) / n_source, 4) if n_source else 0.0,
        "n_approx_tokens": sum(1 for g in groups if g.approx and not g.target_ids),
    }
    return targets, groups, info


# ------------------------------------------------------------------ CTC models


@dataclass
class ModelHandle:
    """A loaded CTC model plus everything the aligner needs to feed it."""

    model_id: str
    language: str
    engine: str
    vocab: dict[str, int]
    blank: int
    frame_stride_s: float
    device: str
    n_params: int
    cache_dir: str
    model: Any = None
    processor: Any = None
    g2p: Callable[[str], list[str]] | None = None
    load_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "engine": self.engine,
            "language": self.language,
            "device": self.device,
            "n_params": self.n_params,
            "frame_stride_s": round(self.frame_stride_s, 6),
            "vocab_size": len(self.vocab),
            "cache_dir": self.cache_dir,
            "load_s": round(self.load_s, 3),
        }

    # ---------------------------------------------------------- inference
    def log_probs(self, samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
        """Frame log-posteriors ``(T, V)`` for raw mono float audio."""
        if self.engine == ENGINE_MMS_FA:  # pragma: no cover - opt-in path
            return self._log_probs_mms(samples, sample_rate)
        return self._log_probs_hf(samples, sample_rate)

    def _log_probs_hf(self, samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
        import torch  # noqa: F401  (import cost only on the heavy path)
        proc = self.processor
        x = np.asarray(samples, dtype=np.float32)
        inputs = proc(x, sampling_rate=sample_rate, return_tensors="pt", padding=False)
        with torch.inference_mode():
            values = inputs.input_values.to(self.device)
            logits = self.model(values).logits.float()
        lp = torch.log_softmax(logits, dim=-1)[0].cpu().numpy()
        return lp, self.frame_stride_s

    def _log_probs_mms(self, samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
        return self.model(samples, sample_rate)  # pragma: no cover


def _hf_vocab(processor: Any, model: Any) -> tuple[dict[str, int], int, float]:
    tokenizer = getattr(processor, "tokenizer", None)
    vocab: dict[str, int] = {}
    if tokenizer is not None and hasattr(tokenizer, "get_vocab"):
        vocab = {str(k): int(v) for k, v in tokenizer.get_vocab().items()}
    if not vocab:  # pragma: no cover - defensive
        cfg_vocab = getattr(getattr(model, "config", None), "vocab_size", None)
        raise media.MediaError("E_ALIGN_MODEL_UNSUPPORTED",
                               f"model has no char vocabulary (vocab_size={cfg_vocab})")
    blank = getattr(getattr(model, "config", None), "pad_token_id", None)
    if blank is None or int(blank) not in vocab.values():
        blank = int(vocab.get("<pad>", 0))
    stride = getattr(getattr(model, "config", None), "inputs_to_logits_ratio", None)
    stride_s = float(stride) / 16000.0 if stride else 0.02
    return vocab, int(blank), stride_s  # type: ignore[return-value]


@lru_cache(maxsize=4)
def load_model(model_id: str | None = None, *, language: str = "zh",
               device: str = "cpu", allow_download: bool = True,
               engine: str = ENGINE_HF) -> ModelHandle:
    """Load (and cache) the CTC model for ``language``.

    ``allow_download=False`` (or ``SEMOTION_FORCED_OFFLINE=1``) forces a
    ``local_files_only`` load; a missing cache raises ``E_ALIGN_MODEL_MISSING``
    with the exact cache path, never a silent network call.
    """
    t0 = time.time()
    lang = str(language or "zh").lower()
    model_id = model_id or DEFAULT_MODELS.get("zh" if lang.startswith("zh") else "en")
    if not model_id:
        raise media.MediaError("E_ALIGN_MODEL_UNSUPPORTED",
                               f"no default CTC model for language {language!r}")
    if os.environ.get("SEMOTION_FORCED_OFFLINE") == "1":
        allow_download = False
    offline = not allow_download

    if engine == ENGINE_MMS_FA:
        raise media.MediaError(
            "E_ALIGN_MODEL_UNSUPPORTED",
            "MMS_FA is an English-phoneme CTC model: it cannot align Mandarin and "
            "needs a grapheme-to-phoneme step for English. Use the default HF "
            "engines (zh character CTC / en letter CTC); see mms_fa_probe() and "
            "docs/FORCED-ALIGNMENT.md")

    # Import lazily: torch/transformers are optional heavy dependencies.
    try:
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC
    except Exception as exc:  # pragma: no cover - environment dependent
        raise media.MediaError(
            "E_ALIGN_MODEL_MISSING",
            "torch/transformers are not installed in this venv",
            stderr_tail=str(exc)) from exc

    snapshot = _snapshot_dir(model_id)
    cache = hf_home()
    cache.mkdir(parents=True, exist_ok=True)
    if snapshot is not None:
        # load straight from the local snapshot: this is the offline path and it
        # never depends on HF_HOME/family defaults
        target, local_only = str(snapshot), True
        extra: dict[str, Any] = {}
    elif offline:
        raise media.MediaError(
            "E_ALIGN_MODEL_MISSING",
            f"CTC model {model_id!r} is not cached (looked in "
            f"{[str(c) for c in _cache_candidates()]}) and downloads are disabled")
    else:
        target, local_only = model_id, False
        extra = {"cache_dir": str(cache / "hub")}
    try:
        processor = AutoProcessor.from_pretrained(
            target, local_files_only=local_only, **extra)
        model = Wav2Vec2ForCTC.from_pretrained(
            target, local_files_only=local_only, **extra)
    except Exception as exc:
        raise media.MediaError(
            "E_ALIGN_MODEL_MISSING",
            f"cannot load CTC model {model_id!r} (cache {cache}); "
            "run once with network access or set HF_HOME",
            stderr_tail=f"{type(exc).__name__}: {str(exc)[:800]}") from exc

    model.eval()
    model.to(device)
    vocab, blank, stride_s = _hf_vocab(processor, model)
    n_params = sum(p.numel() for p in model.parameters())
    handle = ModelHandle(
        model_id=model_id, language=lang, engine=ENGINE_HF, vocab=vocab,
        blank=blank, frame_stride_s=stride_s, device=device, n_params=int(n_params),
        cache_dir=str(cache), model=model, processor=processor,
        load_s=time.time() - t0,
    )
    return handle


def mms_fa_probe() -> dict[str, Any]:
    """Report the torchaudio ``MMS_FA`` bundle without using it for alignment.

    MMS_FA is the model this task asked for first, so it is evaluated here and
    the result is recorded rather than hidden: it loads and runs (315 M params,
    cached under ``TORCH_HOME``), but its label set is 29 *English/IPA phone*
    symbols.  It therefore cannot align Mandarin at all, and aligning English
    text with it needs a grapheme-to-phoneme step (espeak-style IPA), which this
    project does not install.  The shipped engines are the HuggingFace CTC
    models (Chinese character CTC / English letter CTC), which need no G2P.
    """
    info: dict[str, Any] = {
        "engine": ENGINE_MMS_FA,
        "usable_for_alignment": False,
        "reason": "English-phoneme label set: no Mandarin support, and text->IPA "
                  "needs a G2P dependency that is not installed",
    }
    try:
        from torchaudio.pipelines import MMS_FA
        info["labels"] = list(MMS_FA.get_labels())
        info["n_labels"] = len(info["labels"])
        info["sample_rate"] = int(MMS_FA.sample_rate)
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info
    try:
        import torch
        model = MMS_FA.get_model()
        info["n_params"] = int(sum(p.numel() for p in model.parameters()))
        info["torch_home"] = os.environ.get("TORCH_HOME")
    except Exception as exc:  # pragma: no cover - weight download
        info["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return info


def _snapshot_bytes(snap: Path) -> int:
    """Real bytes of a snapshot's files (hub blobs are symlinked; count each once)."""
    seen: set[str] = set()
    total = 0
    for fp in snap.rglob("*"):
        if not (fp.is_file() or fp.is_symlink()):
            continue
        try:
            real = fp.resolve()
        except OSError:  # pragma: no cover - broken link
            continue
        if not real.is_file() or str(real) in seen:
            continue
        seen.add(str(real))
        try:
            total += real.stat().st_size
        except OSError:  # pragma: no cover
            pass
    return total


def _snapshot_dir(mid: str) -> Path | None:
    """Local snapshot of ``mid`` with weights present (searches every cache root)."""
    for base in _cache_candidates():
        root = base / "hub" / ("models--" + mid.replace("/", "--"))
        if not root.exists():
            continue
        for snap in sorted((root / "snapshots").glob("*")):
            has_cfg = (snap / "config.json").exists()
            has_w = any((snap / f).exists() for f in ("model.safetensors", "pytorch_model.bin"))
            if has_cfg and has_w:
                return snap
    return None


def available(language: str = "zh", *, model_id: str | None = None) -> bool:
    """True when the model for ``language`` is cached locally (no download, no load)."""
    lang = str(language).lower()
    mid = model_id or DEFAULT_MODELS.get("zh" if lang.startswith("zh") else "en")
    if not mid:
        return False
    return _snapshot_dir(mid) is not None


def model_info(language: str = "zh", *, model_id: str | None = None) -> dict[str, Any]:
    """Report model, cache path/size, device — without downloading anything."""
    lang = str(language).lower()
    mid = model_id or DEFAULT_MODELS.get("zh" if lang.startswith("zh") else "en")
    cache = hf_home()
    snap = _snapshot_dir(mid) if mid else None
    slug_dir = (snap.parent.parent if snap is not None
                else cache / "hub" / ("models--" + mid.replace("/", "--")) if mid else None)
    return {
        "model_id": mid,
        "language": lang,
        "cached": snap is not None,
        "snapshot": str(snap) if snap else None,
        "cache_dir": str(cache),
        "model_cache_dir": str(slug_dir) if slug_dir else None,
        "model_cache_bytes": _snapshot_bytes(snap) if snap is not None else 0,
        "hf_home_bytes": _dir_size(cache),
        "offline_env": os.environ.get("SEMOTION_FORCED_OFFLINE"),
    }


# ------------------------------------------------------------------- chunking


def _speech_chunks(features: energy.AudioFeatures, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Speech islands, merged across short gaps, split when longer than max."""
    hop_s = features.hop_s
    min_chunk = float(params["min_chunk_s"])
    merge_gap = float(params["chunk_merge_gap_s"])
    max_chunk = float(params["max_chunk_s"])
    islands = [(float(s), float(e)) for s, e in features.regions if e - s >= min_chunk]
    merged: list[tuple[float, float]] = []
    for start, end in islands:
        if merged and start - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    if not merged:
        merged = [(0.0, float(features.duration_s))]

    chunks: list[dict[str, Any]] = []
    for start, end in merged:
        duration = end - start
        if max_chunk and duration > max_chunk:
            n_parts = int(math.ceil(duration / max_chunk))
            # cut at local energy minima, never at a hard boundary
            cuts = [start]
            for k in range(1, n_parts):
                target = start + duration * k / n_parts
                half = duration / n_parts * 0.25
                lo = max(start, target - half)
                hi = min(end, target + half)
                f_lo = int(lo / hop_s)
                f_hi = int(hi / hop_s)
                if f_hi - f_lo >= 2:
                    idx = f_lo + int(np.argmin(features.energy_db[f_lo:f_hi]))
                    cuts.append(idx * hop_s)
                else:
                    cuts.append(target)
            cuts.append(end)
            for a, b in zip(cuts[:-1], cuts[1:]):
                chunks.append({"start_s": float(a), "end_s": float(b), "part": True})
        else:
            chunks.append({"start_s": float(start), "end_s": float(end), "part": False})
    chunks.sort(key=lambda c: c["start_s"])
    return chunks


def _frame_chunks(duration_s: float, frame_stride_s: float, max_chunk_s: float,
                  log_probs: np.ndarray, blank: int) -> list[dict[str, Any]]:
    """Chunk a pre-computed posterior matrix (used by align_posteriors)."""
    if not max_chunk_s or duration_s <= max_chunk_s:
        return [{"start_s": 0.0, "end_s": float(duration_s), "part": False}]
    n_parts = int(math.ceil(duration_s / max_chunk_s))
    t_total = log_probs.shape[0]
    cuts = [0]
    for k in range(1, n_parts):
        target = int(round(k * t_total / n_parts))
        half = max(2, int(t_total / n_parts * 0.25))
        lo, hi = max(1, target - half), min(t_total - 1, target + half)
        if hi - lo < 2:
            cuts.append(target)
            continue
        # a silence proxy: the frame with the most blank posterior mass
        cuts.append(lo + int(np.argmax(log_probs[lo:hi, blank])))
    cuts.append(t_total)
    cuts = sorted(set(cuts))
    return [
        {"start_s": a * frame_stride_s, "end_s": b * frame_stride_s, "part": True}
        for a, b in zip(cuts[:-1], cuts[1:])
    ]


def _allocate_tokens(tokens: Sequence[energy.Token],
                     chunks: Sequence[dict[str, Any]],
                     segments: Sequence[dict[str, Any]]) -> list[list[int]]:
    """Assign token indices to chunks, preserving order and per-segment layout.

    Tokens are allocated per ASR segment (whose text and bounds are trusted)
    proportionally to the segment's overlap with each chunk.  A segment that
    touches no chunk is assigned to the chunk containing its start (or the
    nearest one).  Each chunk keeps its tokens in transcript order.

    The split uses cumulative proportional rounding, so counts are monotone and
    can never exceed the available tokens; a chunk may end up with none (breath,
    music) and is simply skipped by the aligner.
    """
    n_chunks = len(chunks)
    if n_chunks <= 1:
        return [list(range(len(tokens)))]
    seg_bounds = {str(s["id"]): (float(s["start_s"]), float(s["end_s"])) for s in segments}
    by_seg: dict[str, list[int]] = {}
    for i, tok in enumerate(tokens):
        by_seg.setdefault(tok.segment_id, []).append(i)

    per_chunk: list[list[int]] = [[] for _ in range(n_chunks)]
    for seg_id, idxs in by_seg.items():
        start, end = seg_bounds.get(seg_id, (0.0, 0.0))
        overlaps: list[tuple[int, float]] = []
        for ci, ch in enumerate(chunks):
            ov = min(end, float(ch["end_s"])) - max(start, float(ch["start_s"]))
            if ov > 0.0:
                overlaps.append((ci, ov))
        if not overlaps:
            mid = 0.5 * (start + end)
            ci = min(range(n_chunks), key=lambda k: min(
                abs(float(chunks[k]["start_s"]) - mid),
                abs(float(chunks[k]["end_s"]) - mid)))
            per_chunk[ci].extend(idxs)
            continue
        overlaps.sort()
        total = sum(ov for _, ov in overlaps)
        n = len(idxs)
        prev = 0
        cum = 0.0
        for k, (ci, ov) in enumerate(overlaps):
            cum += ov
            cut = n if k == len(overlaps) - 1 else int(round(cum / total * n))
            cut = max(prev, min(cut, n))
            if cut > prev:
                per_chunk[ci].extend(idxs[prev:cut])
            prev = cut

    for chunk_tokens in per_chunk:
        chunk_tokens.sort()
    return per_chunk


# --------------------------------------------------------------- doc assembly


@lru_cache(maxsize=1)
def _t2s_converter():
    """OpenCC Traditional->Simplified converter, or ``None`` when unavailable.

    The zh CTC model is trained on Simplified text; several fixtures are
    Traditional.  Looking a Traditional character up in a Simplified vocabulary
    drops it (and a dropped character distorts its neighbours), so the *lookup
    key* is converted while the emitted character text stays the original.
    """
    try:
        from opencc import OpenCC
        return OpenCC("t2s")
    except Exception:
        return None


def _vocab_key(ch: str, vocab: dict[str, int]) -> str:
    """Vocabulary key for a source character (identity when already in vocab)."""
    if ch in vocab:
        return ch
    conv = _t2s_converter()
    if conv is not None:
        simplified = conv.convert(ch)
        if simplified in vocab:
            return simplified
    return ch


def _is_punct(ch: str) -> bool:
    """Whitespace / punctuation / separator (align.py's rule, kept local)."""
    if ch.isspace():
        return True
    import unicodedata as _ud
    return _ud.category(ch).startswith("P") or ch in "…—·~～"


def _pct(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else float("nan")


def _quality(transcript_doc: dict[str, Any], features: energy.AudioFeatures | None,
             chars: Sequence[dict[str, Any]], syllables: Sequence[dict[str, Any]],
             target_spans: Sequence[dict[str, Any]], params: dict[str, Any],
             info: dict[str, Any]) -> dict[str, Any]:
    durations = [float(s["end_s"]) - float(s["start_s"]) for s in syllables]
    violations = 0
    prev = -1.0
    for c in chars:
        start, end = float(c["start_s"]), float(c["end_s"])
        if start + 1e-6 < prev or end < start:
            violations += 1
        prev = max(prev, start)

    confs = [float(c["conf"]) for c in chars]
    logps = [float(t["logp"]) for t in target_spans if np.isfinite(t["logp"])]
    low = float(params["low_conf"])
    q: dict[str, Any] = {
        "n_chars": len(chars),
        "n_syllables": len(syllables),
        "n_words": len(transcript_doc.get("words") or []),
        "monotonicity_violations": int(violations),
        "syllable_ms": {
            "n": len(durations),
            "min": round(min(durations) * 1e3, 1) if durations else None,
            "p10": round(_pct(durations, 10) * 1e3, 1) if durations else None,
            "median": round(float(np.median(durations)) * 1e3, 1) if durations else None,
            "p90": round(_pct(durations, 90) * 1e3, 1) if durations else None,
            "max": round(max(durations) * 1e3, 1) if durations else None,
        },
        "posterior": {
            "n_tokens": len(logps),
            "mean_logp": round(float(np.mean(logps)), 4) if logps else None,
            "median_conf": round(float(np.median(confs)), 4) if confs else None,
            "mean_conf": round(float(np.mean(confs)), 4) if confs else None,
            "p10_conf": round(_pct(confs, 10), 4) if confs else None,
            "min_conf": round(min(confs), 4) if confs else None,
            "low_conf_rate": round(float(np.mean([c < low for c in confs])), 4) if confs else None,
        },
        "coverage": {
            "source_symbols": info.get("n_source_symbols"),
            "ctc_targets": info.get("n_targets"),
            "oov": info.get("n_oov"),
            "oov_chars": info.get("oov_chars"),
            "coverage": info.get("coverage"),
            "script_converted": info.get("n_script_converted"),
            "approx_tokens": info.get("n_approx_tokens"),
        },
        "low_confidence_syllables": int(sum(1 for c in chars if float(c["conf"]) < low)),
        "n_chunks": info.get("n_chunks"),
        "chunk_feasible": info.get("chunk_feasible"),
        "alignment_score_per_frame": info.get("alignment_score_per_frame"),
        "degenerate_targets": info.get("degenerate_targets"),
    }
    if len(chars) > 1:
        onset_med = float(np.median(np.diff([float(c["start_s"]) for c in chars])))
        q["onset_median_ms"] = round(onset_med * 1e3, 1)
    else:
        q["onset_median_ms"] = None

    if features is not None:
        hop_s = features.hop_s
        starts_frame = [int(round(float(c["start_s"]) / hop_s)) for c in chars]
        peak_frames = _local_peaks(features.onset)
        w = max(1, int(round(0.150 / hop_s)))
        percentiles = []
        hits = 0
        n_starts = 0
        for f in starts_frame:
            if f <= 0 or f >= features.n_frames - 1:
                continue
            win = features.onset[max(0, f - w):f + w + 1]
            if win.size < 3:
                continue
            percentiles.append(float((win < features.onset[f]).mean()))
            local = features.onset[max(0, f - 4):f + 5]
            if local.size and features.onset[f] >= float(local.max()) * 0.98:
                hits += 1
            n_starts += 1
        distances = ([float(np.min(np.abs(peak_frames - f)) * hop_s * 1e3)
                      for f in starts_frame] if peak_frames.size else [])
        q["onset_envelope"] = {
            "onset_percentile_mean": round(float(np.mean(percentiles)), 4) if percentiles else None,
            "peak_hit_rate": round(hits / n_starts, 4) if n_starts else None,
            "nearest_peak_ms": {
                "n": len(distances),
                "median_ms": round(float(np.median(distances)), 1) if distances else None,
                "within_50ms_rate": round(float(np.mean([d <= 50.0 for d in distances])), 4)
                if distances else None,
            },
        }
        q["speech_fraction"] = round(features.speech_fraction, 4)
    return q


def _local_peaks(env: np.ndarray, *, min_value: float = 0.20,
                 prominence: float = 0.15, min_sep: int = 4) -> np.ndarray:
    """Prominent local maxima of an onset envelope (same idea as align.py)."""
    n = int(env.size)
    if n < 3:
        return np.empty(0, dtype=np.int64)
    candidates: list[int] = []
    for i in range(1, n - 1):
        if env[i] < min_value or env[i] < env[i - 1] or env[i] <= env[i + 1]:
            continue
        lo, hi = max(0, i - 5), min(n, i + 6)
        if env[i] - float(env[lo:hi].min()) < prominence:
            continue
        candidates.append(i)
    kept: list[int] = []
    for i in sorted(candidates, key=lambda k: -float(env[k])):
        if all(abs(i - j) >= min_sep for j in kept):
            kept.append(i)
    return np.array(sorted(kept), dtype=np.int64)


def _assemble(transcript: dict[str, Any], *, duration: float, frame_stride_s: float,
              params: dict[str, Any], groups: Sequence[_TokenTargets],
              targets: Sequence[_Target], spans: Sequence[dict[str, Any]],
              fires: dict[str, Any], language: str, fps: float | None,
              features: energy.AudioFeatures | None, model: ModelHandle | None,
              segments_in: Sequence[dict[str, Any]], elapsed_load_s: float,
              elapsed_align_s: float) -> dict[str, Any]:
    """Turn per-target frame spans into a ``semotion.alignment/1`` document."""
    L = len(targets)
    if L == 0:
        raise media.MediaError("E_ALIGN_MODEL_UNSUPPORTED",
                               "no CTC targets could be built from the transcript")
    starts = np.array([int(s["start_frame"]) for s in spans], dtype=np.int64)
    ends = np.array([int(s["end_frame"]) for s in spans], dtype=np.int64)
    confs = np.array([float(s["conf"]) for s in spans], dtype=np.float64)
    ends = np.maximum(ends, starts + 1)
    total_frames = int(round(duration / frame_stride_s))
    total_frames = max(2, total_frames)
    starts = np.clip(starts, 0, total_frames - 1)
    ends = np.clip(ends, 1, total_frames)
    # enforce strict monotonicity: a token may not start before the previous ends
    for i in range(1, L):
        if starts[i] < ends[i - 1]:
            starts[i] = ends[i - 1]
        starts[i] = min(int(starts[i]), total_frames - 1)
        if ends[i] <= starts[i]:
            ends[i] = starts[i] + 1
    ends = np.clip(ends, 1, total_frames)

    def t(frame: int) -> float:
        return round(float(frame) * frame_stride_s, 6)

    # ------------------------------------------------------------ syllables
    syl_docs: list[dict[str, Any]] = []
    target_span_by_id: dict[int, tuple[int, int]] = {}
    for i, tgt in enumerate(targets):
        target_span_by_id[i] = (int(starts[i]), int(ends[i]))
        syl_docs.append({
            "index": i,
            "token_index": int(tgt.token_index),
            "segment_id": tgt.segment_id,
            "text": tgt.text,
            "start_s": t(starts[i]),
            "end_s": t(ends[i]),
            "conf": round(float(np.clip(confs[i], 0.0, 1.0)), 3),
            "island": fires.get("island_of_target", {}).get(i),
            "synthetic": bool(tgt.synthetic),
        })

    # ---------------------------------------------------------- token spans
    token_span: dict[int, tuple[int, int]] = {}
    token_conf: dict[int, float] = {}
    for g in groups:
        ids = [i for i in g.target_ids]
        if ids:
            a = int(starts[ids[0]])
            b = int(ends[ids[-1]])
            token_span[g.token.index] = (a, b)
            token_conf[g.token.index] = float(np.mean(confs[ids]))
    # tokens without acoustic targets (approx) get interpolated spans between
    # their neighbours, inside their own segment bounds
    ordered = [g.token for g in groups]
    for k, tok in enumerate(ordered):
        if tok.index in token_span:
            continue
        prev_end = token_span.get(ordered[k - 1].index, (None, 0))[1] if k else 0
        nxt = None
        for j in range(k + 1, len(ordered)):
            if ordered[j].index in token_span:
                nxt = token_span[ordered[j].index][0]
                break
        seg = next((s for s in segments_in if str(s.get("id")) == tok.segment_id), {})
        lo = prev_end if prev_end is not None else int(float(seg.get("start_s", 0.0)) / frame_stride_s)
        hi = nxt if nxt is not None else int(float(seg.get("end_s", duration)) / frame_stride_s)
        if hi <= lo:
            hi = lo + 1
        token_span[tok.index] = (int(lo), int(hi))
        token_conf[tok.index] = float(params.get("approx_conf", 0.10))

    # --------------------------------------------------------- words / chars
    words: list[dict[str, Any]] = []
    chars: list[dict[str, Any]] = []
    seg_text_by_id = {str(s.get("id")): str(s.get("text") or "").strip() for s in segments_in}
    # consuming the target sequence in token order to attribute synthetic chars
    for g in groups:
        tok = g.token
        if tok.index not in token_span:  # pragma: no cover - defensive
            continue
        a_frame, b_frame = token_span[tok.index]
        start, end = t(a_frame), t(b_frame)
        end = max(end, start + frame_stride_s)
        conf = float(np.clip(token_conf.get(tok.index, 0.0), 0.0, 1.0))
        seg_text = seg_text_by_id.get(tok.segment_id, "")
        word = {
            "id": f"w{len(words)}",
            "segment_id": tok.segment_id,
            "text": tok.text,
            "kind": tok.kind,
            "n_syllables": int(tok.n_syll),
            "start_s": round(start, 6),
            "end_s": round(end, 6),
            "conf": round(conf, 3),
            "char_start": tok.char_start,
            "char_end": tok.char_end,
            "trailing_punct": tok.trailing_punct,
            "group_id": tok.group_id,
            "syllable_ids": [int(i) for i in g.target_ids] if g.target_ids else [],
            "approx": bool(not g.target_ids),
        }
        words.append(word)

        if tok.kind == "han" and tok.n_syll == 1:
            chars.append({
                "id": f"c{len(chars)}", "segment_id": tok.segment_id,
                "token_id": word["id"], "text": seg_text[tok.char_start:tok.char_end],
                "kind": "han", "start_s": round(start, 6), "end_s": round(end, 6),
                "conf": round(conf, 3), "char_start": tok.char_start,
                "char_end": tok.char_end, "group_id": tok.group_id, "approx": False,
            })
            continue

        # multi-target tokens: distribute the token span over its printable
        # source characters (numbers / Latin / English words) exactly like
        # align.py, marked approx so captions use the word span.
        letters = [k for k in range(tok.char_start, tok.char_end)
                   if not _is_punct(seg_text[k])] if seg_text else []
        if not letters:
            letters = list(range(tok.char_start, tok.char_end)) or [tok.char_start]
        n_letter = len(letters)
        per = (end - start) / n_letter
        for j, ci_ in enumerate(letters):
            chars.append({
                "id": f"c{len(chars)}", "segment_id": tok.segment_id,
                "token_id": word["id"], "text": seg_text[ci_:ci_ + 1] if seg_text else "",
                "kind": tok.kind, "start_s": round(start + per * j, 6),
                "end_s": round(start + per * (j + 1), 6), "conf": round(conf, 3),
                "char_start": ci_, "char_end": ci_ + 1,
                "group_id": tok.group_id, "approx": True,
                "target_text": "".join(targets[i].text for i in g.target_ids),
            })

    for c in chars:  # token spans may overlap after interpolation; keep them ordered
        c["start_s"] = round(float(c["start_s"]), 6)
        c["end_s"] = round(float(c["end_s"]), 6)
    chars.sort(key=lambda c: (float(c["start_s"]), int(c.get("char_start") or 0)))

    # ------------------------------------------------------------- segments
    seg_docs: list[dict[str, Any]] = []
    for seg in segments_in:
        sid = str(seg.get("id"))
        seg_words = [w for w in words if w["segment_id"] == sid]
        seg_docs.append({
            "id": sid,
            "start_s": round(float(seg["start_s"]), 6),
            "end_s": round(float(seg["end_s"]), 6),
            "text": str(seg.get("text") or "").strip(),
            "n_tokens": len(seg_words),
            "n_syllables": int(sum(w["n_syllables"] for w in seg_words)),
            "speech_start_s": round(min(w["start_s"] for w in seg_words), 6) if seg_words else None,
            "speech_end_s": round(max(w["end_s"] for w in seg_words), 6) if seg_words else None,
        })

    # ------------------------------------------------------------- quality
    info = dict(fires)
    targets_span_docs = [
        {"start_s": t(starts[i]), "end_s": t(ends[i]), "logp": float(spans[i]["logp"]),
         "conf": float(spans[i]["conf"]), "text": targets[i].text}
        for i in range(L)
    ]
    quality = _quality({**transcript, "words": words}, features, chars, syl_docs,
                       targets_span_docs, params, info)
    quality["n_syllables"] = len(syl_docs)

    # ------------------------------------------------------------- warnings
    warnings: list[str] = []
    if info.get("n_script_converted"):
        warnings.append(f"{info['n_script_converted']} characters were looked up in "
                        "Simplified form (OpenCC t2s); the emitted text is unchanged")
    if info.get("n_oov"):
        warnings.append(f"{info['n_oov']} transcript symbols are out of the CTC "
                        f"vocabulary: {info.get('oov_chars')}")
    if info.get("n_approx_tokens"):
        warnings.append(f"{info['n_approx_tokens']} tokens (Latin words in zh text) "
                        "have no acoustic target and were interpolated")
    if quality["monotonicity_violations"]:
        warnings.append(f"{quality['monotonicity_violations']} monotonicity violations")
    if quality["low_confidence_syllables"]:
        warnings.append(f"{quality['low_confidence_syllables']} low-confidence syllables")
    if not info.get("chunk_feasible", True):
        warnings.append("a chunk had no feasible CTC path; it was merged or uniformly split")
    med_ms = quality["syllable_ms"]["median"]
    if str(language).lower().startswith(("zh", "ja", "ko")) and med_ms is not None \
            and not (80.0 <= med_ms <= 420.0):
        warnings.append(f"median syllable {med_ms} ms outside the expected 80-420 ms band")
    if info.get("degenerate_targets"):
        warnings.append(f"{info['degenerate_targets']} targets had no frames on the path")

    method: dict[str, Any] = {
        "name": METHOD,
        "version": METHOD_REV,
        "truth_class": "MACHINE_WEAK",
        "engine": model.engine if model else "external-posteriors",
        "device": model.device if model else "none",
        "model": model.model_id if model else "external-posteriors",
        "model_info": model.to_dict() if model else None,
        "params": {k: params[k] for k in sorted(params)},
        "runtime_s": {"load": round(elapsed_load_s, 3),
                      "align": round(elapsed_align_s, 3)},
        "notes": [
            "transcript text is trusted; CTC posteriors decide every boundary",
            "monotonic CTC Viterbi (torchaudio.functional.forced_align or NumPy)",
            "long clips are chunked at pauses; token allocation is per ASR segment",
            "confidence = mean posterior of the chosen CTC state over its frames",
            f"frame stride {frame_stride_s * 1e3:.1f} ms",
        ],
    }
    source = transcript.get("source") or {}
    return {
        "schema": SCHEMA,
        "source": {
            "path": source.get("path"),
            "sha256": source.get("sha256"),
            "duration_s": round(float(duration), 6),
            "fps": float(fps if fps is not None else (source.get("fps") or 25.0)),
            "width": source.get("width"),
            "height": source.get("height"),
            "language": str(language),
            "audio_rate": int(features.sample_rate) if features is not None else 16000,
        },
        "method": method,
        "segments": seg_docs,
        "syllables": syl_docs,
        "words": words,
        "chars": chars,
        "quality": quality,
        "warnings": warnings,
    }


# --------------------------------------------------------------- core pipeline


def _chunk_plan(chunks: Sequence[dict[str, Any]], groups: Sequence[_TokenTargets],
                segs: Sequence[dict[str, Any]], frame_stride_s: float,
                ) -> list[tuple[int, int, list[int]]]:
    """Frames + token ids per chunk, merging chunks too short for their tokens.

    A CTC path needs at least ``2*N+1`` frames for ``N`` targets, so a chunk that
    cannot hold its share is merged forward into the next one (the audio is then
    aligned as one longer island).  Only if the resulting single chunk is still
    too short does the aligner fall back to a uniform split — reported as
    ``quality.chunk_feasible == false``, never hidden.
    """
    per_chunk_tokens = _allocate_tokens([g.token for g in groups], chunks, segs)
    plan: list[tuple[int, int, list[int]]] = []
    for ci, ch in enumerate(chunks):
        a = int(round(float(ch["start_s"]) / frame_stride_s))
        b = max(a + 1, int(round(float(ch["end_s"]) / frame_stride_s)))
        plan.append((a, b, list(per_chunk_tokens[ci])))
    merged: list[tuple[int, int, list[int]]] = []
    for a, b, tok_ids in plan:
        if merged:
            pa, _pb, ptoks = merged[-1]
            need = 2 * (len(ptoks) + len(tok_ids)) + 1
            if (not tok_ids) or (b - a) < 2 * len(tok_ids) + 1 or (b - pa) < need:
                merged[-1] = (pa, b, ptoks + tok_ids)
                continue
        merged.append((a, b, tok_ids))
    return merged


def align_posteriors(transcript: dict[str, Any], log_probs: np.ndarray,
                     vocab: dict[str, int], *, blank: int = 0,
                     frame_stride_s: float = 0.02, language: str = "zh",
                     params: dict[str, Any] | None = None,
                     model: ModelHandle | None = None,
                     source: dict[str, Any] | None = None,
                     fps: float | None = None,
                     chunks_override: Sequence[dict[str, Any]] | None = None,
                     ) -> dict[str, Any]:
    """Align a transcript against a pre-computed posterior matrix.

    This is the model-free entry point: callers may bring posteriors from any
    CTC model (or a synthetic one, which is how the tests and the negative
    control run without downloading 1 GB of weights).  ``log_probs`` is
    ``(T, V)`` log-softmaxed, rows ``frame_stride_s`` apart.  ``blank`` is the
    CTC blank id.  With ``params["max_chunk_s"]`` set, the matrix is cut at the
    frames with the most blank mass, exercising the same chunking code as the
    model path.
    """
    started = time.time()
    p = {**DEFAULT_PARAMS, **(params or {})}
    lp = np.asarray(log_probs, dtype=np.float64)
    if lp.ndim != 2:
        raise media.MediaError("E_ALIGN_FAILED", f"log_probs must be (T,V), got {lp.shape}")
    t_total = lp.shape[0]
    duration = float(t_total) * float(frame_stride_s)

    segs = [s for s in (transcript.get("segments") or []) if (s.get("text") or "").strip()]
    segs.sort(key=lambda s: float(s["start_s"]))
    if not segs:
        raise media.MediaError("E_ALIGN_FAILED", "transcript has no non-empty segments")
    tokens: list[energy.Token] = []
    for seg in segs:
        sid = str(seg.get("id"))
        for tok in energy.tokenize_text(str(seg.get("text") or "").strip(), language):
            tok.segment_id = sid
            tok.index = len(tokens)
            tokens.append(tok)
    if not tokens:
        raise media.MediaError("E_ALIGN_FAILED", "transcript text has no alignable tokens")

    targets, groups, info = build_targets(tokens, vocab, language=language)
    if info["coverage"] < 0.85 and info["n_source_symbols"] > 3:
        raise media.MediaError(
            "E_ALIGN_MODEL_UNSUPPORTED",
            f"CTC vocabulary covers only {info['coverage']:.0%} of the transcript "
            f"(OOV: {info['oov_chars'][:16]}); refusing to fake the rest — "
            f"check the language/script or pick another model")

    src = {**(transcript.get("source") or {}), **(source or {})}
    src.setdefault("path", None)
    doc_transcript = {**transcript, "source": src}
    chunks = list(chunks_override) if chunks_override else _frame_chunks(
        duration, frame_stride_s, float(p["max_chunk_s"]), lp, blank)

    spans, fires = _run_chunks(
        chunks, targets, groups, segs, blank, float(frame_stride_s), p,
        lambda a, b, _c: (lp[max(0, int(round(a / frame_stride_s))):
                             max(1, int(round(b / frame_stride_s)))], frame_stride_s))

    doc = _assemble(doc_transcript, duration=duration, frame_stride_s=frame_stride_s,
                    params=p, groups=groups, targets=targets, spans=spans,
                    fires=fires, language=language, fps=fps,
                    features=None, model=model, segments_in=segs,
                    elapsed_load_s=0.0, elapsed_align_s=time.time() - started)
    return doc


def _run_chunks(chunks: Sequence[dict[str, Any]], targets: Sequence[_Target],
                groups: Sequence[_TokenTargets], segs: Sequence[dict[str, Any]],
                blank: int, frame_stride_s: float, params: dict[str, Any],
                posterior_fn: Callable[[float, float, str], tuple[np.ndarray, float]],
                ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Chunked CTC alignment: allocate tokens, align each chunk, concat spans."""
    min_token_frames = int(params["min_token_frames"])
    merged = _chunk_plan(chunks, groups, segs, frame_stride_s)
    spans: list[dict[str, Any]] = []
    island_of_target: dict[int, int] = {}
    score_acc = 0.0
    frames_acc = 0
    feasible = True
    degenerate = 0

    for ci, (a, b, tok_ids) in enumerate(merged):
        tok_set = set(tok_ids)
        ids = [tid for g in groups if g.token.index in tok_set for tid in g.target_ids]
        if not ids:
            continue
        if ids != sorted(ids):  # pragma: no cover - invariant guard
            raise media.MediaError(
                "E_ALIGN_FAILED",
                "token allocation to chunks is not monotonic; refusing to align")
        lp, _chunk_stride = posterior_fn(float(a) * frame_stride_s,
                                         float(b) * frame_stride_s, "chunk")
        n_frames = int(min(lp.shape[0], max(1, b - a)))
        res = forced_align_spans(lp[:n_frames], [targets[i].vocab_id for i in ids],
                                 blank=blank, min_frames=min_token_frames,
                                 engine=str(params.get("ctc_engine", "auto")))
        if not res["feasible"]:
            feasible = False
            for k, tid in enumerate(ids):
                s = int(round(a + (b - a) * k / len(ids)))
                e = int(round(a + (b - a) * (k + 1) / len(ids)))
                spans.append({"start_frame": s, "end_frame": max(e, s + 1),
                              "conf": 0.0, "logp": float("-inf")})
                island_of_target[tid] = ci
            continue
        score_acc += float(res["path_score"])
        frames_acc += int(res["n_frames"])
        degenerate += int(res.get("degenerate_targets", 0))
        for k, tid in enumerate(ids):
            # CTC emits one symbol over a few frames and then blanks: a symbol's
            # *span* is from its first emission frame to the next symbol's first
            # emission frame (the trailing blanks belong to it), and the last
            # symbol of a chunk runs to the end of that speech island.  Using
            # only the emission frames would report every syllable as 20 ms.
            start = int(a + res["starts"][k])
            stop = int(a + res["starts"][k + 1]) if k + 1 < len(ids) else int(b)
            spans.append({
                "start_frame": start,
                "end_frame": max(stop, start + min_token_frames),
                "conf": float(np.exp(max(-50.0, res["logp"][k]))),
                "logp": float(res["logp"][k]),
            })
            island_of_target[tid] = ci

    if len(spans) != len(targets):  # pragma: no cover - allocation invariant
        raise media.MediaError(
            "E_ALIGN_FAILED",
            f"chunked alignment produced {len(spans)} spans for {len(targets)} targets")
    fires = {
        "n_chunks": len(chunks),
        "n_merged_chunks": len(merged),
        "chunk_feasible": feasible,
        "alignment_score_per_frame": round(score_acc / frames_acc, 4) if frames_acc else None,
        "island_of_target": island_of_target,
        "degenerate_targets": degenerate,
    }
    return spans, fires


# ------------------------------------------------------------------ public API


def align_transcript(transcript: dict[str, Any], *,
                     samples: Sequence[float] | None = None,
                     video: str | Path | None = None,
                     language: str | None = None,
                     params: dict[str, Any] | None = None,
                     fps: float | None = None,
                     model: ModelHandle | None = None,
                     model_id: str | None = None,
                     device: str = "cpu",
                     allow_download: bool = True,
                     engine: str = ENGINE_HF,
                     allow_fallback: bool = False) -> dict[str, Any]:
    """Acoustic CTC alignment of ``transcript`` against its audio.

    Signature mirrors :func:`semotion.align.align_transcript` so the pipeline can
    switch engines without touching call sites.  Provide audio as ``samples``
    (mono float at 16 kHz) or a ``video`` path.

    ``allow_fallback=True`` runs the envelope-DP aligner instead when the CTC
    model cannot be loaded, and the returned document then has
    ``method.name == "envelope-dp/1"`` (the caller can tell them apart).
    """
    t0 = time.time()
    p = {**DEFAULT_PARAMS, **(params or {})}
    lang = str(language or transcript.get("language") or "zh")
    if samples is None and video is None:
        raise media.MediaError("E_ALIGN_FAILED", "need audio: pass samples= or video=")
    if samples is None:
        samples, _rate = energy.load_audio(video, rate=int(p["audio_rate"]))
    samples = np.asarray(samples, dtype=np.float32).flatten()
    duration = samples.size / float(p["audio_rate"])
    features = energy.audio_features(samples, sample_rate=int(p["audio_rate"]), params={
        k: p[k] for k in ("vad_min_speech_s", "vad_fill_gap_s", "vad_floor_db",
                          "vad_high_frac", "hop_s") if k in p
    })

    if model_id is None:
        model_id = DEFAULT_MODELS.get("zh" if str(lang).lower().startswith("zh") else "en")
    try:
        h = model or load_model(model_id, language=lang, device=device,
                                allow_download=allow_download, engine=engine)
    except Exception:
        if not allow_fallback:
            raise
        doc = energy.align_transcript(transcript, samples=samples, language=lang, fps=fps)
        doc.setdefault("warnings", []).append(
            "CTC model unavailable; fell back to the envelope-DP aligner")
        return doc

    load_s = h.load_s
    segs = [s for s in (transcript.get("segments") or []) if (s.get("text") or "").strip()]
    segs.sort(key=lambda s: float(s["start_s"]))
    if not segs:
        raise media.MediaError("E_ALIGN_FAILED", "transcript has no non-empty segments")
    tokens: list[energy.Token] = []
    for seg in segs:
        sid = str(seg.get("id"))
        for tok in energy.tokenize_text(str(seg.get("text") or "").strip(), lang):
            tok.segment_id = sid
            tok.index = len(tokens)
            tokens.append(tok)
    if not tokens:
        raise media.MediaError("E_ALIGN_FAILED", "transcript text has no alignable tokens")
    targets, groups, info = build_targets(tokens, h.vocab, language=lang)
    if info["coverage"] < 0.85 and info["n_source_symbols"] > 3:
        raise media.MediaError(
            "E_ALIGN_MODEL_UNSUPPORTED",
            f"CTC vocabulary covers only {info['coverage']:.0%} of the transcript "
            f"(OOV: {info['oov_chars'][:16]}); refusing to fake the rest — "
            f"check the language/script or pick another model")

    chunks = _speech_chunks(features, p)
    pad = float(p["chunk_pad_s"])

    def posterior_fn(start_s: float, end_s: float, _kind: str) -> tuple[np.ndarray, float]:
        lo = max(0.0, start_s - pad)
        hi = min(duration, end_s + pad)
        a = int(round(lo * p["audio_rate"]))
        b = int(round(hi * p["audio_rate"]))
        lp, stride = h.log_probs(samples[a:b], int(p["audio_rate"]))
        # shift chunk frames so local frame 0 == start_s (the pad is context and
        # the caller's a/b already index absolute time)
        offset_frames = int(round((start_s - lo) / stride))
        return lp[max(0, offset_frames):max(0, offset_frames) + int(round((end_s - start_s) / stride))], stride

    started_align = time.time()
    spans, fires = _run_chunks(chunks, targets, groups, segs, h.blank,
                              h.frame_stride_s, p, posterior_fn)
    fires.update(info)
    segs_in = [{"id": str(s.get("id")), "start_s": float(s["start_s"]),
                "end_s": float(s["end_s"]), "text": str(s.get("text") or "").strip()}
               for s in segs]

    doc = _assemble({**transcript, "source": {**(transcript.get("source") or {}),
                                              **({"path": str(Path(video).resolve())} if video else {})}},
                    duration=duration, frame_stride_s=h.frame_stride_s, params=p,
                    groups=groups, targets=targets, spans=spans, fires=fires,
                    language=lang, fps=fps, features=features, model=h,
                    segments_in=segs, elapsed_load_s=load_s,
                    elapsed_align_s=time.time() - started_align)
    doc["method"]["runtime_s"]["total"] = round(time.time() - t0, 3)
    return doc


def align_video(video: str | Path, transcript: dict[str, Any] | str | Path, *,
                language: str | None = None, params: dict[str, Any] | None = None,
                fps: float | None = None, model_id: str | None = None,
                device: str = "cpu", allow_download: bool = True,
                allow_fallback: bool = False) -> dict[str, Any]:
    """Align a transcript document/path against a video (or wav) file."""
    if isinstance(transcript, (str, Path)):
        transcript = json.loads(Path(transcript).read_text(encoding="utf-8"))
    return align_transcript(transcript, video=video, language=language, params=params,
                            fps=fps, model_id=model_id, device=device,
                            allow_download=allow_download, allow_fallback=allow_fallback)


# ------------------------------------------------------------------- IO helpers


def save(doc: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(doc: dict[str, Any]) -> list[str]:
    """Schema check: the shared ``semotion.alignment/1`` validator plus CTC keys."""
    problems = energy.validate(doc)
    method = doc.get("method") or {}
    if method.get("name") != METHOD:
        problems.append(f"method.name must be {METHOD!r}")
    if not method.get("model"):
        problems.append("method.model missing")
    if not method.get("device"):
        problems.append("method.device missing")
    q = doc.get("quality") or {}
    if "posterior" not in q:
        problems.append("quality.posterior missing")
    return problems
