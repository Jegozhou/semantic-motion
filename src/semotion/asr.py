"""Semantic Motion — speech recognition layer (faster-whisper, CPU).

Contract: ``semotion.transcript/1`` (docs/INTERFACES.md §3.1).

Design notes
------------
* Engine: **faster-whisper / CTranslate2** (not the openai-whisper CLI). The
  openai-whisper MPS path is unusable for word timestamps on this machine
  (float64 op not implemented on MPS), and its CLI output has no word times.
  faster-whisper runs on CPU int8: ~0.33x real time for a 24 s clip.
* Audio is decoded through :mod:`semotion.media` only (never a raw ffmpeg
  call), as 16 kHz mono PCM s16le, then read with the stdlib ``wave`` module.
  ``WhisperModel.decode_audio`` is deliberately **not** used: the bundled pyav
  version is incompatible with it.
* The model is resolved to a **local CTranslate2 snapshot directory** whenever
  one exists (env ``SEMOTION_WHISPER_DIR`` or the shared model shelf), so the
  call never touches the network and never downloads. Only when no local copy
  is found does it fall back to the Hugging Face repo id (which may download).
* Determinism: the transcript content is a pure function of (video bytes,
  model, language, decode options). The only run-varying fields are
  ``provenance.created_utc`` / ``command`` / ``elapsed_s``; a cache hit returns
  the byte-identical stored document.
* Cache: ``artifacts/cache/asr/<source_sha256>_<model>_<device>.json``
  (``force=True`` re-runs; a mismatch in schema/sha never poisons a hit).
* Truth class is always ``MACHINE_WEAK``: ASR output may never certify itself
  as ground truth (spec §5).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import tempfile
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from . import media

__all__ = [
    "SCHEMA",
    "DEFAULT_MODEL",
    "MODEL_SEARCH_ROOTS",
    "transcribe",
    "load",
    "validate",
    "words_in_range",
    "segments_in_range",
    "word_by_id",
    "find_phrase",
    "resolve_model_path",
    "cache_path",
]

SCHEMA = "semotion.transcript/1"
DEFAULT_MODEL = "medium"
AUDIO_RATE = 16000

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Where local CTranslate2 snapshots may live. Overridable per call via env.
MODEL_SEARCH_ROOTS: list[Path] = [
    Path("/Volumes/ZHITAI/models/faster-whisper"),
    Path("/Volumes/ZHITAI/models"),
    Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub",
    PROJECT_ROOT / "assets" / "models" / "faster-whisper",
]

# One loaded model per (path, device, compute_type) per process: loading the
# 1.5 GB medium model takes ~2.5 s cold, ~1 s warm, so tests reuse it.
_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


# --------------------------------------------------------------------------- paths

def cache_root() -> Path:
    """Root of the deterministic artifact cache."""
    return Path(os.environ.get("SEMOTION_CACHE_ROOT") or (PROJECT_ROOT / "artifacts" / "cache"))


def _slug(text: str, *, limit: int = 48) -> str:
    text = str(text)
    if "/" in text or "\\" in text or text.endswith(".bin"):
        # A filesystem path is not a stable, readable cache key: use a digest.
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
        text = f"local-{Path(text).name}-{digest}"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    return (slug or "model")[:limit]


def cache_path(source_sha256: str, model: str = DEFAULT_MODEL, device: str = "cpu") -> Path:
    """Cache file for one (video, model, device) triple.

    The key always contains the source sha256 *and* the model name, so two
    different models can never collide.
    """
    return cache_root() / "asr" / f"{source_sha256}_{_slug(model)}_{_slug(device)}.json"


def resolve_model_path(model: str = DEFAULT_MODEL) -> str:
    """Resolve a model name to a local CTranslate2 directory when possible.

    Returns a directory path containing ``model.bin`` (offline, preferred) or
    the original name (Hugging Face repo id fallback).
    """
    raw = os.environ.get("SEMOTION_WHISPER_DIR")
    roots = ([Path(raw)] if raw else []) + MODEL_SEARCH_ROOTS

    candidate = Path(model)
    if candidate.is_dir() and (candidate / "model.bin").exists():
        return str(candidate)
    if candidate.is_file():
        return str(candidate)

    wanted = _slug(model).lower()
    found: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        # 1) standard HF hub layout: <root>/models--<org>--faster-whisper-<size>/snapshots/<rev>/model.bin
        for pattern in (root / "models--*" / "snapshots" / "*" / "model.bin",
                        root / "models--*faster-whisper-" / "*" / "model.bin",
                        root / "*" / "model.bin"):
            for hit in sorted(glob.glob(str(pattern))):
                parent = Path(hit).parent
                # the model shelf name must mention the requested size
                haystack = f"{parent}".lower() + f" {parent.parent.name}".lower()
                if wanted and wanted not in haystack:
                    continue
                found.append(str(parent))
        # 2) plain <root>/<model>/model.bin
        plain = root / model / "model.bin"
        if plain.exists():
            found.append(str(plain.parent))
    if found:
        # shortest path first keeps the choice stable and readable
        return sorted(set(found), key=lambda p: (len(p), p))[0]
    return str(model)


# ------------------------------------------------------------------------- model

def _load_model(model: str, device: str, compute_type: str):
    key = (model, device, compute_type)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    try:
        from faster_whisper import WhisperModel  # heavy import, keep it local
    except Exception as exc:  # pragma: no cover - environment guard
        raise media.MediaError("E_ASR_FAILED", f"faster-whisper unavailable: {exc}") from exc
    try:
        instance = WhisperModel(model, device=device, compute_type=compute_type)
    except Exception as exc:
        raise media.MediaError(
            "E_ASR_FAILED", f"could not load model {model!r} ({device}, {compute_type}): {exc}"
        ) from exc
    _MODEL_CACHE[key] = instance
    return instance


def _engine_version() -> str:
    try:
        import faster_whisper

        return str(faster_whisper.__version__)
    except Exception:  # pragma: no cover
        return "unknown"


def _ctranslate_version() -> str:
    try:
        import ctranslate2

        return str(ctranslate2.__version__)
    except Exception:  # pragma: no cover
        return "unknown"


# ------------------------------------------------------------------------ audio

def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read a PCM wav as float32 in [-1, 1]; faster-whisper needs raw samples."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise media.MediaError("E_ASR_FAILED", f"expected 16-bit PCM wav: {path}")
        rate = w.getframerate()
        frames = w.getnframes()
        raw = w.readframes(frames)
    if not raw:
        raise media.MediaError("E_ASR_FAILED", f"empty audio stream decoded from video: {path}")
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, rate


def _decode_audio(video: Path, *, work_dir: Path | None = None) -> tuple[np.ndarray, int]:
    """Extract 16 kHz mono PCM via media.extract_audio and read it as float32."""
    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix="semotion-asr-") as tmp:
            wav = media.extract_audio(video, Path(tmp) / "audio.wav", rate=AUDIO_RATE, channels=1)
            return _read_wav_mono(wav)
    wav = media.extract_audio(video, Path(work_dir) / "audio.wav", rate=AUDIO_RATE, channels=1)
    return _read_wav_mono(wav)


# ------------------------------------------------------------------- transcripts

def _char_span(text: str, cursor: int, token: str) -> tuple[int, int, int]:
    """Locate ``token`` inside ``text`` starting at ``cursor`` (char indices).

    faster-whisper word strings are the concatenation of a segment's tokens
    (sometimes with a leading space). We walk the segment text to recover the
    character offset so downstream code can map anchors back to characters.
    """
    stripped = token.strip()
    if not stripped:
        return cursor, cursor, cursor
    idx = text.find(stripped, cursor)
    if idx < 0:
        # Fallback: keep the walk monotone instead of inventing an offset.
        idx = min(cursor, max(0, len(text) - len(stripped)))
        idx = max(0, idx)
    return idx, idx + len(stripped), idx + len(stripped)


def _build_document(video: Path, mi: media.MediaInfo, sha: str, language: str,
                    model_arg: str, resolved_model: str, device: str,
                    compute_type: str, beam_size: int, raw_segments: Iterable[Any],
                    info: Any, command: str, elapsed_s: float) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    for s_index, seg in enumerate(raw_segments):
        sid = f"s{s_index}"
        text = (seg.text or "").strip()
        segments.append({
            "id": sid,
            "start_s": round(float(seg.start), 6),
            "end_s": round(float(seg.end), 6),
            "text": text,
            "avg_logprob": round(float(seg.avg_logprob), 6) if seg.avg_logprob is not None else None,
            "no_speech_prob": (round(float(seg.no_speech_prob), 6)
                               if getattr(seg, "no_speech_prob", None) is not None else None),
        })
        cursor = 0
        for w in (getattr(seg, "words", None) or []):
            token = (w.word or "")
            if not token.strip():
                continue
            c0, c1, cursor = _char_span(text, cursor, token)
            words.append({
                "id": f"w{len(words)}",
                "segment_id": sid,
                "text": token.strip(),
                "start_s": round(float(w.start), 6),
                "end_s": round(float(w.end), 6),
                "probability": round(float(w.probability), 6) if w.probability is not None else None,
                "char_start": int(c0),
                "char_end": int(c1),
            })

    return {
        "schema": SCHEMA,
        "source": {
            "path": str(video.resolve()),
            "sha256": sha,
            "duration_s": round(float(mi.duration), 6),
            "width": int(mi.width),
            "height": int(mi.height),
            "fps": round(float(mi.fps), 6),
        },
        "language": language,
        "model": {
            "name": "faster-whisper",
            "family": "whisper",
            "size": model_arg,
            "device": device,
            "compute_type": compute_type,
            "version": _engine_version(),
            "ctranslate2_version": _ctranslate_version(),
            "path": resolved_model,
            "beam_size": int(beam_size),
            "word_timestamps": True,
        },
        "segments": segments,
        "words": words,
        "provenance": {
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "command": command,
            "truth_class": "MACHINE_WEAK",
            "engine": "faster-whisper (CTranslate2)",
            "elapsed_s": round(float(elapsed_s), 3),
            "language_probability": (round(float(getattr(info, "language_probability", 0.0) or 0.0), 6)),
            "n_segments": len(segments),
            "n_words": len(words),
        },
    }


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------- public API

def transcribe(video: Path, out_json: Path, *, model: str = DEFAULT_MODEL,
               device: str = "cpu", language: str = "zh", force: bool = False,
               compute_type: str = "int8", beam_size: int = 5) -> dict[str, Any]:
    """Transcribe ``video`` and write/return a ``semotion.transcript/1`` document.

    Parameters
    ----------
    video : Path
        Any container readable by ffmpeg that has an audio stream.
    out_json : Path
        Destination for the transcript document (parent dirs are created).
    model : str
        Model size name (``medium``) *or* a local CTranslate2 directory.
    force : bool
        Ignore a valid cache entry and re-run the model.

    Raises
    ------
    media.MediaError
        ``E_INPUT_MISSING`` / ``E_NO_VIDEO`` / ``E_FFMPEG`` for media problems,
        ``E_ASR_FAILED`` for model or decoding failures.
    """
    video = Path(video)
    out_json = Path(out_json)
    mi = media.info(video)              # raises E_INPUT_MISSING / E_NO_VIDEO / E_FFMPEG
    sha = media.sha256_file(video)

    cached = cache_path(sha, model, device)
    if not force and cached.exists():
        try:
            doc = json.loads(cached.read_text(encoding="utf-8"))
        except Exception:
            doc = None
        if (isinstance(doc, dict) and doc.get("schema") == SCHEMA
                and doc.get("source", {}).get("sha256") == sha
                and doc.get("model", {}).get("size") == model
                and doc.get("model", {}).get("device") == device):
            _write_json(out_json, doc)
            return doc

    resolved = resolve_model_path(model)
    t_start = time.time()
    samples, rate = _decode_audio(video)
    model_obj = _load_model(resolved, device, compute_type)

    command = (f"faster-whisper {model} ({resolved}) device={device} "
               f"compute_type={compute_type} language={language} word_timestamps=True "
               f"beam_size={beam_size} vad_filter=False :: {video}")

    try:
        raw_segments, info = model_obj.transcribe(
            samples,
            language=language or None,
            beam_size=int(beam_size),
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,
            temperature=0.0,
        )
        raw_segments = list(raw_segments)   # force the generator inside the guard
    except Exception as exc:
        raise media.MediaError("E_ASR_FAILED", f"transcription failed for {video}: {exc}",
                               cmd=["faster-whisper", model, str(video)]) from exc

    elapsed = time.time() - t_start
    if not raw_segments:
        raise media.MediaError("E_ASR_FAILED",
                               f"no speech recognized in {video} (audio {len(samples)/rate:.2f}s)")

    doc = _build_document(video, mi, sha, language, model, resolved, device,
                          compute_type, beam_size, raw_segments, info, command, elapsed)
    if int(doc["source"].get("fps") or 0) and doc["words"]:
        max_end = doc["words"][-1]["end_s"]
        if max_end > mi.duration + 0.5:
            # A word ending far past the media means the decode was misaligned.
            raise media.MediaError(
                "E_ASR_FAILED",
                f"word times exceed media duration ({max_end:.3f}s > {mi.duration:.3f}s)")

    _write_json(cached, doc)
    _write_json(out_json, doc)
    return doc


# ------------------------------------------------------------------------- helpers

def load(path: str | Path) -> dict[str, Any]:
    """Read a transcript JSON from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(transcript: dict[str, Any]) -> list[str]:
    """Return schema problems (empty list == valid). Mirrors INTERFACES §3.1."""
    problems: list[str] = []
    if transcript.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}")
    src = transcript.get("source") or {}
    for key in ("path", "sha256", "duration_s", "width", "height", "fps"):
        if key not in src:
            problems.append(f"source.{key} missing")
    model = transcript.get("model") or {}
    for key in ("name", "size", "device", "version"):
        if key not in model:
            problems.append(f"model.{key} missing")
    prov = transcript.get("provenance") or {}
    if prov.get("truth_class") != "MACHINE_WEAK":
        problems.append("provenance.truth_class must be MACHINE_WEAK (ASR cannot self-certify)")
    duration = float(src.get("duration_s") or 0.0)

    seen_seg: set[str] = set()
    for i, seg in enumerate(transcript.get("segments") or []):
        sid = seg.get("id")
        if not sid or sid in seen_seg:
            problems.append(f"segments[{i}].id missing or duplicated: {sid!r}")
        seen_seg.add(sid)
        if not (0.0 <= float(seg.get("start_s", -1)) < float(seg.get("end_s", -1))
                <= duration + 0.5):
            problems.append(f"segments[{i}] bad interval {seg.get('start_s')}-{seg.get('end_s')}")

    seen_word: set[str] = set()
    for i, w in enumerate(transcript.get("words") or []):
        wid = w.get("id")
        if not wid or wid in seen_word:
            problems.append(f"words[{i}].id missing or duplicated: {wid!r}")
        seen_word.add(wid)
        if w.get("segment_id") not in seen_seg:
            problems.append(f"words[{i}] unknown segment_id {w.get('segment_id')!r}")
        if not (0.0 <= float(w.get("start_s", -1)) <= float(w.get("end_s", -1))
                <= duration + 0.5):
            problems.append(f"words[{i}] bad interval {w.get('start_s')}-{w.get('end_s')}")
        if int(w.get("char_start", 0)) > int(w.get("char_end", 0)):
            problems.append(f"words[{i}] char_start > char_end")
        if w.get("probability") is None:
            problems.append(f"words[{i}].probability missing")
    return problems


def words_in_range(transcript: dict[str, Any], t0: float, t1: float, *,
                   mode: str = "overlap") -> list[dict[str, Any]]:
    """Words touching ``[t0, t1)`` (``overlap``) or whose midpoint falls inside.

    Times come straight from the transcript; nothing is interpolated here.
    """
    lo, hi = float(min(t0, t1)), float(max(t0, t1))
    out = []
    for w in transcript.get("words") or []:
        start, end = float(w["start_s"]), float(w["end_s"])
        if mode == "midpoint":
            if lo <= (start + end) / 2.0 < hi:
                out.append(w)
        elif start < hi and end > lo:
            out.append(w)
        else:
            continue
    return out


def segments_in_range(transcript: dict[str, Any], t0: float, t1: float) -> list[dict[str, Any]]:
    """Segments overlapping ``[t0, t1)``."""
    lo, hi = float(min(t0, t1)), float(max(t0, t1))
    return [s for s in transcript.get("segments") or []
            if float(s["start_s"]) < hi and float(s["end_s"]) > lo]


def word_by_id(transcript: dict[str, Any], word_id: str) -> dict[str, Any] | None:
    for w in transcript.get("words") or []:
        if w.get("id") == word_id:
            return w
    return None


def _clean_phrase(text: str) -> str:
    """Strip punctuation/whitespace used for matching, keep characters."""
    return re.sub(r"[\s，。、！？,.!?;；:：\"'“”‘’（）()\[\]【】—…·]+", "", text or "")


def find_phrase(transcript: dict[str, Any], phrase: str, *, start_s: float | None = None,
                end_s: float | None = None) -> dict[str, Any] | None:
    """Locate ``phrase`` as a run of consecutive ASR words.

    Returns ``{"text", "start_s", "end_s", "word_ids", "segment_ids"}`` or
    ``None``. Matching ignores punctuation, so ``"其实是节奏"`` matches the words
    ``其实`` + ``是`` + ``节`` + ``奏。``. Times are always word-derived.
    """
    target = _clean_phrase(phrase)
    if not target:
        return None
    words = [w for w in transcript.get("words") or []
             if (start_s is None or float(w["end_s"]) > float(start_s))
             and (end_s is None or float(w["start_s"]) < float(end_s))]
    for i, first in enumerate(words):
        acc = ""
        ids: list[str] = []
        seg_ids: list[str] = []
        for w in words[i:]:
            piece = _clean_phrase(w.get("text", ""))
            if not piece:
                continue
            acc += piece
            ids.append(w["id"])
            if w.get("segment_id") not in seg_ids:
                seg_ids.append(w.get("segment_id"))
            if acc == target:
                return {"text": acc, "start_s": float(first["start_s"]),
                        "end_s": float(w["end_s"]), "word_ids": ids,
                        "segment_ids": seg_ids}
            if not target.startswith(acc):
                break
    return None
