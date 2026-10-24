"""Speech recognition with word timestamps + acoustic boundary refinement + sentence building.

Backends: faster-whisper (default, CPU/CUDA), mlx-whisper (Apple Silicon, if installed),
openai-whisper (if installed). All return the same word list; timing is then refined against the
10 ms VAD/onset envelope so that visuals never land in silence before a word is actually spoken.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .audio import AudioFeatures, emphasis_scores
from .text import CLAUSE_END, SENT_END, cjk_ratio, is_cjk, norm, to_simplified
from .util import SMError, log, warn

# English words that cannot end a visual beat (articles, prepositions, conjunctions, auxiliaries)
DANGLING = {"a", "an", "the", "to", "of", "and", "or", "but", "in", "on", "at", "for", "with", "from", "by", "into",
            "that", "which", "who", "is", "are", "was", "were", "be", "our", "my", "your", "their", "his", "her",
            "its", "this", "these", "those", "so", "if", "than", "as", "about", "we're", "it's", "i'm", "very"}

PROMPTS = {
    "zh": "以下是普通话的句子，使用简体中文和标点符号。",
    "en": "Hello. This is a clearly punctuated transcript of a talk.",
}


def _faster_whisper(audio: np.ndarray, model_name: str, language: Optional[str], models_dir: str, threads: int):
    from faster_whisper import WhisperModel  # type: ignore

    device, compute = "cpu", "int8"
    try:
        import ctranslate2  # type: ignore

        if ctranslate2.get_cuda_device_count() > 0:
            device, compute = "cuda", "float16"
    except Exception:
        pass
    log(f"ASR faster-whisper model={model_name} device={device}/{compute}")
    model = WhisperModel(model_name, device=device, compute_type=compute, download_root=models_dir,
                         cpu_threads=threads or max(1, (os.cpu_count() or 2)))
    prompt = PROMPTS.get(language or "", None)
    if language is None:
        # detect first so that the prompt matches the language
        try:
            lang, prob, _ = model.detect_language(audio[: 16000 * 30])
            language = lang
            prompt = PROMPTS.get(lang)
            log(f"detected language {lang} ({prob:.2f})")
        except Exception:
            pass
    segs, info = model.transcribe(audio, language=language, beam_size=5, word_timestamps=True, vad_filter=True,
                                  vad_parameters=dict(min_silence_duration_ms=250, speech_pad_ms=200),
                                  initial_prompt=prompt, condition_on_previous_text=False, temperature=[0.0, 0.2, 0.4])
    words = []
    for s in segs:
        sw = list(s.words or [])
        for k, w in enumerate(sw):
            d = {"text": w.word, "start": float(w.start), "end": float(w.end), "p": float(w.probability)}
            if k == len(sw) - 1:
                d["seg_end"] = True
            words.append(d)
    return words, info.language


def _whisper_like(audio: np.ndarray, model_name: str, language: Optional[str], backend: str):
    if backend == "mlx":
        import mlx_whisper  # type: ignore

        repo = model_name if "/" in model_name else f"mlx-community/whisper-{model_name}"
        res = mlx_whisper.transcribe(audio, path_or_hf_repo=repo, word_timestamps=True, language=language,
                                     initial_prompt=PROMPTS.get(language or ""), condition_on_previous_text=False)
    else:
        import whisper  # type: ignore

        m = whisper.load_model(model_name)
        res = m.transcribe(audio, word_timestamps=True, language=language, initial_prompt=PROMPTS.get(language or ""),
                           condition_on_previous_text=False)
    words = []
    for s in res.get("segments", []):
        for w in s.get("words", []) or []:
            words.append({"text": w["word"], "start": float(w["start"]), "end": float(w["end"]), "p": float(w.get("probability", 1.0))})
    return words, res.get("language", language)


def transcribe(audio16k: np.ndarray, model: str = "large-v3-turbo", language: Optional[str] = None,
               backend: str = "auto", models_dir: str = "", threads: int = 0) -> tuple[list[dict], str]:
    if backend == "auto":
        backend = "faster-whisper"
        try:
            import platform

            if platform.system() == "Darwin" and platform.machine() == "arm64":
                import mlx_whisper  # type: ignore  # noqa: F401

                backend = "mlx"
        except Exception:
            pass
    try:
        if backend == "faster-whisper":
            return _faster_whisper(audio16k, model, language, models_dir, threads)
        return _whisper_like(audio16k, model, language, backend)
    except ImportError as e:
        raise SMError(f"ASR backend '{backend}' unavailable ({e}). Run the skill's setup script.")


# ----------------------------------------------------------------------------------------------
def refine_words(words: list[dict], af: AudioFeatures, max_shift: float = 0.30,
                 rules: tuple = ("soft_start", "silence_end"), walk_back: float = 0.12,
                 win: float = 0.06, soft_frac: float = 0.2, soft_max: float = 0.8,
                 soft_pre: float = 0.02) -> list[dict]:
    """Snap word boundaries to the acoustic envelope.

    Defaults were chosen with the timing benchmark (tests/bench_timing.py, docs/TEST_REPORT.md):

    * ``soft_start`` - a word whose Whisper onset sits in non-speech is moved forward to the first hop whose
      loudness rises above ``noise_floor + soft_frac * (speech_level - noise_floor)`` (searching up to
      ``soft_max`` s or the next word), minus ``soft_pre`` s. This fixes the typical Whisper failure where a
      sentence-initial word after a pause is stamped hundreds of ms early, without touching words that already
      start inside speech (moving those made results worse).
    * ``silence_end`` - an end that falls in silence moves back to the last voiced hop.

    Experimental rules kept for benchmarking: ``silence_start`` (hard VAD snap) and ``pause_onset`` (snap
    the first word after a pause to the strongest onset within +-win; made p90 error 40-50 ms worse).
    Boundaries are made monotonic afterwards.
    """
    hop = af.hop
    vad = af.vad
    n = len(vad)
    out = []
    for i, w in enumerate(words):
        s, e = w["start"], w["end"]
        rs, re_ = s, e
        i0 = af.idx(s)
        if not vad[i0] and "soft_start" in rules:
            nxt = words[i + 1]["start"] if i + 1 < len(words) else s + soft_max
            lim = af.idx(min(s + soft_max, max(e, nxt)))
            thr = max(af.noise_floor + 6.0, af.noise_floor + soft_frac * (af.speech_level - af.noise_floor))
            j = i0
            while j < lim and af.db[j] < thr:
                j += 1
            if j < lim:
                ns = max(s, j * hop - soft_pre)
                if ns > s:
                    e = max(e, ns + min(e - s, 0.25))
                    s = ns
        elif not vad[i0] and "silence_start" in rules:
            j = i0
            lim = min(n - 1, af.idx(min(e - 0.02, s + max_shift)))
            while j < lim and not vad[j]:
                j += 1
            if vad[j]:
                s = j * hop
        elif "pause_onset" in rules:
            prev_end = out[-1]["end"] if out else -1.0
            if s - prev_end > 0.15:  # word after a pause: look for the true onset just before/after
                a, b = max(0, af.idx(s - win)), min(n - 1, af.idx(s + win))
                k = a + int(np.argmax(af.onset[a:b + 1])) if b > a else i0
                # walk back to the voicing start of that onset
                while k > 0 and vad[k - 1] and (s - (k - 1) * hop) < walk_back:
                    k -= 1
                s = k * hop
        i1 = af.idx(e)
        if not vad[min(n - 1, max(0, i1 - 1))] and "silence_end" in rules:
            j = i1
            lim = af.idx(max(s + 0.04, e - max_shift))
            while j > lim and not vad[max(0, j - 1)]:
                j -= 1
            e = j * hop
        if e <= s + 0.02:
            e = s + max(0.04, min(0.25, rs and (re_ - rs)))
        out.append({**w, "start": round(s, 3), "end": round(e, 3), "asr_start": round(rs, 3), "asr_end": round(re_, 3)})
    for i in range(1, len(out)):
        if out[i]["start"] < out[i - 1]["start"]:
            out[i]["start"] = out[i - 1]["start"]
        if out[i - 1]["end"] > out[i]["start"]:
            out[i - 1]["end"] = round(max(out[i - 1]["start"] + 0.02, out[i]["start"]), 3)
    return out


def _clean_words(words: list[dict], language: str, af: Optional[AudioFeatures]) -> list[dict]:
    res = []
    for w in words:
        t = w["text"]
        if language and language.startswith("zh"):
            t = to_simplified(t)
        if not t.strip():
            continue
        if af is not None and w.get("p", 1) < 0.15:
            a, b = af.idx(w["start"]), af.idx(w["end"])
            if b > a and af.vad[a:b].mean() < 0.2:
                continue  # likely hallucination in silence
        res.append({**w, "text": t})
    return res


def build_transcript(raw_words: list[dict], language: str, af: Optional[AudioFeatures], refine: bool = True) -> dict:
    words = _clean_words(raw_words, language, af)
    if refine and af is not None:
        words = refine_words(words, af)
    for i, w in enumerate(words):
        w["id"] = f"w{i:04d}"
    cjk = language.startswith(("zh", "ja", "ko")) if language else cjk_ratio("".join(w["text"] for w in words)) > 0.5
    sentences = []
    cur: list[int] = []

    def units(ids: list[int]) -> int:
        txt = "".join(words[i]["text"] for i in ids)
        if cjk:
            import re as _re

            return sum(1 for c in txt if is_cjk(c)) + len(_re.findall(r"[A-Za-z0-9]+", txt))
        return len(txt.split())

    def flush():
        if cur:
            sentences.append(list(cur))
            cur.clear()

    # sentence/clause segmentation tuned so that one "sentence" is one visual beat
    soft_u, soft_gap_u, force_u, max_u = (10, 8, 30, 40) if cjk else (9, 6, 22, 30)
    for i, w in enumerate(words):
        cur.append(i)
        txt = w["text"].strip()
        gap = (words[i + 1]["start"] - w["end"]) if i + 1 < len(words) else 9.0
        u = units(cur)
        last = txt[-1:] if txt else ""
        # never end a beat on a dangling function word ("... and the" | "good news ...") unless the speaker
        # really stops; whisper segment boundaries often fall there in unpunctuated speech
        dangling = (not cjk) and txt.lower().strip(",") in DANGLING
        hard = (last in SENT_END) or (last == "." and not cjk) or gap >= 0.65 or \
            (w.get("seg_end") and (gap >= 0.2 or u >= 6) and not dangling)
        soft = ((last in CLAUSE_END and u >= soft_u) or (gap >= 0.40 and u >= soft_gap_u)) and not dangling
        force = u >= force_u and gap >= 0.15 and not dangling
        if hard or soft or force:
            flush()
        elif u >= max_u:
            # too long without a natural boundary: split retrospectively at the best point
            # (largest gap, whisper segment end or clause punctuation; never after a dangling word)
            best, best_sc = None, -1e9
            n = len(cur)
            for k in range(3, n - 3):
                wi = cur[k]
                g = words[wi + 1]["start"] - words[wi]["end"]
                tk = words[wi]["text"].strip()
                sc = 4.0 * g + (0.5 if words[wi].get("seg_end") else 0.0) + (0.8 if tk[-1:] in CLAUSE_END else 0.0)
                sc -= 10.0 if (not cjk and tk.lower().strip(",") in DANGLING) else 0.0
                sc -= 0.6 * abs(k + 1 - n / 2) / n
                if sc > best_sc:
                    best, best_sc = k, sc
            if best is None:
                flush()
            else:
                sentences.append(cur[:best + 1])
                rest = cur[best + 1:]
                cur.clear()
                cur.extend(rest)
    flush()
    # merge tiny fragments ("come.") into the previous beat when the speaker did not pause between them
    merged: list[list[int]] = []
    for ids in sentences:
        if merged and units(ids) <= (2 if cjk else 2):
            prev = merged[-1]
            gap = words[ids[0]]["start"] - words[prev[-1]]["end"]
            ptxt = words[prev[-1]]["text"].strip()
            if gap < 0.35 and not (ptxt[-1:] in SENT_END or (ptxt.endswith(".") and not cjk)) and units(prev) < max_u:
                prev.extend(ids)
                continue
        merged.append(list(ids))
    sentences = merged
    sent_objs = []
    for si, ids in enumerate(sentences):
        ws = [words[i] for i in ids]
        text = ("" if cjk else " ").join(w["text"].strip() for w in ws) if not cjk else "".join(w["text"] for w in ws).strip()
        if not cjk:
            text = "".join(w["text"] for w in ws).strip()
        start, end = ws[0]["start"], ws[-1]["end"]
        voiced = sum(max(0.0, w["end"] - w["start"]) for w in ws) or (end - start) or 1.0
        n_units = units(ids)
        nxt = words[ids[-1] + 1]["start"] if ids[-1] + 1 < len(words) else None
        sent_objs.append({
            "id": si + 1, "w0": ids[0], "w1": ids[-1], "text": text, "start": start, "end": end,
            "rate": round(n_units / max(0.3, end - start), 2), "rate_unit": "char/s" if cjk else "word/s",
            "pause_after": round((nxt - end) if nxt is not None else 0.0, 3),
        })
    if af is not None and words:
        emph = emphasis_scores(af, [(w["start"], w["end"]) for w in words])
        for w, z in zip(words, emph):
            w["emph"] = round(z, 2)
    for s in sent_objs:
        for i in range(s["w0"], s["w1"] + 1):
            words[i]["s"] = s["id"]
    return {"language": language, "cjk": bool(cjk), "words": words, "sentences": sent_objs,
            "refined": bool(refine and af is not None)}


def char_timeline(tr: dict) -> list[dict]:
    """Normalized characters with interpolated times: [{c, t0, t1, w, s}] for anchor search."""
    out = []
    for wi, w in enumerate(tr["words"]):
        n = norm(w["text"])
        if not n:
            continue
        d = (w["end"] - w["start"]) / len(n)
        for k, c in enumerate(n):
            out.append({"c": c, "t0": w["start"] + k * d, "t1": w["start"] + (k + 1) * d, "w": wi, "s": w.get("s", 0)})
    return out
