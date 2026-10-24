"""Analysis stage orchestration with caching (each sub-stage is keyed by its inputs)."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import audio as audio_mod
from .media import probe, read_audio
from .paths import models_dir
from .project import Project, env_threads
from .style import safe_rect
from .util import SMError, log, read_json, sha1_obj, warn, write_json


def _parse_srt(path: Path) -> list[dict]:
    """SRT/VTT -> pseudo word list (each cue split into characters/words with interpolated times)."""
    txt = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", txt.replace("\r", ""))
    words = []
    tre = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
    for b in blocks:
        m = tre.search(b)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        t0 = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        t1 = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        body = " ".join(l.strip() for l in b[m.end():].split("\n") if l.strip())
        body = re.sub(r"<[^>]+>", "", body)
        if not body:
            continue
        from .text import is_cjk

        toks = re.findall(r"[\u3400-\u9fff\uf900-\ufaff][，。！？、；：,.!?]*|[^\s\u3400-\u9fff]+\s*", body)
        toks = [t for t in toks if t.strip()]
        if not toks:
            continue
        d = (t1 - t0) / len(toks)
        for k, t in enumerate(toks):
            words.append({"text": (t if is_cjk(t[0]) else " " + t.strip()), "start": t0 + k * d, "end": t0 + (k + 1) * d, "p": 1.0})
    return words


def analyze(proj: Project, language: Optional[str] = None, asr_model: Optional[str] = None, force: bool = False,
            skip_vision: bool = False, transcript_file: Optional[str] = None, sample_fps: float = 4.0) -> dict:
    t_all = time.time()
    st = proj.settings
    if language:
        st["language"] = language
    if asr_model:
        st["asr_model"] = asr_model
    proj.save()
    src = proj.source
    src_key = proj.state["source"]["sha1_head"]
    timings = {}

    # ---- probe
    t0 = time.time()
    info = probe(src)
    write_json(proj.p("analysis", "media.json"), info.to_dict())
    if not info.has_audio:
        raise SMError("the video has no audio track — Semantic Motion needs speech to anchor motion graphics")
    if info.vfr_suspected:
        warn("variable frame rate suspected — rendering will normalize to constant frame rate "
             f"{info.fps_str}; audio stays untouched")
    timings["probe"] = round(time.time() - t0, 2)

    # ---- audio features
    t0 = time.time()
    key = sha1_obj({"src": src_key, "v": 2})
    if force or not proj.stage_fresh("audio", key, ["analysis/audio.npz"]):
        pcm = read_audio(src, 16000)
        af = audio_mod.analyze(pcm, 16000)
        np.savez_compressed(proj.p("analysis", "audio.npz"), **af.to_npz_dict())
        proj.mark_stage("audio", key, duration=af.duration, speech_level=af.speech_level, noise_floor=af.noise_floor)
    else:
        af = proj.audio()
        pcm = None
    timings["audio"] = round(time.time() - t0, 2)

    # ---- ASR
    t0 = time.time()
    tf_key = None
    if transcript_file:
        tf = Path(transcript_file)
        tf_key = sha1_obj({"f": str(tf.resolve()), "m": tf.stat().st_mtime})
    key = sha1_obj({"src": src_key, "model": st["asr_model"], "lang": st.get("language"), "tf": tf_key, "v": 2})
    if force or not proj.stage_fresh("asr", key, ["analysis/asr_raw.json"]):
        if transcript_file:
            tf = Path(transcript_file)
            if tf.suffix.lower() in (".srt", ".vtt"):
                words = _parse_srt(tf)
                lang = st.get("language") or ("zh" if any("\u4e00" <= c <= "\u9fff" for w in words for c in w["text"]) else "en")
            else:
                j = read_json(tf)
                words = j["words"] if isinstance(j, dict) else j
                lang = (j.get("language") if isinstance(j, dict) else None) or st.get("language") or "zh"
            backend = f"import:{tf.name}"
        else:
            from .asr import transcribe

            if pcm is None:
                pcm = read_audio(src, 16000)
            log(f"ASR on {len(pcm) / 16000:.1f}s of audio (model {st['asr_model']}) — this is the slowest step on CPU")
            words, lang = transcribe(pcm, st["asr_model"], st.get("language"), models_dir=str(models_dir() / "whisper"),
                                     threads=env_threads())
            backend = "whisper"
        write_json(proj.p("analysis", "asr_raw.json"), {"language": lang, "model": st["asr_model"], "backend": backend,
                                                       "words": words, "elapsed": round(time.time() - t0, 1)})
        proj.mark_stage("asr", key, language=lang, n_words=len(words), elapsed=round(time.time() - t0, 1))
    raw = read_json(proj.p("analysis", "asr_raw.json"))
    timings["asr"] = round(time.time() - t0, 2)

    # ---- transcript (cheap; always rebuilt)
    from .asr import build_transcript

    tr = build_transcript(raw["words"], raw["language"], af, refine=True)
    tr["asr_model"] = raw.get("model")
    write_json(proj.p("analysis", "transcript.json"), tr)

    # ---- vision
    t0 = time.time()
    if not skip_vision:
        key = sha1_obj({"src": src_key, "fps": sample_fps, "v": 3})
        if force or not proj.stage_fresh("vision", key, ["analysis/vision.npz"]):
            from .vision import analyze as vanalyze

            vd = vanalyze(info, sample_fps=sample_fps)
            vd.save(str(proj.p("analysis", "vision.npz")))
            proj.mark_stage("vision", key, shots=len(vd.shots), samples=int(len(vd.t)))
        write_shots(proj)
    timings["vision"] = round(time.time() - t0, 2)
    timings["total"] = round(time.time() - t_all, 2)
    proj.state["analysis_timings"] = timings
    proj.save()
    log(f"analysis done in {timings['total']}s: {timings}")
    return timings


def write_shots(proj: Project) -> list:
    from .vision import shot_composition

    vd = proj.vision()
    info = proj.media()
    safe = safe_rect(proj.settings.get("platform", "auto"), info.orientation)
    out = []
    for si, s in enumerate(vd.shots):
        comp = shot_composition(vd, si, safe)
        s = dict(s)
        s["speaker"] = bool((s.get("face_ratio", 0) > 0.3 and s.get("person_cov", 0) > 0.03) or s.get("person_cov", 0) > 0.2)
        out.append({**s, **comp})
    write_json(proj.p("analysis", "shots.json"), out)
    return out
