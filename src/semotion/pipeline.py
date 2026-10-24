"""End-to-end v2 pipeline: video -> perception -> alignment -> design -> render.

One command produces the deliverable plus its editable engineering files:

    python -m semotion.pipeline run --video talk.mp4 --out out/talk

Artifacts in the output directory:

    final.mp4                 the deliverable (original audio kept)
    final_overlay.mp4         overlay-only pass (QA attribution)
    motion_ir.json            editable Motion IR (semotion.motion-ir/2)
    captions.json/.srt/.ass   frame-accurate captions
    collision.md              storyboard/contact sheet
    qa_metrics.json           objective checks (when --verify)
    run_summary.json          everything needed to reproduce the run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from . import design2, director, media, policy as policy_mod, render2, theme2 as T

SCHEMA = "semotion.run/2"

# words and phrases that carry a claim worth emphasising (Chinese-first, with EN)
CONTRAST = ("但是", "但", "其实", "不是", "而是", "所以", "因为", "关键", "最重要",
            "真正", "本质", "反而", "however", "actually", "but ", "because")
CONCLUSION = ("就是", "意味着", "说明", "结论", "等于", "一定是", "永远是",
              "means", "the point is")
SUPERLATIVE = ("最", "第一", "唯一", "永远", "一定", "从来", "绝对", "千万",
               "best", "worst", "always", "never")
DISCOURSE_START = ("但是", "但", "所以", "因为", "其实", "然后", "而且", "如果", "就是",
                   "可能", "我的", "每一年", "网球", "奥运会", "它", "这", "那", "你",
                   "我", "等待", "终于", "后来", "最后", "首先", "其次")
ENUM = ("第一", "第二", "第三", "第四", "第五", "首先", "其次", "然后", "最后",
        "一是", "二是", "三是", "一方面", "另一方面", "first", "second", "third",
        "finally", "next")


@dataclass
class RunConfig:
    video: str
    out: str
    transcript: str | None = None
    perception: str | None = None
    anchors: str | None = None
    decisions: str | None = None
    captions_mode: str = "auto"        # auto | off | file
    captions_path: str | None = None
    draft: bool = False
    verify: bool = True
    auto_repair: bool = True
    max_repair_rounds: int = 2
    max_elements: int = 8
    min_gap_s: float = 0.9
    language: str | None = None
    force: bool = False
    policy: str | None = None
    director_mode: str = "auto"     # auto | llm | rules | off
    brief_only: bool = False
    aligner: str = "auto"           # auto | energy | ctc (auto = CTC if cached, else DP)
    consensus: bool = False         # also run CTC and record timing agreement


def _log(msg: str) -> None:
    print(f"[semotion] {msg}", flush=True)


def _load(path: str | Path | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _write(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------- anchors
def _rms_db(samples, rate: int, t0: float, t1: float) -> float:
    import numpy as _np
    a = max(0, int(t0 * rate)); b = min(len(samples), int(t1 * rate))
    if b - a < rate * 0.02:
        return -90.0
    seg = _np.asarray(samples[a:b], dtype=_np.float32)
    rms = float(_np.sqrt(_np.mean(seg * seg) + 1e-12))
    return 20.0 * _np.log10(rms + 1e-9)


def auto_anchors(alignment: dict, *, max_elements: int = 8,
                 min_gap_s: float = 0.9, audio=None,
                 rate: int = 16000) -> dict:
    """Deterministic semantic director: only claims, numbers, lists get motion.

    Phrases are cut on pauses (not on ASR segment ids — whisper splits words like
    结/束 across segments), then capped at a readable length. Classification is
    conservative on purpose: plain narration gets no overlay.
    """
    chars = alignment.get("chars") or []
    duration = float((alignment.get("source") or {}).get("duration_s") or 0.0)
    levels = []
    if audio is not None:
        for c in chars:
            levels.append(_rms_db(audio, rate, float(c["start_s"]), float(c["end_s"])))
        speech = float(sorted(levels)[int(len(levels) * 0.5)]) if levels else -20.0
        silent_db = speech - 10.0
    else:
        speech, silent_db = -20.0, -34.0

    def real_pause(i: int) -> tuple[bool, float]:
        if i <= 0 or i >= len(chars):
            return False, -90.0
        t0 = float(chars[i - 1]["end_s"]); t1 = float(chars[i]["start_s"])
        if t1 - t0 < 0.10:
            return False, -90.0
        if audio is None:
            return (t1 - t0) >= 0.26, -min(0.26, t1 - t0)
        db = _rms_db(audio, rate, t0, t1)
        return db < silent_db, -db

    phrases = _split_phrases(chars, max_chars=13, real_pause=real_pause)

    cands: list[dict] = []
    for ph in phrases:
        text = ph["text"]
        if len(text) < 4 or ph["end_s"] - ph["start_s"] < 0.45:
            continue
        kind, score = None, 0.0
        if _has_number(text):
            kind, score = "stat", 0.9
        elif any(text.startswith(e) for e in ENUM):
            kind, score = "list", 0.84
        elif any(k in text for k in CONTRAST):
            kind, score = "keyword", 0.78
        elif any(k in text for k in SUPERLATIVE):
            kind, score = "keyword", 0.72
        elif any(k in text for k in CONCLUSION) and len(text) <= 16:
            kind, score = "card", 0.66
        if not kind:
            continue
        score -= max(0.0, (len(text) - 14) * 0.015)
        cands.append({"kind": kind, "text": text, "start_s": ph["start_s"],
                      "end_s": ph["end_s"], "score": round(score, 3),
                      "segment_id": ph["segment_id"], "word_ids": ph["word_ids"]})

    # repeated parallel structure -> one list ("每一年… / 每一年… / …")
    merged: list[dict] = []
    for c in cands:
        if merged:
            prev = merged[-1]
            rep = _common_prefix(prev["text"], c["text"])
            if (len(rep) >= 2 and c["start_s"] - prev["end_s"] < 2.6
                    and len(prev.get("items") or [prev["text"]]) < 4
                    and len(prev["text"]) <= 14):
                prev["kind"] = "list"
                prev.setdefault("items", [prev["text"]]).append(c["text"])
                prev["end_s"] = c["end_s"]
                prev["score"] = min(0.9, prev["score"] + 0.05)
                continue
        if c["kind"] == "list":
            c["items"] = [c["text"]]
        merged.append(c)

    merged.sort(key=lambda c: (-c["score"], c["start_s"]))
    kept: list[dict] = []
    for c in merged:
        if len(kept) >= max_elements:
            break
        if any(abs(c["start_s"] - k["start_s"]) < min_gap_s for k in kept):
            continue
        kept.append(c)
    kept.sort(key=lambda c: c["start_s"])
    anchors = []
    for i, c in enumerate(kept):
        anchors.append({
            "id": f"a{i:02d}", "kind": c["kind"], "text": c["text"],
            "start_s": round(c["start_s"], 3), "end_s": round(c["end_s"], 3),
            "confidence": c["score"], "source": "RULE_V2", "segment_id": c["segment_id"],
            "word_ids": c.get("word_ids") or [],
            "items": c.get("items") or [],
        })
    return {
        "schema": "semotion.anchors/1",
        "source_sha256": (alignment.get("source") or {}).get("sha256"),
        "transcript_sha256": None,
        "duration_s": duration, "language": (alignment.get("source") or {}).get("language"),
        "anchors": anchors,
        "provenance": {"method": "semotion.pipeline.auto_anchors (deterministic rules)",
                       "n_anchors": len(anchors), "truth_class": "MACHINE"},
    }


def _common_prefix(a: str, b: str) -> str:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return a[:n]


def _split_phrases(chars: list[dict], *, max_chars: int = 13, real_pause=None) -> list[dict]:
    """Cut the character stream into readable phrases.

    A cut is accepted only where the audio is actually silent — Whisper leaves
    fake 300 ms gaps inside words (结/束), and splitting there truncates meaning.
    """
    if not chars:
        return []
    cuts = [0]
    for i in range(1, len(chars)):
        if real_pause and real_pause(i)[0]:
            cuts.append(i)
    cuts.append(len(chars))
    rough = [chars[a:b] for a, b in zip(cuts, cuts[1:]) if b > a]
    phrases: list[dict] = []
    for chunk in rough:
        phrases.extend(_cap_length(chunk, max_chars, real_pause))
    return phrases


def _cap_length(chunk: list[dict], max_chars: int, real_pause=None) -> list[dict]:
    out: list[dict] = []
    cur = list(chunk)
    while len(cur) > max_chars:
        lo = max(5, int(max_chars * 0.62))
        hi = min(len(cur) - 3, max_chars)
        best, best_quiet = None, -1e9
        for i in range(lo, hi + 1):
            silent, quiet = real_pause(i) if real_pause else (False, 0.0)
            score = (100.0 if silent else 0.0) + quiet * 0.05 + (i / max_chars) * 0.4
            nxt = "".join(c.get("text") or "" for c in cur[i:i + 2])
            if nxt[:2] in DISCOURSE_START or nxt[:1] in DISCOURSE_START:
                score += 1.2
            if cur[i].get("segment_id") != cur[i - 1].get("segment_id"):
                score += 0.35
            if score > best_quiet:
                best, best_quiet = i, score
        best = best or max_chars
        out.append(_phrase(cur[:best]))
        cur = cur[best:]
    if cur:
        out.append(_phrase(cur))
    return out


def _phrase(chars: list[dict], segments: dict | None = None) -> dict:
    text = "".join(c.get("text") or "" for c in chars)
    seg_ids = [c.get("segment_id") for c in chars if c.get("segment_id")]
    return {"text": text, "start_s": float(chars[0]["start_s"]),
            "end_s": float(chars[-1]["end_s"]),
            "segment_id": seg_ids[0] if seg_ids else None,
            "word_ids": [c.get("token_id") for c in chars if c.get("token_id")],
            "n_chars": len(chars)}


def _has_number(text: str) -> bool:
    import re
    if re.search(r"\d+(?:\.\d+)?\s*(%|％|倍|万|亿|千|k|K|M|元|块|美元|分|秒|分钟|小时|天|年|个月)?", text):
        return True
    return bool(re.search(r"[零〇一二两三四五六七八九十百千万]+\s*(成|折|%|％)", text))


def _timing_consensus(energy: dict, ctc: dict) -> dict:
    """Compare two independent alignments char-by-char (no truth, only agreement)."""
    a = energy.get("chars") or []
    b = ctc.get("chars") or []
    deltas: list[float] = []
    per_char: list[dict] = []
    for i in range(min(len(a), len(b))):
        if str(a[i].get("text") or "") != str(b[i].get("text") or ""):
            continue
        d = (float(b[i].get("start_s") or 0.0) - float(a[i].get("start_s") or 0.0)) * 1000.0
        deltas.append(abs(d))
        per_char.append({"i": i, "char": a[i].get("text"), "delta_ms": round(d, 1)})
    if not deltas:
        return {"schema": "semotion.timing-consensus/1", "n": 0,
                "note": "no comparable characters"}
    deltas_sorted = sorted(deltas)
    n = len(deltas_sorted)
    return {
        "schema": "semotion.timing-consensus/1",
        "methods": [energy.get("method", {}).get("name"), ctc.get("method", {}).get("name")],
        "n": n,
        "median_abs_ms": round(deltas_sorted[n // 2], 1),
        "p90_abs_ms": round(deltas_sorted[min(n - 1, int(n * 0.9))], 1),
        "within_50ms_rate": round(sum(1 for d in deltas if d <= 50) / n, 4),
        "per_char": per_char,
        "honesty": "agreement between two machines is not accuracy; no human labels exist here",
    }


# --------------------------------------------------------------------- helpers
def _run_module(module: str, argv: list[str], log: Path, timeout: int = 3600) -> int:
    cmd = [sys.executable, "-m", module] + argv
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")
        fh.flush()
        return subprocess.call(cmd, stdout=fh, stderr=fh, timeout=timeout)


def _ensure_asr(cfg: RunConfig, out: Path, log: Path) -> dict:
    if cfg.transcript:
        doc = _load(cfg.transcript)
        if doc:
            return doc
    cache = out / "transcript.json"
    if cache.exists() and not cfg.force:
        return json.loads(cache.read_text(encoding="utf-8"))
    from . import asr
    _log("transcribing (faster-whisper, cached model)")
    asr.transcribe(Path(cfg.video), cache)
    return json.loads(cache.read_text(encoding="utf-8"))


def _ensure_perception(cfg: RunConfig, out: Path, log: Path) -> dict:
    if cfg.perception:
        doc = _load(cfg.perception)
        if doc:
            doc["_path"] = str(Path(cfg.perception).resolve())
            return doc
    cache = out / "perception.json"
    if cache.exists() and not cfg.force:
        doc = json.loads(cache.read_text(encoding="utf-8"))
        doc["_path"] = str(cache.resolve())
        return doc
    _log("perception: person mask / pose / hands / burned-in text (Apple Vision)")
    rc = _run_module("semotion.perception",
                     ["--video", cfg.video, "--out", str(cache),
                      "--mask-dir", str(out / "masks")], log)
    if rc != 0:
        _log(f"perception failed rc={rc}; continuing with vision fallback")
        return {}
    doc = json.loads(cache.read_text(encoding="utf-8"))
    doc["_path"] = str(cache.resolve())
    return doc


# ------------------------------------------------------------------------ main
def run(cfg: RunConfig) -> dict:
    t0 = time.time()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "pipeline.log"
    log.write_text("", encoding="utf-8")
    pol = policy_mod.load(cfg.policy)
    _log(f"policy: {pol.name} v{pol.version} ({pol.sha256})")
    info = media.info(cfg.video)
    _log(f"source {info.width}x{info.height} @ {info.fps:.2f} fps, {info.duration:.2f}s")

    transcript = _ensure_asr(cfg, out, log)
    _log(f"transcript: {len(transcript.get('words') or [])} words, "
         f"{len(transcript.get('segments') or [])} segments")

    # precise character timing
    from . import align as align_mod
    alignment_path = out / "alignment.json"
    if alignment_path.exists() and not cfg.force:
        alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    else:
        lang = (cfg.language or transcript.get("language") or "zh")
        from . import forced as forced_mod
        use_ctc = cfg.aligner == "ctc"
        if cfg.aligner == "auto":
            try:
                use_ctc = bool(forced_mod.available(lang))
            except Exception:
                use_ctc = False
            if not use_ctc:
                _log("aligner: CTC model not cached — using the envelope DP "
                     "(pass --aligner ctc to download it)")
        if use_ctc:
            _log(f"aligning characters with the CTC acoustic model ({lang})")
            alignment = forced_mod.align_video(cfg.video, transcript, language=lang,
                                               allow_download=(cfg.aligner == "ctc"))
            _write(alignment, alignment_path)
            _log(f"aligner: {alignment.get('method', {}).get('name')} "
                 f"chars={len(alignment.get('chars') or [])}")
        else:
            _log("aligning characters to the audio envelope")
            alignment = align_mod.align_video(cfg.video, transcript)
            _write(alignment, alignment_path)

    # optional consensus: an independent acoustic alignment, used only as a confidence signal
    if cfg.consensus and not (alignment.get("method") or {}).get("name", "").startswith("ctc"):
        try:
            from . import forced as forced_mod
            _log("consensus: cross-checking timing with the CTC model")
            ctc = forced_mod.align_video(cfg.video, transcript, allow_download=False,
                                         allow_fallback=True)
            consensus = _timing_consensus(alignment, ctc)
            alignment["consensus"] = consensus
            _write(alignment, alignment_path)
            _write(consensus, out / "alignment_consensus.json")
            _log(f"consensus: n={consensus['n']} median|Δ|={consensus['median_abs_ms']} ms "
                 f"within50={consensus['within_50ms_rate']:.0%}")
        except Exception as exc:  # model absent or failed — never fatal
            _log(f"consensus skipped: {exc}")

    perception = _ensure_perception(cfg, out, log)
    _log(f"perception: {len(perception.get('frames') or [])} samples, "
         f"{len((perception.get('regions') or {}).get('quiet_zones') or [])} quiet zones")

    # captions
    captions_doc = None
    if cfg.captions_mode != "off":
        from . import captions as cap_mod
        try:
            if cfg.captions_path:
                captions_doc = _load(cfg.captions_path)
            else:
                captions_doc = cap_mod.build_captions(alignment, fps=info.fps,
                                                      width=info.width, height=info.height)
            if captions_doc:
                _write(captions_doc, out / "captions.json")
                try:
                    cap_mod.write_srt(captions_doc, out / "captions.srt")
                    cap_mod.write_ass(captions_doc, out / "captions.ass")
                except Exception as exc:  # optional exports
                    _log(f"caption export skipped: {exc}")
                _log(f"captions: {len(captions_doc.get('cues') or [])} cues")
        except Exception as exc:
            _log(f"captions unavailable: {exc}")

    audio = None
    try:
        from . import align as _al
        audio, _rate = _al.load_audio(cfg.video)
    except Exception as exc:
        _log(f"audio envelope unavailable ({exc}); pause detection falls back to gaps")
        _rate = 16000

    # anchors
    anchors_doc = _load(cfg.anchors)
    if anchors_doc:
        anchors = anchors_doc.get("anchors") or []
        _log(f"anchors: {len(anchors)} (supplied)")
    else:
        auto = auto_anchors(alignment, max_elements=cfg.max_elements,
                            min_gap_s=cfg.min_gap_s, audio=audio, rate=_rate)
        anchors = auto["anchors"]
        _write(auto, out / "anchors.json")
        _log(f"anchors: {len(anchors)} (deterministic candidates)")

    # director: the model (or the host agent) chooses what deserves motion
    source_meta = {"path": str(Path(cfg.video).resolve()),
                   "sha256": media.sha256_file(cfg.video),
                   "width": info.width, "height": info.height, "fps": info.fps,
                   "duration_s": info.duration, "frames": info.frame_count}
    canvas = {"width": info.width, "height": info.height, "fps": info.fps}
    brief = director.build_brief(source=source_meta, canvas=canvas,
                                 transcript=transcript, alignment=alignment,
                                 anchors=anchors, perception=perception,
                                 policy=pol, style="smart_balanced")
    director.write_brief(brief, out / "director_brief.json")
    decisions = _load(cfg.decisions)
    director_info: dict = {"mode": cfg.director_mode, "source": "none"}
    if decisions:
        try:
            decisions = director.validate_decisions(decisions, brief)
            director_info = {"mode": "supplied", "source": "file",
                             "decisions": len(decisions.get("decisions") or [])}
            _log(f"director: {director_info['decisions']} supplied decisions (validated)")
        except director.DecisionError as exc:
            _log(f"director: supplied decisions rejected ({exc}); falling back")
            decisions = None
            director_info = {"mode": cfg.director_mode, "source": "file_rejected",
                             "error": str(exc)}
    if decisions is None and cfg.director_mode in ("auto", "llm"):
        generated = director.direct(brief, out_dir=out)
        if generated:
            decisions = generated
            director_info = {"mode": "llm", "source": "model",
                             "model": (generated.get("director") or {}).get("model"),
                             "decisions": len(generated.get("decisions") or []),
                             "prompt_summary": generated.get("prompt_summary")}
            _log(f"director: model chose {director_info['decisions']} elements")
        elif cfg.director_mode == "llm":
            raise SystemExit("director: --director llm requested but no usable model "
                             "response (set SM_LLM_BASE_URL/SM_LLM_API_KEY/SM_LLM_MODEL)")
        else:
            director_info = {"mode": "rules", "source": "fallback",
                             "reason": "no model configured"}
    if cfg.brief_only:
        summary = {"schema": SCHEMA, "video": source_meta,
                   "brief": str(out / "director_brief.json"),
                   "anchors": len(anchors), "policy": pol.as_dict(),
                   "elapsed_s": round(time.time() - t0, 1)}
        _write(summary, out / "run_summary.json")
        _log(f"brief written -> {out / 'director_brief.json'} ({len(anchors)} candidates)")
        return summary

    source = source_meta
    frames = render2.grab_frames(cfg.video, [info.duration * f for f in
                                             (0.08, 0.25, 0.45, 0.65, 0.85)])
    dcfg = design2.DesignConfig(language=cfg.language or transcript.get("language") or "zh",
                                policy=pol)
    ir = design2.design(source=source, canvas=canvas,
                        anchors=([] if cfg.director_mode == "off" else anchors),
                        transcript=transcript, perception=perception,
                        vision=_load(out / "vision.json"), decisions=decisions,
                        frames=frames, cfg=dcfg)
    _write(ir, out / "motion_ir.json")
    _log(f"design: {len(ir['elements'])} elements, coverage "
         f"{ir['layout_audit']['coverage']:.0%}")

    # render + verify (+ deterministic repair loop)
    repair_history: list[dict] = []
    metrics = None
    manifest = None
    rounds = 0
    while True:
        _log("rendering" + (" (draft)" if cfg.draft else " (full quality)")
             + (f" [round {rounds + 1}]" if rounds else ""))
        manifest = render2.render(
            ir, cfg.video, out / "final.mp4", captions=captions_doc,
            perception=perception, overlay_pass=True, draft=cfg.draft,
            log_path=out / "render.log",
            progress=lambda i, n: print(f"\r  {i}/{n}", end="", flush=True))
        print()
        render2.contact_sheet(ir, out / "final.mp4", out / "storyboard.png")
        if not (cfg.verify and (out / "final_overlay.mp4").exists()):
            break
        _log("verifying rendered output")
        qa_dir = out / "qa"
        qa_dir.mkdir(exist_ok=True)
        _run_module("semotion.verify2", [
            "--ir", str(out / "motion_ir.json"), "--source", cfg.video,
            "--rendered", str(out / "final.mp4"),
            "--masks", str(out / "perception.json") if (out / "perception.json").exists()
            else str(cfg.perception or ""),
            "--overlay-only", str(out / "final_overlay.mp4"),
            "--ink-only", str(out / "final_ink.mp4"),
            "--out-dir", str(qa_dir)], log)
        metrics = _load(qa_dir / "qa_metrics.json")
        verdict = (metrics or {}).get("verdict")
        _log(f"verify: {verdict}")
        if verdict == "PASS" or not cfg.auto_repair or rounds >= cfg.max_repair_rounds:
            break
        # deterministic repair: rewrite the IR, then re-render and re-verify
        repair_dir = out / "repair"
        repair_dir.mkdir(exist_ok=True)
        _log("repairing failed layout (deterministic ladder)")
        _run_module("semotion.repair", [
            "--ir", str(out / "motion_ir.json"),
            "--metrics", str(qa_dir / "qa_metrics.json"),
            "--masks", str(out / "perception.json") if (out / "perception.json").exists()
            else str(cfg.perception or ""),
            "--out-dir", str(repair_dir)], log)
        repaired = _load(repair_dir / "motion_ir.repaired.json")
        rep_doc = _load(repair_dir / "repairs.json") or {}
        change = {"round": rounds + 1,
                  "before_verdict": verdict,
                  "summary": rep_doc.get("summary"),
                  "actions": [{"element": r.get("element_id"), "action": r.get("action"),
                               "reason": r.get("reason")} for r in rep_doc.get("repairs") or []]}
        repair_history.append(change)
        if not repaired or (rep_doc.get("summary") or {}).get("repaired", 0) == 0:
            _log("repair found nothing it can fix deterministically; stopping")
            break
        if not (out / "motion_ir.initial.json").exists():
            (out / "motion_ir.initial.json").write_text(
                json.dumps(ir, ensure_ascii=False, indent=2), encoding="utf-8")
        ir = repaired
        _write(ir, out / "motion_ir.json")
        rounds += 1

    summary = {
        "schema": SCHEMA,
        "video": source, "canvas": canvas,
        "policy": pol.as_dict(),
        "director": director_info,
        "elements": len(ir["elements"]),
        "coverage": ir["layout_audit"]["coverage"],
        "captions": len((captions_doc or {}).get("cues") or []),
        "perception_used": bool(perception.get("frames")),
        "render": manifest["output"] if manifest else None,
        "elapsed_s": round(time.time() - t0, 1),
        "repair_rounds": rounds,
        "repair_history": repair_history,
        "artifacts": {}, "checks": {},
    }
    for name in ("final.mp4", "final_overlay.mp4", "final_ink.mp4", "motion_ir.json",
                 "motion_ir.initial.json", "captions.json", "captions.srt", "captions.ass",
                 "storyboard.png", "perception.json", "alignment.json", "anchors.json",
                 "qa/qa_metrics.json", "repair/repairs.json"):
        p = out / name
        if p.exists():
            summary["artifacts"][name] = str(p)

    if metrics:
        summary["checks"] = {
            "verdict": metrics.get("verdict"),
            "checks": {k: v.get("verdict") for k, v in (metrics.get("checks") or {}).items()},
        }
        shutil.copy(out / "qa" / "qa_metrics.json", out / "qa_metrics.json")
        if (out / "qa" / "contact_sheet.png").exists():
            shutil.copy(out / "qa" / "contact_sheet.png", out / "qa_contact_sheet.png")
    _write(summary, out / "run_summary.json")
    _log(f"done in {summary['elapsed_s']}s -> {out / 'final.mp4'}")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="semotion.pipeline",
                                 description="Semantic Motion v2 end-to-end pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process one talking-head video")
    r.add_argument("--video", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--transcript")
    r.add_argument("--perception")
    r.add_argument("--anchors")
    r.add_argument("--decisions")
    r.add_argument("--captions", default="auto", choices=["auto", "off", "file"])
    r.add_argument("--captions-path")
    r.add_argument("--draft", action="store_true")
    r.add_argument("--aligner", default="auto", choices=["auto", "energy", "ctc"],
                   help="auto: CTC when cached else envelope DP; energy: fast DP; "
                        "ctc: acoustic model (downloads ~1.3 GB if missing)")
    r.add_argument("--consensus", action="store_true",
                   help="also run the CTC aligner and record timing agreement")
    r.add_argument("--no-verify", action="store_true")
    r.add_argument("--policy", help="design policy JSON (default policies/smart_balanced.json)")
    r.add_argument("--director", default="auto", choices=["auto", "llm", "rules", "off"],
                   help="auto: model if configured else rules; llm: require a model; "
                        "rules: deterministic only; off: no elements")
    r.add_argument("--brief-only", action="store_true",
                   help="emit director_brief.json for the host agent and stop")
    r.add_argument("--no-repair", action="store_true",
                   help="verify but do not auto-repair failed layouts")
    r.add_argument("--max-repair-rounds", type=int, default=2)
    r.add_argument("--max-elements", type=int, default=8)
    r.add_argument("--min-gap", type=float, default=0.9)
    r.add_argument("--language")
    r.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    cfg = RunConfig(video=args.video, out=args.out, transcript=args.transcript,
                    perception=args.perception, anchors=args.anchors,
                    decisions=args.decisions, captions_mode=args.captions,
                    captions_path=args.captions_path, draft=args.draft,
                    verify=not args.no_verify, auto_repair=not args.no_repair,
                    max_repair_rounds=args.max_repair_rounds,
                    aligner=args.aligner, consensus=args.consensus,
                    policy=args.policy, director_mode=args.director,
                    brief_only=args.brief_only, max_elements=args.max_elements,
                    min_gap_s=args.min_gap, language=args.language, force=args.force)
    summary = run(cfg)
    verdict = (summary.get("checks") or {}).get("verdict")
    if verdict == "FAIL":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
