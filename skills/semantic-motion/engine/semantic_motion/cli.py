"""`sm` command line. Every command takes a video file or a .smproj directory."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path

from . import __version__
from .util import SMError, log, read_json, warn, write_json


def _proj(target, create=True, **settings):
    from .project import Project, find_project

    if target is None:
        return find_project(None)
    return Project.open(target, create=create, **settings)


def _need_analysis(p):
    if not p.has("analysis", "transcript.json") or not p.has("analysis", "vision.npz"):
        raise SMError(f"project not analyzed yet — run: sm analyze {p.root}")


# ------------------------------------------------------------------------------------------ commands
def cmd_doctor(a):
    import importlib
    import platform
    import subprocess

    ok = True
    rep = {"engine": __version__, "python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "cv2", "skia", "mediapipe", "faster_whisper", "jieba", "onnxruntime"):
        try:
            m = importlib.import_module(mod)
            rep[mod] = getattr(m, "__version__", "ok")
        except Exception as e:  # pragma: no cover
            rep[mod] = f"MISSING ({e.__class__.__name__}: {str(e)[:80]})"
            ok = ok and mod in ("onnxruntime",)
    try:
        from .media import which_ffmpeg

        ff, fp = which_ffmpeg()
        v = subprocess.run([ff, "-version"], capture_output=True, text=True).stdout.split("\n")[0]
        filt = subprocess.run([ff, "-hide_banner", "-filters"], capture_output=True, text=True).stdout
        rep["ffmpeg"] = v
        rep["ffmpeg_zscale(HDR)"] = "yes" if " zscale " in filt else "no (HDR sources will not be tone-mapped)"
        enc = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        rep["libx264"] = "yes" if "libx264" in enc else "NO"
        ok = ok and "libx264" in enc
    except Exception as e:
        rep["ffmpeg"] = f"MISSING: {e}"
        ok = False
    try:
        from .render.fonts import FontBook

        rep["fonts"] = FontBook().describe()
    except Exception as e:
        rep["fonts"] = f"error {e}"
    from .paths import MODEL_URLS, models_dir

    md = models_dir()
    rep["models_dir"] = str(md)
    rep["vision_models"] = {k: (md / k).exists() for k in MODEL_URLS}
    wd = md / "whisper"
    rep["whisper_models"] = sorted(p.name for p in wd.glob("models--*")) if wd.exists() else []
    try:
        from .vision import _Models

        m = _Models(use_pose=False, use_text=False)
        m.close()
        rep["mediapipe_runtime"] = "ok"
    except Exception as e:
        rep["mediapipe_runtime"] = f"FAILED: {str(e)[:160]} (Linux: install libEGL/libGLESv2, e.g. libgl1 libegl1 libgles2)"
        ok = False
    rep["ok"] = ok
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0 if ok else 1


def cmd_setup(a):
    from .paths import MODEL_URLS, model_path, models_dir

    for k in MODEL_URLS:
        log(f"model {k}: {model_path(k)}")
    if not a.skip_asr:
        from faster_whisper import WhisperModel  # type: ignore

        log(f"fetching ASR model {a.asr_model} (first time only)…")
        WhisperModel(a.asr_model, device="cpu", compute_type="int8", download_root=str(models_dir() / "whisper"))
    log("setup complete")
    return 0


def cmd_analyze(a):
    from .brief import write_brief
    from .pipeline import analyze

    p = _proj(a.target, platform=a.platform, style=a.style)
    analyze(p, language=a.language, asr_model=a.asr_model, force=a.force, transcript_file=a.transcript)
    b = write_brief(p)
    print(json.dumps({"project": str(p.root), "brief": b, "timings": p.state.get("analysis_timings")}, ensure_ascii=False))
    return 0


def cmd_brief(a):
    from .brief import write_brief

    p = _proj(a.target)
    _need_analysis(p)
    path = write_brief(p)
    if a.print:
        print(Path(path).read_text(encoding="utf-8"))
    else:
        print(path)
    return 0


def cmd_transcript(a):
    p = _proj(a.target)
    tr = p.transcript()
    for s in tr["sentences"]:
        print(f"S{s['id']}\t{s['start']:.2f}-{s['end']:.2f}\t{s['text']}")
        if a.words:
            for i in range(s["w0"], s["w1"] + 1):
                w = tr["words"][i]
                print(f"   {w['id']}\t{w['start']:.3f}-{w['end']:.3f}\tp={w.get('p', 1):.2f}\t{w['text'].strip()}")
    return 0


def cmd_auto_plan(a):
    from .autoplan import auto_plan

    p = _proj(a.target)
    _need_analysis(p)
    if p.has("plan.json") and not a.force:
        raise SMError("plan.json exists (use --force to replace it; the old one stays in plans/)")
    p.snapshot_plan("before auto-plan")
    plan = auto_plan(p, speaker=a.speaker, style=p.settings.get("style", "balanced"))
    p.write_plan(plan, note="auto-plan")
    print(json.dumps({"plan": str(p.p("plan.json")), "elements": len(plan["elements"])}, ensure_ascii=False))
    return 0


def _compile(p, plan=None, write=True):
    from .compiler import compile_project

    return compile_project(p, plan, write=write)


def _print_issues(issues, limit=60):
    for i in issues[:limit]:
        el = f"[{i.get('element')}] " if i.get("element") else ""
        print(f"  {i['level'].upper():5} {i['code']:18} {el}{i['msg']}")


def cmd_validate(a):
    p = _proj(a.target)
    _need_analysis(p)
    plan = read_json(a.plan) if a.plan else p.plan()
    tl, rep = _compile(p, plan, write=False)
    from .qa import pre_qa

    pre = pre_qa(p, tl, rep)
    print(f"elements: {rep['n_elements']}/{rep['n_planned']} placed · compile errors {rep['errors']} · QA errors {pre['errors']} "
          f"warnings {pre['warnings']}")
    _print_issues(pre["checks"])
    for e in tl["elements"]:
        a_ = e.get("anchor") or {}
        print(f"  {e['id']:12} {e['type']:11} in {e['t_in']:6.2f} land {e['t_land']:6.2f} out {e['t_out']:6.2f}  "
              f"anchor “{(a_.get('text') or '')[:16]}”@{a_.get('t', 0):.2f}  box {[int(v) for v in e.get('box', [0, 0, 0, 0])]} {e.get('backing')}")
    return 0 if pre["ok"] else 2


def cmd_compile(a):
    p = _proj(a.target)
    _need_analysis(p)
    tl, rep = _compile(p)
    print(json.dumps({"timeline": str(p.p("build", "timeline.json")), "ok": rep["ok"], "errors": rep["errors"],
                      "warnings": rep["warnings"]}, ensure_ascii=False))
    _print_issues(rep["issues"])
    return 0 if rep["ok"] else 2


def _parse_range(r):
    if not r:
        return 0.0, None
    a, _, b = r.partition("-")
    return float(a or 0), (float(b) if b else None)


def cmd_preview(a):
    import cv2
    import numpy as np

    from .media import EncodeSettings, preview_size
    from .qa import full_reveal_time
    from .render.engine import render_stills, render_video

    p = _proj(a.target)
    _need_analysis(p)
    tl, rep = _compile(p)
    info = p.media()
    if a.range:
        s, e = _parse_range(a.range)
        w, h = preview_size(info, a.width)
        out = str(p.p("renders", f"preview_{s:.1f}-{(e or info.duration):.1f}.mp4"))
        r = render_video(tl, info, out, w, h, start=s, end=e, settings=EncodeSettings(preset="veryfast", crf=23))
        print(json.dumps(r))
        return 0
    if a.at:
        times = [float(x) for x in a.at.split(",")]
    else:
        ids = set(a.elements.split(",")) if a.elements else None
        times = []
        for el in tl["elements"]:
            if el["type"] == "captions" or (ids and el["id"] not in ids):
                continue
            times.append(full_reveal_time(el))
    ims = render_stills(tl, info, times, width=a.width)
    cols = 2 if len(ims) > 1 else 1
    th, tw = ims[0].shape[:2]
    rows = (len(ims) + cols - 1) // cols
    grid = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for k, im in enumerate(ims):
        r, c = divmod(k, cols)
        grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = im[:, :, :3]
    out = str(p.p("qa", "preview.jpg"))
    cv2.imwrite(out, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 86])
    print(json.dumps({"preview": out, "times": [round(t, 2) for t in times]}))
    _print_issues(rep["issues"])
    return 0


# ------------------------------------------------------------------------------------------ build / repair
REPAIRABLE = {"NO_SPACE", "FACE_OCCLUDED", "HEAD_OCCLUDED", "BOX_ON_FACE", "COLLISION", "LOW_CONTRAST", "EMPTY_CARD",
              "TEXT_OCCLUDED", "OUTSIDE_SAFE"}
SMALLER = {"xl": "l", "l": "m", "m": "s", "s": "xs"}


def _repair(plan: dict, checks: list, tl: dict) -> list:
    """Apply deterministic fixes for known failure codes. Returns list of applied repairs."""
    by_id = {e.get("id", f"e{i + 1}"): e for i, e in enumerate(plan.get("elements", []))}
    done = []
    seen = set()
    for c in checks:
        code, eid = c["code"], c.get("element")
        if code not in REPAIRABLE or not eid or eid not in by_id or (eid, code) in seen:
            continue
        seen.add((eid, code))
        e = by_id[eid]
        pl = e.get("placement")
        if isinstance(pl, str):
            pl = {"region": pl}
        pl = dict(pl or {})
        if code in ("FACE_OCCLUDED", "HEAD_OCCLUDED", "BOX_ON_FACE", "COLLISION", "TEXT_OCCLUDED", "OUTSIDE_SAFE") and pl.get("box"):
            pl.pop("box")
            e["placement"] = pl
            done.append({"element": eid, "code": code, "fix": "removed explicit box; solver places it"})
        elif code == "NO_SPACE":
            cur = pl.get("size", "m")
            if cur in SMALLER:
                pl["size"] = SMALLER[cur]
                e["placement"] = pl
                done.append({"element": eid, "code": code, "fix": f"size {cur} -> {pl['size']}"})
            else:
                e["enabled"] = False
                done.append({"element": eid, "code": code, "fix": "disabled (no face-safe space at any size)"})
        elif code == "LOW_CONTRAST":
            if e.get("backing") != "glass":
                e["backing"] = "glass"
                done.append({"element": eid, "code": code, "fix": "backing -> glass"})
        elif code == "EMPTY_CARD":
            items = (e.get("content") or {}).get("items") or (e.get("content") or {}).get("steps") or []
            first = next((it.get("at") for it in items if isinstance(it, dict) and it.get("at")), None)
            if first and e.get("at") != first:
                e["at"] = first
                done.append({"element": eid, "code": code, "fix": f"at -> first item anchor {first!r}"})
    return done


def cmd_build(a):
    from .media import EncodeSettings, preview_size
    from .qa import contact_sheet, determinism, post_qa, pre_qa, timeline_strip, write_report
    from .render.engine import render_overlay, render_video

    p = _proj(a.target)
    _need_analysis(p)
    t_all = time.time()
    plan = p.plan()
    p.snapshot_plan("build")
    repairs = []
    for it in range(4):
        tl, rep = _compile(p, plan)
        pre = pre_qa(p, tl, rep)
        if a.no_repair or it == 3:
            break
        fixes = _repair(plan, [c for c in pre["checks"] if c["level"] in ("error", "warn")], tl)
        if not fixes:
            break
        repairs += [{**f, "round": it + 1} for f in fixes]
        log(f"auto-repair round {it + 1}: {len(fixes)} fix(es)")
    if repairs:
        p.write_plan(plan, note=f"auto-repair ({len(repairs)} fixes)")
    info = p.media()
    preview = bool(a.preview)
    if preview:
        w, h = preview_size(info, a.width or 960)
        out = str(p.p("renders", "preview.mp4"))
        settings = EncodeSettings(preset="veryfast", crf=23)
    else:
        w, h = info.width, info.height
        out = str(p.p("renders", "final.mp4"))
        settings = EncodeSettings(preset=a.preset, crf=a.crf)
    log(f"rendering {'preview' if preview else 'final'} {w}x{h} -> {out}")
    rr = render_video(tl, info, out, w, h, settings=settings)
    post = post_qa(p, tl, out)
    extra = {"render": rr, "repairs": repairs, "timeline_hash": tl["hash"]}
    if a.determinism:
        extra["determinism"] = determinism(p, tl)
    cs = contact_sheet(p, tl, out, str(p.p("qa", "contact_sheet.jpg")))
    ts = timeline_strip(p, tl, str(p.p("qa", "timeline.png")))
    if a.overlay and not preview:
        extra["overlay"] = render_overlay(tl, info, str(p.p("renders", "overlay.mov")))
    report = write_report(p, pre, post, extra)
    ok = pre["ok"] and post["ok"]
    p.mark_stage("build", tl["hash"], ok=ok, output=out, preview=preview, seconds=round(time.time() - t_all, 1))
    summary = {"ok": ok, "output": out, "contact_sheet": cs, "timeline": ts, "report": str(p.p("qa", "report.md")),
               "render": {k: rr[k] for k in ("frames", "drawn_frames", "seconds", "fps", "audio")},
               "pre_qa": {"errors": pre["errors"], "warnings": pre["warnings"], **pre["metrics"]},
               "post_qa": {"errors": post["errors"], "warnings": post["warnings"]}, "repairs": repairs}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    issues = [c for c in pre["checks"] + post["checks"] if c["level"] in ("error", "warn")]
    if issues:
        print("issues:")
        _print_issues(issues)
    return 0 if ok else 2


def cmd_render(a):
    from .media import EncodeSettings, preview_size
    from .render.engine import render_overlay, render_video

    p = _proj(a.target)
    tl = read_json(p.p("build", "timeline.json")) if not a.recompile else _compile(p)[0]
    info = p.media()
    s, e = _parse_range(a.range)
    if a.overlay:
        print(json.dumps(render_overlay(tl, info, str(p.p("renders", "overlay.mov")))))
        return 0
    if a.preview:
        w, h = preview_size(info, a.width or 960)
        settings = EncodeSettings(preset="veryfast", crf=23)
        out = str(p.p("renders", "preview.mp4"))
    else:
        w, h = info.width, info.height
        settings = EncodeSettings(preset=a.preset, crf=a.crf)
        out = str(p.p("renders", "final.mp4"))
    if a.out:
        out = a.out
    print(json.dumps(render_video(tl, info, out, w, h, start=s, end=e, settings=settings)))
    return 0


def cmd_qa(a):
    from .qa import contact_sheet, determinism, post_qa, pre_qa, timeline_strip, write_report

    p = _proj(a.target)
    tl = read_json(p.p("build", "timeline.json"))
    rep = read_json(p.p("build", "compile_report.json")) if p.has("build", "compile_report.json") else None
    pre = pre_qa(p, tl, rep)
    video = a.video or (str(p.p("renders", "final.mp4")) if p.has("renders", "final.mp4") else
                        (str(p.p("renders", "preview.mp4")) if p.has("renders", "preview.mp4") else None))
    post = post_qa(p, tl, video) if video else None
    extra = {}
    if a.determinism:
        extra["determinism"] = determinism(p, tl)
    if video:
        contact_sheet(p, tl, video, str(p.p("qa", "contact_sheet.jpg")))
    timeline_strip(p, tl, str(p.p("qa", "timeline.png")))
    write_report(p, pre, post, extra)
    print(p.p("qa", "report.md").read_text(encoding="utf-8"))
    return 0 if pre["ok"] and (post is None or post["ok"]) else 2


def cmd_export(a):
    p = _proj(a.target)
    dest = Path(a.dest) if a.dest else p.p("export")
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for rel in ("renders/final.mp4", "renders/overlay.mov", "plan.json", "build/timeline.json", "brief.md", "qa/report.md",
                "qa/report.json", "qa/contact_sheet.jpg", "qa/timeline.png", "analysis/transcript.json"):
        if p.has(*rel.split("/")):
            shutil.copy2(p.p(*rel.split("/")), dest / Path(rel).name)
            copied.append(Path(rel).name)
    stem = p.source.stem
    if (dest / "final.mp4").exists():
        os.replace(dest / "final.mp4", dest / f"{stem}.motion.mp4")
        copied = [f"{stem}.motion.mp4" if c == "final.mp4" else c for c in copied]
    print(json.dumps({"export": str(dest), "files": copied}, ensure_ascii=False))
    return 0


def cmd_status(a):
    p = _proj(a.target, create=False)
    st = p.state
    out = {"project": str(p.root), "source": st["source"]["path"], "settings": st["settings"],
           "stages": {k: {kk: vv for kk, vv in v.items() if kk != "key"} for k, v in st.get("stages", {}).items()},
           "plan": p.has("plan.json"), "snapshots": len(p.snapshots()),
           "final": p.has("renders", "final.mp4"), "preview": p.has("renders", "preview.mp4")}
    if p.has("qa", "report.json"):
        r = read_json(p.p("qa", "report.json"))
        out["last_qa"] = {"pre_errors": r["pre"]["errors"], "post_errors": (r.get("post") or {}).get("errors")}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


def cmd_undo(a):
    p = _proj(a.target, create=False)
    prev = p.undo()
    print(f"restored {prev.name} -> plan.json")
    return 0


def cmd_snapshot(a):
    p = _proj(a.target, create=False)
    s = p.snapshot_plan(a.note or "manual")
    print(s)
    return 0


def _field_diff(x, y, path: str = "") -> list[str]:
    """Readable field-level differences between two JSON values."""
    def short(v):
        t = json.dumps(v, ensure_ascii=False)
        return t if len(t) <= 90 else t[:87] + "..."
    if isinstance(x, dict) and isinstance(y, dict):
        out = []
        for k in list(x) + [k for k in y if k not in x]:
            p = f"{path}.{k}" if path else str(k)
            if k not in y:
                out.append(f"{p}: {short(x[k])} -> (removed)")
            elif k not in x:
                out.append(f"{p}: (none) -> {short(y[k])}")
            elif x[k] != y[k]:
                out.extend(_field_diff(x[k], y[k], p))
        return out
    if isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        out = []
        for i, (u, v) in enumerate(zip(x, y)):
            if u != v:
                out.extend(_field_diff(u, v, f"{path}[{i}]"))
        return out
    return [f"{path}: {short(x)} -> {short(y)}"]


def cmd_diff(a):
    p = _proj(a.target, create=False)
    snaps = p.snapshots()
    if len(snaps) < 1:
        raise SMError("no snapshots")
    B = read_json(p.p("plan.json")) if not a.b else read_json(p.p("plans", a.b))
    if a.a:
        ref = p.p("plans", a.a)
    else:
        # latest snapshot that differs from the current plan (the plan may or may not have been built yet)
        ref = next((s for s in reversed(snaps) if read_json(s) != B), snaps[-1])
    A = read_json(ref)
    print(f"plan diff: {Path(ref).name} -> {'plan.json' if not a.b else a.b}")
    ea = {e.get("id"): e for e in A.get("elements", [])}
    eb = {e.get("id"): e for e in B.get("elements", [])}
    n = 0
    for k in sorted(set(ea) | set(eb), key=str):
        if k not in ea:
            print(f"+ {k} ({eb[k].get('type')}): {json.dumps(eb[k].get('content', {}), ensure_ascii=False)[:160]}")
            n += 1
        elif k not in eb:
            print(f"- {k} ({ea[k].get('type')})")
            n += 1
        elif ea[k] != eb[k]:
            for line in _field_diff(ea[k], eb[k]):
                print(f"~ {k}.{line}")
            n += 1
    for line in _field_diff({k: v for k, v in A.items() if k != "elements"}, {k: v for k, v in B.items() if k != "elements"}):
        print(f"~ {line}")
        n += 1
    if not n:
        print("no changes")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sm", description="Semantic Motion — semantic motion graphics for talking-head video")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(fn=fn)
        return sp

    add("doctor", cmd_doctor, "check dependencies, models, fonts")
    sp = add("setup", cmd_setup, "download models")
    sp.add_argument("--asr-model", default="large-v3-turbo")
    sp.add_argument("--skip-asr", action="store_true")
    sp = add("analyze", cmd_analyze, "probe + ASR + audio + vision analysis, writes brief.md")
    sp.add_argument("target")
    sp.add_argument("--language", default=None)
    sp.add_argument("--asr-model", default=None)
    sp.add_argument("--platform", default=None)
    sp.add_argument("--style", default=None)
    sp.add_argument("--transcript", default=None, help="use an existing transcript (.srt/.vtt or words .json) instead of ASR")
    sp.add_argument("--force", action="store_true")
    sp = add("brief", cmd_brief, "(re)write brief.md")
    sp.add_argument("target", nargs="?")
    sp.add_argument("--print", action="store_true")
    sp = add("transcript", cmd_transcript, "print sentences (and word ids)")
    sp.add_argument("target", nargs="?")
    sp.add_argument("--words", action="store_true")
    sp = add("auto-plan", cmd_auto_plan, "heuristic baseline plan.json")
    sp.add_argument("target", nargs="?")
    sp.add_argument("--speaker", default=None, help="'Name|Role' for a lower third")
    sp.add_argument("--force", action="store_true")
    sp = add("validate", cmd_validate, "compile + pre-render QA without writing")
    sp.add_argument("target", nargs="?")
    sp.add_argument("--plan", default=None)
    sp = add("compile", cmd_compile, "plan.json -> build/timeline.json")
    sp.add_argument("target", nargs="?")
    sp = add("preview", cmd_preview, "stills of every element (or --at times / --range clip)")
    sp.add_argument("target", nargs="?")
    sp.add_argument("--at", default=None)
    sp.add_argument("--elements", default=None)
    sp.add_argument("--range", default=None)
    sp.add_argument("--width", type=int, default=960)
    sp = add("build", cmd_build, "compile + auto-repair + render + QA + contact sheet")
    sp.add_argument("target", nargs="?")
    sp.add_argument("--preview", action="store_true")
    sp.add_argument("--width", type=int, default=None)
    sp.add_argument("--crf", type=int, default=18)
    sp.add_argument("--preset", default="medium")
    sp.add_argument("--no-repair", action="store_true")
    sp.add_argument("--determinism", action="store_true")
    sp.add_argument("--overlay", action="store_true", help="also export graphics-only ProRes 4444 with alpha")
    sp = add("render", cmd_render, "render from build/timeline.json")
    sp.add_argument("target", nargs="?")
    sp.add_argument("--preview", action="store_true")
    sp.add_argument("--recompile", action="store_true")
    sp.add_argument("--range", default=None)
    sp.add_argument("--width", type=int, default=None)
    sp.add_argument("--crf", type=int, default=18)
    sp.add_argument("--preset", default="medium")
    sp.add_argument("--overlay", action="store_true")
    sp.add_argument("--out", default=None)
    sp = add("qa", cmd_qa, "run QA on the current timeline/render")
    sp.add_argument("target", nargs="?")
    sp.add_argument("--video", default=None)
    sp.add_argument("--determinism", action="store_true")
    sp = add("export", cmd_export, "collect deliverables")
    sp.add_argument("target", nargs="?")
    sp.add_argument("--dest", default=None)
    sp = add("status", cmd_status, "project state")
    sp.add_argument("target", nargs="?")
    sp = add("undo", cmd_undo, "restore the previous plan snapshot")
    sp.add_argument("target", nargs="?")
    sp = add("snapshot", cmd_snapshot, "store plan.json as a numbered snapshot")
    sp.add_argument("target", nargs="?")
    sp.add_argument("--note", default="")
    sp = add("diff", cmd_diff, "diff plan.json against the previous snapshot")
    sp.add_argument("target", nargs="?")
    sp.add_argument("--a", default=None)
    sp.add_argument("--b", default=None)
    a = ap.parse_args(argv)
    try:
        return int(a.fn(a) or 0)
    except SMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
