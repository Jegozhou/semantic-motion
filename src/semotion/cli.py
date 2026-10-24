"""Semantic Motion — command line entry point.

    PYTHONPATH=src .venv/bin/python -m semotion.cli <command> [options]

Exit codes are part of the contract (see docs/INTERFACES.md §1):
0 ok, 2 usage, 3 IR invalid, 4 media/input, 5 generation, 6 layout, 7 render, 8 verify.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import media

EXIT_USAGE = 2
EXIT_IR = 3
EXIT_MEDIA = 4
EXIT_GEN = 5
EXIT_LAYOUT = 6
EXIT_RENDER = 7
EXIT_VERIFY = 8


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dump(obj, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ subcommands

def cmd_fixtures(a) -> int:
    from . import fixtures

    res = fixtures.build_all(a.specs, a.out, only=a.only)
    for r in res:
        print(f"{r['fixture_id']}: {r['duration_s']:.2f}s {r['width']}x{r['height']} "
              f"{len(r['units'])} units -> {r['abs_path']}")
    return 0


def cmd_baseline(a) -> int:
    from . import baseline

    rep = baseline.run_g0_baseline(Path(a.input), Path(a.out))
    print(json.dumps({k: rep[k] for k in ("verdict", "outputs")}, ensure_ascii=False, indent=2))
    return 0 if rep["verdict"] == "PASS" else EXIT_MEDIA


def cmd_asr(a) -> int:
    from . import asr

    t = asr.transcribe(Path(a.video), Path(a.out), model=a.model, device=a.device,
                       language=a.language, force=a.force)
    print(f"transcript: {a.out}  segments={len(t['segments'])} words={len(t['words'])}")
    return 0


def cmd_vision(a) -> int:
    from . import vision

    v = vision.analyze(Path(a.video), Path(a.out), fps_scan=a.fps_scan, force=a.force)
    print(f"vision: {a.out}  faces={len(v['faces'])} tracks={len(v['face_tracks'])} "
          f"shots={len(v['shots'])}")
    return 0


def cmd_anchors(a) -> int:
    from . import anchors

    tr = _load(a.transcript)
    an = anchors.classify(tr)
    if a.video:
        an = anchors.refine_onsets(Path(a.video), an)
    if not an.get("duration_s"):
        an["duration_s"] = tr.get("source", {}).get("duration_s")
    _dump(an, Path(a.out))
    print(f"anchors: {a.out}  n={len(an['anchors'])}")
    return 0


def cmd_plan(a) -> int:
    from . import semantic

    an = _load(a.anchors)
    vis = _load(a.vision) if a.vision else {}
    w, h = (int(x) for x in a.canvas.lower().split("x"))
    plan = semantic.decide(an, vis, {"width": w, "height": h, "fps": a.fps},
                           style=a.style)
    _dump(plan, Path(a.out))
    print(json.dumps(semantic.summarize(plan), ensure_ascii=False))
    return 0


def cmd_ir(a) -> int:
    from . import ir as ir_mod

    tr = _load(a.transcript)
    vis = _load(a.vision) if a.vision else {}
    an = _load(a.anchors)
    pl = _load(a.plan)
    doc = ir_mod.build(tr, vis, an, pl, Path(a.source),
                       {"width": a.width, "height": a.height, "fps": a.fps})
    errs = ir_mod.validate(doc)
    if errs:
        print("IR INVALID:", *errs[:20], sep="\n  ")
        return EXIT_IR
    ir_mod.dump(doc, Path(a.out))
    print(f"IR ok: {a.out}  elements={len(doc['elements'])} tracks={len(doc['tracks'])} "
          f"hash={doc['provenance']['ir_sha256'][:16]}")
    return 0


def cmd_render(a) -> int:
    from . import ir as ir_mod
    from . import render as render_mod

    doc = ir_mod.load(Path(a.ir))
    errs = ir_mod.validate(doc)
    if errs:
        print("IR INVALID (refusing to render):", *errs[:10], sep="\n  ")
        return EXIT_IR
    man = render_mod.render(doc, Path(a.source), Path(a.out), preview=a.preview)
    print(json.dumps({"output": man["output"], "commands": len(man.get("commands", []))},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_verify(a) -> int:
    from . import verify as verify_mod

    doc = _load(a.ir)
    man = _load(a.render)
    vis = _load(a.vision) if a.vision else {}
    m = verify_mod.metrics(doc, man, source=Path(a.source), rendered=Path(a.rendered),
                           truth_json=Path(a.truth) if a.truth else None,
                           vision=vis, gate=a.gate,
                           work=Path(a.out).parent)
    _dump(m, Path(a.out))
    print(json.dumps({"verdicts": m["verdicts"], "timing": {
        k: m["timing"].get(k) for k in ("samples", "p50_ms", "p90_ms", "max_ms", "verdict")}},
        ensure_ascii=False, indent=2))
    return 0 if all(v == "PASS" for v in m["verdicts"].values()) else EXIT_VERIFY


# ------------------------------------------------------------------- end-to-end

def cmd_run(a) -> int:
    """Full pipeline: input video -> transcript -> vision -> anchors -> plan -> IR
    -> render -> metrics. Uses fixture ground truth when the input is a fixture."""
    from . import anchors, asr, ir as ir_mod, render as render_mod, semantic, verify, vision

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(exist_ok=True)
    work = outdir / "work"
    work.mkdir(exist_ok=True)

    if a.fixture:
        clip_dir = Path(a.clips_root) / a.fixture
        video = clip_dir / f"{a.fixture}.mp4"
        truth = clip_dir / "ground_truth.json"
    else:
        video = Path(a.input)
        truth = Path(a.truth) if a.truth else None
    if not video.exists():
        print(f"input not found: {video}")
        return EXIT_MEDIA

    steps: list[dict] = []
    started = _now()

    def step(name: str, fn):
        t0 = datetime.now(timezone.utc)
        try:
            res = fn()
            steps.append({"step": name, "status": "ok",
                          "seconds": round((datetime.now(timezone.utc) - t0).total_seconds(), 2)})
            return res
        except Exception as e:  # noqa: BLE001 - pipeline boundary
            steps.append({"step": name, "status": "failed", "error": str(e)})
            raise

    try:
        tr_path = outdir / "transcript.json"
        transcript = step("asr", lambda: asr.transcribe(video, tr_path, model=a.model,
                                                       device=a.device, language=a.language,
                                                       force=a.force))
        vis_path = outdir / "vision.json"
        vis = step("vision", lambda: vision.analyze(video, vis_path, force=a.force))

        def _anchors():
            an = anchors.classify(transcript)
            an = anchors.refine_onsets(video, an)
            an["duration_s"] = float(transcript["source"]["duration_s"])
            _dump(an, outdir / "anchors.json")
            return an

        anchors_doc = step("anchors", _anchors)

        canvas = {"width": transcript["source"]["width"], "height": transcript["source"]["height"],
                  "fps": transcript["source"]["fps"]}
        plan = step("plan", lambda: semantic.decide(anchors_doc, vis, canvas, style=a.style))
        _dump(plan, outdir / "plan.json")

        ir_doc = step("ir", lambda: ir_mod.build(transcript, vis, anchors_doc, plan, video, canvas))
        errs = ir_mod.validate(ir_doc)
        if errs:
            _dump({"errors": errs}, outdir / "ir_errors.json")
            print("IR INVALID:", *errs[:20], sep="\n  ")
            return EXIT_IR
        ir_mod.dump(ir_doc, outdir / "motion_ir.json")

        rendered = outdir / f"{video.stem}_motion.mp4"
        man = step("render", lambda: render_mod.render(ir_doc, video, rendered, preview=a.preview))
        _dump(man, outdir / "render_manifest.json")

        m = step("verify", lambda: verify.metrics(
            ir_doc, man, source=video, rendered=rendered,
            truth_json=truth if (truth and Path(truth).exists()) else None,
            vision=vis, gate=a.gate, work=outdir))
        _dump(m, outdir / "metrics.json")

        summary = {
            "gate": a.gate,
            "started_utc": started, "finished_utc": _now(),
            "input": {"video": str(video),
                      "sha256": media.sha256_file(video),
                      "truth": str(truth) if truth and Path(truth).exists() else None},
            "artifacts": {
                "transcript": str(tr_path), "vision": str(vis_path),
                "anchors": str(outdir / "anchors.json"), "plan": str(outdir / "plan.json"),
                "motion_ir": str(outdir / "motion_ir.json"),
                "render": str(rendered), "render_manifest": str(outdir / "render_manifest.json"),
                "metrics": str(outdir / "metrics.json"),
            },
            "decisions": semantic.summarize(plan),
            "verdicts": m["verdicts"],
            "timing": {k: m["timing"].get(k) for k in
                       ("samples", "p50_ms", "p90_ms", "max_ms", "source", "verdict")},
            "steps": steps,
            "exit": 0 if all(v == "PASS" for v in m["verdicts"].values()) else EXIT_VERIFY,
        }
        _dump(summary, outdir / "run_summary.json")
        print(json.dumps(summary["verdicts"], ensure_ascii=False))
        print(f"run summary: {outdir / 'run_summary.json'}")
        return summary["exit"]
    except media.MediaError as e:
        _dump({"error": e.as_dict(), "steps": steps}, outdir / "error.json")
        print("MEDIA ERROR:", e)
        return EXIT_MEDIA
    except Exception as e:  # noqa: BLE001
        _dump({"error": repr(e), "traceback": traceback.format_exc(), "steps": steps},
              outdir / "error.json")
        print("ERROR:", e)
        return EXIT_GEN


# ---------------------------------------------------------------------- argparse

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="semotion", description="Semantic Motion pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fixtures", help="build synthetic calibration fixtures")
    fs = f.add_subparsers(dest="action", required=True)
    fb = fs.add_parser("build")
    fb.add_argument("--specs", default="tests/fixtures/specs")
    fb.add_argument("--out", default="tests/fixtures/clips")
    fb.add_argument("--only", nargs="*")
    fb.set_defaults(func=cmd_fixtures)

    b = sub.add_parser("baseline", help="G0 pass-through / re-encode baseline")
    bs = b.add_subparsers(dest="action", required=True)
    br = bs.add_parser("run")
    br.add_argument("--input", required=True)
    br.add_argument("--out", required=True)
    br.set_defaults(func=cmd_baseline)

    a_ = sub.add_parser("asr", help="transcribe with word-level timing")
    as_ = a_.add_subparsers(dest="action", required=True)
    ar = as_.add_parser("run")
    ar.add_argument("--video", required=True)
    ar.add_argument("--out", required=True)
    ar.add_argument("--model", default="medium")
    ar.add_argument("--device", default="cpu")
    ar.add_argument("--language", default="zh")
    ar.add_argument("--force", action="store_true")
    ar.set_defaults(func=cmd_asr)

    v = sub.add_parser("vision", help="faces / shots / safe zones")
    vs = v.add_subparsers(dest="action", required=True)
    vr = vs.add_parser("analyze")
    vr.add_argument("--video", required=True)
    vr.add_argument("--out", required=True)
    vr.add_argument("--fps-scan", type=float, default=5.0)
    vr.add_argument("--force", action="store_true")
    vr.set_defaults(func=cmd_vision)

    an = sub.add_parser("anchors", help="semantic anchors")
    ans = an.add_subparsers(dest="action", required=True)
    anb = ans.add_parser("build")
    anb.add_argument("--transcript", required=True)
    anb.add_argument("--video")
    anb.add_argument("--out", required=True)
    anb.set_defaults(func=cmd_anchors)

    pl = sub.add_parser("plan", help="director decisions")
    pl.add_argument("--anchors", required=True)
    pl.add_argument("--vision")
    pl.add_argument("--canvas", default="1080x1920")
    pl.add_argument("--fps", type=float, default=25.0)
    pl.add_argument("--style", default="smart_balanced")
    pl.add_argument("--out", required=True)
    pl.set_defaults(func=cmd_plan)

    i = sub.add_parser("ir", help="Motion IR")
    is_ = i.add_subparsers(dest="action", required=True)
    ib = is_.add_parser("build")
    ib.add_argument("--transcript", required=True)
    ib.add_argument("--vision")
    ib.add_argument("--anchors", required=True)
    ib.add_argument("--plan", required=True)
    ib.add_argument("--source", required=True)
    ib.add_argument("--width", type=int, required=True)
    ib.add_argument("--height", type=int, required=True)
    ib.add_argument("--fps", type=float, required=True)
    ib.add_argument("--out", required=True)
    ib.set_defaults(func=cmd_ir)

    r = sub.add_parser("render", help="render the IR")
    r.add_argument("--ir", required=True)
    r.add_argument("--source", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--preview", action="store_true")
    r.set_defaults(func=cmd_render)

    ve = sub.add_parser("verify", help="metrics")
    ve.add_argument("--ir", required=True)
    ve.add_argument("--render", required=True)
    ve.add_argument("--source", required=True)
    ve.add_argument("--rendered", required=True)
    ve.add_argument("--truth")
    ve.add_argument("--vision")
    ve.add_argument("--gate", default="G1")
    ve.add_argument("--out", required=True)
    ve.set_defaults(func=cmd_verify)

    ru = sub.add_parser("run", help="end-to-end pipeline")
    ru.add_argument("--input")
    ru.add_argument("--fixture")
    ru.add_argument("--clips-root", default="tests/fixtures/clips")
    ru.add_argument("--truth")
    ru.add_argument("--outdir", required=True)
    ru.add_argument("--model", default="medium")
    ru.add_argument("--device", default="cpu")
    ru.add_argument("--language", default="zh")
    ru.add_argument("--style", default="smart_balanced")
    ru.add_argument("--gate", default="G1")
    ru.add_argument("--preview", action="store_true")
    ru.add_argument("--force", action="store_true")
    ru.set_defaults(func=cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except media.MediaError as e:
        print("MEDIA ERROR:", e)
        return EXIT_MEDIA


if __name__ == "__main__":
    sys.exit(main())
