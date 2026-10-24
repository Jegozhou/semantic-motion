#!/usr/bin/env python3
"""Gate acceptance runner — produces the evidence a gate report needs.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/gate_runner.py \
        --gate G1 --out artifacts/g1 \
        --fixtures fx_a_zh_portrait fx_b_zh_landscape fx_c_zh_cuts \
        --determinism

For every fixture it runs the real CLI end-to-end, then:
  * pools timing errors across fixtures (the independent-event sample the spec asks for),
  * re-renders one fixture and compares decoded frames (reproducibility),
  * extracts review frames for human inspection,
  * writes `gate_summary.json` with explicit PASS / FAIL / NOT RUN / BLOCKED.

Nothing is marked PASS unless the underlying metrics file says so.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semotion import media, verify  # noqa: E402

PY = str(ROOT / ".venv/bin/python")
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([PY, "-m", "semotion.cli", *args], cwd=ROOT, env=ENV,
                          capture_output=True, text=True)


def _read(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def extract_frames(video: Path, out_dir: Path, times: list[float]) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for t in times:
        dst = out_dir / f"{video.stem}_t{t:05.2f}.png"
        p = subprocess.run([media.FFMPEG, "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video),
                            "-frames:v", "1", "-vf", "scale=540:-2", str(dst)],
                           capture_output=True, text=True)
        if p.returncode == 0 and dst.exists():
            made.append(str(dst))
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="G1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixtures", nargs="+", required=True)
    ap.add_argument("--clips-root", default="tests/fixtures/clips")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--style", default="smart_balanced",
                    help="director policy: smart_balanced | calibration")
    ap.add_argument("--determinism", action="store_true")
    ap.add_argument("--frames", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    pooled_errors: list[dict] = []
    started = datetime.now(timezone.utc)

    for fid in args.fixtures:
        clip = Path(args.clips_root) / fid / f"{fid}.mp4"
        if not clip.exists():
            results.append({"fixture": fid, "status": "BLOCKED", "reason": "clip missing"})
            continue
        rundir = out / fid
        cp = run_cli(["run", "--fixture", fid, "--clips-root", args.clips_root,
                      "--outdir", str(rundir), "--gate", args.gate,
                      "--model", args.model, "--language", args.language,
                      "--style", args.style])
        summary = _read(rundir / "run_summary.json")
        metrics = _read(rundir / "metrics.json")
        entry = {"fixture": fid, "exit": cp.returncode,
                 "stdout_tail": (cp.stdout or "")[-500:], "stderr_tail": (cp.stderr or "")[-800:],
                 "run_summary": summary is not None, "metrics": metrics is not None}
        if metrics:
            entry["verdicts"] = metrics["verdicts"]
            entry["timing"] = {k: metrics["timing"].get(k) for k in
                               ("samples", "p50_ms", "p90_ms", "max_ms", "source", "verdict")}
            for row in metrics["timing"].get("rows", []):
                if row.get("error_ms") is not None:
                    pooled_errors.append({"fixture": fid, "effect": row["effect"],
                                          "error_ms": row["error_ms"]})
            rendered = Path(rundir) / f"{fid}_motion.mp4"
            entry["rendered"] = str(rendered) if rendered.exists() else None
            if rendered.exists():
                entry["review_frames"] = extract_frames(
                    rendered, out / "frames", [t for t in _frame_times(metrics, args.frames)])
        results.append(entry)

    # ---- pooled timing distribution (independent events across fixtures)
    pooled = None
    if pooled_errors:
        e = sorted(x["error_ms"] for x in pooled_errors)
        pooled = {
            "n": len(e),
            "p50_ms": round(statistics.median(e), 2),
            "p90_ms": round(e[max(0, int(round(0.9 * (len(e) - 1))))], 2),
            "max_ms": round(max(e), 2),
            "threshold_p90_ms": verify.P90_THRESHOLD_MS,
            "verdict": "PASS" if e[max(0, int(round(0.9 * (len(e) - 1))))] <= verify.P90_THRESHOLD_MS else "FAIL",
            "samples_needed_for_p90_claim": 50,
            "claim_scope": ("P90 claim" if len(e) >= 50 else
                            "exploratory only (<50 independent events)"),
            "rows": pooled_errors,
        }

    # ---- reproducibility on the first fixture that produced a render
    determinism = None
    if args.determinism:
        first = next((r for r in results if r.get("rendered")), None)
        if first:
            fid = first["fixture"]
            rundir = out / fid
            ir_path = rundir / "motion_ir.json"
            clip = Path(args.clips_root) / fid / f"{fid}.mp4"
            again = rundir / f"{fid}_motion_rerender.mp4"
            cp = run_cli(["render", "--ir", str(ir_path), "--source", str(clip),
                          "--out", str(again)])
            if again.exists():
                determinism = verify.determinism(_read(ir_path), Path(first["rendered"]), again)
                determinism["fixture"] = fid
                determinism["rerender_stdout"] = (cp.stdout or "")[-400:]
            else:
                determinism = {"verdict": "FAIL", "error": (cp.stderr or "")[-800:]}

    verdicts = [r.get("verdicts", {}).get("timing") for r in results]
    summary = {
        "gate": args.gate,
        "style": args.style,
        "started_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fixtures": args.fixtures,
        "results": results,
        "pooled_timing": pooled,
        "determinism": determinism,
        "tool_versions": media.tool_versions(),
    }
    (out / "gate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")

    print(f"gate summary -> {out / 'gate_summary.json'}")
    for r in results:
        print(f"  {r['fixture']:<24} {r.get('verdicts', r.get('status'))}")
    if pooled:
        print(f"  pooled timing  n={pooled['n']} p50={pooled['p50_ms']}ms p90={pooled['p90_ms']}ms "
              f"max={pooled['max_ms']}ms -> {pooled['verdict']} ({pooled['claim_scope']})")
    if determinism:
        print(f"  determinism    {determinism.get('verdict')} "
              f"(frame_hashes_equal={determinism.get('frame_hashes_equal')})")
    return 0


def _frame_times(metrics: dict, count: int) -> list[float]:
    """Pick review times inside the rendered effect windows."""
    rows = metrics.get("timing", {}).get("rows", [])
    times = [r["window_start_s"] + 0.12 for r in rows[:count]]
    if not times:
        times = [1.0, 3.0, 5.0][:count]
    return times


if __name__ == "__main__":
    raise SystemExit(main())
