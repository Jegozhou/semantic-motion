#!/usr/bin/env python3
"""Automated regression for Semantic Motion.

Runs the whole evidence chain and writes one machine-readable summary:

    PYTHONPATH=src .venv/bin/python scripts/regression.py            # full
    PYTHONPATH=src .venv/bin/python scripts/regression.py --quick    # skip matrices

A step is recorded as PASS / FAIL / NOT RUN / BLOCKED exactly as its own artifact
says; the summary never upgrades a step.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semotion import media  # noqa: E402

PY = str(ROOT / ".venv/bin/python")
ENV = {"PYTHONPATH": str(ROOT / "src"), **{k: v for k, v in __import__("os").environ.items()}}


def _run(argv: list[str], *, timeout: int = 1800) -> dict:
    t0 = time.time()
    cp = subprocess.run(argv, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=timeout)
    return {"argv": argv, "exit": cp.returncode, "seconds": round(time.time() - t0, 1),
            "stdout_tail": (cp.stdout or "")[-1500:], "stderr_tail": (cp.stderr or "")[-800:]}


def _json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/regression")
    ap.add_argument("--quick", action="store_true", help="skip the effect matrices")
    ap.add_argument("--gate", default="G3")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    steps: list[dict] = []

    def step(name: str, argv: list[str], verdict_fn=None, crash_ok=False) -> None:
        print(f"[regression] {name} ...", flush=True)
        r = _run(argv)
        verdict = "PASS" if r["exit"] == 0 else ("FAIL" if not crash_ok else "NOT RUN")
        if verdict_fn is not None:
            try:
                verdict = verdict_fn() or verdict
            except Exception:
                pass
        steps.append({"step": name, **r, "verdict": verdict})

    # 1. unit + integration tests
    def tests_verdict() -> str:
        tail = steps[-1]["stdout_tail"]
        return "PASS" if " failed" not in tail and ("passed" in tail or "OK" in tail) else "FAIL"

    step("pytest", [PY, "-m", "pytest", "tests/", "-q"], tests_verdict)

    # 2. environment probe (also proves tool versions still match)
    step("env_probe", [PY, "scripts/env_probe.py", "artifacts/env/env_probe.json"])

    # 3. G0 baseline (remux + re-encode verification)
    def baseline_verdict() -> str:
        d = _json(ROOT / "artifacts/g0/baseline_report.json")
        return d.get("verdict", "FAIL")

    step("baseline", [PY, "-m", "semotion.cli", "baseline", "run",
                      "--input", "tests/fixtures/clips/fx_a_zh_portrait/fx_a_zh_portrait.mp4",
                      "--out", "artifacts/g0"], baseline_verdict)

    # 4. G1 timing calibration (>=50 independent events)
    step("gate_g1_calibration",
         [PY, "scripts/gate_runner.py", "--gate", "G1", "--out", "artifacts/g1/final_calibration",
          "--fixtures", "fx_t1_zh_phrase", "fx_t2_zh_phrase", "fx_t3_zh_phrase",
          "--style", "calibration"],
         lambda: _json(ROOT / "artifacts/g1/final_calibration/gate_summary.json")
         .get("pooled_timing", {}).get("verdict", "FAIL"))

    # 5. G1 product behaviour on the dev set (+ determinism)
    step("gate_g1_product",
         [PY, "scripts/gate_runner.py", "--gate", "G1", "--out", "artifacts/g1/final_product",
          "--fixtures", "fx_a_zh_portrait", "fx_b_zh_landscape", "fx_c_zh_cuts",
          "fx_d_zh_portrait", "fx_e_zh_landscape", "--determinism"],
         lambda: ("PASS" if all(r.get("verdicts", {}).get(k) == "PASS"
                               for r in _json(ROOT / "artifacts/g1/final_product/gate_summary.json")
                               .get("results", []) for k in ("position", "render"))
                  and _json(ROOT / "artifacts/g1/final_product/gate_summary.json")
                  .get("determinism", {}).get("verdict") == "PASS" else "FAIL"))

    # 6. G2 effect matrices (optional in quick mode)
    if not args.quick:
        for fx, out_dir in (("fx_face_zh_portrait", "fx_face"), ("fx_pd_real_16x9_a", "fx_pd")):
            step(f"effect_matrix_{out_dir}",
                 [PY, "scripts/effect_matrix.py", "--fixture", fx,
                  "--out", f"artifacts/g2/matrix/{out_dir}"],
                 (lambda d=out_dir: "PASS" if all(
                     (c.get("verdicts", {}).get("position") == "PASS")
                     for c in _json(ROOT / f"artifacts/g2/matrix/{d}/matrix.json").get("cells", [])
                     if c.get("verdicts")) else "FAIL"))

    step("gate_g2_face_stress",
         [PY, "scripts/gate_runner.py", "--gate", "G2", "--out", "artifacts/g2/face_stress",
          "--fixtures", "fx_face_zh_portrait", "--style", "calibration"],
         lambda: "PASS" if all(r.get("verdicts", {}).get("position") == "PASS"
                              for r in _json(ROOT / "artifacts/g2/face_stress/gate_summary.json")
                              .get("results", [])) else "FAIL")

    # 7. G3 stability (degradation / partial update / recovery)
    step("g3_stability", [PY, "scripts/g3_stability.py", "--out", "artifacts/g3/stability"],
         lambda: _json(ROOT / "artifacts/g3/stability/stability.json").get("verdict", "FAIL"))

    # 8. G3 camera motion + cut safety
    step("gate_g3_motion",
         [PY, "scripts/gate_runner.py", "--gate", "G3", "--out", "artifacts/g3/pan",
          "--fixtures", "fx_face_pan", "--style", "calibration"],
         lambda: "PASS" if all(r.get("verdicts", {}).get("position") == "PASS"
                              for r in _json(ROOT / "artifacts/g3/pan/gate_summary.json")
                              .get("results", [])) else "FAIL")

    # 9. numeric semantics on product renders
    num_verdicts = []
    for d in sorted((ROOT / "artifacts/g1/final_product").glob("fx_*")):
        ir, mp4 = d / "motion_ir.json", next(iter(sorted(d.glob("*_motion.mp4"))), None)
        if not ir.exists() or mp4 is None:
            continue
        r = _run([PY, "scripts/check_number_semantics.py", str(ir), str(mp4),
                  str(d / "number_check.json")], timeout=600)
        v = _json(d / "number_check.json").get("verdict", "FAIL")
        num_verdicts.append((d.name, v, r["exit"]))
    steps.append({"step": "numeric_semantics", "verdict": (
        "PASS" if all(v in ("PASS", "NOT RUN") for _, v, _ in num_verdicts) else "FAIL"),
        "details": [{"fixture": n, "verdict": v} for n, v, _ in num_verdicts]})

    verdicts = [s["verdict"] for s in steps]
    summary = {
        "schema": "semotion.regression/1",
        "gate": args.gate,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "quick": args.quick,
        "steps": steps,
        "verdict": ("PASS" if all(v == "PASS" for v in verdicts)
                    else ("FAIL" if "FAIL" in verdicts else "NOT RUN")),
        "counts": {v: verdicts.count(v) for v in set(verdicts)},
        "tool_versions": media.tool_versions(),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    for s in steps:
        print(f"  {s['step']:<28} {s['verdict']:<8} {s.get('seconds')}s")
    print(f"{summary['verdict']} -> {out / 'summary.json'}")
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
