#!/usr/bin/env python3
"""G2 effect matrix: render each effect family on its own, then all four together.

For every (fixture × effect) cell this script builds an IR containing *only* that
effect family, renders it, verifies timing/position/render, and (for numbers) OCR
checks the settled value. It also runs a `combo` cell using the real director.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/effect_matrix.py \
        --out artifacts/g2/matrix/fx_a --fixture fx_a_zh_portrait
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semotion import anchors as anchors_mod, asr, ir as ir_mod, media, render as render_mod  # noqa: E402
from semotion import semantic, verify, vision as vision_mod  # noqa: E402

EFFECT_KINDS = {
    "keyword": {"keyword", "comparison", "process", "conclusion"},
    "number": {"number"},
    "list": {"list_item"},
    "card": {"card"},
}


def _tile(anchors: list[dict], duration: float, effect: str, *, items_map=None) -> list[dict]:
    """One decision per anchor, windows tiled so they never overlap."""
    rows = sorted([a for a in anchors if a.get("kind") in EFFECT_KINDS[effect]],
                  key=lambda a: float(a.get("start_s", 0.0)))
    decisions = []
    for i, a in enumerate(rows):
        start = float(a["start_s"])
        nxt = float(rows[i + 1]["start_s"]) if i + 1 < len(rows) else duration
        end = min(float(a["end_s"]) + 0.45, nxt - 0.04, duration - 0.05)
        if end - start < 0.32:
            continue
        items = (items_map or {}).get(a["id"])
        decisions.append({
            "id": f"{effect[:2]}{i + 1:02d}",
            "anchor_id": a["id"], "anchor_ids": [a["id"]],
            "effect": effect, "emphasis": "high",
            "reason": f"matrix_{effect}", "confidence": float(a.get("confidence", 0.8)),
            "decision_model": "matrix-v1",
            "content": {"text": semantic.to_simplified(a.get("text", "")), "items": items},
            "window": {"start_s": round(start, 6), "end_s": round(end, 6)},
        })
    return decisions


def _list_items(anchors: list[dict], transcript: dict) -> tuple[list[dict], dict]:
    """Group consecutive list_item anchors; compose row text with the director helper."""
    rows = sorted([a for a in anchors if a.get("kind") == "list_item"],
                  key=lambda a: float(a.get("start_s", 0.0)))
    groups: list[list[dict]] = []
    for a in rows:
        if groups and float(a["start_s"]) - float(groups[-1][-1]["end_s"]) <= semantic.LIST_MERGE_GAP_S:
            groups[-1].append(a)
        else:
            groups.append([a])
    items_map: dict[str, list[str]] = {}
    for g in groups:
        text = semantic._compose_list_items(g, anchors, transcript)
        items_map[g[0]["id"]] = [semantic.to_simplified(t) for t in text]
    # keep only the first anchor of each group as the list element
    firsts = [{"kind": "list_item", "id": g[0]["id"], "text": g[0].get("text", ""),
               "start_s": g[0]["start_s"], "end_s": g[-1]["end_s"],
               "confidence": min(float(x.get("confidence", 0.8)) for x in g)} for g in groups]
    # the last group's element must end at its own last anchor, not the next group
    for f, g in zip(firsts, groups):
        f["end_s"] = float(g[-1]["end_s"])
    return firsts, items_map


def run_cell(fixture: str, effect: str, *, clips_root: Path, out_dir: Path) -> dict:
    clip = clips_root / fixture / f"{fixture}.mp4"
    truth = clips_root / fixture / "ground_truth.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    mi = media.info(clip)
    canvas = {"width": mi.width, "height": mi.height, "fps": mi.fps}

    transcript = asr.transcribe(clip, out_dir / "transcript.json",
                               model="medium", device="cpu",
                               language="en" if fixture.startswith("fx_pd") else "zh")
    vis = vision_mod.analyze(clip, out_dir / "vision.json")
    anchors_doc = anchors_mod.classify(transcript)
    anchors_doc = anchors_mod.refine_onsets(clip, anchors_doc)
    anchors_doc["duration_s"] = float(transcript["source"]["duration_s"])
    (out_dir / "anchors.json").write_text(json.dumps(anchors_doc, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    duration = float(transcript["source"]["duration_s"])
    if effect == "combo":
        plan = semantic.decide(anchors_doc, vis, canvas, style=semantic.STYLE_CALIBRATION,
                              transcript=transcript)
        plan["style"] = "calibration_combo"
    else:
        if effect == "list":
            rows, items_map = _list_items(anchors_doc["anchors"], transcript)
        else:
            rows, items_map = anchors_doc["anchors"], {}
        decisions = _tile(rows, duration, effect, items_map=items_map)
        plan = {"schema": "semotion.plan/1", "style": f"matrix_{effect}", "canvas": canvas,
                "duration_s": duration, "decisions": decisions,
                "notes": {"policy": f"effect matrix: {effect} only", "skipped": []},
                "provenance": {"method": "matrix-v1"}}

    if not plan.get("decisions"):
        return {"fixture": fixture, "effect": effect, "status": "NO_ANCHORS",
                "note": "this material has no anchor of this effect family — NOT RUN, "
                        "not a code failure"}

    ir_doc = ir_mod.build(transcript, vis, anchors_doc, plan, clip, canvas)
    errors = ir_mod.validate(ir_doc)
    if errors:
        (out_dir / "ir_errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
        return {"fixture": fixture, "effect": effect, "status": "IR_INVALID", "errors": errors[:5]}

    ir_mod.dump(ir_doc, out_dir / "motion_ir.json")
    rendered = out_dir / f"{fixture}_{effect}.mp4"
    man = render_mod.render(ir_doc, clip, rendered)
    (out_dir / "render_manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    m = verify.metrics(ir_doc, man, source=clip, rendered=rendered,
                       truth_json=truth if truth.exists() else None,
                       vision=vis, gate="G2", work=out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    number_check = None
    if any(e["type"] == "number" for e in ir_doc["elements"]):
        cp = subprocess.run([str(ROOT / ".venv/bin/python"),
                             str(ROOT / "scripts/check_number_semantics.py"),
                             str(out_dir / "motion_ir.json"), str(rendered),
                             str(out_dir / "number_check.json")],
                            capture_output=True, text=True, cwd=ROOT)
        number_check = json.loads((out_dir / "number_check.json").read_text())["verdict"] \
            if (out_dir / "number_check.json").exists() else cp.stdout

    return {
        "fixture": fixture, "effect": effect,
        "elements": len(ir_doc["elements"]),
        "ir_hash": ir_doc["provenance"]["ir_sha256"][:16],
        "rendered": str(rendered),
        "verdicts": m["verdicts"],
        "timing": {k: m["timing"].get(k) for k in ("samples", "p50_ms", "p90_ms", "max_ms")},
        "position": {
            "planned_face": m["position"]["planned"]["face_overlap"]["value"],
            "observed_face": m["position"]["observed"]["face_overlap"]["value"],
            "observed_face_samples": m["position"]["observed"]["face_overlap"]["samples"],
            "out_of_bounds": m["position"]["observed"]["out_of_bounds"]["value"],
            "containment": m["position"]["raster_containment"]["value"],
            "collision": m["position"]["collision"]["value"],
        },
        "number_check": number_check,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips-root", default="tests/fixtures/clips")
    ap.add_argument("--effects", nargs="*",
                    default=["keyword", "number", "list", "card", "combo"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for effect in args.effects:
        cell_dir = out / effect
        print(f"[matrix] {args.fixture} / {effect} ...", flush=True)
        cells.append(run_cell(args.fixture, effect, clips_root=Path(args.clips_root),
                              out_dir=cell_dir))
    doc = {"schema": "semotion.effect_matrix/1", "fixture": args.fixture,
           "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "cells": cells, "tool_versions": media.tool_versions()}
    (out / "matrix.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    for c in cells:
        print(f"  {c['effect']:<8} elements={c.get('elements')} "
              f"{c.get('verdicts', c.get('status'))} timing={c.get('timing')} "
              f"num={c.get('number_check')}")
    print(f"matrix -> {out / 'matrix.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
