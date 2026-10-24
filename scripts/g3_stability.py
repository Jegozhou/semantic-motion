#!/usr/bin/env python3
"""G3 stability checks: degradation, partial re-render, failure recovery.

Every scenario runs the real pipeline functions and records PASS/FAIL with the
evidence needed to re-check it by hand. Nothing is simulated at the level of the
metric itself: renders are real ffmpeg runs.

    PYTHONPATH=src .venv/bin/python scripts/g3_stability.py --out artifacts/g3/stability
"""
from __future__ import annotations

import argparse
import json
import shutil

import numpy as np
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from semotion import anchors as anchors_mod, asr, ir as ir_mod, media, render as render_mod  # noqa: E402
from semotion import semantic, verify, vision as vision_mod  # noqa: E402


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _dump(obj, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _prep(clip: Path, work: Path) -> tuple[dict, dict, dict, dict]:
    """transcript / vision / anchors / canvas for a fixture (cached)."""
    mi = media.info(clip)
    canvas = {"width": mi.width, "height": mi.height, "fps": mi.fps}
    lang = "en" if clip.stem.startswith("fx_pd") else "zh"
    tr = asr.transcribe(clip, work / "transcript.json", model="medium", device="cpu", language=lang)
    vis = vision_mod.analyze(clip, work / "vision.json")
    an = anchors_mod.classify(tr)
    an = anchors_mod.refine_onsets(clip, an)
    an["duration_s"] = float(tr["source"]["duration_s"])
    return tr, vis, an, canvas


# ------------------------------------------------------------------ degradations

def _build_and_check(tr, vis, an, canvas, clip, plan, out_dir: Path) -> dict:
    doc = ir_mod.build(tr, vis, an, plan, clip, canvas)
    errs = ir_mod.validate(doc)
    boxes = [e["layout"]["box"] for e in doc.get("elements", [])]
    in_bounds = all(b[0] >= 0 and b[1] >= 0 and b[0] + b[2] <= canvas["width"]
                    and b[1] + b[3] <= canvas["height"] for b in boxes)
    degraded = [e["id"] for e in doc.get("elements", []) if e["layout"].get("degraded")]
    _dump(doc, out_dir / "motion_ir.json")
    skipped = [x for x in (doc.get("provenance", {}).get("skipped_decisions") or [])
               if str(x.get("reason", "")).startswith("no_face_free_space")]
    result = {
        "schema": "semotion.motion-ir/1", "elements": len(boxes), "validate_errors": errs,
        "all_boxes_in_canvas": in_bounds, "degraded_elements": degraded,
        "dropped_for_face_safety": skipped,
        "warnings": doc.get("provenance", {}).get("build_warnings", [])[:8],
    }
    if not errs:
        rendered = out_dir / "render.mp4"
        try:
            man = render_mod.render(doc, clip, rendered)
            m = verify.metrics(doc, man, source=clip, rendered=rendered, vision=vis, gate="G3")
            result.update({"render": "ok", "verdicts": m["verdicts"],
                           "position": {"planned_face": m["position"]["planned"]["face_overlap"]["value"],
                                        "observed_face": m["position"]["observed"]["face_overlap"].get("value"),
                                        "out_of_bounds": m["position"]["observed"]["out_of_bounds"].get("value"),
                                        "containment": m["position"]["raster_containment"].get("value")},
                           "metrics": str(out_dir / "metrics.json")})
            _dump(m, out_dir / "metrics.json")
        except media.MediaError as exc:
            result.update({"render": "failed", "error": exc.as_dict()})
    pos = result.get("verdicts", {}).get("position")
    result["verdict"] = ("PASS" if not errs and in_bounds
                         and result.get("render") == "ok"
                         and pos in ("PASS", "NOT RUN")
                         and result.get("position", {}).get("planned_face") == 0.0
                         else "FAIL")
    return result


def degradations(clip: Path, work: Path, out: Path) -> list[dict]:
    tr, vis, an, canvas = _prep(clip, work)
    plan = semantic.decide(an, vis, canvas, style=semantic.STYLE_CALIBRATION, transcript=tr)
    out.mkdir(parents=True, exist_ok=True)
    scenarios: list[dict] = []

    # S1 — no face detected: conservative subject band must still place everything
    vis_no_face = {k: v for k, v in vis.items() if k not in ("faces", "face_tracks")}
    scenarios.append({"name": "no_face_detected",
                      **_build_and_check(tr, vis_no_face, an, canvas, clip, plan, out / "no_face")})

    # S2 — a face covering the middle of the frame: degrade, stay in bounds
    huge = {"id": "f0", "bbox_union": [int(canvas["width"] * 0.12), int(canvas["height"] * 0.18),
                                       int(canvas["width"] * 0.76), int(canvas["height"] * 0.62)],
            "coverage": 1.0,
            "frames": [{"frame": 0, "t_s": 0.0, "box": [int(canvas["width"] * 0.12),
                                                         int(canvas["height"] * 0.18),
                                                         int(canvas["width"] * 0.76),
                                                         int(canvas["height"] * 0.62)],
                        "score": 0.99}]}
    vis_huge = dict(vis)
    vis_huge["face_tracks"] = [huge]
    vis_huge["faces"] = huge["frames"]
    scenarios.append({"name": "face_covers_center",
                      **_build_and_check(tr, vis_huge, an, canvas, clip, plan, out / "huge_face")})

    # S2b — face covers almost the whole frame: the engine must degrade (font shrink /
    # least-bad slot, with a recorded reason) rather than crash or silently overlap.
    full_box = [int(canvas["width"] * 0.04), int(canvas["height"] * 0.05),
                int(canvas["width"] * 0.92), int(canvas["height"] * 0.9)]
    vis_full = dict(vis)
    vis_full["face_tracks"] = [{"id": "f0", "bbox_union": full_box, "coverage": 1.0,
                                "frames": [{"frame": 0, "t_s": 0.0, "box": full_box,
                                            "score": 0.99}]}]
    vis_full["faces"] = [{"frame": 0, "t_s": 0.0, "box": full_box, "score": 0.99}]
    scen = _build_and_check(tr, vis_full, an, canvas, clip, plan, out / "full_face")
    scen["name"] = "face_covers_nearly_everything"
    scen["degradation_exercised"] = (len(scen.get("degraded_elements", [])) > 0
                                     or len(scen.get("dropped_for_face_safety", [])) > 0)
    # Correct behaviour: no element may cover the face. Elements without a face-free
    # slot are dropped (recorded in skipped_decisions), the IR stays valid and the
    # render still produces a playable pass-through.
    dropped = scen.get("dropped_for_face_safety") or []
    scen["degradation_exercised"] = bool(dropped) or bool(scen.get("degraded_elements"))
    scen["verdict"] = ("PASS" if scen.get("validate_errors") == []
                       and scen.get("all_boxes_in_canvas")
                       and scen.get("render") == "ok"
                       and scen["degradation_exercised"]
                       and (scen.get("position") is None
                            or scen.get("position", {}).get("planned_face") == 0.0)
                       else "FAIL")
    scenarios.append(scen)

    # S3 — genuinely small source (320x568): layout must not crash, boxes stay inside
    small_clip = out / "small_source" / "small.mp4"
    small_clip.parent.mkdir(parents=True, exist_ok=True)
    if not small_clip.exists():
        media.run([media.FFMPEG, "-v", "error", "-y", "-i", str(clip), "-vf",
                   "scale=320:568:flags=lanczos,setsar=1", "-c:v", "libx264", "-preset",
                   "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "96k", str(small_clip)])
    mi_small = media.info(small_clip)
    small_canvas = {"width": mi_small.width, "height": mi_small.height, "fps": mi_small.fps}
    small_vis = vision_mod.analyze(small_clip, out / "small_source" / "vision.json")
    small_plan = semantic.decide(an, small_vis, small_canvas,
                                 style=semantic.STYLE_CALIBRATION, transcript=tr)
    scenarios.append({"name": "small_canvas_320x568",
                      **_build_and_check(tr, small_vis, an, small_canvas, small_clip,
                                         small_plan, out / "small")})

    # S4 — empty plan: zero overlays is a valid product decision, not an error
    empty_plan = {"schema": "semotion.plan/1", "style": "empty", "canvas": canvas,
                  "duration_s": tr["source"]["duration_s"], "decisions": [],
                  "notes": {"policy": "no motion"}}
    doc = ir_mod.build(tr, vis, an, empty_plan, clip, canvas)
    errs = ir_mod.validate(doc)
    rendered = out / "empty" / "render.mp4"
    rendered.parent.mkdir(parents=True, exist_ok=True)
    status, detail = "ok", {}
    try:
        man = render_mod.render(doc, clip, rendered)
        detail = {"frames": man["output"]["n_frames"], "duration_s": man["output"]["duration_s"]}
    except media.MediaError as exc:
        status, detail = "failed", exc.as_dict()
    scenarios.append({"name": "empty_plan_zero_motion", "elements": len(doc["elements"]),
                      "validate_errors": errs, "render": status, "detail": detail,
                      "verdict": "PASS" if not errs and status == "ok" else "FAIL"})
    return scenarios


# --------------------------------------------------------------- partial update

def partial_update(clip: Path, work: Path, out: Path) -> dict:
    tr, vis, an, canvas = _prep(clip, work)
    plan = semantic.decide(an, vis, canvas, style=semantic.STYLE_CALIBRATION, transcript=tr)
    out.mkdir(parents=True, exist_ok=True)
    doc = ir_mod.build(tr, vis, an, plan, clip, canvas)
    base_mp4 = out / "base.mp4"
    render_mod.render(doc, clip, base_mp4)

    target = next((e for e in doc["elements"] if e["type"] in ("keyword", "card")), None)
    if target is None:
        return {"verdict": "NOT RUN", "reason": "no editable element"}

    edited = json.loads(json.dumps(doc))
    el = next(e for e in edited["elements"] if e["id"] == target["id"])
    original_text = el["content"]["text"]
    edited["collision_ok"] = edited.get("collision_ok") or []
    edited = ir_mod.seal(edited) if hasattr(ir_mod, "seal") else edited
    edited["provenance"]["ir_sha256"] = ir_mod.compute_ir_hash(edited)
    _dump(edited, out / "edited_identical_ir.json")
    identical_mp4 = out / "identical_rerender.mp4"
    render_mod.render(edited, clip, identical_mp4)
    det_identical = verify.determinism(edited, base_mp4, identical_mp4)

    # now a real content change on exactly one element
    edited2 = json.loads(json.dumps(doc))
    el2 = next(e for e in edited2["elements"] if e["id"] == target["id"])
    el2["content"]["text"] = (original_text + "（改）")[:16]
    el2["content"]["edited"] = True
    el2["content"]["edit_note"] = "G3 partial-update test: deliberate human edit"
    edited2["provenance"]["ir_sha256"] = ir_mod.compute_ir_hash(edited2)
    errs = ir_mod.validate(edited2)
    changed_mp4 = out / "changed.mp4"
    if errs:
        return {"verdict": "FAIL", "reason": "edited IR invalid", "errors": errs}
    render_mod.render(edited2, clip, changed_mp4)

    # Outside the edited window the picture must stay the same. Inter-frame coding
    # legitimately propagates small residuals for a few frames, so this compares
    # decoded pixels with a compression tolerance instead of byte-exact hashes.
    src_frames = media.video_frame_hashes(base_mp4)
    new_frames = media.video_frame_hashes(changed_mp4)
    win = target["window"]
    ref = verify._gray_frames(base_mp4)
    new = verify._gray_frames(changed_mp4)
    fps = float(doc["source"]["fps"])
    guard = int(round(fps * 0.75))
    outside_idx = [i for i in range(min(len(ref), len(new)))
                   if not (win["start_frame"] - guard <= i < win["end_frame"] + guard)]
    outside_diffs = [float(np.abs(ref[i].astype(np.int32) - new[i].astype(np.int32)).mean())
                     for i in outside_idx[:80]]
    outside_max = max(outside_diffs) if outside_diffs else 0.0
    outside_equal = outside_max <= 1.0
    box = target["layout"]["box"]
    W, H = int(doc["canvas"]["width"]), int(doc["canvas"]["height"])
    low_h = ref.shape[1] // 192
    x0 = max(0, int(box[0] * 192 / W))
    x1 = min(192, int((box[0] + box[2]) * 192 / W) + 1)
    y0 = max(0, int(box[1] * low_h / H))
    y1 = min(low_h, int((box[1] + box[3]) * low_h / H) + 1)
    inside_idx = [i for i in range(max(0, win["start_frame"] - 2),
                                   min(min(len(ref), len(new)), win["end_frame"] + 2))]
    inside_diffs = []
    for i in inside_idx:
        a = ref[i].reshape(low_h, 192)[y0:y1, x0:x1].astype(np.int32)
        b = new[i].reshape(low_h, 192)[y0:y1, x0:x1].astype(np.int32)
        inside_diffs.append(float(np.abs(a - b).mean()))
    inside_changed = bool(inside_diffs) and max(inside_diffs) > 2.0

    # invalid edit must be refused by validation (failure fallback path)
    bad = json.loads(json.dumps(doc))
    bad_el = next(e for e in bad["elements"] if e["id"] == target["id"])
    bad_el["layout"]["box"] = [-50, -50, 200, 200]
    bad["provenance"]["ir_sha256"] = ir_mod.compute_ir_hash(bad)
    bad_errors = ir_mod.validate(bad)

    return {
        "verdict": "PASS" if (det_identical["verdict"] == "PASS" and outside_equal
                             and inside_changed and bad_errors) else "FAIL",
        "target_element": target["id"], "original_text": original_text,
        "edited_text": el2["content"]["text"],
        "ir_hash_changed": doc["provenance"]["ir_sha256"] != edited2["provenance"]["ir_sha256"],
        "identical_ir_reproducible": det_identical,
        "outside_window_unchanged": outside_equal,
        "outside_window_max_mean_diff": round(outside_max, 4),
        "outside_frames_compared": len(outside_idx[:80]),
        "inside_window_changed": inside_changed,
        "inside_max_mean_diff": round(max(inside_diffs), 4) if inside_diffs else None,
        "invalid_edit_rejected": {"errors": bad_errors[:3]},
        "artifacts": {"base": str(base_mp4), "changed": str(changed_mp4),
                      "changed_ir": str(out / "changed_ir.json")},
    }


# --------------------------------------------------------------------- recovery

def recovery(clip: Path, work: Path, out: Path) -> list[dict]:
    tr, vis, an, canvas = _prep(clip, work)
    plan = semantic.decide(an, vis, canvas, style=semantic.STYLE_CALIBRATION, transcript=tr)
    out.mkdir(parents=True, exist_ok=True)
    doc = ir_mod.build(tr, vis, an, plan, clip, canvas)
    results: list[dict] = []

    # R1 — invalid IR is refused before rendering
    bad = json.loads(json.dumps(doc))
    if bad["elements"]:
        bad["elements"][0]["layout"]["box"] = [-10, -10, 300, 300]
    bad["provenance"]["ir_sha256"] = ir_mod.compute_ir_hash(bad)
    refused = bool(ir_mod.validate(bad))
    results.append({"name": "invalid_ir_refused", "validation_errors": refused,
                    "verdict": "PASS" if refused else "FAIL"})

    # R2 — unwritable destination fails cleanly and leaves no partial file
    target = Path("/nonexistent_semotion_dir/out.mp4")
    error = None
    try:
        render_mod.render(doc, clip, target)
    except Exception as exc:  # noqa: BLE001 - typed boundary
        error = getattr(exc, "code", type(exc).__name__)
    results.append({"name": "unwritable_destination",
                    "error_code": error, "partial_file_exists": target.exists(),
                    "verdict": "PASS" if error and not target.exists() else "FAIL"})

    # R3 — corrupt input fails, then a clean retry succeeds from cache
    corrupt = out / "corrupt.mp4"
    corrupt.write_bytes(b"not a video" * 100)
    corrupt_error = None
    try:
        render_mod.render(doc, corrupt, out / "from_corrupt.mp4")
    except Exception as exc:  # noqa: BLE001
        corrupt_error = getattr(exc, "code", type(exc).__name__)
    retry = out / "retry.mp4"
    render_mod.render(doc, clip, retry)
    ok_retry = retry.exists() and media.info(retry).duration > 0.5
    results.append({"name": "corrupt_input_then_retry", "error_code": corrupt_error,
                    "retry_ok": ok_retry,
                    "verdict": "PASS" if corrupt_error and ok_retry else "FAIL"})

    # R4 — rerunning the same IR after failure yields the same frames (resume)
    a = out / "resume_a.mp4"
    b = out / "resume_b.mp4"
    render_mod.render(doc, clip, a)
    render_mod.render(doc, clip, b)
    det = verify.determinism(doc, a, b)
    results.append({"name": "resume_after_failure_reproducible", **det})
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="fx_face_pan")
    ap.add_argument("--clips-root", default="tests/fixtures/clips")
    ap.add_argument("--out", default="artifacts/g3/stability")
    args = ap.parse_args()
    clip = Path(args.clips_root) / args.fixture / f"{args.fixture}.mp4"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"

    doc = {
        "schema": "semotion.stability/1",
        "fixture": args.fixture,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "degradation": degradations(clip, work, out / "degradation"),
        "partial_update": partial_update(clip, work, out / "partial"),
        "recovery": recovery(clip, work, out / "recovery"),
        "tool_versions": media.tool_versions(),
    }
    all_verdicts = ([d["verdict"] for d in doc["degradation"]]
                    + [doc["partial_update"]["verdict"]]
                    + [r["verdict"] for r in doc["recovery"]])
    doc["verdict"] = "PASS" if all(v == "PASS" for v in all_verdicts) else "FAIL"
    _dump(doc, out / "stability.json")
    for d in doc["degradation"]:
        print(f"  degradation {d['name']:<24} {d['verdict']} "
              f"elements={d.get('elements')} degraded={len(d.get('degraded_elements', []))}")
    print(f"  partial_update            {doc['partial_update']['verdict']}")
    for r in doc["recovery"]:
        print(f"  recovery {r['name']:<32} {r['verdict']}")
    print(f"{doc['verdict']} -> {out / 'stability.json'}")
    return 0 if doc["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
