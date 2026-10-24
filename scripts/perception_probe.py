"""Perception v2 probe — run ``semotion.perception`` on a clip and report.

This is the evidence generator for the perception task (not part of the
runtime pipeline). It writes:

* ``<outdir>/perception.json``          the ``semotion.perception/1`` document
* ``<outdir>/masks/NNNN.png``           downscaled 8-bit person masks
* ``<outdir>/perception.json.debug.png``2x2 debug sheet
* ``<outdir>/probe_report.json``        timings, mask quality, zones, text regions
* ``<outdir>/determinism.json``         only with ``--determinism``

Usage::

    PYTHONPATH=src .venv/bin/python scripts/perception_probe.py \\
        --video inputs/zhengqinwen_25s.mp4 \\
        --outdir artifacts/p2/perception/zhengqinwen_25s --determinism

Exit code 0 when the document validates and (with ``--determinism``) the two
runs produce identical normalised JSON and identical mask PNGs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from semotion import perception  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _normalise(doc: dict) -> dict:
    """Drop run-varying fields: timestamps, absolute paths, mask directory."""
    out = copy.deepcopy(doc)
    out.pop("provenance", None)
    src = out.get("source") or {}
    if "path" in src:
        src["path"] = "<source>"
    for frame in out.get("frames") or []:
        mask = frame.get("mask")
        if mask:
            frame["mask"] = f"mask:{Path(mask).name}"
    return out


def _mask_hashes(mask_dir: Path) -> dict[str, str]:
    return {p.name: _sha256(p) for p in sorted(mask_dir.glob("*.png"))}


def _diff_summary(a: str, b: str, limit: int = 24) -> list[str]:
    import difflib

    lines = list(difflib.unified_diff(a.splitlines(), b.splitlines(),
                                      fromfile="run1", tofile="run2", lineterm=""))
    return lines[:limit]


def _hand_box_mask_coverage(doc: dict, masks: Path) -> dict:
    """Fraction of each reported hand box covered by that sample's person mask."""
    from PIL import Image

    if not any(f.get("hands") for f in doc.get("frames") or []):
        return {"frames": 0, "boxes": 0, "mean": None, "min": None, "max": None}
    width, height = doc["regions"]["frame_size"]
    values = []
    for frame in doc["frames"]:
        if not frame.get("hands"):
            continue
        mask_path = masks / Path(frame["mask"]).name
        if not mask_path.exists():
            continue
        mask = np.asarray(Image.open(mask_path).convert("L"))
        gh, gw = mask.shape
        for hand in frame["hands"]:
            x, y, w, h = hand["bbox"]
            gx0 = max(0, int(x * gw / width))
            gx1 = min(gw, max(gx0 + 1, int(np.ceil((x + w) * gw / width))))
            gy0 = max(0, int(y * gh / height))
            gy1 = min(gh, max(gy0 + 1, int(np.ceil((y + h) * gh / height))))
            sub = mask[gy0:gy1, gx0:gx1] > 127
            if sub.size:
                values.append(float(sub.mean()))
    if not values:
        return {"frames": 0, "boxes": 0, "mean": None, "min": None, "max": None}
    return {"frames": sum(1 for f in doc["frames"] if f.get("hands")), "boxes": len(values),
            "mean": round(float(np.mean(values)), 4),
            "min": round(float(np.min(values)), 4),
            "max": round(float(np.max(values)), 4)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--fps", type=float, default=perception.DEFAULT_SAMPLE_HZ)
    parser.add_argument("--det-max-side", type=int, default=perception.DEFAULT_DET_MAX_SIDE)
    parser.add_argument("--seg-quality", choices=["accurate", "balanced", "fast"],
                        default="balanced")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--determinism", action="store_true",
                        help="run twice (cache bypassed) and compare JSON + masks")
    parser.add_argument("--tag", default="", help="free-form label stored in the report")
    args = parser.parse_args(argv)

    video = Path(args.video)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "perception.json"
    masks = outdir / "masks"
    debug = outdir / "perception.json.debug.png"

    options = perception.Options(sample_hz=float(args.fps),
                                 det_max_side=int(args.det_max_side),
                                 seg_quality=args.seg_quality)
    meta: dict = {}
    doc = perception.analyze(video, json_path, mask_dir=masks, debug_path=debug,
                             options=options, force=args.force, meta=meta)
    problems = perception.validate(doc)
    regions = doc.get("regions") or {}
    frames = doc.get("frames") or []
    zones = regions.get("quiet_zones") or []
    cov = [float(f["mask_coverage"]) for f in frames] or [0.0]
    hand_frames = [float(f["mask_coverage"]) for f in frames if f.get("hands")]
    hand_box_cov = _hand_box_mask_coverage(doc, masks)
    big_zones = [z for z in zones if z["bbox"][2] >= 400 and z["bbox"][3] >= 200]
    report = {
        "schema": "semotion.perception-probe/1",
        "tag": args.tag,
        "video": str(video.resolve()),
        "video_sha256": doc["source"]["sha256"],
        "source": doc["source"],
        "backend": doc["backend"],
        "runtime_s": meta.get("runtime_s"),
        "phase_s": meta.get("phase_s"),
        "cache": meta.get("cache"),
        "cache_key": meta.get("cache_key"),
        "n_samples": len(frames),
        "sample_hz": doc.get("sample_hz"),
        "mask": {
            "mask_scale": options.mask_scale,
            "mask_pngs": len(list(masks.glob("*.png"))),
            "mean_coverage": round(sum(cov) / len(cov), 4),
            "min_coverage": round(min(cov), 4),
            "max_coverage": round(max(cov), 4),
            "frames_with_zero_coverage": sum(1 for c in cov if c == 0.0),
            "zero_coverage_frames": [f["frame"] for f in frames if f["mask_coverage"] == 0.0],
            "frames_with_hands": len(hand_frames),
            "mean_coverage_on_hand_frames": (round(sum(hand_frames) / len(hand_frames), 4)
                                             if hand_frames else None),
            "hand_box_mask_coverage": hand_box_cov,
            "mask_failures": (doc.get("provenance") or {}).get("mask_failures", []),
            "face_without_mask_count": (doc.get("provenance") or {}).get(
                "face_without_mask_count", 0),
        },
        "pose": {
            "frames_with_pose": sum(1 for f in frames if f.get("pose")),
            "pose_failures_count": (doc.get("provenance") or {}).get("pose_failures_count", 0),
        },
        "hands": {
            "frames_with_hands": len(hand_frames),
            "samples_with_hands": int((doc.get("provenance") or {}).get("hand_samples", 0)),
        },
        "zones": {
            "count": len(zones),
            "zero_person_overlap": sum(1 for z in zones if z["person_overlap"] == 0.0),
            "fit_400x200": len(big_zones),
            "items": zones,
        },
        "existing_text": regions.get("existing_text") or [],
        "legacy": {
            "faces": len(doc.get("faces") or []),
            "face_tracks": len(doc.get("face_tracks") or []),
            "shots": len(doc.get("shots") or []),
            "safe_zones": sorted((doc.get("safe_zones") or {}).keys()),
        },
        "validation": {"ok": not problems, "problems": problems},
        "paths": {
            "json": str(json_path.resolve()),
            "masks": str(masks.resolve()),
            "debug_sheet": str(debug.resolve()) if debug.exists() else None,
        },
    }

    if args.determinism:
        run2 = outdir / "_determinism_run2"
        run2.mkdir(parents=True, exist_ok=True)
        meta2: dict = {}
        doc2 = perception.analyze(video, run2 / "perception.json",
                                  mask_dir=run2 / "masks",
                                  debug_path=run2 / "perception.json.debug.png",
                                  options=options, force=True, meta=meta2)
        first = json.dumps(_normalise(doc), ensure_ascii=False, indent=1, sort_keys=True)
        second = json.dumps(_normalise(doc2), ensure_ascii=False, indent=1, sort_keys=True)
        hashes1 = _mask_hashes(masks)
        hashes2 = _mask_hashes(run2 / "masks")
        debug_identical = (debug.exists() and (run2 / "perception.json.debug.png").exists()
                           and _sha256(debug) == _sha256(run2 / "perception.json.debug.png"))
        det = {
            "schema": "semotion.perception-determinism/1",
            "json_identical": first == second,
            "mask_count_run1": len(hashes1),
            "mask_count_run2": len(hashes2),
            "masks_identical": hashes1 == hashes2,
            "debug_sheet_identical": bool(debug_identical),
            "runtime_run1_s": meta.get("runtime_s"),
            "runtime_run2_s": meta2.get("runtime_s"),
            "cache_run1": meta.get("cache"),
            "cache_run2": meta2.get("cache"),
            "json_diff": [] if first == second else _diff_summary(first, second),
            "mask_diff": [] if hashes1 == hashes2 else [
                name for name in sorted(set(hashes1) | set(hashes2))
                if hashes1.get(name) != hashes2.get(name)][:24],
            "verdict": "PASS" if (first == second and hashes1 == hashes2) else "FAIL",
        }
        (outdir / "determinism.json").write_text(
            json.dumps(det, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        report["determinism"] = {k: det[k] for k in
                                 ("json_identical", "masks_identical",
                                  "debug_sheet_identical", "verdict")}

    (outdir / "probe_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    print(f"== {video.name} ==")
    print(f"out          : {json_path.resolve()}")
    print(f"runtime      : {report['runtime_s']} s  cache={report['cache']}  "
          f"phases={report['phase_s']}")
    print(f"samples      : {report['n_samples']} @ {report['sample_hz']} Hz "
          f"({doc['source']['width']}x{doc['source']['height']})")
    print(f"mask coverage: mean {report['mask']['mean_coverage']}  "
          f"zero-cov frames {report['mask']['frames_with_zero_coverage']}  "
          f"hand-frame mean {report['mask']['mean_coverage_on_hand_frames']}  "
          f"hand-box mean {report['mask']['hand_box_mask_coverage']['mean']}")
    print(f"pose         : {report['pose']['frames_with_pose']}/{report['n_samples']} frames")
    print(f"hands        : {report['hands']['frames_with_hands']} samples")
    print(f"text regions : {len(report['existing_text'])}")
    for entry in report["existing_text"]:
        print(f"   {entry['kind']:8s} {entry['t0']:6.2f}-{entry['t1']:6.2f}s "
              f"{entry['bbox']} conf={entry['conf']} {entry['text']!r}")
    print(f"quiet zones  : {report['zones']['count']} "
          f"(person_overlap==0: {report['zones']['zero_person_overlap']}, "
          f">=400x200: {report['zones']['fit_400x200']})")
    for rank, zone in enumerate(zones, 1):
        print(f"   Z{rank:<2d} {zone['score']:.3f} {tuple(zone['bbox'])} {zone['kind']:11s} "
              f"person={zone['person_overlap']} text={zone['text_overlap']}")
    print(f"validation   : {'OK' if not problems else 'FAIL'}")
    for problem in problems:
        print(f"   - {problem}")
    if args.determinism:
        det = json.loads((outdir / "determinism.json").read_text(encoding="utf-8"))
        print(f"determinism  : {det['verdict']}  json={det['json_identical']} "
              f"masks={det['masks_identical']} debug={det['debug_sheet_identical']}")
        for line in det["json_diff"]:
            print(f"   {line}")
    return 0 if not problems and (not args.determinism
                                  or det["verdict"] == "PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
