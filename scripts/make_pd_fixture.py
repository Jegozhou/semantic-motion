#!/usr/bin/env python3
"""Build a fixture clip from the public-domain US-government source video.

Source: White House "Weekly Address" (2009-01-24), Wikimedia Commons,
Public domain (work of the U.S. federal government, 17 U.S.C. §105).
https://commons.wikimedia.org/wiki/File:Weekly_Address_(2009-01-24),_small.ogv

Honesty note: this is real footage of a real person, but it is English speech and
it has **no human word labels**, so it is registered as `MACHINE_WEAK` for timing
and is *not* used to claim Chinese or human-labelled accuracy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semotion import media  # noqa: E402

SOURCE = ROOT / "assets/media/pd_weekly_address_2009-01-24.ogv"
CLIPS = ROOT / "tests/fixtures/clips"

LICENSE = {
    "license": "Public domain (U.S. Government work, 17 U.S.C. §105)",
    "license_url": "https://commons.wikimedia.org/wiki/File:Weekly_Address_(2009-01-24),_small.ogv",
    "attribution_required": False,
    "note": "White House Weekly Address, 2009-01-24. Real footage of a real person; "
            "English speech; no human word-level labels.",
}

CLIPS_SPEC = [
    {"id": "fx_pd_real_16x9_a", "start_s": 6.0, "duration_s": 24.0, "width": 1280, "height": 720},
    {"id": "fx_pd_real_16x9_b", "start_s": 66.0, "duration_s": 24.0, "width": 1280, "height": 720},
]


def build(spec: dict) -> dict:
    fid = spec["id"]
    out_dir = CLIPS / fid
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{fid}.mp4"
    logs = out_dir / "_logs"
    logs.mkdir(exist_ok=True)
    vf = (f"scale={spec['width']}:{spec['height']}:flags=lanczos,"
          f"fps=30,setsar=1,format=yuv420p")
    media.run([
        media.FFMPEG, "-v", "error", "-y", "-ss", str(spec["start_s"]), "-t", str(spec["duration_s"]),
        "-i", str(SOURCE), "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", str(out),
    ], log=logs / "build.log")
    mi = media.info(out)
    doc = {
        "fixture_id": fid,
        "kind": "REAL_PD",
        "truth_class": "MACHINE_WEAK",
        "note": "Real footage (public domain US government video), English speech. "
                "No human word labels: timing is NOT independently labelled.",
        "source": {"name": "Weekly Address (2009-01-24), small.ogv",
                   "provider": "Wikimedia Commons", **LICENSE},
        "abs_path": str(out),
        "sha256": media.sha256_file(out),
        "duration_s": round(mi.duration, 6),
        "width": mi.width, "height": mi.height, "fps": mi.fps,
        "n_frames": mi.frame_count,
        "language": "en",
        "people": 1,
        "composition": "talking head, center; bookshelf background",
        "camera_motion": "static",
        "label_types": ["face", "speech"],
        "split": "dev",
        "derived_from": {"path": str(SOURCE.relative_to(ROOT)),
                         "trim_start_s": spec["start_s"], "trim_duration_s": spec["duration_s"],
                         "upscaled": f"854x480 -> {spec['width']}x{spec['height']}"},
        "tool_versions": media.tool_versions(),
    }
    (out_dir / "fixture.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    return doc


if __name__ == "__main__":
    for spec in CLIPS_SPEC:
        d = build(spec)
        print(f"{d['fixture_id']}: {d['duration_s']:.2f}s {d['width']}x{d['height']} "
              f"-> {d['abs_path']}")
