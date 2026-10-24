"""Semantic Motion — baseline pass-through / re-encode checks (G0).

Two operations that must work before any semantic work is attempted:

* ``remux``      — copy video+audio streams into a fresh MP4 container;
* ``reencode``   — decode and re-encode to H.264/AAC (the render pipeline's target).

Verification compares the decoded media, never just file bytes: frame count,
duration, decoded-audio identity (PCM hash for stream copies) and per-frame
pixel differences with an explicit tolerance.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import media


def remux(src: Path, dst: Path, *, log: Path | None = None) -> dict[str, Any]:
    media.run([media.FFMPEG, "-v", "error", "-y", "-i", str(src),
               "-c", "copy", "-movflags", "+faststart", str(dst)], log=log)
    return media.info(dst).__dict__ | {"raw": None}


def reencode(src: Path, dst: Path, *, crf: int = 18, preset: str = "medium",
             log: Path | None = None) -> dict[str, Any]:
    media.run([media.FFMPEG, "-v", "error", "-y", "-i", str(src),
               "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
               "-ar", "48000", "-movflags", "+faststart", str(dst)], log=log)
    return media.info(dst).__dict__ | {"raw": None}


def verify_copy(src: Path, dst: Path, *, tolerance_ms: float = 0.0,
                work: Path | None = None) -> dict[str, Any]:
    """Stream-copy check: identical decoded audio and identical decoded frames."""
    a, b = media.info(src), media.info(dst)
    work = Path(work or dst.parent / "_verify")
    work.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Any] = {}

    checks["duration"] = {
        "src_s": round(a.duration, 6), "dst_s": round(b.duration, 6),
        "delta_s": round(abs(a.duration - b.duration), 6),
        "pass": abs(a.duration - b.duration) <= tolerance_ms / 1000.0,
    }
    checks["streams"] = {
        "video": [a.n_video_streams, b.n_video_streams],
        "audio": [a.n_audio_streams, b.n_audio_streams],
        "pass": (a.n_video_streams, a.n_audio_streams) == (b.n_video_streams, b.n_audio_streams),
    }
    checks["geometry"] = {
        "src": [a.width, a.height, a.fps], "dst": [b.width, b.height, b.fps],
        "pass": (a.width, a.height, round(a.fps, 3)) == (b.width, b.height, round(b.fps, 3)),
    }
    ha = media.audio_pcm_sha256(src, log=work / "audio_src.log")
    hb = media.audio_pcm_sha256(dst, log=work / "audio_dst.log")
    checks["audio_pcm_identity"] = {"src": ha, "dst": hb, "pass": ha == hb}
    fa = media.video_frame_hashes(src, log=work / "frames_src.log")
    fb = media.video_frame_hashes(dst, log=work / "frames_dst.log")
    checks["frames"] = {
        "src": len(fa), "dst": len(fb),
        "identical_hashes": fa[:len(fb)] == fb,
        "pass": len(fa) == len(fb) and fa == fb,
    }

    ok = all(c["pass"] for c in checks.values())
    return {
        "operation": "stream_copy",
        "src": str(src), "dst": str(dst),
        "src_sha256": media.sha256_file(src), "dst_sha256": media.sha256_file(dst),
        "checks": checks,
        "verdict": "PASS" if ok else "FAIL",
        "tool_versions": media.tool_versions(),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def verify_reencode(src: Path, dst: Path, *, max_frame_diff: float = 12.0,
                    duration_tolerance_frames: int = 1,
                    work: Path | None = None) -> dict[str, Any]:
    """Lossy re-encode check with explicit tolerances (pixel + timing)."""
    a, b = media.info(src), media.info(dst)
    work = Path(work or dst.parent / "_verify")
    work.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Any] = {}

    frame_budget = duration_tolerance_frames / max(a.fps, 1.0)
    checks["duration"] = {
        "src_s": round(a.duration, 6), "dst_s": round(b.duration, 6),
        "delta_s": round(abs(a.duration - b.duration), 6),
        "budget_s": round(frame_budget, 6),
        "pass": abs(a.duration - b.duration) <= frame_budget,
    }
    checks["geometry"] = {
        "src": [a.width, a.height], "dst": [b.width, b.height],
        "pass": (a.width, a.height) == (b.width, b.height),
    }
    checks["streams"] = {
        "dst_video": b.video_codec, "dst_audio": b.audio_codec,
        "pass": b.video_codec == "h264" and b.audio_codec in ("aac", None),
    }
    diff = media.frame_diff(src, dst, width=320, height=180)
    checks["pixels"] = {
        **diff,
        "tolerance_mean_abs": max_frame_diff,
        "pass": diff["max_diff"] is not None and diff["max_diff"] <= max_frame_diff,
    }
    checks["audio_present"] = {"dst_audio": b.audio_codec, "pass": b.audio_codec is not None or a.audio_codec is None}

    ok = all(c["pass"] for c in checks.values())
    return {
        "operation": "reencode",
        "src": str(src), "dst": str(dst),
        "src_sha256": media.sha256_file(src), "dst_sha256": media.sha256_file(dst),
        "checks": checks,
        "verdict": "PASS" if ok else "FAIL",
        "tool_versions": media.tool_versions(),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def run_g0_baseline(src: Path, out_dir: Path) -> dict[str, Any]:
    """Full G0 baseline: remux + re-encode + verification + logs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    remuxed = out_dir / "baseline_remux.mp4"
    encoded = out_dir / "baseline_reencode.mp4"
    remux(src, remuxed, log=logs / "remux.log")
    reencode(src, encoded, log=logs / "reencode.log")
    v1 = verify_copy(src, remuxed, work=logs / "verify_remux")
    v2 = verify_reencode(src, encoded, work=logs / "verify_reencode")

    report = {
        "gate": "G0",
        "input": {
            "path": str(src),
            "sha256": media.sha256_file(src),
            "info": media.info(src).__dict__ | {"raw": None},
        },
        "outputs": {
            "remux": {"path": str(remuxed), "sha256": media.sha256_file(remuxed),
                      "size_bytes": remuxed.stat().st_size},
            "reencode": {"path": str(encoded), "sha256": media.sha256_file(encoded),
                         "size_bytes": encoded.stat().st_size},
        },
        "verification": {"remux": v1, "reencode": v2},
        "verdict": "PASS" if v1["verdict"] == v2["verdict"] == "PASS" else "FAIL",
        "tool_versions": media.tool_versions(),
        "commands": {
            "remux": f"ffmpeg -y -i {src} -c copy -movflags +faststart {remuxed}",
            "reencode": f"ffmpeg -y -i {src} -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p "
                        f"-c:a aac -b:a 192k -ar 48000 -movflags +faststart {encoded}",
        },
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out_dir / "baseline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
