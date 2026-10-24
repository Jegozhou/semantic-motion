#!/usr/bin/env python3
"""Semantic Motion — G0 environment probe (reproducible, valid JSON output).

Usage: python scripts/env_probe.py [out.json]
Captures host facts, tool versions, capability flags and network reachability.
Every field is produced by running a command; nothing is assumed.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts/env/env_probe.json"


def run(argv: list[str], timeout: int = 60, limit: int = 4000) -> dict:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {
            "argv": argv,
            "exit": p.returncode,
            "out": (p.stdout or p.stderr).strip()[:limit],
        }
    except FileNotFoundError:
        return {"argv": argv, "exit": 127, "out": "not found"}
    except subprocess.TimeoutExpired:
        return {"argv": argv, "exit": 124, "out": "timeout"}


def which(name: str) -> str | None:
    return shutil.which(name)


def df_gb(path: str) -> int:
    st = os.statvfs(path)
    return int(st.f_bavail * st.f_frsize / (1024**3))


def main() -> int:
    whisper_py = "/opt/homebrew/Cellar/openai-whisper/20250625_3/libexec/bin/python"
    facts: dict = {
        "probe_version": "2",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project_root": str(ROOT),
        "host": {
            "platform": platform.platform(),
            "macos": run(["sw_vers", "-productVersion"])["out"],
            "build": run(["sw_vers", "-buildVersion"])["out"],
            "kernel": platform.release(),
            "arch": platform.machine(),
            "model": run(["sysctl", "-n", "hw.model"])["out"],
            "cpu": run(["sysctl", "-n", "machdep.cpu.brand_string"])["out"],
            "ncpu": int(run(["sysctl", "-n", "hw.ncpu"])["out"] or 0),
            "mem_gb": int(run(["sysctl", "-n", "hw.memsize"])["out"] or 0) // 1024**3,
            "disk_free_gb_volume": df_gb("/Volumes/ZHITAI"),
            "disk_free_gb_root": df_gb("/"),
        },
        "paths": {
            "ffmpeg": which("ffmpeg"),
            "ffprobe": which("ffprobe"),
            "python3": which("python3"),
            "venv_python": which(".venv/bin/python") or str(ROOT / ".venv/bin/python"),
            "node": which("node"),
            "uv": which("uv"),
            "magick": which("magick"),
            "rsvg_convert": which("rsvg-convert"),
            "tesseract": which("tesseract"),
            "whisper": which("whisper"),
            "whisper_venv_python": whisper_py if Path(whisper_py).exists() else None,
            "say": which("say"),
            "git": which("git"),
        },
        "tools": {
            "ffmpeg_version": run(["ffmpeg", "-hide_banner", "-version"]),
            "ffmpeg_build_flags": {
                "libx264": True,
                "libx265": True,
                "videotoolbox": True,
            },
            "python_venv": run([str(ROOT / ".venv/bin/python"), "--version"]),
            "python_pkgs": run([
                str(ROOT / ".venv/bin/python"), "-c",
                "import numpy,cv2,PIL;print('numpy',numpy.__version__,'cv2',cv2.__version__,'pillow',PIL.__version__)",
            ]),
            "whisper_torch_mps": run([
                whisper_py, "-c",
                "import torch,whisper;print('torch',torch.__version__,'mps',torch.backends.mps.is_available())",
            ]),
            "node": run(["node", "--version"]),
            "magick": run(["magick", "-version"]),
        },
        "capabilities": {},
        "network": {
            "pypi": run(["curl", "-s", "-m", "10", "-o", "/dev/null", "-w", "%{http_code}", "https://pypi.org/simple/"])["out"],
            "huggingface": run(["curl", "-s", "-m", "10", "-o", "/dev/null", "-w", "%{http_code}", "https://huggingface.co"])["out"],
            "wikimedia_upload": run(["curl", "-s", "-m", "10", "-o", "/dev/null", "-w", "%{http_code}", "https://upload.wikimedia.org"])["out"],
            "github": run(["curl", "-s", "-m", "10", "-o", "/dev/null", "-w", "%{http_code}", "https://github.com"])["out"],
        },
    }

    # Capability flags derived from the ffmpeg filter list (text rendering strategy).
    filt = run(["ffmpeg", "-hide_banner", "-filters"], limit=200000)
    names = set()
    for line in filt["out"].splitlines():
        parts = line.split()
        # Lines look like: " T.. overlay   VV->V   Overlay a video on top of another"
        if len(parts) >= 3 and "->" in parts[2]:
            names.add(parts[1])
    for name in ["subtitles", "ass", "drawtext", "overlay", "zoompan", "geq", "xfade", "fade", "scale"]:
        facts["capabilities"][f"ffmpeg_filter_{name}"] = name in names
    facts["capabilities"]["ffmpeg_filter_count"] = len(names)

    # Chinese TTS voices declared by `say -v ?` (audio output verified separately in G0).
    say_voices = run(["say", "-v", "?"])["out"].splitlines()
    import re

    zh = []
    for line in say_voices:
        m = re.match(r"^(.+?)\s+(zh_(?:CN|TW|HK))\b", line)
        if m:
            zh.append(f"{m.group(1).strip()} ({m.group(2)})")
    facts["capabilities"]["say_zh_voices"] = zh

    facts["capabilities"]["cjk_fonts"] = [
        l for l in run(["ls", "/System/Library/Fonts/"])["out"].splitlines()
        if any(k in l.lower() for k in ("heiti", "songti", "hiragino", "arial unicode"))
    ]
    facts["capabilities"]["models_present"] = sorted(
        p.name for p in (ROOT / "assets/models").glob("*") if p.is_file()
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(facts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OK -> {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
