"""Semantic Motion — media utilities.

Deterministic wrappers around ffmpeg/ffprobe plus hashing helpers used by every
other module. No module in this package calls ffmpeg directly; all calls go
through here so that logging, hashing and error codes stay consistent.
"""
from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

# Tool versions are pinned into artifacts so a render can be reproduced.
TOOL_VERSIONS: dict[str, str] = {}


class MediaError(RuntimeError):
    """Typed media failure with a stable error code (see docs/INTERFACES.md)."""

    def __init__(self, code: str, message: str, *, cmd: Sequence[str] | None = None,
                 stderr_tail: str = "") -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.cmd = list(cmd or [])
        self.stderr_tail = stderr_tail

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "cmd": " ".join(shlex.quote(c) for c in self.cmd),
            "stderr_tail": self.stderr_tail[-3000:],
        }


def _tool_version(binary: str) -> str:
    if binary in TOOL_VERSIONS:
        return TOOL_VERSIONS[binary]
    try:
        out = subprocess.run([binary, "-version"], capture_output=True, text=True, timeout=30)
        line = (out.stdout or out.stderr).splitlines()[0]
    except Exception:  # pragma: no cover - environment guard
        line = "unavailable"
    TOOL_VERSIONS[binary] = line
    return line


def tool_versions() -> dict[str, str]:
    return {"ffmpeg": _tool_version(FFMPEG), "ffprobe": _tool_version(FFPROBE)}


def run(argv: Sequence[str], *, timeout: int = 3600, log: Path | None = None) -> str:
    """Run a subprocess, raising MediaError with stderr tail on failure."""
    argv = [str(a) for a in argv]
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            "$ " + " ".join(shlex.quote(a) for a in argv) + "\n\n"
            + (p.stdout or "") + ("\n--- stderr ---\n" + p.stderr if p.stderr else ""),
            encoding="utf-8",
        )
    if p.returncode != 0:
        raise MediaError("E_FFMPEG", f"{argv[0]} exited {p.returncode}",
                         cmd=argv, stderr_tail=p.stderr or p.stdout)
    return p.stdout


def probe(path: str | Path) -> dict[str, Any]:
    """ffprobe JSON for a media file (format + streams)."""
    path = Path(path)
    if not path.exists():
        raise MediaError("E_INPUT_MISSING", f"input not found: {path}")
    out = run([FFPROBE, "-v", "error", "-print_format", "json",
               "-show_format", "-show_streams", str(path)], timeout=120)
    return json.loads(out)


@dataclass(frozen=True)
class MediaInfo:
    path: str
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str | None
    n_video_streams: int
    n_audio_streams: int
    rotation: int = 0
    raw: dict[str, Any] = field(default_factory=dict, compare=False)
    # Exact decoded frame count when the container reports it (nb_frames).
    # duration*fps can be off by a whole frame for VFR/tail-padded files, which
    # must never be confused with a render fault.
    stream_frames: int | None = None

    @property
    def frame_count(self) -> int:
        if self.stream_frames:
            return int(self.stream_frames)
        return int(round(self.duration * self.fps))


def info(path: str | Path) -> MediaInfo:
    path = Path(path)
    pr = probe(path)
    vstreams = [s for s in pr.get("streams", []) if s.get("codec_type") == "video"]
    astreams = [s for s in pr.get("streams", []) if s.get("codec_type") == "audio"]
    if not vstreams:
        raise MediaError("E_NO_VIDEO", f"no video stream in {path}")
    v = vstreams[0]
    num, den = (v.get("r_frame_rate") or "25/1").split("/")
    fps = float(num) / float(den) if float(den) else 25.0
    rotation = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    nb = v.get("nb_frames")
    try:
        nb_frames = int(nb) if nb is not None else None
    except (TypeError, ValueError):
        nb_frames = None
    return MediaInfo(
        path=str(path),
        duration=float(pr["format"].get("duration", 0.0)),
        width=int(v["width"]),
        height=int(v["height"]),
        fps=fps,
        video_codec=v.get("codec_name", "?"),
        audio_codec=astreams[0].get("codec_name") if astreams else None,
        n_video_streams=len(vstreams),
        n_audio_streams=len(astreams),
        rotation=rotation,
        raw=pr,
        stream_frames=nb_frames,
    )


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def audio_pcm_sha256(path: str | Path, *, rate: int = 48000, channels: int = 2,
                     log: Path | None = None) -> str:
    """Hash decoded audio PCM (container-independent audio identity)."""
    with tempfile.NamedTemporaryFile(suffix=".s16le", delete=False) as tf:
        raw = Path(tf.name)
    try:
        run([FFMPEG, "-v", "error", "-y", "-i", str(path), "-vn",
             "-ac", str(channels), "-ar", str(rate), "-f", "s16le", str(raw)], log=log)
        return sha256_file(raw)
    finally:
        raw.unlink(missing_ok=True)


def video_frame_hashes(path: str | Path, *, width: int = 320, height: int = 180,
                       log: Path | None = None) -> list[str]:
    """Per-frame hash of downscaled decoded frames (pixel-identity check)."""
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tf:
        raw = Path(tf.name)
    try:
        run([FFMPEG, "-v", "error", "-y", "-i", str(path),
             "-vf", f"scale={width}:{height}:flags=bilinear,format=rgb24",
             "-f", "rawvideo", str(raw)], log=log)
        data = raw.read_bytes()
        frame_size = width * height * 3
        return [hashlib.sha256(data[i:i + frame_size]).hexdigest()
                for i in range(0, len(data) - frame_size + 1, frame_size)]
    finally:
        raw.unlink(missing_ok=True)


def frame_diff(a: str | Path, b: str | Path, *, width: int = 320, height: int = 180) -> dict[str, Any]:
    """Max/mean absolute pixel difference between two videos' decoded frames."""
    import numpy as np  # local import: keeps --help fast

    ra, rb = _raw_frames(a, width, height), _raw_frames(b, width, height)
    n = min(len(ra), len(rb))
    if n == 0:
        return {"frames_compared": 0, "max_diff": None, "mean_diff": None}
    diffs = [float(np.abs(ra[i] - rb[i]).mean()) for i in range(n)]
    return {
        "frames_compared": n,
        "frames_a": len(ra),
        "frames_b": len(rb),
        "max_diff": max(diffs),
        "mean_diff": sum(diffs) / len(diffs),
    }


def _raw_frames(path: str | Path, width: int, height: int):
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tf:
        raw = Path(tf.name)
    try:
        run([FFMPEG, "-v", "error", "-y", "-i", str(path),
             "-vf", f"scale={width}:{height}:flags=bilinear,format=gray",
             "-f", "rawvideo", str(raw)])
        data = np.frombuffer(raw.read_bytes(), dtype=np.uint8)
        fs = width * height
        return data[: (len(data) // fs) * fs].reshape(-1, fs).astype(np.int16)
    finally:
        raw.unlink(missing_ok=True)


def extract_audio(path: str | Path, out_wav: str | Path, *, rate: int = 16000,
                  channels: int = 1, log: Path | None = None) -> Path:
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    run([FFMPEG, "-v", "error", "-y", "-i", str(path), "-vn",
         "-ac", str(channels), "-ar", str(rate), "-c:a", "pcm_s16le", str(out_wav)], log=log)
    return out_wav


def ffconcat_list(paths: Iterable[str | Path], dest: Path) -> Path:
    """Write an ffmpeg concat-demuxer list with absolute paths."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"file '{Path(p).resolve()}'" for p in paths]
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest
