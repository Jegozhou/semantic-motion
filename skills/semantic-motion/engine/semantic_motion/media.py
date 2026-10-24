"""Media probing and frame-accurate decode/encode through ffmpeg pipes.

Design notes
- Decoding always yields constant-frame-rate RGBA frames at the source's average frame rate
  (phone footage is often VFR); audio is never resampled in time, so A/V sync is preserved.
- Colour: we pass the source matrix/range explicitly to swscale (its default is BT.601, a classic
  source of colour shifts) and encode BT.709 with explicit metadata.
- HDR (HLG / PQ, e.g. iPhone footage) is tone-mapped to SDR BT.709 so overlays and output colours
  are predictable on social platforms.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .util import SMError, log

STANDARD_RATES = [Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001), Fraction(30),
                  Fraction(48), Fraction(50), Fraction(60000, 1001), Fraction(60), Fraction(120)]


def which_ffmpeg() -> tuple[str, str]:
    ff, fp = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ff or not fp:
        raise SMError("ffmpeg/ffprobe not found on PATH. Install ffmpeg (macOS: `brew install ffmpeg`).")
    return ff, fp


@dataclass
class MediaInfo:
    path: str
    width: int                 # display width (after rotation)
    height: int                # display height (after rotation)
    fps: float
    fps_str: str               # rational string for ffmpeg, e.g. "30000/1001"
    duration: float
    nb_frames: int
    video_codec: str
    pix_fmt: str
    rotation: int = 0
    color_space: str = ""
    color_range: str = ""
    color_transfer: str = ""
    color_primaries: str = ""
    is_hdr: bool = False
    vfr_suspected: bool = False
    has_audio: bool = False
    audio_codec: str = ""
    audio_sr: int = 0
    audio_channels: int = 0
    container: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def orientation(self) -> str:
        r = self.width / max(1, self.height)
        return "portrait" if r < 0.9 else ("square" if r < 1.1 else "landscape")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["orientation"] = self.orientation
        return d


def _snap_rate(fr: Fraction) -> Fraction:
    best = min(STANDARD_RATES, key=lambda s: abs(float(s) - float(fr)))
    return best if abs(float(best) - float(fr)) < 0.02 * float(best) else fr.limit_denominator(1001)


def probe(path: str | Path) -> MediaInfo:
    _, ffprobe = which_ffmpeg()
    path = str(path)
    if not Path(path).exists():
        raise SMError(f"input not found: {path}")
    out = subprocess.run([ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SMError(f"ffprobe failed: {out.stderr.strip()[:400]}")
    j = json.loads(out.stdout)
    vs = [s for s in j.get("streams", []) if s.get("codec_type") == "video" and s.get("disposition", {}).get("attached_pic", 0) == 0]
    if not vs:
        raise SMError("no video stream found")
    v = vs[0]
    a = next((s for s in j.get("streams", []) if s.get("codec_type") == "audio"), None)
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                rot = int(round(float(sd["rotation"])))
            except Exception:
                pass
    if "rotate" in (v.get("tags") or {}):
        try:
            rot = int(v["tags"]["rotate"])
        except Exception:
            pass
    w, h = int(v["width"]), int(v["height"])
    if abs(rot) % 180 == 90:
        w, h = h, w
    def _frac(s: str) -> Fraction:
        try:
            n, d = s.split("/")
            return Fraction(int(n), int(d)) if int(d) else Fraction(0)
        except Exception:
            return Fraction(0)
    avg = _frac(v.get("avg_frame_rate", "0/0"))
    rr = _frac(v.get("r_frame_rate", "0/0"))
    fr = avg if avg > 0 else rr
    if fr <= 0:
        fr = Fraction(30)
    vfr = bool(avg > 0 and rr > 0 and abs(float(avg) - float(rr)) > 0.05 * float(rr))
    fr = _snap_rate(fr)
    dur = float(j.get("format", {}).get("duration") or v.get("duration") or 0)
    if v.get("duration"):
        dur = min(dur, float(v["duration"])) if dur else float(v["duration"])
    trc = v.get("color_transfer", "") or ""
    info = MediaInfo(
        path=path, width=w, height=h, fps=float(fr), fps_str=f"{fr.numerator}/{fr.denominator}",
        duration=dur, nb_frames=int(round(dur * float(fr))), video_codec=v.get("codec_name", ""),
        pix_fmt=v.get("pix_fmt", ""), rotation=rot, color_space=v.get("color_space", "") or "",
        color_range=v.get("color_range", "") or "", color_transfer=trc,
        color_primaries=v.get("color_primaries", "") or "",
        is_hdr=trc in ("arib-std-b67", "smpte2084"), vfr_suspected=vfr,
        has_audio=a is not None, audio_codec=(a or {}).get("codec_name", ""),
        audio_sr=int((a or {}).get("sample_rate", 0) or 0), audio_channels=int((a or {}).get("channels", 0) or 0),
        container=j.get("format", {}).get("format_name", ""),
    )
    return info


def _in_matrix(info: MediaInfo) -> str:
    cs = info.color_space
    if cs in ("bt709",):
        return "bt709"
    if cs in ("smpte170m", "bt470bg", "bt601"):
        return "bt601"
    if cs.startswith("bt2020"):
        return "bt2020"
    return "bt709" if info.height >= 720 or info.width >= 1280 else "bt601"


def _in_range(info: MediaInfo) -> str:
    return "pc" if info.color_range in ("pc", "jpeg") else "tv"


def decode_filter(info: MediaInfo, width: int, height: int, fps_str: Optional[str] = None, fast: bool = False) -> str:
    """Filter chain producing CFR RGBA frames at width x height."""
    fps_str = fps_str or info.fps_str
    chain = [f"fps=fps={fps_str}"]
    if info.is_hdr:
        # HLG/PQ -> SDR BT.709 (Hable). Requires an ffmpeg build with zimg (zscale).
        chain += ["zscale=t=linear:npl=100", "format=gbrpf32le", "zscale=p=bt709", "tonemap=tonemap=hable:desat=0",
                  f"zscale=t=bt709:m=bt709:r=tv:w={width}:h={height}", "format=rgba"]
    else:
        flags = "bilinear" if fast else "bicubic+accurate_rnd+full_chroma_int"
        chain.append(f"scale={width}:{height}:flags={flags}:in_color_matrix={_in_matrix(info)}:in_range={_in_range(info)}")
        chain.append("format=rgba")
    return ",".join(chain)


class FrameReader:
    """Iterate CFR RGBA frames (H, W, 4) uint8 from a video via an ffmpeg pipe."""

    def __init__(self, info: MediaInfo, width: Optional[int] = None, height: Optional[int] = None,
                 start: float = 0.0, duration: Optional[float] = None, fps_str: Optional[str] = None,
                 fast: bool = False, threads: int = 0):
        ffmpeg, _ = which_ffmpeg()
        self.info = info
        self.w = int(width or info.width)
        self.h = int(height or info.height)
        self.frame_bytes = self.w * self.h * 4
        cmd = [ffmpeg, "-v", "error", "-nostdin"]
        if threads:
            cmd += ["-threads", str(threads)]
        if start > 0:
            cmd += ["-ss", f"{start:.6f}"]
        cmd += ["-i", info.path]
        if duration is not None:
            cmd += ["-t", f"{duration:.6f}"]
        cmd += ["-map", "0:v:0", "-vf", decode_filter(info, self.w, self.h, fps_str, fast), "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"]
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=self.frame_bytes * 2)

    def __iter__(self) -> Iterator[np.ndarray]:
        assert self.proc.stdout is not None
        while True:
            buf = self.proc.stdout.read(self.frame_bytes)
            if not buf or len(buf) < self.frame_bytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 4).copy()
        self.close()

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdout.close()  # type: ignore[union-attr]
            except Exception:
                pass
            self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


@dataclass
class EncodeSettings:
    crf: int = 18
    preset: str = "medium"
    codec: str = "libx264"
    audio_bitrate: str = "256k"
    threads: int = 0
    extra: list = field(default_factory=list)


class FrameWriter:
    """Encode RGBA frames to H.264 + original audio (copied when possible)."""

    def __init__(self, out_path: str | Path, width: int, height: int, fps_str: str, source: Optional[MediaInfo],
                 settings: EncodeSettings, start: float = 0.0, duration: Optional[float] = None):
        ffmpeg, _ = which_ffmpeg()
        self.out_path = str(out_path)
        Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
        self.w, self.h = width, height
        cmd = [ffmpeg, "-v", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-r", fps_str, "-i", "pipe:0"]
        audio_mode = None
        if source is not None and source.has_audio:
            if start > 0:
                cmd += ["-ss", f"{start:.6f}"]
            if duration is not None:
                cmd += ["-t", f"{duration:.6f}"]
            cmd += ["-i", source.path]
            can_copy = (source.audio_codec == "aac" and start == 0 and duration is None)
            audio_mode = "copy" if can_copy else "aac"
        cmd += ["-map", "0:v:0"]
        if audio_mode:
            cmd += ["-map", "1:a:0"]
        vf = "scale=out_color_matrix=bt709:out_range=tv:flags=bicubic+accurate_rnd+full_chroma_int,format=yuv420p"
        cmd += ["-vf", vf, "-c:v", settings.codec, "-preset", settings.preset, "-crf", str(settings.crf),
                "-profile:v", "high", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-color_range", "tv", "-g", str(max(12, int(round(2 * _fps(fps_str)))))]
        if settings.threads:
            cmd += ["-threads", str(settings.threads)]
        if audio_mode == "copy":
            cmd += ["-c:a", "copy"]
        elif audio_mode == "aac":
            cmd += ["-c:a", "aac", "-b:a", settings.audio_bitrate, "-ar", str(source.audio_sr or 48000)]  # type: ignore[union-attr]
        cmd += settings.extra
        cmd += ["-movflags", "+faststart", self.out_path]
        self.audio_mode = audio_mode
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        assert frame.shape == (self.h, self.w, 4), frame.shape
        try:
            self.proc.stdin.write(memoryview(np.ascontiguousarray(frame)))  # type: ignore[union-attr]
        except BrokenPipeError:
            err = self.proc.stderr.read().decode(errors="ignore") if self.proc.stderr else ""
            raise SMError(f"encoder died: {err[-800:]}")
        self.frames += 1

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        rc = self.proc.wait()
        err = self.proc.stderr.read().decode(errors="ignore") if self.proc.stderr else ""
        if rc != 0:
            raise SMError(f"ffmpeg encode failed ({rc}): {err[-1200:]}")


def _fps(fps_str: str) -> float:
    n, d = fps_str.split("/") if "/" in fps_str else (fps_str, "1")
    return float(n) / float(d)


def read_audio(path: str | Path, sr: int = 16000, start: float = 0.0, duration: Optional[float] = None) -> np.ndarray:
    """Decode mono float32 PCM."""
    ffmpeg, _ = which_ffmpeg()
    cmd = [ffmpeg, "-v", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.6f}"]
    cmd += ["-map", "0:a:0", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0:
        raise SMError(f"audio decode failed: {out.stderr.decode(errors='ignore')[-400:]}")
    return np.frombuffer(out.stdout, np.float32).copy()


def grab_frame(info: MediaInfo, t: float, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
    ffmpeg, _ = which_ffmpeg()
    w = int(width or info.width)
    h = int(height or info.height)
    vf = decode_filter(info, w, h).split(",", 1)[1]  # no fps filter for single grabs
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-ss", f"{max(0.0, t):.6f}", "-i", info.path, "-frames:v", "1",
           "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0 or len(out.stdout) < w * h * 4:
        raise SMError(f"frame grab failed at {t:.2f}s: {out.stderr.decode(errors='ignore')[-300:]}")
    return np.frombuffer(out.stdout[: w * h * 4], np.uint8).reshape(h, w, 4).copy()


def even(x: float) -> int:
    v = int(round(x))
    return v - (v % 2)


def preview_size(info: MediaInfo, long_side: int = 960) -> tuple[int, int]:
    s = long_side / max(info.width, info.height)
    if s >= 1:
        return info.width, info.height
    return even(info.width * s), even(info.height * s)
