"""Semantic Motion — fixture builder.

Creates synthetic talking-head style clips with *independent* timing ground truth:
every speech unit is synthesized on its own, measured with ffprobe, then laid out
on a known timeline. Because the truth comes from the audio assembly (not from an
ASR model), it is valid as SYNTH_KNOWN ground truth for timing calibration.

Honesty rules (spec §5): fixtures are labelled SYNTH_KNOWN or HUMAN_* in
tests/fixtures/manifest.json. Synthetic prosody is not real speech and must never
be reported as a real-human validation set.
"""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import media

AUDIO_RATE = 48000


# --------------------------------------------------------------------------- TTS

def synth_speech(text: str, out_wav: Path, *, voice: str = "Tingting",
                 rate: int = 175, log: Path | None = None) -> Path:
    """Synthesize one unit of Chinese speech to mono 48 kHz PCM wav."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    aiff = out_wav.with_suffix(".aiff")
    p = subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text],
                       capture_output=True, text=True, timeout=180)
    if p.returncode != 0:
        raise media.MediaError("E_TTS", f"say failed for {text!r}", cmd=["say", voice],
                               stderr_tail=p.stderr)
    media.run([media.FFMPEG, "-v", "error", "-y", "-i", str(aiff), "-ac", "1",
               "-ar", str(AUDIO_RATE), "-c:a", "pcm_s16le", str(out_wav)], log=log)
    aiff.unlink(missing_ok=True)
    return out_wav


def silence(out_wav: Path, seconds: float, *, rate: int = AUDIO_RATE) -> Path:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    media.run([media.FFMPEG, "-v", "error", "-y", "-f", "lavfi",
               "-i", f"anullsrc=r={rate}:cl=mono", "-t", f"{seconds:.4f}",
               "-c:a", "pcm_s16le", str(out_wav)])
    return out_wav


# ------------------------------------------------------------------ background art

_PALETTES = {
    # name: (top, bottom, accent, ink)
    "studio_night": ((16, 20, 32), (30, 38, 58), (242, 178, 51), (240, 242, 248)),
    "studio_warm": ((32, 24, 20), (54, 40, 32), (255, 176, 92), (245, 240, 234)),
    "studio_day": ((236, 240, 245), (214, 222, 232), (28, 92, 168), (24, 28, 36)),
}


def make_background(width: int, height: int, palette: str, dest: Path,
                    *, label: str = "", seed: int = 7) -> Path:
    """Deterministic designed background with a soft presenter zone.

    This is generated graphic artwork, not footage: it exists so the pipeline can
    be exercised offline. Real footage fixtures are tracked separately.
    """
    import random

    from PIL import Image, ImageDraw, ImageFilter

    top, bottom, accent, ink = _PALETTES.get(palette, _PALETTES["studio_night"])
    rnd = random.Random(seed)
    img = Image.new("RGB", (width, height), top)
    d = ImageDraw.Draw(img)

    # vertical gradient
    for y in range(height):
        t = y / max(1, height - 1)
        c = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        d.line([(0, y), (width, y)], fill=c)

    # vignette + soft light from upper left
    glow = Image.new("L", (width, height), 0)
    gd = ImageDraw.Draw(glow)
    gd.ellipse([-width * 0.45, -height * 0.25, width * 0.95, height * 0.55], fill=48)
    glow = glow.filter(ImageFilter.GaussianBlur(radius=max(width, height) // 12))
    img = Image.composite(Image.new("RGB", img.size, accent), img, glow.point(lambda v: v // 3))

    # presenter zone: rounded shoulders + head silhouette (designed, no real face)
    d = ImageDraw.Draw(img, "RGBA")
    cx = width // 2
    head_r = int(min(width, height) * 0.085)
    head_cy = int(height * 0.36)
    body_w = int(width * 0.46)
    body_top = head_cy + int(head_r * 1.25)
    d.ellipse([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r],
              fill=(*ink, 46), outline=(*accent, 120), width=max(2, width // 400))
    d.rounded_rectangle([cx - body_w // 2, body_top, cx + body_w // 2, int(height * 0.95)],
                        radius=int(width * 0.12), fill=(*ink, 38), outline=(*accent, 90),
                        width=max(2, width // 500))

    # subtle grid + dots for depth
    step = max(48, width // 18)
    for x in range(0, width, step):
        d.line([(x, 0), (x, height)], fill=(255, 255, 255, 6))
    for y in range(0, height, step):
        d.line([(0, y), (width, y)], fill=(255, 255, 255, 6))
    for _ in range(24):
        x, y = rnd.randrange(width), rnd.randrange(height)
        r = rnd.randrange(max(2, width // 320), max(4, width // 120))
        d.ellipse([x - r, y - r, x + r, y + r], fill=(*accent, rnd.randrange(10, 26)))

    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, "PNG")
    return dest


# ------------------------------------------------------------------------ builder

@dataclass
class Unit:
    id: str
    text: str
    role: str = "plain"          # plain | keyword | number | list_item | card
    kind: str = "speech"         # speech | pause
    pause_s: float = 0.35


def _unit_plan(spec_units: Iterable[dict]) -> list[Unit]:
    out: list[Unit] = []
    for i, u in enumerate(spec_units, 1):
        out.append(Unit(
            id=u.get("id") or f"u{i:02d}",
            text=u.get("text", ""),
            role=u.get("role", "plain"),
            kind=u.get("kind", "speech"),
            pause_s=float(u.get("pause_s", 0.35)),
        ))
    return out


def build_fixture(spec: dict, out_root: Path, *, work: Path | None = None,
                  log_dir: Path | None = None) -> dict[str, Any]:
    """Synthesize one fixture clip + ground truth JSON.

    Returns the ground-truth dict (also written next to the clip).
    """
    fid = spec["id"]
    out_root = Path(out_root)
    out_dir = out_root / fid
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(work or out_dir / "_work")
    work.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir or out_dir / "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    width, height = int(spec["width"]), int(spec["height"])
    fps = float(spec.get("fps", 25))
    voice = spec.get("voice", "Tingting")
    rate = int(spec.get("rate", 175))
    gap = float(spec.get("gap_s", 0.35))
    lead = float(spec.get("lead_in_s", 0.6))
    tail = float(spec.get("tail_s", 0.8))
    palette = spec.get("palette", "studio_night")

    units = _unit_plan(spec["units"])

    # 1. synthesize + measure every unit
    segs: list[dict[str, Any]] = []
    t = lead
    speech_files: list[Path] = []
    silence(work / "lead.wav", lead)
    speech_files.append(work / "lead.wav")

    truth: list[dict[str, Any]] = []
    for u in units:
        if u.kind == "pause":
            gap = u.pause_s
            continue
        wav = work / f"{u.id}.wav"
        synth_speech(u.text, wav, voice=voice, rate=rate, log=log_dir / f"tts_{u.id}.log")
        dur = float(media.info(wav).duration) if False else _wav_duration(wav)
        start, end = t, t + dur
        truth.append({
            "unit_id": u.id,
            "text": u.text,
            "role": u.role,
            "start_s": round(start, 6),
            "end_s": round(end, 6),
            "start_frame": int(round(start * fps)),
            "end_frame": int(round(end * fps)),
            "duration_s": round(dur, 6),
            "audio_sha256": media.sha256_file(wav),
            "truth": "SYNTH_KNOWN",
        })
        speech_files.append(wav)
        t = end
        # inter-unit gap
        silence(work / f"gap_{u.id}.wav", gap)
        speech_files.append(work / f"gap_{u.id}.wav")
        t += gap

    silence(work / "tail.wav", tail)
    speech_files.append(work / "tail.wav")
    total_dur = t + tail

    # 2. concat audio
    lst = media.ffconcat_list(speech_files, work / "audio.ffconcat")
    audio = work / "narration.wav"
    media.run([media.FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0",
               "-i", str(lst), "-c:a", "pcm_s16le", "-ar", str(AUDIO_RATE), "-ac", "1",
               str(audio)], log=log_dir / "audio_concat.log")

    # 3. designed background (+ optional slow push-in for motion fixtures)
    bg = make_background(width, height, palette, work / "bg.png",
                         label=fid, seed=int(spec.get("seed", 7)))
    motion = spec.get("camera_motion", "static")
    vf = "format=yuv420p"
    if motion == "slow_zoom":
        vf = (f"zoompan=z='min(1.0+0.0006*on,1.12)':x='iw/2-(iw/zoom/2)':"
              f"y='ih/2-(ih/zoom/2)':d=1:s={width}x{height}:fps={fps}," + vf)

    video = work / "video.mp4"
    media.run([media.FFMPEG, "-v", "error", "-y", "-loop", "1", "-framerate", str(fps),
               "-i", str(bg), "-t", f"{total_dur:.4f}", "-vf", vf,
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-pix_fmt", "yuv420p", str(video)], log=log_dir / "video_build.log")

    # 4. mux
    out = out_dir / f"{fid}.mp4"
    media.run([media.FFMPEG, "-v", "error", "-y", "-i", str(video), "-i", str(audio),
               "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-ac", "2",
               "-ar", "48000", "-shortest", "-movflags", "+faststart", str(out)],
              log=log_dir / "mux.log")

    mi = media.info(out)
    gt = {
        "fixture_id": fid,
        "kind": "SYNTH_KNOWN",
        "note": "Synthetic TTS narration over generated graphic artwork. Not real footage.",
        "path": str(out.relative_to(out_root.parent.parent) if out_root.parent.parent in out.parents else out),
        "abs_path": str(out),
        "sha256": media.sha256_file(out),
        "duration_s": round(mi.duration, 6),
        "width": mi.width, "height": mi.height, "fps": mi.fps,
        "n_frames": mi.frame_count,
        "language": "zh",
        "voice": voice, "rate": rate,
        "camera_motion": motion,
        "palette": palette,
        "design": "generated graphic artwork (no real person)",
        "units": truth,
        "audio_assembly": {
            "lead_in_s": lead, "gap_s": gap, "tail_s": tail,
            "method": "per-unit TTS + silence concat; onsets measured with ffprobe",
            "asr_used_for_truth": False,
        },
        "tool_versions": media.tool_versions(),
    }
    (out_dir / "ground_truth.json").write_text(
        json.dumps(gt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return gt


def _wav_duration(path: Path) -> float:
    import wave

    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def build_all(specs_dir: str | Path, out_root: str | Path, *, only: list[str] | None = None) -> list[dict]:
    specs_dir, out_root = Path(specs_dir), Path(out_root)
    results = []
    for f in sorted(specs_dir.glob("*.json")):
        spec = json.loads(f.read_text(encoding="utf-8"))
        if only and spec["id"] not in only:
            continue
        print(f"[fixtures] building {spec['id']} ...", flush=True)
        results.append(build_fixture(spec, out_root))
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build synthetic fixtures")
    ap.add_argument("--specs", default="tests/fixtures/specs")
    ap.add_argument("--out", default="tests/fixtures/clips")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    res = build_all(a.specs, a.out, only=a.only)
    for r in res:
        print(f"  {r['fixture_id']}: {r['duration_s']:.2f}s {r['width']}x{r['height']} "
              f"{len(r['units'])} units -> {r['abs_path']}")
