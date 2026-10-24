"""Render a validated Motion IR to MP4.

Pipeline (docs/INTERFACES.md §3.6, spec §3.5):

1. refuse an invalid IR or a source whose hash differs from the frozen IR;
2. rasterise every element window to an RGBA PNG sequence (``raster.py``);
3. overlay each sequence on the source video in track z-order, at its exact
   frame window, and re-encode the source audio once (content preserved);
4. verify duration / frame count / resolution / audio energy-correlation;
5. write ``render_manifest.json`` with hashes, commands and tool versions.

Only the whitelisted primitives in the IR are consumed; nothing in the document
is executed. Temporary frame sequences are deleted unless ``keep_frames``.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from . import ir as ir_mod
from . import media, raster
from .effects import base

SCHEMA = "semotion.render/1"
ENCODE_ARGS = ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
               "-ar", "48000", "-movflags", "+faststart"]
PREVIEW_CRF = "28"
AUDIO_MIN_CORR = 0.99
AUDIO_MAX_RMS_DELTA_DB = 0.6


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _decode_pcm(path: Path, *, rate: int = 48000, channels: int = 2,
                log: Path | None = None) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".s16le", delete=False) as tf:
        raw = Path(tf.name)
    try:
        media.run([media.FFMPEG, "-v", "error", "-y", "-i", str(path), "-vn",
                   "-ac", str(channels), "-ar", str(rate), "-f", "s16le", str(raw)],
                  log=log)
        return raw.read_bytes()
    finally:
        raw.unlink(missing_ok=True)


def _audio_stats(pcm: bytes, *, rate: int = 48000, channels: int = 2) -> dict:
    import numpy as np

    n = len(pcm) // 2
    samples = np.frombuffer(pcm[: n * 2], dtype="<i2").astype(np.float64)
    if samples.size == 0:
        return {"duration_s": 0.0, "rms_db": -120.0, "samples": 0,
                "sha256": _sha256_bytes(pcm)}
    rms = float(np.sqrt(np.mean(samples ** 2)))
    rms_db = 20.0 * math.log10(max(rms, 1e-9) / 32768.0)
    return {"duration_s": round(samples.size / float(rate * channels), 6),
            "rms_db": round(rms_db, 3), "samples": int(samples.size),
            "sha256": _sha256_bytes(pcm)}


def _correlation(a: bytes, b: bytes) -> float:
    import numpy as np

    na, nb = len(a) // 2, len(b) // 2
    n = min(na, nb)
    if n < 1000:
        return 0.0
    x = np.frombuffer(a[: n * 2], dtype="<i2").astype(np.float64)
    y = np.frombuffer(b[: n * 2], dtype="<i2").astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    den = float(np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y)))
    if den <= 0:
        return 0.0
    return float(np.sum(x * y) / den)


def _frame_count(path: Path) -> int:
    out = media.run([media.FFPROBE, "-v", "error", "-select_streams", "v:0",
                     "-count_frames", "-show_entries", "stream=nb_read_frames",
                     "-of", "default=nokey=1:noprint_wrappers=1", str(path)])
    try:
        return int((out or "0").strip().splitlines()[0])
    except (ValueError, IndexError):
        return 0


def _z_by_element(ir: dict) -> dict[str, int]:
    z: dict[str, int] = {}
    for track in ir.get("tracks") or []:
        try:
            tz = int(track.get("z"))
        except (TypeError, ValueError):
            tz = 20
        for eid in track.get("elements") or []:
            z[str(eid)] = tz
    return z


def _argv_for(ir: dict, source: Path, out: Path, seqs: Sequence[dict], *,
              preview: bool) -> tuple[list[str], str]:
    fps = float(ir["source"]["fps"])
    n_frames = int(ir["source"]["n_frames"])
    fps_str = f"{fps:g}"
    argv: list[str] = [media.FFMPEG, "-hide_banner", "-y", "-v", "error",
                       "-i", str(source)]
    filters: list[str] = ["[0:v]setpts=PTS-STARTPTS[base]"]
    prev = "base"
    for i, seq in enumerate(seqs):
        el = seq["element"]
        argv += ["-framerate", fps_str, "-start_number", str(seq["start_frame"]),
                 "-i", str(Path(seq["frames_dir"]) / "f%05d.png")]
        start_s = float(el["window"]["start_s"])
        end_s = float(el["window"]["end_s"])
        filters.append(f"[{i + 1}:v]format=rgba,setpts=PTS-STARTPTS+{start_s:.6f}/TB[o{i}]")
        nxt = f"v{i}"
        filters.append(f"[{prev}][o{i}]overlay=0:0:eof_action=pass:"
                       f"enable='gte(t,{start_s:.6f})*lt(t,{end_s:.6f})'[{nxt}]")
        prev = nxt
    if preview:
        filters.append(f"[{prev}]scale=iw/2:ih/2,format=yuv420p[vout]")
    else:
        filters.append(f"[{prev}]format=yuv420p[vout]")

    argv += ["-filter_complex", ";".join(filters),
             "-map", "[vout]", "-map", "0:a?",
             "-frames:v", str(n_frames),
             "-r", fps_str,
             "-c:v", "libx264", "-preset", "medium",
             "-crf", PREVIEW_CRF if preview else "18",
             "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
             "-movflags", "+faststart"]
    argv.append(str(out))
    return argv, ";".join(filters)


def render(ir: dict, source: str | Path, out: str | Path, *, preview: bool = False,
           workers: int = 1, work: str | Path | None = None,
           keep_frames: bool = False) -> dict:
    """Render ``ir`` onto ``source`` and return the ``semotion.render/1`` manifest."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.parent / "logs" / f"render_{out.stem}.log"
    ir_hash = str((ir.get("provenance") or {}).get("ir_sha256") or "")
    workdir = Path(work) if work else Path("artifacts/tmp") / (
        f"render_{out.stem}_{(ir_hash or 'nohash')[:12]}")

    try:
        errs = ir_mod.validate(ir)
        if errs:
            raise media.MediaError("E_IR_INVALID",
                                   "refusing to render an invalid IR: " + "; ".join(errs[:5]))
        source = Path(source)
        if not source.exists():
            raise media.MediaError("E_INPUT_MISSING", f"source not found: {source}")
        src_sha = media.sha256_file(source)
        if src_sha != ir["source"]["sha256"]:
            raise media.MediaError(
                "E_IR_INVALID",
                f"source sha256 {src_sha[:12]}… != IR source sha256 "
                f"{str(ir['source']['sha256'])[:12]}… (rendering other material would "
                "break reproducibility)")

        fps = float(ir["source"]["fps"])
        n_frames = int(ir["source"]["n_frames"])
        width, height = int(ir["canvas"]["width"]), int(ir["canvas"]["height"])
        zmap = _z_by_element(ir)
        elements = sorted(ir.get("elements") or [],
                          key=lambda e: (zmap.get(str(e.get("id")), 20),
                                         int(e["window"]["start_frame"])))
        workdir.mkdir(parents=True, exist_ok=True)
        seqs: list[dict] = []
        empty = 0
        rendered_total = 0
        for el in elements:
            eid = str(el.get("id"))
            seq_dir = workdir / eid
            paths = raster.render_element_png(el, {"width": width, "height": height,
                                                   "fps": fps}, seq_dir)
            if not paths:
                empty += 1
                shutil.rmtree(seq_dir, ignore_errors=True)
                continue
            seqs.append({"element": el, "frames_dir": str(seq_dir),
                         "start_frame": int(el["window"]["start_frame"]),
                         "end_frame": int(el["window"]["end_frame"]),
                         "frames": len(paths), "z": zmap.get(eid, 20)})
            rendered_total += len(paths)

        if not seqs:
            # nothing to draw: still produce a valid, re-encoded pass-through
            argv = [media.FFMPEG, "-hide_banner", "-y", "-v", "error",
                    "-i", str(source), "-map", "0:v:0", "-map", "0:a?",
                    "-frames:v", str(n_frames)]
            if preview:
                argv += ["-vf", "scale=iw/2:ih/2"]
            argv += ["-c:v", "libx264", "-preset", "medium",
                     "-crf", PREVIEW_CRF if preview else "18", "-pix_fmt", "yuv420p",
                     "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                     "-movflags", "+faststart", str(out)]
        else:
            argv, _filt = _argv_for(ir, source, out, seqs, preview=preview)

        media.run(argv, timeout=1800, log=log_path)
        # ---- verification
        man_probe = media.probe(out)
        vstreams = [s for s in man_probe.get("streams", []) if s.get("codec_type") == "video"]
        astreams = [s for s in man_probe.get("streams", []) if s.get("codec_type") == "audio"]
        if not vstreams:
            raise media.MediaError("E_RENDER_FAILED", "rendered file has no video stream")
        v = vstreams[0]
        out_w, out_h = int(v.get("width")), int(v.get("height"))
        expected_w, expected_h = (width // 2, height // 2) if preview else (width, height)
        out_frames = _frame_count(out)
        out_duration = float(man_probe.get("format", {}).get("duration", 0.0))
        src_duration = float(ir["source"]["duration_s"])
        duration_delta = abs(out_duration - src_duration)
        problems: list[str] = []
        if (out_w, out_h) != (expected_w, expected_h):
            problems.append(f"resolution {out_w}x{out_h} != expected {expected_w}x{expected_h}")
        if out_frames != n_frames:
            problems.append(f"frame count {out_frames} != source {n_frames}")
        if duration_delta > 1.0 / fps + 0.01:
            problems.append(f"duration delta {duration_delta:.3f}s exceeds 1 frame "
                            f"({1.0 / fps:.3f}s)")
        if not astreams:
            problems.append("rendered file has no audio stream")

        src_pcm = _decode_pcm(source, log=log_path)
        out_pcm = _decode_pcm(out, log=log_path)
        src_audio = _audio_stats(src_pcm)
        out_audio = _audio_stats(out_pcm)
        audio_delta = abs(out_audio["duration_s"] - src_audio["duration_s"])
        rms_delta = abs(out_audio["rms_db"] - src_audio["rms_db"])
        corr = _correlation(src_pcm, out_pcm)
        if not problems:
            if audio_delta > 1.0 / fps + 0.05:
                problems.append(f"audio duration delta {audio_delta:.3f}s")
            if rms_delta > AUDIO_MAX_RMS_DELTA_DB:
                problems.append(f"audio RMS delta {rms_delta:.2f} dB")
            if corr < AUDIO_MIN_CORR:
                problems.append(f"audio correlation {corr:.4f} < {AUDIO_MIN_CORR}")

        manifest = {
            "schema": SCHEMA,
            "ir_sha256": ir_hash,
            "source": {"path": str(source.resolve()), "sha256": src_sha},
            "output": {
                "path": str(out.resolve()),
                "sha256": media.sha256_file(out),
                "size_bytes": out.stat().st_size,
                "duration_s": round(out_duration, 6),
                "n_frames": out_frames,
                "width": out_w, "height": out_h,
                "fps": fps,
                "video_codec": v.get("codec_name"),
                "audio_codec": astreams[0].get("codec_name") if astreams else None,
                "pix_fmt": v.get("pix_fmt"),
            },
            "audio": {
                "pcm_sha256": out_audio["sha256"],
                "source": "passthrough_decode_encode",
                "delta_vs_source_s": round(audio_delta, 6),
                "rms_delta_db": round(rms_delta, 3),
                "correlation": round(corr, 6),
                "method": ("decoded 48 kHz stereo PCM: duration + RMS energy + "
                           "zero-mean correlation vs source (AAC re-encode prevents "
                           "sample-exact identity)"),
                "source_duration_s": src_audio["duration_s"],
            },
            "overlay": {
                "fps": fps,
                "frames_dir": str(workdir.resolve()),
                "rendered_frames": rendered_total,
                "empty_frames": empty,
                "elements": [{"id": s["element"]["id"],
                              "type": s["element"]["type"],
                              "z": s["z"],
                              "start_frame": s["start_frame"],
                              "end_frame": s["end_frame"],
                              "frames": s["frames"],
                              "frames_dir": s["frames_dir"]} for s in seqs],
                "frames_retained": bool(keep_frames),
            },
            "commands": [" ".join(_quote(c) for c in argv)],
            "command_argv": argv,
            "tool_versions": media.tool_versions(),
            "created_utc": _now_utc(),
            "preview": bool(preview),
            "checks": {
                "duration_delta_s": round(duration_delta, 6),
                "duration_tolerance_s": round(1.0 / fps, 6),
                "frames_expected": n_frames,
                "resolution_expected": [expected_w, expected_h],
                "verdict": "PASS" if not problems else "FAIL",
                "problems": problems,
            },
            "log": str(log_path),
        }
        manifest_path = out.parent / "render_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        manifest["manifest_path"] = str(manifest_path.resolve())
        _append_log(log_path, f"\nrender manifest: {manifest_path}\n"
                              f"verdict: {manifest['checks']['verdict']}\n")
        if problems:
            raise media.MediaError("E_RENDER_FAILED",
                                   "render verification failed: " + "; ".join(problems),
                                   cmd=argv)
        return manifest
    except media.MediaError as exc:
        _append_log(log_path, "\nERROR " + json.dumps(exc.as_dict(), ensure_ascii=False) + "\n")
        if exc.code in ("E_IR_INVALID", "E_INPUT_MISSING"):
            raise
        raise media.MediaError("E_RENDER_FAILED", exc.message, cmd=exc.cmd,
                               stderr_tail=exc.stderr_tail) from exc
    except Exception as exc:  # noqa: BLE001 - render boundary
        import traceback
        _append_log(log_path, "\nUNEXPECTED\n" + traceback.format_exc() + "\n")
        raise media.MediaError("E_RENDER_FAILED", f"{type(exc).__name__}: {exc}") from exc
    finally:
        if not keep_frames:
            shutil.rmtree(workdir, ignore_errors=True)


def _quote(s: str) -> str:
    import shlex
    return shlex.quote(s)


def _append_log(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text)
    except OSError:
        pass
