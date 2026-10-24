"""Deterministic compositor: timeline.json + source video -> rendered video (or stills).

Frames are decoded at constant frame rate in BT.709 RGBA, the virtual camera (punch/push) is
applied with cubic resampling, and graphics are drawn with Skia in reference (source) coordinates
scaled to the output size. Frames with no graphics and no camera move are passed through
untouched, so the speaker's image is only ever altered where the plan asks for it.
"""
from __future__ import annotations

import math
import time
from typing import Iterable, Optional

import numpy as np
import skia

from ..elements import get_element_class, load_all
from ..elements.base import RenderCtx
from ..media import EncodeSettings, FrameReader, FrameWriter, MediaInfo, grab_frame, which_ffmpeg
from ..render.easing import ease, prog
from ..render.fonts import FontBook
from ..util import SMError, log

LAYER_ORDER = {"secondary": 0, "primary": 1, "overlay": 2}


class Compositor:
    def __init__(self, timeline: dict, out_w: Optional[int] = None, out_h: Optional[int] = None,
                 only: Optional[set] = None):
        load_all()
        self.tl = timeline
        self.W, self.H = int(timeline["W"]), int(timeline["H"])
        self.ow, self.oh = int(out_w or self.W), int(out_h or self.H)
        self.s = self.ow / self.W
        fam = (timeline.get("font") or {}).get("family")
        self.fb = FontBook(fam)
        self.ctx = RenderCtx(self.W, self.H, self.fb, timeline["style"], timeline["orientation"])
        self.elements = []
        for spec in timeline["elements"]:
            if only and spec["id"] not in only:
                continue
            cls = get_element_class(spec["type"])
            el = cls(spec, self.ctx)
            self.elements.append((LAYER_ORDER.get(spec.get("layer", "primary"), 1), spec["t_in"], el))
        self.elements.sort(key=lambda x: (x[0], x[1]))
        self.camera = timeline.get("camera") or []
        self.sampling = skia.SamplingOptions(skia.CubicResampler.Mitchell())

    def camera_at(self, t: float) -> tuple[float, tuple[float, float]]:
        for c in self.camera:
            t0, t1 = c["t0"], c["t1"]
            if t < t0 or t >= t1 + c.get("out_dur", 0.0):
                continue
            z = c["zoom"]
            curve = [0.65, 0.0, 0.35, 1.0]
            if c["move"] == "push_in":
                k = ease([0.45, 0.0, 0.55, 1.0], prog(t, t0, max(0.01, c["in_dur"])))
                zz = 1 + (z - 1) * k
            elif c["in_dur"] > 0 and t < t0 + c["in_dur"]:
                zz = 1 + (z - 1) * ease(curve, prog(t, t0, c["in_dur"]))
            else:
                zz = z
            if t >= t1:
                od = c.get("out_dur", 0.0)
                if od <= 0:
                    return 1.0, (0.5, 0.5)
                zz = zz + (1 - zz) * ease(curve, prog(t, t1, od))
            return zz, (c["focus"][0], c["focus"][1])
        return 1.0, (0.5, 0.5)

    def active(self, t: float) -> list:
        return [el for _, _, el in self.elements if el.t_in - 1e-6 <= t < el.t_end]

    def needs_draw(self, t: float) -> bool:
        z, _ = self.camera_at(t)
        return abs(z - 1) > 1e-4 or bool(self.active(t))

    def compose(self, frame: np.ndarray, t: float, alpha_only: bool = False) -> np.ndarray:
        """frame: (oh, ow, 4) uint8 RGBA. Returns a new array (input is not modified)."""
        z, f = self.camera_at(t)
        act = self.active(t)
        if not act and abs(z - 1) <= 1e-4 and not alpha_only:
            return frame
        out = np.zeros_like(frame) if alpha_only else frame.copy()
        surf = skia.Surface(out)
        with surf as canvas:
            if abs(z - 1) > 1e-4 and not alpha_only:
                img = skia.Image.fromarray(frame, colorType=skia.kRGBA_8888_ColorType)
                fx, fy = f[0] * self.ow, f[1] * self.oh
                canvas.save()
                canvas.translate(fx, fy)
                canvas.scale(z, z)
                canvas.translate(-fx, -fy)
                canvas.drawImage(img, 0, 0, self.sampling)
                canvas.restore()
            if act:
                canvas.save()
                canvas.scale(self.s, self.s)
                for el in act:
                    el.draw(canvas, t)
                canvas.restore()
        if not alpha_only:
            out[:, :, 3] = 255
        return out


def _frame_range(tl: dict, start: float, end: Optional[float]) -> tuple[int, int]:
    fps = tl["fps"]
    n = int(round(tl["duration"] * fps))
    f0 = max(0, int(round(start * fps)))
    f1 = n if end is None else min(n, int(round(end * fps)))
    return f0, f1


def render_video(tl: dict, info: MediaInfo, out_path: str, width: Optional[int] = None, height: Optional[int] = None,
                 start: float = 0.0, end: Optional[float] = None, settings: Optional[EncodeSettings] = None,
                 progress: bool = True) -> dict:
    settings = settings or EncodeSettings()
    ow, oh = int(width or info.width), int(height or info.height)
    comp = Compositor(tl, ow, oh)
    f0, f1 = _frame_range(tl, start, end)
    fps = tl["fps"]
    dur = None if (f0 == 0 and end is None) else (f1 - f0) / fps
    reader = FrameReader(info, ow, oh, start=f0 / fps if f0 else 0.0, duration=dur, fps_str=tl["fps_str"])
    writer = FrameWriter(out_path, ow, oh, tl["fps_str"], info, settings, start=f0 / fps if f0 else 0.0, duration=dur)
    t_start = time.time()
    drawn = 0
    i = f0
    last_log = time.time()
    try:
        for frame in reader:
            if i >= f1:
                break
            t = i / fps
            out = comp.compose(frame, t)
            if out is not frame:
                drawn += 1
            writer.write(out)
            i += 1
            if progress and time.time() - last_log > 10:
                el = time.time() - t_start
                log(f"render {i - f0}/{f1 - f0} frames ({(i - f0) / max(el, 1e-6):.1f} fps)")
                last_log = time.time()
    finally:
        reader.close()
        writer.close()
    el = time.time() - t_start
    n = i - f0
    if n < (f1 - f0) - 1:
        log(f"note: decoder delivered {n} of {f1 - f0} expected frames")
    return {"frames": n, "drawn_frames": drawn, "passthrough_frames": n - drawn, "seconds": round(el, 2),
            "fps": round(n / max(el, 1e-6), 2), "size": [ow, oh], "audio": writer.audio_mode, "out": out_path,
            "range": [round(f0 / fps, 3), round(i / fps, 3)]}


def render_overlay(tl: dict, info: MediaInfo, out_path: str, width: Optional[int] = None, height: Optional[int] = None) -> dict:
    """Graphics-only ProRes 4444 with straight alpha, for finishing in an NLE (glass blur becomes a flat tint)."""
    import subprocess

    ffmpeg, _ = which_ffmpeg()
    ow, oh = int(width or info.width), int(height or info.height)
    comp = Compositor(tl, ow, oh)
    f0, f1 = _frame_range(tl, 0, None)
    cmd = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{ow}x{oh}", "-r", tl["fps_str"],
           "-i", "pipe:0", "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-vendor", "apl0", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    blank = np.zeros((oh, ow, 4), np.uint8)
    t0 = time.time()
    for i in range(f0, f1):
        t = i / tl["fps"]
        if comp.needs_draw(t):
            fr = comp.compose(blank, t, alpha_only=True).astype(np.float32)
            a = fr[:, :, 3:4]
            rgb = np.where(a > 0, fr[:, :, :3] * 255.0 / np.maximum(a, 1), 0)
            fr = np.concatenate([np.clip(rgb, 0, 255), a], axis=2).astype(np.uint8)
        else:
            fr = blank
        proc.stdin.write(fr.tobytes())  # type: ignore[union-attr]
    proc.stdin.close()  # type: ignore[union-attr]
    if proc.wait() != 0:
        raise SMError("overlay encode failed: " + proc.stderr.read().decode(errors="ignore")[-600:])  # type: ignore[union-attr]
    return {"frames": f1 - f0, "seconds": round(time.time() - t0, 2), "out": out_path}


def render_stills(tl: dict, info: MediaInfo, times: Iterable[float], width: Optional[int] = None) -> list[np.ndarray]:
    ow = int(width or info.width)
    oh = int(round(info.height * ow / info.width / 2) * 2)
    comp = Compositor(tl, ow, oh)
    fps = tl["fps"]
    out = []
    for t in times:
        # snap to the exact frame time the video render would use
        fi = int(round(t * fps))
        tt = fi / fps
        fr = grab_frame(info, tt, ow, oh)
        out.append(comp.compose(fr, tt))
    return out
