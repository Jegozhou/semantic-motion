"""Deterministic single-pass renderer for the v2 Motion IR.

Pipeline: decode source frames -> composite backdrops + elements in order ->
draw captions -> encode H.264 with the original audio stream. The same IR and the
same source bytes produce the same frames, and an optional overlay-only pass is
written so that QA can attribute every changed pixel to the motion layer.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import effects2 as FX
from . import media
from . import policy as policy_mod
from . import theme2 as T

SCHEMA = "semotion.render/2"
BASE_ENCODE = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "high",
               "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
               "-movflags", "+faststart"]


# --------------------------------------------------------------------- captions
@dataclass
class CaptionSegment:
    start_s: float
    end_s: float
    lines: list[str]


class CaptionTrack:
    def __init__(self, segments: list[CaptionSegment], *, W: int, H: int, fps: float,
                 book: T.FontBook, palette: T.Palette, bottom: int | None = None):
        self.segments = segments
        self.W, self.H = W, H
        self.fps = fps
        self.book = book
        self.palette = palette
        cap = (policy_mod.DEFAULT.captions or {})
        portrait = H > W
        aspect = "portrait" if portrait else "landscape"
        line_height = float((cap.get("line_height_ratio") or {}).get(aspect, 0.075)) * H
        factor = float(cap.get("line_factor") or 1.46)
        lo = float(cap.get("min_size_ratio") or 0.018) * H
        hi = float(cap.get("max_size_ratio") or 0.09) * H
        self.size = int(max(lo, min(hi, round(line_height / max(1.0, factor)))))
        base_bottom = int(float((cap.get("bottom_ratio") or {}).get(aspect, 0.052)) * H)
        self.bottom = bottom if bottom is not None else base_bottom
        self.pitch = int(self.size * factor)
        self._grad_cache: dict = {}
        self._tile_cache: dict = {}

    def active(self, t: float) -> CaptionSegment | None:
        for s in self.segments:
            if s.start_s - 1e-6 <= t <= s.end_s + 1e-6:
                return s
        return None

    def draw(self, frame: Image.Image, t: float) -> Image.Image:
        seg = self.active(t)
        if not seg or not seg.lines:
            return frame
        fin = min(0.14, max(0.02, (seg.end_s - seg.start_s) * 0.3))
        op = 1.0
        if t < seg.start_s + fin:
            op = T.ease("out_cubic", (t - seg.start_s) / max(1e-3, fin))
        if t > seg.end_s - fin:
            op = min(op, T.ease("in_quad", (seg.end_s - t) / max(1e-3, fin)))
        if op <= 0:
            return frame

        font = self.book.get("semibold", self.size)
        lines = seg.lines[:2]
        block_h = self.pitch * len(lines)
        y = self.H - self.bottom - block_h
        band_top = int(y - self.size * 1.05)
        if band_top < self.H:
            key = band_top
            grad = self._grad_cache.get(key)
            if grad is None:
                h = self.H - band_top
                grad = T.linear_gradient((self.W, h),
                                         [(0.0, (0, 0, 0, 0)), (0.45, (0, 0, 0, 92)),
                                          (1.0, (0, 0, 0, 158))], direction="v")
                self._grad_cache[key] = grad
            if op < 1:
                g = grad.copy()
                g.putalpha(g.getchannel("A").point(lambda v: int(v * op)))
            else:
                g = grad
            frame.alpha_composite(g, (0, band_top))

        for i, ln in enumerate(lines):
            tile = self._tile_cache.get(ln)
            if tile is None:
                tw = int(T.text_width(font, ln)) + 40
                th = self.pitch + 30
                tile = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
                td = ImageDraw.Draw(tile)
                td.text((20, 15 + 2), ln, font=font, fill=(0, 0, 0, 170))
                tile = tile.filter(ImageFilter.GaussianBlur(7))
                td = ImageDraw.Draw(tile)
                td.text((20, 15), ln, font=font, fill=T.alpha(self.palette.ink, 0.98))
                self._tile_cache[ln] = tile
            if op < 0.999:
                tile = _apply_alpha(tile, op)
            lx = int((self.W - (tile.width - 40)) / 2) - 20
            frame.alpha_composite(tile, (lx, int(y + i * self.pitch) - 15))
        return frame


# --------------------------------------------------------------------- helpers
def _sha256_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _fit_from_dict(d: dict) -> FX.Fit:
    return FX.Fit(kind=d.get("kind", "keyword"), width=int(d.get("width") or 10),
                  height=int(d.get("height") or 10), pad=int(d.get("pad") or 0),
                  display_px=int(d.get("display_px") or 0),
                  title_px=int(d.get("title_px") or 0),
                  body_px=int(d.get("body_px") or 0),
                  kicker_px=int(d.get("kicker_px") or 0),
                  pitch=int(d.get("pitch") or 0),
                  lines=list(d.get("lines") or []),
                  body_lines=list(d.get("body_lines") or []),
                  kicker=str(d.get("kicker") or ""),
                  emphasis=list(d.get("emphasis") or []),
                  num_str=str(d.get("num_str") or ""), unit=str(d.get("unit") or ""),
                  label=str(d.get("label") or ""), rows=list(d.get("rows") or []),
                  bars=list(d.get("bars") or []), compare=dict(d.get("compare") or {}),
                  quote=dict(d.get("quote") or {}), rule=bool(d.get("rule")),
                  align=str(d.get("align") or "left"))


def _scrim_2d(w: int, h: int, rgb, a: float, side: str,
              profile: list | None = None, x0: int = 0,
              soft: int = 46) -> Image.Image:
    """Directional scrim with a soft vertical falloff, so it reads as lighting.

    A plain rectangle reads as a pasted panel; multiplying the horizontal ramp by
    a gentle vertical bell makes the gradient blend into the frame at every edge.
    """
    hgrad = T.linear_gradient((w, h), [
        (0.0, (*rgb, 0)), (0.10, (*rgb, int(255 * a * 0.62))),
        (0.32, (*rgb, int(255 * a * 0.92))), (0.72, (*rgb, int(255 * a * 0.98))),
        (1.0, (*rgb, int(255 * a)))], "h") if side == "right" else T.linear_gradient((w, h), [
        (0.0, (*rgb, int(255 * a))), (0.28, (*rgb, int(255 * a * 0.98))),
        (0.68, (*rgb, int(255 * a * 0.92))), (0.90, (*rgb, int(255 * a * 0.62))),
        (1.0, (*rgb, 0))], "h")
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)
    v = 0.70 + 0.30 * np.exp(-((y - 0.44) / 0.62) ** 2)
    al = np.asarray(hgrad.getchannel("A"), dtype=np.float32)
    if profile:
        prof = sorted((int(p[0]), int(p[1])) for p in profile)
        ys = np.array([p[0] for p in prof], dtype=np.float32)
        xs = np.array([p[1] for p in prof], dtype=np.float32)
        rows = np.arange(h, dtype=np.float32)
        edge = np.interp(rows, ys, xs, left=xs[0], right=xs[-1])
        cols = np.arange(w, dtype=np.float32) + float(x0)
        # hard zero for a few pixels past the silhouette edge, then a soft ramp:
        # guarantees the scrim never tints an anti-aliased person pixel
        guard = 6.0
        if side == "right":
            ramp = np.clip((cols[None, :] - edge[:, None] - guard) / max(1.0, soft), 0.0, 1.0)
        else:
            ramp = np.clip((edge[:, None] - cols[None, :] - guard) / max(1.0, soft), 0.0, 1.0)
        al = al * ramp
    al = np.clip(al * v[:, None], 0, 255)
    out = hgrad.copy()
    out.putalpha(Image.fromarray(al.astype(np.uint8), "L"))
    return out


def _backdrop_image(el: dict, W: int, H: int, base: Image.Image | None) -> Image.Image | None:
    b = el.get("backdrop")
    if not b:
        return None
    x, y, w, h = [int(v) for v in b["box"]]
    w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
    kind = b.get("kind")
    a = float(b.get("alpha") or 0.85)
    sc = T.rgba(el.get("_scrim_rgb") or (7, 10, 16))
    rgb = (sc[0], sc[1], sc[2])
    prof = b.get("profile")
    if kind == "scrim_right":
        return _scrim_2d(w, h, rgb, a, "right", prof, x)
    if kind == "scrim_left":
        return _scrim_2d(w, h, rgb, a, "left", prof, x)
    if kind == "band_top":
        return T.linear_gradient((w, h), [
            (0.0, (*rgb, int(255 * a))), (0.72, (*rgb, int(255 * a * 0.86))),
            (1.0, (*rgb, 0))], "v")
    if kind == "band_bottom":
        return T.linear_gradient((w, h), [
            (0.0, (*rgb, 0)), (0.38, (*rgb, int(255 * a * 0.82))),
            (1.0, (*rgb, int(255 * a)))], "v")
    radius = int(min(w, h) * 0.10)
    blur = float(b.get("blur_px") or 24)
    if base is not None:
        panel = T.blur_crop(base, (x, y, w, h), blur, tint=rgb,
                            tint_amount=min(0.92, 0.5 + 0.35 * a))
    else:
        panel = Image.new("RGBA", (w, h), (*rgb, int(255 * a)))
    mask = T.rounded_mask((w, h), radius)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(panel, (0, 0), mask)
    border = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    ImageDraw.Draw(border).rounded_rectangle(
        [0, 0, w * 2 - 1, h * 2 - 1], radius=radius * 2,
        outline=(255, 255, 255, 34), width=2)
    out.alpha_composite(border.resize((w, h), Image.LANCZOS))
    return out


def _apply_alpha(img: Image.Image, op: float) -> Image.Image:
    op = T.clamp(op)
    if op >= 0.999:
        return img
    out = img.copy()
    out.putalpha(out.getchannel("A").point(lambda v: int(v * op)))
    return out


def _element_transform(el: dict, t: float) -> tuple[float, int, float]:
    m = el.get("motion") or {}
    enter = m.get("enter") or {}
    exit_ = m.get("exit") or {}
    edur = float(enter.get("dur_s") or 0.4)
    xdur = float(exit_.get("dur_s") or 0.18)
    dur = max(0.001, float(el["window"]["end_s"]) - float(el["window"]["start_s"]))
    ein = T.ease(enter.get("ease") or "out_expo", T.progress(t, 0.0, max(0.05, edur)))
    # the word owns the frame: type is already readable on the anchor frame
    op = 0.80 + 0.20 * T.ease("out_cubic",
                              T.progress(t, 0.0, max(0.06, min(0.18, edur * 0.42))))
    dy = (1 - ein) * float(enter.get("offset_y") or 0)
    scale = 1.0
    if str(enter.get("style") or "") in ("scale", "pop"):
        s_from = float(enter.get("scale_from") or 0.94)
        scale = s_from + (1.0 - s_from) * ein
    t_exit_start = max(0.0, dur - xdur)
    if t > t_exit_start and xdur > 0:
        xp = T.clamp((t - t_exit_start) / xdur)
        op *= (1 - T.ease("in_quad", xp))
        dy += -float(exit_.get("offset_y") or 0) * T.ease("in_quad", xp)
    return T.clamp(op), int(round(dy)), scale


def _encode_cmd(out: Path, W: int, H: int, fps: float, has_audio: bool,
                source: Path, draft: bool) -> list[str]:
    crf = "28" if draft else "18"
    preset = "veryfast" if draft else "medium"
    cmd = [media.FFMPEG, "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-"]
    if has_audio:
        cmd += ["-i", str(source), "-map", "0:v:0", "-map", "1:a:0", "-shortest"]
    else:
        cmd += ["-map", "0:v:0"]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", crf,
            "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.2"]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    cmd += ["-movflags", "+faststart", str(out)]
    return cmd


# ----------------------------------------------------------------------- render
def render(ir: dict, source: str | Path, out: str | Path, *,
           captions: dict | None = None, perception: dict | None = None,
           overlay_pass: bool = True, draft: bool = False,
           log_path: str | Path | None = None, progress: Any = None,
           max_frames: int | None = None, ink_pass: bool = True) -> dict:
    t_start = time.time()
    source, out = Path(source), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    W = int(ir["canvas"]["width"]); H = int(ir["canvas"]["height"])
    fps = float(ir["canvas"]["fps"])
    info = media.info(source)
    if (info.width, info.height) != (W, H):
        W, H = info.width, info.height

    palette = T.Palette()
    th = ir.get("theme") or {}
    if th.get("accent"):
        palette.accent = T.rgba(th["accent"]); palette.accent_soft = T.alpha(palette.accent, 0.24)
    if th.get("ink"):
        palette.ink = T.rgba(th["ink"])
    if th.get("scrim"):
        palette.scrim = T.rgba(th["scrim"])[:3]
    book = T.FontBook()
    ctx = FX.Ctx(book=book, palette=palette, canvas_w=W, canvas_h=H, fps=fps,
                 language=(ir.get("semantic") or {}).get("language") or "zh")

    elements: list[dict] = []
    import copy as _copy
    for el in ir.get("elements") or []:
        e = dict(el)
        e["_scrim_rgb"] = palette.scrim
        e["_fit"] = _fit_from_dict(el.get("fit") or {})
        ctx_e = ctx
        if e.get("accent"):
            pal = _copy.copy(palette)
            pal.accent = T.rgba(e["accent"])
            pal.accent_soft = T.alpha(pal.accent, 0.24)
            ctx_e = _copy.copy(ctx)
            ctx_e.palette = pal
        e["_ctx"] = ctx_e
        elements.append(e)
    elements.sort(key=lambda e: float(e["window"]["start_s"]))

    cap_track = None
    if captions:
        segs = []
        for s in (captions.get("cues") or captions.get("segments") or []):
            lines = [str(x) for x in (s.get("lines") or [s.get("text", "")]) if str(x).strip()][:2]
            if lines:
                segs.append(CaptionSegment(float(s["start_s"]), float(s["end_s"]), lines))
        if segs:
            bottom = None
            obs = ((perception or {}).get("regions") or {}).get("existing_text") or []
            tops = []
            for o in obs:
                bb = o.get("bbox") or []
                if len(bb) >= 4 and float(bb[1]) > 0.62 * H and float(bb[2]) > 0.02 * W:
                    conf = float(o.get("conf") or o.get("confidence") or 1.0)
                    if conf >= 0.5:
                        tops.append(int(bb[1]))
            if tops:
                cap = policy_mod.DEFAULT.captions or {}
                portrait = H > W
                base = int(float((cap.get("bottom_ratio") or {}).get(
                    "portrait" if portrait else "landscape", 0.052)) * H)
                lift = int(float(cap.get("max_lift_ratio") or 0.06) * H)
                bottom = max(int(0.035 * H), min(base + lift,
                                                 H - min(tops) + int(0.018 * H)))
            cap_track = CaptionTrack(segs, W=W, H=H, fps=fps, book=book, palette=palette,
                                     bottom=bottom)

    backdrop_cache = {e["id"]: _backdrop_image(e, W, H, None) for e in elements
                      if e.get("backdrop") and e["backdrop"].get("kind") != "card"}

    n_frames = int(info.frame_count) if info.stream_frames else int(round(info.duration * fps))
    n_frames = max(1, n_frames)
    if max_frames:
        n_frames = min(n_frames, max_frames)

    dec_cmd = [media.FFMPEG, "-v", "error", "-i", str(source),
               "-vf", f"fps={fps},scale={W}:{H}:flags=lanczos",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    enc_cmd = _encode_cmd(out, W, H, fps, info.n_audio_streams > 0, source, draft)
    ovl_path = out.with_name(out.stem + "_overlay.mp4")
    ovl_cmd = _encode_cmd(ovl_path, W, H, fps, False, source, True)

    log_file = open(log_path, "w", encoding="utf-8") if log_path else None
    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    ovl = subprocess.Popen(ovl_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE) \
        if overlay_pass else None
    ink_path = out.with_name(out.stem + "_ink.mp4")
    ink_cmd = _encode_cmd(ink_path, W, H, fps, False, source, True)
    inkp = subprocess.Popen(ink_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE) \
        if ink_pass else None
    frame_bytes = W * H * 3
    written = 0
    t0 = time.time()
    try:
        while written < n_frames:
            buf = dec.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            base = Image.frombytes("RGB", (W, H), buf).convert("RGBA")
            ov = Image.new("RGBA", (W, H), (0, 0, 0, 255)) if overlay_pass else None
            ink = Image.new("RGBA", (W, H), (0, 0, 0, 255)) if ink_pass else None
            t = written / fps
            for e in elements:
                win = e["window"]
                if not (win["start_s"] - 1e-6 <= t <= win["end_s"] + 1e-6):
                    continue
                b = e.get("backdrop")
                if not b:
                    continue
                bdur = float(b.get("enter_s") or 0.3)
                bx = float(b.get("exit_s") or 0.2)
                op = 0.86 + 0.14 * T.ease("out_cubic", T.progress(t, win["start_s"], bdur))
                rem = win["end_s"] - t
                if rem < bx:
                    op *= T.ease("in_quad", rem / max(1e-3, bx))
                op = T.clamp(op)
                if op <= 0.003:
                    continue
                panel = backdrop_cache.get(e["id"])
                if b.get("kind") == "card":
                    panel = _backdrop_image(e, W, H, base)
                if panel is None:
                    continue
                x, y = [int(v) for v in b["box"][:2]]
                p = _apply_alpha(panel, op)
                base.alpha_composite(p, (x, y))
                if ov is not None:
                    ov.alpha_composite(p, (x, y))
            for e in elements:
                win = e["window"]
                if not (win["start_s"] - 1e-6 <= t <= win["end_s"] + 1e-6):
                    continue
                lt = t - win["start_s"]
                img = FX.paint_element(e, e.get("_ctx") or ctx, e["_fit"], lt,
                                       dur=win["end_s"] - win["start_s"])
                op, dy, scale = _element_transform(e, lt)
                if op <= 0.003:
                    continue
                if op < 0.999:
                    img = _apply_alpha(img, op)
                bx_, by_, bw_, bh_ = [int(v) for v in e["box"]]
                if abs(scale - 1.0) > 0.001:
                    nw, nh = max(1, int(img.width * scale)), max(1, int(img.height * scale))
                    img = img.resize((nw, nh), Image.LANCZOS)
                    bx_ += (bw_ - nw) // 2
                    by_ += (bh_ - nh) // 2
                box = (0, 0, min(bw_, img.width), min(bh_, img.height))
                base.alpha_composite(img.crop(box), (bx_, by_ + dy))
                if ov is not None:
                    ov.alpha_composite(img.crop(box), (bx_, by_ + dy))
                if ink is not None:
                    ink.alpha_composite(img.crop(box), (bx_, by_ + dy))
            if cap_track:
                base = cap_track.draw(base, t)
                if ov is not None:
                    ov = cap_track.draw(ov, t)
            try:
                enc.stdin.write(base.convert("RGB").tobytes())
                if ovl is not None and ov is not None:
                    ovl.stdin.write(ov.convert("RGB").tobytes())
                if inkp is not None and ink is not None:
                    inkp.stdin.write(ink.convert("RGB").tobytes())
            except (BrokenPipeError, IOError):
                break
            written += 1
            if progress and (written % 50 == 0 or written == n_frames):
                progress(written, n_frames)
        for p in (enc, ovl, inkp):
            if p:
                try:
                    p.stdin.close()
                except Exception:
                    pass
        # stop the decoder if we deliberately read only part of the stream
        try:
            dec.stdout.close()
        except Exception:
            pass
        if dec.poll() is None:
            dec.terminate()
        enc.wait(timeout=900)
        if ovl:
            ovl.wait(timeout=900)
        if inkp:
            inkp.wait(timeout=900)
        try:
            dec.wait(timeout=20)
        except Exception:
            dec.kill()
    finally:
        if log_file:
            for name, p in (("decoder", dec), ("encoder", enc), ("overlay", ovl), ("ink", inkp)):
                if not p:
                    continue
                if p.poll() is None:
                    p.kill()
                try:
                    err = p.stderr.read().decode("utf-8", "replace") if p.stderr else ""
                except Exception:
                    err = ""
                if err.strip():
                    log_file.write(f"[{name}]\n{err}\n")
            log_file.close()
        for p in (dec, enc, ovl, inkp):
            if p and p.poll() is None:
                p.kill()

    elapsed = time.time() - t_start
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {"path": str(source), "sha256": media.sha256_file(source),
                   "width": info.width, "height": info.height, "fps": info.fps,
                   "duration_s": info.duration, "frames": info.frame_count},
        "output": {"path": str(out),
                   "sha256": media.sha256_file(out) if out.exists() else None,
                   "width": W, "height": H, "fps": fps, "frames": written,
                   "duration_s": round(written / fps, 4),
                   "bytes": out.stat().st_size if out.exists() else 0},
        "overlay_pass": str(ovl_path) if overlay_pass and ovl_path.exists() else None,
        "ink_pass": str(ink_path) if ink_pass and ink_path.exists() else None,
        "ir": {"schema": ir.get("schema"), "elements": len(elements),
               "sha256": _sha256_obj(ir.get("elements"))},
        "captions": ({"segments": len(captions.get("cues") or captions.get("segments") or [])}
                     if captions else None),
        "encode": {"video": "libx264", "crf": 28 if draft else 18, "draft": draft,
                   "audio": "aac 192k" if info.n_audio_streams else None},
        "elapsed_s": round(elapsed, 2),
        "fps_real": round(written / max(1e-6, elapsed), 2),
        "tools": media.tool_versions(),
        "command_decode": " ".join(dec_cmd),
        "command_encode": " ".join(enc_cmd),
    }
    man_path = out.with_name(out.stem + ".manifest.json")
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


# --------------------------------------------------------------------- frames
def grab_frames(source: str | Path, times: Sequence[float],
                size: tuple[int, int] | None = None) -> list[Image.Image]:
    src = Path(source)
    out: list[Image.Image] = []
    for t in times:
        cmd = [media.FFMPEG, "-v", "error", "-ss", f"{max(0.0, t):.4f}", "-i", str(src),
               "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
        try:
            raw = subprocess.run(cmd, check=True, capture_output=True).stdout
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            if size and img.size != size:
                img = img.resize(size, Image.LANCZOS)
            out.append(img)
        except Exception:
            continue
    return out


def contact_sheet(ir: dict, source: str | Path, out_png: str | Path, *,
                  max_cells: int = 6, label: bool = True) -> str | None:
    els = list(ir.get("elements") or [])
    if not els:
        return None
    W = int(ir["canvas"]["width"]); H = int(ir["canvas"]["height"])
    picks = els[:max_cells]
    cells = []
    for e in picks:
        t = (float(e["window"]["start_s"]) + float(e["window"]["end_s"])) / 2
        fr = grab_frames(source, [t])
        cells.append((e, t, fr[0] if fr else None))
    if not any(c[2] for c in cells):
        return None
    cols = min(3, len(cells))
    rows = int(math.ceil(len(cells) / cols))
    thumb_w = 640
    thumb_h = int(thumb_w * H / W)
    pad, head = 14, 36
    sheet = Image.new("RGB", (cols * thumb_w + (cols + 1) * pad,
                              rows * (thumb_h + head) + (rows + 1) * pad), (16, 18, 22))
    d = ImageDraw.Draw(sheet)
    font = T.FontBook().get("semibold", 17)
    for i, (e, t, img) in enumerate(cells):
        r, c = divmod(i, cols)
        x = pad + c * (thumb_w + pad)
        y = pad + r * (thumb_h + head + pad)
        if img is not None:
            sheet.paste(img.resize((thumb_w, thumb_h), Image.LANCZOS), (x, y))
        if label:
            c_ = e.get("content") or {}
            txt = (f"{e.get('type')} · {t:.2f}s · "
                   f"{str(c_.get('text') or c_.get('title') or c_.get('value') or '')[:26]}")
            d.text((x + 2, y + thumb_h + 9), txt, font=font, fill=(210, 215, 225))
    sheet.save(out_png)
    return str(out_png)
