"""Quality assurance.

pre  (timeline level, no render): anchors, timing deltas, reading time, face/body/text occlusion
     per analysis sample, safe area, collisions, contrast estimate, density / clean-screen ratio.
post (rendered file): container/stream checks, frame count & duration, audio fidelity (bit-exact
     packet hash when copied, PCM correlation otherwise), A/V start alignment, pass-through PSNR
     on frames without graphics, graphics-present check inside every element box, determinism.
Visual: contact sheet of every element taken from the *rendered output* + timeline strip.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from .layout import box_stats, build_fields
from .media import grab_frame, probe, read_audio, which_ffmpeg
from .style import contrast_ratio
from .util import log, write_json

FACE_IOU_EPS = 0.0


def _rect_inter(a, b) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    return ix * iy


def element_box_at(e: dict) -> list:
    """Footprint used by QA. The compiler stores the *final* box (solver scale step and placement.size already
    applied; `scale` is only the canvas factor the renderer uses to reach it), so it must not be scaled again.
    Verified against rendered alpha bounding boxes: rendered = box + soft shadow (~1.5u) on each side."""
    return list(e["box"])


def pre_qa(proj, tl: dict, compile_report: Optional[dict] = None) -> dict:
    vd = proj.vision()
    W, H = tl["W"], tl["H"]
    safe = tl["safe"]
    style = tl["style"]
    checks = []
    metrics = {}

    def add(level, code, msg, el=None, **kw):
        checks.append({"level": level, "code": code, "element": el, "msg": msg, **kw})

    els = [e for e in tl["elements"] if e["type"] != "captions"]
    land_deltas = []
    for e in els:
        a = e.get("anchor") or {}
        box = element_box_at(e)
        # --- timing
        if a.get("t") is not None:
            d = (e["t_land"] - a["t"]) * 1000
            land_deltas.append(abs(d))
            if abs(d) > 120 and not (e.get("motion") or {}).get("lead"):
                add("info", "LAND_SHIFTED", f"lands {d:+.0f} ms from its spoken anchor (conflict/cut alignment)", e["id"])
        vis = e["t_out"] - e["t_land"]
        if vis < 0.85 * e.get("read_time", 1.0):
            add("warn", "SHORT_READ", f"visible {vis:.1f}s but needs ~{e.get('read_time', 1):.1f}s to read", e["id"])
        # --- occlusion per analysis sample
        # an element that hard-cuts out on a shot boundary must not be judged against the next shot's sample
        t_end_q = e["t_end"] - (0.08 if e.get("exit_mode") == "cut" else 0.0)
        idx = vd.samples_between(e["t_in"], t_end_q)
        bn = [box[0] / W, box[1] / H, box[2] / W, box[3] / H]
        cam = tl.get("camera") or []
        face_hits = 0
        for i in idx:
            ts = float(vd.t[i])
            z, f = 1.0, (0.5, 0.5)
            for c in cam:
                if c["t0"] <= ts < c["t1"]:
                    z, f = c["zoom"], c["focus"]
            for fb in vd.faces[int(i)]:
                x, y, w, h = fb[:4]
                cx, cy = (x + w / 2 - f[0]) * z + f[0], (y + h / 2 - f[1]) * z + f[1]
                fz = [cx - w * z / 2, cy - h * z / 2, w * z, h * z]
                if _rect_inter(bn, fz) > FACE_IOU_EPS:
                    face_hits += 1
                    break
        fl = build_fields(vd, e["t_in"], t_end_q, tuple(safe), cam,
                          chin_margin=0.06 if tl.get("orientation") == "portrait" else None)
        st = box_stats(fl, box, W, H)
        e_q = {"face_hits": face_hits, "samples": int(len(idx)), **st}
        if face_hits:
            add("error", "FACE_OCCLUDED", f"box touches the face in {face_hits}/{len(idx)} sampled frames", e["id"], **e_q)
        if st.get("head_max", 0) > 0.5:
            add("error", "HEAD_OCCLUDED", f"box covers head/hair (p={st['head_max']})", e["id"])
        if st.get("text_max", 0) > 0.5:
            add("error", "TEXT_OCCLUDED", "box covers burned-in text (subtitles/logos)", e["id"])
        if st.get("person_frac", 0) > 0.35 and e["type"] not in ("lower_third", "callout"):
            # over the chest is the norm in vertical video; in landscape it usually means a missed open side
            add("info" if tl.get("orientation") == "portrait" else "warn", "BODY_OVERLAP", f"{st['person_frac'] * 100:.0f}% of the box is over the speaker's body", e["id"])
        # --- safe area
        sx0, sy0, sx1, sy1 = safe[0] * W, safe[1] * H, safe[2] * W, safe[3] * H
        if box[0] < sx0 - 2 or box[1] < sy0 - 2 or box[0] + box[2] > sx1 + 2 or box[1] + box[3] > sy1 + 2:
            add("error", "OUTSIDE_SAFE", "box leaves the platform safe area", e["id"])
        # --- contrast estimate for unbacked text
        if e.get("backing", "none") == "none":
            bg = st.get("lum_p90", 0.3)
            l_bg = bg ** 2.2
            cr = contrast_ratio(1.0, l_bg)
            e_q["contrast_est"] = round(cr, 2)
            if cr < 3.0:
                add("warn", "LOW_CONTRAST", f"white text on bright background (est. {cr:.1f}:1)", e["id"])
        else:
            e_q["contrast_est"] = "plate"
        e["qa"] = e_q
    # --- collisions between simultaneously visible boxes
    for i, a in enumerate(els):
        for b in els[i + 1:]:
            if a["t_in"] < b["t_end"] and b["t_in"] < a["t_end"]:
                if _rect_inter(element_box_at(a), element_box_at(b)) > 0:
                    add("error", "COLLISION", f"overlaps {b['id']} on screen", a["id"])
    # --- density & clean ratio
    dur = tl["duration"]
    prim = [e for e in els if e.get("layer") == "primary"]
    cover = np.zeros(int(dur * 10) + 1, bool)
    for e in prim:
        cover[int(e["t_in"] * 10): int(e["t_end"] * 10)] = True
    clean = 1 - cover.mean() if len(cover) else 1.0
    per_min = len(els) / max(dur / 60, 1e-6)
    dens = style["density"]
    metrics.update(elements=len(els), per_minute=round(per_min, 2), clean_ratio=round(float(clean), 3),
                   land_delta_ms_max=round(max(land_deltas), 1) if land_deltas else None,
                   land_delta_ms_median=round(float(np.median(land_deltas)), 1) if land_deltas else None,
                   camera_moves=len(tl.get("camera") or []))
    if per_min > dens["max_per_min"]:
        add("warn", "TOO_DENSE", f"{per_min:.1f} elements/min (style max {dens['max_per_min']})")
    if clean < dens["clean_ratio"] * 0.8:
        add("warn", "LOW_CLEAN_RATIO", f"only {clean * 100:.0f}% of the time is graphics-free (target {dens['clean_ratio'] * 100:.0f}%)")
    if per_min < dens["target_per_min"] * 0.35 and dur > 20:
        add("info", "SPARSE", f"{per_min:.1f} elements/min — consider more support for key points")
    # --- carry compile issues
    if compile_report:
        for i in compile_report.get("issues", []):
            if i["level"] in ("error", "warn"):
                checks.append({**i, "source": "compile"})
    n_err = sum(1 for c in checks if c["level"] == "error")
    return {"ok": n_err == 0, "errors": n_err, "warnings": sum(1 for c in checks if c["level"] == "warn"), "checks": checks,
            "metrics": metrics, "elements": {e["id"]: e.get("qa", {}) for e in els}}


# ---------------------------------------------------------------------------------------- post
def _ffprobe(path: str) -> dict:
    _, ffprobe = which_ffmpeg()
    out = subprocess.run([ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", "-count_packets", path],
                         capture_output=True, text=True)
    return json.loads(out.stdout or "{}")


def _audio_md5(path: str) -> Optional[str]:
    ffmpeg, _ = which_ffmpeg()
    out = subprocess.run([ffmpeg, "-v", "error", "-i", path, "-map", "0:a:0", "-c", "copy", "-f", "md5", "-"],
                         capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    d = (a[:, :, :3].astype(np.float32) - b[:, :, :3].astype(np.float32)) ** 2
    mse = float(d.mean())
    return 99.0 if mse < 1e-9 else 10 * math.log10(255 ** 2 / mse)


def post_qa(proj, tl: dict, out_path: str, full: bool = True) -> dict:
    info = proj.media()
    src = str(proj.source)
    checks = []
    metrics: dict = {}

    def add(level, code, msg, **kw):
        checks.append({"level": level, "code": code, "msg": msg, **kw})

    pr = _ffprobe(out_path)
    vs = [s for s in pr.get("streams", []) if s.get("codec_type") == "video"]
    aus = [s for s in pr.get("streams", []) if s.get("codec_type") == "audio"]
    if not vs:
        add("error", "NO_VIDEO", "output has no video stream")
        return {"ok": False, "checks": checks, "metrics": metrics}
    v = vs[0]
    n_frames = int(v.get("nb_read_packets") or v.get("nb_frames") or 0)
    exp_frames = int(round(tl["duration"] * tl["fps"]))
    out_info = probe(out_path)
    metrics.update(width=int(v["width"]), height=int(v["height"]), codec=v.get("codec_name"), pix_fmt=v.get("pix_fmt"),
                   frames=n_frames, expected_frames=exp_frames, duration=round(out_info.duration, 3),
                   source_duration=round(info.duration, 3), color=f"{v.get('color_space')}/{v.get('color_transfer')}/{v.get('color_range')}")
    if abs(n_frames - exp_frames) > 1:
        add("error", "FRAME_COUNT", f"{n_frames} frames, expected {exp_frames}")
    if abs(out_info.duration - info.duration) > max(0.1, 2 / tl["fps"]):
        add("warn", "DURATION", f"duration {out_info.duration:.3f}s vs source {info.duration:.3f}s")
    if info.has_audio:
        if not aus:
            add("error", "NO_AUDIO", "output lost the audio track")
        else:
            a = aus[0]
            metrics["audio_codec"] = a.get("codec_name")
            # A/V start alignment
            v_st = float(v.get("start_time", 0) or 0)
            a_st = float(a.get("start_time", 0) or 0)
            metrics["av_start_offset_ms"] = round((a_st - v_st) * 1000, 2)
            sp = _ffprobe(src)
            sv = next((s for s in sp.get("streams", []) if s.get("codec_type") == "video"), {})
            sa = next((s for s in sp.get("streams", []) if s.get("codec_type") == "audio"), {})
            src_off = (float(sa.get("start_time", 0) or 0) - float(sv.get("start_time", 0) or 0)) * 1000
            metrics["source_av_start_offset_ms"] = round(src_off, 2)
            if abs(metrics["av_start_offset_ms"] - src_off) > 1000 / tl["fps"] / 2 + 1:
                add("warn", "AV_OFFSET", f"audio/video start offset {metrics['av_start_offset_ms']} ms vs source {src_off:.1f} ms")
            m_src, m_out = _audio_md5(src), _audio_md5(out_path)
            if m_src and m_out and m_src == m_out:
                metrics["audio_fidelity"] = "bit-exact (stream copied)"
            else:
                x = read_audio(src, 16000)
                y = read_audio(out_path, 16000)
                n = min(len(x), len(y))
                if n > 16000:
                    # best lag within +-50 ms, then correlation
                    best = (0.0, 0)
                    seg = slice(0, min(n, 16000 * 60))
                    for lag in range(-800, 801, 16):
                        xa = x[seg][max(0, lag): n - max(0, -lag)] if lag >= 0 else x[seg][: n + lag]
                        ya = y[seg][: len(xa)] if lag >= 0 else y[seg][-lag: -lag + len(xa)]
                        if len(xa) < 1000:
                            continue
                        c = float(np.corrcoef(xa[: len(ya)], ya[: len(xa)])[0, 1])
                        if c > best[0]:
                            best = (c, lag)
                    metrics["audio_fidelity"] = f"re-encoded, correlation {best[0]:.4f}, lag {best[1] / 16:.1f} ms"
                    if best[0] < 0.98 or abs(best[1]) > 16 * 22:
                        add("error", "AUDIO_CHANGED", metrics["audio_fidelity"])
    # pass-through and graphics-present checks on sampled frames
    if full:
        W, H = tl["W"], tl["H"]
        cam = tl.get("camera") or []
        busy = []
        for e in tl["elements"]:
            busy.append((e["t_in"] - 0.05, e["t_end"] + 0.05))
        for c in cam:
            busy.append((c["t0"] - 0.05, c["t1"] + c.get("out_dur", 0) + 0.05))
        free_t = []
        dur = tl["duration"]
        for k in range(1, 40):
            t = dur * k / 40
            if all(not (a <= t <= b) for a, b in busy):
                free_t.append(t)
        free_t = free_t[:: max(1, len(free_t) // 5)][:5]
        sw = min(960, info.width)
        sh = int(round(info.height * sw / info.width / 2) * 2)
        ps = []
        for t in free_t:
            fi = int(round(t * tl["fps"]))
            tt = (fi + 0.5) / tl["fps"]
            a = grab_frame(info, tt, sw, sh)
            b = grab_frame(out_info, tt, sw, sh)
            ps.append(_psnr(a, b))
        if ps:
            metrics["passthrough_psnr_db"] = {"min": round(min(ps), 2), "mean": round(float(np.mean(ps)), 2), "n": len(ps)}
            if min(ps) < 32:
                add("error", "PASSTHROUGH_DEGRADED", f"frames without graphics differ from source (PSNR {min(ps):.1f} dB)")
        present = {}
        for e in tl["elements"]:
            if e["type"] == "captions":
                continue
            t = min(e["t_out"] - 0.05, e["t_land"] + e.get("enter_dur", 0.46) + 0.3)
            fi = int(round(t * tl["fps"]))
            tt = (fi + 0.5) / tl["fps"]
            a = grab_frame(info, tt, sw, sh)
            b = grab_frame(out_info, tt, sw, sh)
            x, y, w, h = element_box_at(e)
            k = sw / W
            x0, y0 = max(0, int(x * k)), max(0, int(y * k))
            x1, y1 = min(sw, int((x + w) * k)), min(sh, int((y + h) * k))
            d = float(np.abs(a[y0:y1, x0:x1, :3].astype(np.int16) - b[y0:y1, x0:x1, :3].astype(np.int16)).mean()) if x1 > x0 and y1 > y0 else 0
            present[e["id"]] = round(d, 2)
            if d < 2.0:
                add("error", "GRAPHIC_MISSING", f"{e['id']}: no visible change inside its box at {t:.2f}s (mean diff {d:.2f})")
        metrics["graphics_box_mean_diff"] = present
    n_err = sum(1 for c in checks if c["level"] == "error")
    return {"ok": n_err == 0, "errors": n_err, "warnings": sum(1 for c in checks if c["level"] == "warn"), "checks": checks,
            "metrics": metrics, "file": out_path}


def determinism(proj, tl: dict, seconds: float = 1.0) -> dict:
    """Render the same short range twice and compare per-frame hashes of the encoded video."""
    from .media import EncodeSettings
    from .render.engine import render_video

    e = next((e for e in tl["elements"] if e["type"] != "captions"), None)
    t0 = max(0.0, (e["t_land"] if e else 0.0) - 0.2)
    info = proj.media()
    outs = []
    ffmpeg, _ = which_ffmpeg()
    for k in range(2):
        p = str(proj.p("qa", f"_det_{k}.mp4"))
        render_video(tl, info, p, start=t0, end=t0 + seconds, settings=EncodeSettings(preset="veryfast", crf=20, threads=1),
                     progress=False)
        r = subprocess.run([ffmpeg, "-v", "error", "-i", p, "-map", "0:v:0", "-f", "framemd5", "-"], capture_output=True, text=True)
        outs.append([l.split(",")[-1].strip() for l in r.stdout.splitlines() if l and not l.startswith("#")])
        Path(p).unlink(missing_ok=True)
    same = outs[0] == outs[1] and len(outs[0]) > 0
    return {"ok": same, "frames": len(outs[0]), "range": [round(t0, 3), round(t0 + seconds, 3)]}


# ---------------------------------------------------------------------------------------- visuals
def _skia_text(canvas, txt, x, y, size, fb, color=(1, 1, 1, 1), weight=500):
    import skia

    f = fb.font(size, weight)
    p = skia.Paint(AntiAlias=True, Color4f=skia.Color4f(*color))
    xx = x
    for ch in txt:
        ff = f
        if not fb.has_glyph(f, ch):
            tf = fb.fallback_typeface(ch, weight)
            if tf is not None:
                ff = skia.Font(tf, size)
        canvas.drawString(ch, xx, y, ff, p)
        xx += ff.measureText(ch)
    return xx


def full_reveal_time(e: dict) -> float:
    """The moment an element is fully built (all items / custom layers landed) — used for stills and sheets."""
    c = e.get("content") or {}
    items = c.get("items") or c.get("steps") or c.get("data") or []
    ts = [e["t_land"] + e.get("enter_dur", 0.46) + 0.5]
    ts += [it.get("t_land", 0) + 0.6 for it in items if isinstance(it, dict)]
    ts += [(c.get(k) or {}).get("t_land", 0) + 0.6 for k in ("left", "right") if isinstance(c.get(k), dict)]
    for L in c.get("layers", []) or []:
        if isinstance(L, dict):
            f = L.get("from")
            if isinstance(f, str) and f.startswith("@"):
                ts.append(float(f[1:]) + e.get("enter_dur", 0.46) + 0.2)
            for ks in (L.get("keys") or {}).values():
                for k in ks:
                    if isinstance(k, list) and k and isinstance(k[0], str) and k[0].startswith("@"):
                        ts.append(float(k[0][1:]) + 0.25)
    return min(e["t_out"] - 0.05, max(ts))


def contact_sheet(proj, tl: dict, video_path: str, out_path: str, cols: int = 3, tile_w: int = 640) -> str:
    import cv2
    import skia

    from .render.fonts import FontBook

    info = probe(video_path)
    fb = FontBook()
    els = [e for e in tl["elements"] if e["type"] != "captions"]
    th = int(round(info.height * tile_w / info.width / 2) * 2)
    label_h = 54
    rows = max(1, math.ceil(len(els) / cols))
    sheet = np.zeros((rows * (th + label_h) + 10, cols * tile_w + (cols + 1) * 10, 4), np.uint8)
    sheet[:, :, 3] = 255
    sheet[:, :, :3] = 18
    for k, e in enumerate(els):
        t = full_reveal_time(e)
        fi = int(round(t * tl["fps"]))
        fr = grab_frame(info, (fi + 0.5) / tl["fps"], tile_w, th)
        r, c = divmod(k, cols)
        x0, y0 = 10 + c * (tile_w + 10), 10 + r * (th + label_h)
        sheet[y0:y0 + th, x0:x0 + tile_w] = fr
        surf = skia.Surface(sheet)
        with surf as canvas:
            a = e.get("anchor") or {}
            _skia_text(canvas, f"{e['id']} · {e['type']} · {t:.2f}s", x0 + 4, y0 + th + 20, 15, fb, (1, 1, 1, 1), 700)
            _skia_text(canvas, f"anchor S{a.get('s')} “{(a.get('text') or '')[:24]}” @ {a.get('t', 0):.2f}s  "
                               f"in {e['t_in']:.2f} → out {e['t_out']:.2f}", x0 + 4, y0 + th + 42, 13, fb, (1, 1, 1, 0.7))
    cv2.imwrite(out_path, cv2.cvtColor(sheet[:, :, :3], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out_path


def timeline_strip(proj, tl: dict, out_path: str, width: int = 1800) -> str:
    import cv2
    import skia

    from .render.fonts import FontBook

    fb = FontBook()
    tr = proj.transcript()
    af = proj.audio()
    dur = tl["duration"]
    rows_el = sorted(set(e.get("layer", "primary") for e in tl["elements"]), key=lambda l: {"primary": 0, "secondary": 1}.get(l, 2))
    H = 60 + 40 + 46 + 30 + 34 * (len(rows_el) + 1) + 30
    img = np.zeros((H, width, 4), np.uint8)
    img[:, :, 3] = 255
    img[:, :, :3] = 16
    L, R = 70, width - 20
    X = lambda t: L + (R - L) * t / max(dur, 1e-6)  # noqa: E731
    surf = skia.Surface(img)
    with surf as c:
        p = skia.Paint(AntiAlias=True)
        # time ruler
        for s in range(0, int(dur) + 1):
            p.setColor4f(skia.Color4f(1, 1, 1, 0.25 if s % 5 else 0.6))
            c.drawRect(skia.Rect.MakeXYWH(X(s), 14, 1, 8 if s % 5 else 14), p)
            if s % 5 == 0:
                _skia_text(c, f"{s}s", X(s) + 2, 12, 11, fb, (1, 1, 1, 0.6))
        y = 30
        # loudness + VAD
        _skia_text(c, "audio", 8, y + 22, 12, fb, (1, 1, 1, 0.6))
        db = af.db
        n = len(db)
        for px in range(L, R):
            t = (px - L) / (R - L) * dur
            i = min(n - 1, int(t / af.hop))
            v = max(0.0, min(1.0, (db[i] - af.noise_floor) / max(1.0, af.speech_level - af.noise_floor)))
            p.setColor4f(skia.Color4f(0.3, 0.8, 1.0, 0.9) if af.vad[i] else skia.Color4f(0.5, 0.5, 0.5, 0.5))
            c.drawRect(skia.Rect.MakeXYWH(px, y + 30 - 26 * v, 1, max(1, 26 * v)), p)
        y += 40
        # sentences
        _skia_text(c, "speech", 8, y + 22, 12, fb, (1, 1, 1, 0.6))
        for s in tr["sentences"]:
            x0, x1 = X(s["start"]), X(s["end"])
            p.setColor4f(skia.Color4f(1, 1, 1, 0.10))
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(x0, y + 4, x1, y + 40), 4, 4), p)
            c.save()
            c.clipRect(skia.Rect.MakeLTRB(x0, y, x1, y + 44))
            _skia_text(c, f"S{s['id']} {s['text']}", x0 + 4, y + 26, 12, fb, (1, 1, 1, 0.85))
            c.restore()
        y += 46
        # shots
        _skia_text(c, "shots", 8, y + 18, 12, fb, (1, 1, 1, 0.6))
        for k, s in enumerate(proj.shots()):
            p.setColor4f(skia.Color4f(1, 1, 1, 0.16 if k % 2 else 0.08))
            c.drawRect(skia.Rect.MakeLTRB(X(s["start"]), y + 4, X(s["end"]), y + 26), p)
        y += 30
        colors = {"primary": (1.0, 0.78, 0.24, 0.9), "secondary": (0.3, 0.9, 0.66, 0.9), "overlay": (0.6, 0.6, 1.0, 0.8)}
        for li, layer in enumerate(rows_el):
            _skia_text(c, layer[:9], 8, y + 20, 12, fb, (1, 1, 1, 0.6))
            for e in tl["elements"]:
                if e.get("layer", "primary") != layer:
                    continue
                p.setColor4f(skia.Color4f(*colors.get(layer, (1, 1, 1, 0.8))))
                c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(X(e["t_in"]), y + 4, X(e["t_end"]), y + 30), 4, 4), p)
                if e.get("anchor") and e["anchor"].get("t") is not None:
                    p.setColor4f(skia.Color4f(1, 1, 1, 1))
                    c.drawRect(skia.Rect.MakeXYWH(X(e["anchor"]["t"]) - 1, y, 2, 34), p)
                c.save()
                c.clipRect(skia.Rect.MakeLTRB(X(e["t_in"]), y, X(e["t_end"]), y + 34))
                _skia_text(c, f"{e['id']}", X(e["t_in"]) + 4, y + 22, 12, fb, (0.05, 0.05, 0.05, 1), 700)
                c.restore()
            y += 34
        _skia_text(c, "camera", 8, y + 20, 12, fb, (1, 1, 1, 0.6))
        for cm in tl.get("camera") or []:
            p.setColor4f(skia.Color4f(1.0, 0.45, 0.4, 0.85))
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(X(cm["t0"]), y + 4, X(cm["t1"] + cm.get("out_dur", 0)), y + 30), 4, 4), p)
            _skia_text(c, f"{cm['move']} ×{cm['zoom']}", X(cm["t0"]) + 4, y + 22, 12, fb, (0.05, 0.05, 0.05, 1), 700)
    cv2.imwrite(out_path, cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR))
    return out_path


def write_report(proj, pre: dict, post: Optional[dict], extra: Optional[dict] = None) -> dict:
    rep = {"pre": pre, "post": post, **(extra or {})}
    write_json(proj.p("qa", "report.json"), rep)
    lines = ["# QA report", ""]
    ok = pre["ok"] and (post is None or post["ok"])
    lines.append(f"**Overall: {'PASS' if ok else 'FAIL'}** — pre-render {pre['errors']} errors / {pre['warnings']} warnings"
                 + (f"; post-render {post['errors']} errors / {post['warnings']} warnings" if post else "; not rendered"))
    lines += ["", "## Metrics", "", "```json", json.dumps({"pre": pre["metrics"], "post": (post or {}).get("metrics")},
                                                         ensure_ascii=False, indent=1), "```", ""]
    if extra and extra.get("determinism"):
        lines.append(f"Determinism: {'identical' if extra['determinism']['ok'] else 'DIFFERENT'} framemd5 over "
                     f"{extra['determinism']['frames']} frames")
    lines += ["", "## Findings", ""]
    for c in pre["checks"] + ((post or {}).get("checks") or []):
        lines.append(f"- **{c['level'].upper()}** `{c['code']}` {('[' + c['element'] + '] ') if c.get('element') else ''}{c['msg']}")
    if not pre["checks"] and not (post or {}).get("checks"):
        lines.append("- none")
    lines += ["", "## Visual review", "", "- contact sheet: `qa/contact_sheet.jpg` (frames from the rendered output)",
              "- timeline: `qa/timeline.png`", ""]
    proj.p("qa", "report.md").write_text("\n".join(lines), encoding="utf-8")
    return rep
