"""Semantic Motion — measurable quality metrics (spec §6).

Everything here is computed from real artifacts: the Motion IR, the rendered MP4,
the source MP4 and (when available) independent ground-truth event times. Nothing
is estimated from the same model that produced the timing.

Method summary
--------------
timing
    rendered onset = first frame inside the element window whose decoded pixels
    differ from the source by more than an adaptive threshold (noise floor from
    frames before the window). Target onset = independently labelled event time
    (SYNTH_KNOWN assembly times or HUMAN labels). error = |onset - target|.
position
    two independent views:
      * planned  — element boxes from the IR checked against detected face boxes;
      * observed — changed-pixel bounding box of the rendered frame checked against
        the same face boxes (catches rasterizer overflow that planned boxes miss).
render
    duration / frame-count identity, decoded-audio statistics, determinism.
"""
from __future__ import annotations

import json
import statistics
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from . import media

FAIL = "FAIL"
PASS = "PASS"
NOT_RUN = "NOT RUN"
BLOCKED = "BLOCKED"

P90_THRESHOLD_MS = 100.0
FACE_OVERLAP_THRESHOLD = 0.0
FACE_INSIDE_AREA_RATIO = 0.01      # >1% of the face box covered counts as overlap
LOW_W = 192                        # width of the low-res analysis grid
COLLISION_THRESHOLD = 0.01
COLLISION_AREA_RATIO = 0.05


# --------------------------------------------------------------------- helpers

def _gray_frames(path: Path, width: int = LOW_W, height: int | None = None) -> np.ndarray:
    """Decoded grayscale frames as (n, h*w) int16."""
    mi = media.info(path)
    h = height or max(2, int(round(width * mi.height / mi.width / 2)) * 2)
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tf:
        raw = Path(tf.name)
    try:
        media.run([media.FFMPEG, "-v", "error", "-y", "-i", str(path),
                   "-vf", f"scale={width}:{h}:flags=bilinear,format=gray",
                   "-f", "rawvideo", str(raw)])
        data = np.frombuffer(raw.read_bytes(), dtype=np.uint8)
        fs = width * h
        return data[: (len(data) // fs) * fs].reshape(-1, fs).astype(np.int16)
    finally:
        raw.unlink(missing_ok=True)


def _audio_stats(path: Path) -> dict[str, float]:
    """Decoded audio length, RMS and peak (PCM s16le 48k stereo)."""
    with tempfile.NamedTemporaryFile(suffix=".s16le", delete=False) as tf:
        raw = Path(tf.name)
    try:
        media.run([media.FFMPEG, "-v", "error", "-y", "-i", str(path), "-vn",
                   "-ac", "2", "-ar", "48000", "-f", "s16le", str(raw)])
        data = np.frombuffer(raw.read_bytes(), dtype=np.int16).astype(np.float32) / 32768.0
        if data.size == 0:
            return {"n_samples": 0, "duration_s": 0.0, "rms_db": -120.0, "peak_db": -120.0}
        rms = float(np.sqrt(np.mean(data ** 2)))
        peak = float(np.max(np.abs(data)))
        return {
            "n_samples": int(data.size),
            "duration_s": round(data.size / 2 / 48000, 6),
            "rms_db": round(20 * np.log10(max(rms, 1e-9)), 3),
            "peak_db": round(20 * np.log10(max(peak, 1e-9)), 3),
        }
    finally:
        raw.unlink(missing_ok=True)


_T2S = None


def _to_simplified(text: str) -> str:
    """Fold traditional Chinese to simplified so ASR script variants still match.

    Whisper occasionally emits traditional characters for Mandarin; matching text
    must not silently fail because of that (opencc is optional, pure Python).
    """
    global _T2S
    if _T2S is None:
        try:
            import opencc
            _T2S = opencc.OpenCC("t2s")
        except Exception:  # pragma: no cover - optional dependency
            _T2S = False
    if _T2S:
        return _T2S.convert(text)
    return text


def _norm_text(s: str) -> str:
    return "".join(ch for ch in _to_simplified(str(s)) if ch.isalnum())


def _match_truth(anchor_text: str, units: list[dict]) -> dict | None:
    """Match an anchor to a SYNTH_KNOWN unit by character overlap (>=60%)."""
    a = _norm_text(anchor_text)
    if not a:
        return None
    best, best_score = None, 0.0
    for u in units:
        t = _norm_text(u.get("text", ""))
        if not t:
            continue
        inter = sum(1 for ch in a if ch in t)
        score = inter / max(len(a), len(t))
        if a in t or t in a:
            score = max(score, 0.85)
        if score > best_score:
            best, best_score = u, score
    return best if best_score >= 0.6 else None


def _face_boxes_by_frame(face_frames: list[dict]) -> dict[int, list[list[int]]]:
    out: dict[int, list[list[int]]] = {}
    for f in face_frames:
        out.setdefault(int(f["frame"]), []).append([int(v) for v in f["box"]])
    return out


def _nearest_faces(faces: dict[int, list[list[int]]], frame: int, tol: int = 8) -> list[list[int]]:
    for d in range(tol + 1):
        for cand in (frame, frame - d, frame + d):
            if cand in faces:
                return faces[cand]
    return []


def _inter_area(a: list[float], b: list[float]) -> float:
    x = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    y = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return x * y


# ---------------------------------------------------------------------- metrics

def _unit_matches(anchor_text: str, units: list[dict]) -> list[dict]:
    """Units whose text *starts with* the anchor (prefix match, no timing used).

    Only a unit-initial anchor can be validated against the unit's independently
    measured onset. An anchor that sits inside a longer unit has no independent
    target in a SYNTH_KNOWN assembly, so it is reported as not measurable rather
    than being graded against a number derived from ASR.
    """
    a = _norm_text(anchor_text)
    if len(a) < 2:
        return []
    out = []
    for u in units:
        t = _norm_text(u.get("text", ""))
        if t.startswith(a):
            out.append(u)
    return out


def _global_noise(ir: dict, rendered: np.ndarray, source: np.ndarray) -> float:
    """Changed-pixel ratio baseline from frames outside every element window."""
    n = len(rendered)
    inside = np.zeros(n, dtype=bool)
    for el in ir.get("elements", []):
        w = el["window"]
        inside[max(0, int(w["start_frame"]) - 2):min(n, int(w["end_frame"]) + 2)] = True
    idx = np.nonzero(~inside)[0]
    if idx.size == 0:
        return 0.0
    step = max(1, idx.size // 60)
    sample = idx[::step][:60]
    ratios = (np.abs(rendered[sample].astype(np.int32) - source[sample].astype(np.int32)) > 8).mean(axis=1)
    return float(np.median(ratios))


def _region_series(el: dict, rendered: np.ndarray, source: np.ndarray,
                   width: int, height: int, lo: int, hi: int) -> np.ndarray:
    """Mean |Δ| per frame inside the element's own declared box (low-res grid)."""
    box = el["layout"]["box"]
    x0 = max(0, int(box[0] * LOW_W / width) - 3)
    x1 = min(LOW_W, int((box[0] + box[2]) * LOW_W / width) + 3)
    hh = rendered.shape[1] // LOW_W
    y0 = max(0, int(box[1] * hh / height) - 3)
    y1 = min(hh, int((box[1] + box[3]) * hh / height) + 3)
    if x1 <= x0 or y1 <= y0:
        return np.zeros(hi - lo)
    a = rendered[lo:hi].reshape(-1, hh, LOW_W)[:, y0:y1, x0:x1].astype(np.int32)
    b = source[lo:hi].reshape(-1, hh, LOW_W)[:, y0:y1, x0:x1].astype(np.int32)
    return np.abs(a - b).mean(axis=(1, 2))


def _timing(ir: dict, rendered: np.ndarray, source: np.ndarray, truth: dict | None,
            fps: float, global_noise: float = 0.0) -> dict[str, Any]:
    """Measure rendered onsets and compare with independent truth where available.

    Onset is measured *inside each element's own box* (localised), so a faint fade-in
    on a small card is still detected, and neighbour elements cannot mask it.
    Targets come from SYNTH_KNOWN unit starts for anchors that begin a unit.
    """
    width = int(ir["canvas"]["width"])
    height = int(ir["canvas"]["height"])
    units = (truth or {}).get("units", []) if truth else []
    try:  # prefer the anchors module's shared matching rule when available
        from . import anchors as anchors_mod
        prefix_match = getattr(anchors_mod, "prefix_matches", None)
    except Exception:  # pragma: no cover
        prefix_match = None

    elements = ir.get("elements", [])
    anchor_index = {a.get("id"): a for a in ir.get("semantic", {}).get("anchors", [])}
    rows: list[dict[str, Any]] = []

    for el in elements:
        w = el["window"]
        sf, ef = int(w["start_frame"]), int(w["end_frame"])
        # Search from one frame before the declared start: an earlier neighbour may
        # share the same box, and its tail must not be mistaken for this onset.
        lo, hi = max(0, sf - 1), min(len(rendered), ef + 3)
        if hi - lo < 2:
            continue
        series = _region_series(el, rendered, source, width, height, lo, hi)
        pre_lo, pre_hi = max(0, sf - 10), max(1, sf - 1)
        if pre_hi - pre_lo >= 2:
            pre = _region_series(el, rendered, source, width, height, pre_lo, pre_hi)
            # 20th percentile: the pre-window may still contain the tail of the
            # previous element, and that must not raise the onset gate above the
            # amplitude of the element being measured.
            noise = float(np.percentile(pre, 20))
        else:
            noise = float(global_noise) * 255.0
        peak = float(series.max()) if series.size else 0.0
        thr = max(0.8, noise * 2.5 + 0.4, 0.05 * peak)
        if peak > 2.0:
            thr = min(thr, 0.5 * peak)
        onset_idx = None
        for k, val in enumerate(series):
            if val > thr:
                onset_idx = lo + k
                break
        rendered_onset_s = (onset_idx / fps) if onset_idx is not None else None

        anchor = anchor_index.get(el.get("anchor_ref"))
        anchor_start = float(anchor.get("start_s")) if anchor else None

        target_s, truth_unit = None, None
        if units and anchor:
            cands = _unit_matches(anchor.get("text", ""), units)
            if len(cands) == 1:
                truth_unit = cands[0]
                target_s = float(truth_unit["start_s"])

        rows.append({
            "element_id": el["id"],
            "effect": el.get("type"),
            "text": (el.get("content") or {}).get("text"),
            "window_start_s": w["start_s"],
            "rendered_onset_s": None if rendered_onset_s is None else round(rendered_onset_s, 4),
            "anchor_start_s": anchor_start,
            "fidelity_ms": (None if (rendered_onset_s is None or anchor_start is None)
                            else round(abs(rendered_onset_s - anchor_start) * 1000, 2)),
            "target_onset_s": None if target_s is None else round(target_s, 4),
            "truth_unit": None if not truth_unit else truth_unit.get("unit_id"),
            "error_ms": (None if (rendered_onset_s is None or target_s is None)
                         else round(abs(rendered_onset_s - target_s) * 1000, 2)),
            "detected": rendered_onset_s is not None,
            "measurable_against_truth": target_s is not None,
            "peak_mean_abs_diff": round(peak, 3),
            "detect_threshold": round(thr, 3),
        })

    errors = [r["error_ms"] for r in rows if r["error_ms"] is not None]
    fidelity = [r["fidelity_ms"] for r in rows if r["fidelity_ms"] is not None]
    base = {
        "samples": len(errors),
        "elements": len(rows),
        "detected_onsets": sum(1 for r in rows if r["detected"]),
        "rows": rows,
        "source": (truth or {}).get("kind", "none"),
        "threshold_p90_ms": P90_THRESHOLD_MS,
        "measurement": "localised per-element box diff (mean |Δ|) vs pre-window noise floor",
        "fidelity": {
            "samples": len(fidelity),
            "p50_ms": round(statistics.median(fidelity), 2) if fidelity else None,
            "max_ms": round(max(fidelity), 2) if fidelity else None,
        },
    }
    if not errors:
        base.update({"p50_ms": None, "p90_ms": None, "max_ms": None,
                     "verdict": NOT_RUN,
                     "note": ("truth file present but no element onset could be matched to an "
                              "independent unit start (anchors inside longer units) — this is "
                              "NOT RUN, not a pass and not an environment block")})
        return base
    e = sorted(errors)
    p90 = e[max(0, int(round(0.9 * (len(e) - 1))))]
    by_effect: dict[str, list[float]] = {}
    for r in rows:
        if r["error_ms"] is not None:
            by_effect.setdefault(str(r["effect"]), []).append(r["error_ms"])
    base.update({
        "p50_ms": round(statistics.median(e), 2), "p90_ms": round(p90, 2), "max_ms": round(max(e), 2),
        "by_effect": {k: {"n": len(v), "p50_ms": round(statistics.median(v), 2),
                          "max_ms": round(max(v), 2)} for k, v in by_effect.items()},
        "frame_quantum_ms": round(1000.0 / fps, 2),
        "verdict": PASS if p90 <= P90_THRESHOLD_MS else FAIL,
        "note": ("rendered onset measured inside each element box; target = independent "
                 "SYNTH_KNOWN unit start of a unit the anchor begins. ASR never defines "
                 "the target. n<50 is exploratory, not a P90 claim."),
    })
    return base


def _position(ir: dict, rendered: np.ndarray, source: np.ndarray, vision: dict,
              fps: float, width: int, height: int, *, max_samples: int = 240) -> dict[str, Any]:
    faces = _face_boxes_by_frame(vision.get("faces", []))
    elements = ir.get("elements", [])

    # ---- planned geometry (static boxes per element)
    planned_face_hits = 0
    planned_frames = 0
    planned_oob = 0
    for el in elements:
        box = el["layout"]["box"]
        x, y, w, h = box
        if x < 0 or y < 0 or x + w > width or y + h > height:
            planned_oob += 1
        wnd = el["window"]
        for fr in range(int(wnd["start_frame"]), int(wnd["end_frame"]), max(1, int(fps // 2))):
            fb = _nearest_faces(faces, fr)
            if not fb:
                continue
            planned_frames += 1
            covered = sum(_inter_area(box, f) for f in fb)
            face_area = sum(f[2] * f[3] for f in fb) or 1.0
            if covered / face_area > FACE_INSIDE_AREA_RATIO and not el["layout"].get("allow_face_overlap"):
                planned_face_hits += 1

    # ---- observed geometry (changed pixels vs source)
    obs_face_hits = 0
    obs_frames = 0
    obs_oob = 0
    observed_windows = 0
    containment = {"changed": 0, "inside": 0}
    stride = max(1, len(rendered) // max_samples)
    for el in elements:
        wnd = el["window"]
        lo, hi = int(wnd["start_frame"]), min(int(wnd["end_frame"]), len(rendered))
        observed_any = False
        for fr in range(lo, hi, stride):
            d = np.abs(rendered[fr].astype(np.int32) - source[fr].astype(np.int32))
            mask = d > 8
            if not mask.any():
                continue
            observed_any = True
            obs_frames += 1
            # mask is a flat grayscale frame of width 192; rebuild 2D and map back
            hh = mask.size // LOW_W
            m2 = mask.reshape(hh, LOW_W)
            ys, xs = np.nonzero(m2)
            if xs.size == 0:
                continue
            bx0, bx1 = xs.min() * width / LOW_W, (xs.max() + 1) * width / LOW_W
            by0, by1 = ys.min() * height / hh, (ys.max() + 1) * height / hh
            if bx0 < 1 or by0 < 1 or bx1 > width - 1 or by1 > height - 1:
                obs_oob += 1
            fbox = [bx0, by0, bx1 - bx0, by1 - by0]
            # containment: changed pixels must fall inside the declared box
            pad = 6.0
            box = el["layout"]["box"]
            bx_lo, by_lo = box[0] - pad, box[1] - pad
            bx_hi, by_hi = box[0] + box[2] + pad, box[1] + box[3] + pad
            changed = int(m2.sum())
            inside = int(m2[max(0, int(by_lo * hh / height)):int(by_hi * hh / height),
                            max(0, int(bx_lo * LOW_W / width)):int(bx_hi * LOW_W / width)].sum())
            containment["changed"] += changed
            containment["inside"] += inside
            for f in _nearest_faces(faces, fr):
                covered = _inter_area(fbox, f)
                face_area = f[2] * f[3] or 1.0
                if covered / face_area > FACE_INSIDE_AREA_RATIO:
                    obs_face_hits += 1
                    break
        if observed_any:
            observed_windows += 1

    # ---- unintended collisions among simultaneously visible planned elements
    collision_pairs = 0
    compared = 0
    allowed = {tuple(sorted(p)) for p in ir.get("collision_ok", [])}
    for i, a in enumerate(elements):
        for b in elements[i + 1:]:
            wa, wb = a["window"], b["window"]
            if not (wa["start_frame"] < wb["end_frame"] and wb["start_frame"] < wa["end_frame"]):
                continue
            compared += 1
            box_a, box_b = a["layout"]["box"], b["layout"]["box"]
            inter = _inter_area(box_a, box_b)
            small = min(box_a[2] * box_a[3], box_b[2] * box_b[3]) or 1.0
            if inter / small > COLLISION_AREA_RATIO and (a["id"], b["id"]) not in allowed \
                    and (b["id"], a["id"]) not in allowed:
                collision_pairs += 1

    def _rate(hits: int, samples: int, threshold: float, extra: dict | None = None) -> dict:
        value = (hits / samples) if samples else 0.0
        d = {"value": round(value, 6), "hits": hits, "samples": samples,
             "threshold": threshold,
             "verdict": PASS if (samples and value <= threshold) else (PASS if samples == 0 else FAIL)}
        if extra:
            d.update(extra)
        return d

    return {
        "planned": {
            "face_overlap": _rate(planned_face_hits, planned_frames, FACE_OVERLAP_THRESHOLD),
            "out_of_bounds_elements": {"value": planned_oob / max(1, len(elements)),
                                       "count": planned_oob, "threshold": 0,
                                       "verdict": PASS if planned_oob == 0 else FAIL},
        },
        "observed": {
            "face_overlap": _rate(obs_face_hits, obs_frames, FACE_OVERLAP_THRESHOLD),
            "out_of_bounds": _rate(obs_oob, obs_frames, 0.0),
            "windows_with_visible_change": observed_windows,
            "sampled_frames": obs_frames,
        },
        "raster_containment": {
            "changed_pixels": containment["changed"],
            "inside_declared_box": containment["inside"],
            "value": (round(1.0 - containment["inside"] / containment["changed"], 6)
                      if containment["changed"] else 0.0),
            "threshold": 0.03,
            "verdict": ("PASS" if (not containment["changed"] or
                                   1.0 - containment["inside"] / containment["changed"] <= 0.03)
                        else FAIL),
            "note": "changed pixels outside the IR layout box (+6px pad) / changed pixels",
        },
        "collision": {
            "value": round(collision_pairs / compared, 6) if compared else 0.0,
            "pairs": collision_pairs, "compared_pairs": compared,
            "threshold": COLLISION_THRESHOLD,
            "verdict": PASS if (not compared or collision_pairs / compared <= COLLISION_THRESHOLD) else FAIL,
        },
        "face_detection": {"faces_detected": sum(len(v) for v in faces.values()),
                           "frames_with_faces": len(faces),
                           "note": "no faces detected ⇒ face metrics are vacuous PASS, "
                                   "reported as such in the gate report"},
        "sampling": {"stride_frames": stride, "grid": "192px wide grayscale diff"},
    }


def metrics(ir: dict, render_manifest: dict, *, source: Path, rendered: Path,
            truth_json: Path | None = None, vision: dict | None = None,
            gate: str = "G1", work: Path | None = None) -> dict[str, Any]:
    """Compute the full metric set for one render of one IR."""
    source, rendered = Path(source), Path(rendered)
    fps = float(ir["source"]["fps"])
    width, height = int(ir["canvas"]["width"]), int(ir["canvas"]["height"])
    truth = None
    if truth_json and Path(truth_json).exists():
        truth = json.loads(Path(truth_json).read_text(encoding="utf-8"))

    src_frames = _gray_frames(source)
    out_frames = _gray_frames(rendered)
    global_noise = _global_noise(ir, out_frames, src_frames)
    timing = _timing(ir, out_frames, src_frames, truth, fps, global_noise)
    position = _position(ir, out_frames, src_frames, vision or {}, fps, width, height)

    src_audio = _audio_stats(source)
    out_audio = _audio_stats(rendered)
    mi_src, mi_out = media.info(source), media.info(rendered)
    render_check = {
        "duration_src_s": round(mi_src.duration, 6),
        "duration_rendered_s": round(mi_out.duration, 6),
        "duration_delta_s": round(abs(mi_src.duration - mi_out.duration), 6),
        "duration_tolerance_s": round(1.0 / fps, 6),
        "frames_src": len(src_frames), "frames_rendered": len(out_frames),
        "audio": {
            "src": src_audio, "rendered": out_audio,
            "duration_delta_s": round(abs(src_audio["duration_s"] - out_audio["duration_s"]), 6),
            "rms_delta_db": round(abs(src_audio["rms_db"] - out_audio["rms_db"]), 3),
            "method": "decoded PCM statistics (AAC re-encode prevents sample-exact identity)",
            "pass": abs(src_audio["duration_s"] - out_audio["duration_s"]) <= 0.05
                    and abs(src_audio["rms_db"] - out_audio["rms_db"]) <= 0.5,
        },
        "verdict": None,
    }
    render_check["verdict"] = PASS if (
        render_check["duration_delta_s"] <= 1.0 / fps
        and len(src_frames) == len(out_frames)
        and render_check["audio"]["pass"]
    ) else FAIL

    verdicts = {
        "timing": timing["verdict"],
        "position": _worst([
            position["planned"]["face_overlap"]["verdict"],
            position["planned"]["out_of_bounds_elements"]["verdict"],
            position["observed"]["face_overlap"]["verdict"],
            position["observed"]["out_of_bounds"]["verdict"],
            position["collision"]["verdict"],
            position["raster_containment"]["verdict"],
        ]),
        "render": render_check["verdict"],
    }

    result = {
        "schema": "semotion.metrics/1",
        "gate": gate,
        "input_sha256": ir["source"]["sha256"],
        "ir_sha256": ir.get("provenance", {}).get("ir_sha256"),
        "render_sha256": render_manifest.get("output", {}).get("sha256"),
        "timing": timing,
        "position": position,
        "render": render_check,
        "verdicts": verdicts,
        "evidence": {
            "source": str(source), "rendered": str(rendered),
            "truth": str(truth_json) if truth_json else None,
            "tool_versions": media.tool_versions(),
        },
    }
    result = _clean(result)
    if work:
        work = Path(work)
        work.mkdir(parents=True, exist_ok=True)
        (work / f"metrics_{gate}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _clean(obj):
    """Convert numpy scalars/containers into plain JSON types."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def _worst(vals: list[str]) -> str:
    for level in (BLOCKED, FAIL, NOT_RUN, PASS):
        if level in vals:
            return level
    return PASS


def determinism(ir: dict, first: Path, second: Path, *, tolerance: int = 0) -> dict[str, Any]:
    """Frame-level reproducibility check between two renders of the same IR."""
    first, second = Path(first), Path(second)
    ha = media.video_frame_hashes(first)
    hb = media.video_frame_hashes(second)
    equal = ha == hb
    diff = media.frame_diff(first, second)
    ok = equal if tolerance == 0 else (diff["max_diff"] is not None and diff["max_diff"] <= tolerance)
    return {
        "runs": 2,
        "frames": [len(ha), len(hb)],
        "frame_hashes_equal": equal,
        "max_pixel_diff": diff["max_diff"],
        "mean_pixel_diff": diff["mean_diff"],
        "verdict": PASS if ok else FAIL,
        "policy": f"pixel-identical required (tolerance={tolerance}); byte-identical MP4 not required",
        "sha256": [media.sha256_file(first), media.sha256_file(second)],
    }
