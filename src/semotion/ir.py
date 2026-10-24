"""Motion IR (`semotion.motion-ir/1`) — build, validate, hash, load/dump.

The IR is the only thing the renderer may consume (spec §3.6): a schema-validated
document of whitelisted primitives. Nothing in it is executed; element text is a
string that gets rasterised by ``raster.py``.

Invariants enforced by :func:`validate` (docs/INTERFACES.md §3.5):
 1. frame numbers agree with seconds and stay inside the source;
 2. anchor references exist and the window sits within +-250 ms of the anchor
    (captions exempt);
 3. every box lies fully on the canvas;
 4. an avoid_face element covers at most 1 % of the face-safe box, unless it is
    explicitly allowed with a recorded reason;
 5. tracks only reference existing elements and every element is in exactly one
    track;
 6. simultaneously visible elements are declared in ``collision_ok``;
 7. ``provenance.ir_sha256`` matches :func:`compute_ir_hash`;
 8. ``content.text`` occurs in the frozen anchors or transcript (no invented copy).
"""
from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import media

SCHEMA = "semotion.motion-ir/1"
IR_VERSION = "1.0.0"
SOURCE_SCHEMA = "semotion.transcript/1"
ANCHOR_SCHEMA = "semotion.anchors/1"

ANCHOR_TOLERANCE_S = 0.25           # start may not precede the anchor by more than this
HOLD_TOLERANCE_S = 0.80             # end may trail the anchor end by at most this (hold_s)
FACE_OVERLAP_RATIO = 0.01           # invariant 4 / spec §6
CANVAS_EDGE_MARGIN_PX = 16          # never paint into the last pixels of the frame
FACE_SAFE_MARGIN_PX = 32            # default outward margin around the face union

ELEMENT_TYPES = ("keyword", "number", "list", "card", "caption")
ZOOM_TRACK_Z = {"t_card": 30, "t_content": 20, "t_caption": 10}

FPS_EPS = 1e-6


# --------------------------------------------------------------------------- json
def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_ir_hash(ir: dict) -> str:
    """sha256 of the canonical IR with ``provenance.ir_sha256`` removed."""
    clone = deepcopy(ir)
    prov = clone.get("provenance")
    if isinstance(prov, dict):
        prov.pop("ir_sha256", None)
    return _sha256_text(canonical_json(clone))


def seal(ir: dict) -> dict:
    """Stamp ``provenance.ir_sha256`` in place and return the document."""
    prov = ir.setdefault("provenance", {})
    prov.pop("ir_sha256", None)
    prov["ir_sha256"] = compute_ir_hash(ir)
    return ir


def dump(ir: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ir, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- helpers
_T2S = None


def _to_simplified(text: str) -> str:
    """Fold Traditional Chinese to Simplified (ASR may emit either script).

    Used only for *text containment* checks, never to rewrite stored content.
    """
    global _T2S
    if _T2S is None:
        try:
            import opencc
            _T2S = opencc.OpenCC("t2s")
        except Exception:  # pragma: no cover - optional dependency
            _T2S = False
    if not text or not _T2S:
        return str(text or "")
    return _T2S.convert(str(text))


def _norm_text(text: str) -> str:
    return "".join(ch for ch in _to_simplified(text) if not ch.isspace())


def _now_utc() -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch:
        try:
            ts = datetime.fromtimestamp(int(epoch), tz=timezone.utc)
            return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, OSError):
            pass
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fps_of(transcript: dict, canvas: dict, source_info: media.MediaInfo | None) -> float:
    for candidate in ((transcript.get("source") or {}).get("fps"),
                      (canvas or {}).get("fps"),
                      source_info.fps if source_info else None):
        try:
            f = float(candidate)
        except (TypeError, ValueError):
            continue
        if f > 0:
            return f
    return 25.0


def _frame_count_of(transcript: dict, source_info: media.MediaInfo | None,
                    duration: float, fps: float) -> tuple[int, str, list[str]]:
    """Real decoded frame count first: duration*fps is not authoritative.

    Containers pad the audio tail, so ``duration*fps`` can be a few frames larger
    than the video stream (nb_frames). Trusting the derived number would make a
    correct render look like a frame-count regression.
    """
    warnings: list[str] = []
    stream_frames = getattr(source_info, "stream_frames", None) if source_info else None
    if stream_frames and int(stream_frames) > 0:
        n = int(stream_frames)
        tr_n = (transcript.get("source") or {}).get("n_frames")
        if tr_n is not None and int(tr_n) != n:
            warnings.append(f"transcript n_frames={tr_n} != container nb_frames={n}; "
                            "using the decoded stream count")
        return n, "nb_frames", warnings
    tr_n = (transcript.get("source") or {}).get("n_frames")
    if tr_n is not None:
        try:
            n = int(tr_n)
            if n > 0:
                return n, "transcript", warnings
        except (TypeError, ValueError):
            pass
    return int(round(duration * fps)), "duration*fps", warnings


def _duration_of(transcript: dict, source_info: media.MediaInfo | None) -> float:
    for candidate in ((transcript.get("source") or {}).get("duration_s"),
                      source_info.duration if source_info else None):
        try:
            d = float(candidate)
        except (TypeError, ValueError):
            continue
        if d > 0:
            return d
    return 0.0


def _face_union(vision: dict) -> list[int] | None:
    unions: list[Sequence[float]] = []
    for tr in vision.get("face_tracks") or []:
        b = tr.get("bbox_union")
        if b and len(b) >= 4:
            unions.append(b)
    if unions:
        from .effects.base import union_boxes
        return union_boxes(unions)
    boxes = [f.get("box") for f in vision.get("faces") or [] if f.get("box")]
    if boxes:
        from .effects.base import union_boxes
        return union_boxes(boxes)
    return None


def _transcript_text(transcript: dict) -> str:
    parts: list[str] = []
    for w in transcript.get("words") or []:
        parts.append(str(w.get("text") or ""))
    if not parts:
        for s in transcript.get("segments") or []:
            parts.append(str(s.get("text") or ""))
    return "".join(parts)


def _anchor_index(anchors_doc: dict) -> dict[str, dict]:
    return {str(a.get("id")): a for a in anchors_doc.get("anchors") or [] if a.get("id")}


def _window_bounds(refs: Sequence[str], idx: dict[str, dict]) -> tuple[float, float] | None:
    spans = [(float(idx[r].get("start_s", 0.0)), float(idx[r].get("end_s", 0.0)))
             for r in refs if r in idx]
    if not spans:
        return None
    return min(s for s, _ in spans), max(e for _, e in spans)


# --------------------------------------------------------------------------- style
def _style_for(effect: str, emphasis: str, canvas: dict) -> dict[str, Any]:
    from .effects.base import scaled

    high = emphasis == "high"
    base = {
        "font_family": "NotoSansSC",
        "font_file": "NotoSansSC-Regular.otf",
        "font_path": "assets/fonts/NotoSansSC-Regular.otf",
        "weight": "bold" if high else "regular",
        "color": "#F5F7FA",
        "accent": "#F2B233",
        "bg": "rgba(12,16,28,0.72)",
    }
    if effect == "keyword":
        base.update({"font_size_px": scaled(96 if high else 86, canvas),
                     "padding_px": scaled(36, canvas), "radius_px": scaled(28, canvas),
                     "align": "center", "accent_line": True})
    elif effect == "number":
        base.update({"font_size_px": scaled(84, canvas), "padding_px": scaled(40, canvas),
                     "radius_px": scaled(26, canvas), "align": "center",
                     "accent_line": False, "bg": "rgba(10,13,22,0.80)"})
    elif effect == "list":
        base.update({"font_size_px": scaled(64, canvas), "padding_px": scaled(36, canvas),
                     "radius_px": scaled(28, canvas), "align": "left",
                     "bg": "rgba(10,13,22,0.82)",
                     "min_width_px": int(round(0.62 * float((canvas or {}).get("width") or 1080))),
                     "line_height": 1.35})
    elif effect == "card":
        base.update({"font_size_px": scaled(78, canvas), "padding_px": scaled(40, canvas),
                     "radius_px": scaled(30, canvas), "align": "left",
                     "accent_bar": True, "bg": "rgba(16,21,33,0.78)"})
    else:
        base.update({"font_size_px": scaled(78, canvas), "padding_px": scaled(36, canvas),
                     "radius_px": scaled(24, canvas), "align": "center"})
    return base


def _motion_for(effect: str, emphasis: str) -> dict[str, Any]:
    if effect == "keyword":
        return {"enter": {"type": "pop_up", "dur_s": 0.26, "ease": "out_cubic",
                          "scale_from": 0.86},
                "exit": {"type": "fade", "dur_s": 0.18, "ease": "in_quad"}}
    if effect == "number":
        return {"enter": {"type": "fade_up", "dur_s": 0.24, "ease": "out_cubic",
                          "scale_from": 0.94},
                "exit": {"type": "fade", "dur_s": 0.18, "ease": "in_quad"}}
    if effect == "list":
        return {"enter": {"type": "reveal", "dur_s": 0.20, "ease": "out_cubic",
                          "per_item": True},
                "exit": {"type": "fade", "dur_s": 0.20, "ease": "in_quad"}}
    if effect == "card":
        return {"enter": {"type": "scale_in", "dur_s": 0.28, "ease": "smoothstep",
                          "scale_from": 0.96},
                "exit": {"type": "fade", "dur_s": 0.20, "ease": "in_quad"}}
    return {"enter": {"type": "fade", "dur_s": 0.20, "ease": "smoothstep"},
            "exit": {"type": "fade", "dur_s": 0.18, "ease": "in_quad"}}


def _ordinal_value(text: str) -> int | None:
    import re
    from .effects.number import cn_to_number

    m = re.match(r"^第([一二三四五六七八九十百\d]+)$", text.strip())
    if m:
        v = cn_to_number(m.group(1))
        return int(v) if v is not None else None
    m = re.match(r"^(\d+)[.、)．]?$", text.strip())
    return int(m.group(1)) if m else None


def _list_rows(items: Sequence[str], onsets: Sequence[float]) -> list[dict]:
    """Merge bare ordinals ("第一", "②") with the step they introduce."""
    rows: list[dict] = []
    i = 0
    while i < len(items):
        text = str(items[i])
        at = float(onsets[i]) if i < len(onsets) else 0.0
        ordinal = _ordinal_value(text)
        next_is_ordinal = (i + 1 < len(items)
                           and _ordinal_value(str(items[i + 1])) is not None)
        if ordinal is not None and i + 1 < len(items) and not next_is_ordinal:
            rows.append({"marker": str(ordinal), "text": str(items[i + 1]),
                         "at_s": round(at, 6),
                         "text_at_s": round(float(onsets[i + 1]) if i + 1 < len(onsets) else at, 6)})
            i += 2
            continue
        rows.append({"marker": str(ordinal if ordinal is not None else len(rows) + 1),
                     "text": text, "at_s": round(at, 6), "text_at_s": round(at, 6)})
        i += 1
    return rows


# --------------------------------------------------------------------------- build
def build(transcript: dict, vision: dict, anchors: dict, plan: dict,
          source: str | Path, canvas: dict) -> dict:
    """Freeze transcript/vision/anchors/plan into a validated-ready Motion IR."""
    from . import design  # local import keeps ``import semotion.ir`` dependency-free

    source = Path(source)
    if not source.exists():
        raise media.MediaError("E_INPUT_MISSING", f"source not found: {source}")
    src_info = media.info(source)
    src_sha = media.sha256_file(source)
    fps = _fps_of(transcript, canvas, src_info)
    duration = _duration_of(transcript, src_info) or src_info.duration
    n_frames, frame_source, frame_warnings = _frame_count_of(transcript, src_info,
                                                            duration, fps)
    width = int((canvas or {}).get("width") or src_info.width)
    height = int((canvas or {}).get("height") or src_info.height)

    warnings: list[str] = list(frame_warnings)
    ts_sha = (transcript.get("source") or {}).get("sha256")
    if ts_sha and ts_sha != src_sha:
        warnings.append(f"transcript.source.sha256 != source sha256 ({ts_sha[:12]} vs {src_sha[:12]})")
    if int(src_info.width) != width or int(src_info.height) != height:
        warnings.append(f"canvas {width}x{height} differs from source "
                        f"{src_info.width}x{src_info.height}")

    anchors_doc = anchors or {}
    idx = _anchor_index(anchors_doc)
    face_union = _face_union(vision or {})
    face_margin = int((vision or {}).get("face_safe_margin_px") or FACE_SAFE_MARGIN_PX)
    occupied: list[dict] = []

    elements: list[dict] = []
    skipped: list[dict] = []
    for dec in (plan or {}).get("decisions") or []:
        effect = str(dec.get("effect") or "none")
        if effect not in ("keyword", "number", "list", "card"):
            skipped.append({"decision": dec.get("id"), "reason": f"effect:{effect}"})
            continue
        refs = [str(r) for r in (dec.get("anchor_ids") or [dec.get("anchor_id")]) if r]
        if not refs:
            refs = [str(dec.get("anchor_id"))] if dec.get("anchor_id") else []
        missing = [r for r in refs if r not in idx]
        if missing:
            skipped.append({"decision": dec.get("id"), "reason": f"missing_anchor:{missing}"})
            continue
        bounds = _window_bounds(refs, idx)
        win = dec.get("window") or {}
        start = max(0.0, float(win.get("start_s", bounds[0] if bounds else 0.0)))
        end = min(duration, float(win.get("end_s", bounds[1] if bounds else start + 1.0)))
        if bounds:
            low = max(0.0, bounds[0] - ANCHOR_TOLERANCE_S)
            high = min(duration, bounds[1] + HOLD_TOLERANCE_S)
            if start < low:
                warnings.append(f"{dec.get('id')}: onset clamped to anchor start -250ms "
                                f"({start:.3f}->{low:.3f}s)")
                start = low
            if end > high:
                warnings.append(f"{dec.get('id')}: end clamped to anchor end +800ms hold "
                                f"({end:.3f}->{high:.3f}s)")
                end = high
        if end <= start:
            skipped.append({"decision": dec.get("id"), "reason": "empty_window"})
            continue
        sf, ef = int(round(start * fps)), int(round(end * fps))
        sf = max(0, min(sf, n_frames - 2))
        ef = max(sf + 2, min(ef, n_frames))
        # frame snapping must respect the onset tolerance and the hold budget
        if bounds:
            low = max(0.0, bounds[0] - ANCHOR_TOLERANCE_S) - 0.5 / fps - FPS_EPS
            high = min(duration, bounds[1] + HOLD_TOLERANCE_S) + 0.5 / fps + FPS_EPS
            while sf < ef - 2 and sf / fps < low:
                sf += 1
            while ef > sf + 2 and ef / fps > high:
                ef -= 1
        start_s, end_s = sf / fps, ef / fps
        hold_s = round(max(0.0, end_s - bounds[1]), 6) if bounds else 0.0

        content: dict[str, Any] = {"text": str((dec.get("content") or {}).get("text")
                                               or (idx[refs[0]].get("text") if refs else ""))}
        if effect == "number":
            from .effects.number import parse_numeric
            numeric = parse_numeric(content["text"])
            if numeric:
                content["numeric"] = numeric
                content["numeric_display"] = (content["text"][:numeric["start"]] + numeric["final"]
                                              + content["text"][numeric["end"]:])
            else:
                warnings.append(f"{dec.get('id')}: no numeric token in {content['text']!r}; "
                                "rendering as plain emphasis")
        item_onsets: list[float] = []
        if effect == "list":
            items = [str(x) for x in ((dec.get("content") or {}).get("items")
                                      or [idx[r].get("text", "") for r in refs])]
            item_onsets = []
            for r in refs:
                a = idx.get(r) or {}
                item_onsets.append(round(max(0.0, float(a.get("start_s", start_s)) - start_s), 6))
            content["items"] = items
            content["item_onsets_s"] = item_onsets
            content["rows"] = _list_rows(items, item_onsets)
            if dec.get("anchor_ids") and len(refs) > 1:
                content["anchor_rows"] = refs
        element = {
            "id": f"el_{dec.get('id')}",
            "type": effect,
            "anchor_ref": refs[0],
            "anchor_refs": refs,
            "window": {"start_s": round(start_s, 6), "end_s": round(end_s, 6),
                       "start_frame": sf, "end_frame": ef,
                       "hold_s": hold_s,
                       "anchor_start_s": round(bounds[0], 6) if bounds else None,
                       "anchor_end_s": round(bounds[1], 6) if bounds else None},
            "style": _style_for(effect, str(dec.get("emphasis") or "medium"),
                                {"width": width, "height": height}),
            "motion": _motion_for(effect, str(dec.get("emphasis") or "medium")),
            "content": content,
            "provenance": {
                "decision_id": dec.get("id"),
                "decision_model": dec.get("decision_model") or (plan.get("provenance") or {})
                .get("decision_model", "rules"),
                "prompt_summary": str((plan.get("provenance") or {}).get("method") or "")[:200],
                "confidence": float(dec.get("confidence") or 0.0),
                "reason": dec.get("reason"),
            },
        }
        lay = design.layout(element, {"width": width, "height": height, "fps": fps},
                            vision or {}, [dict(o) for o in occupied])
        element["style"].update(lay.get("style") or {})
        element["layout"] = {k: v for k, v in lay.items() if k != "style"}
        if lay.get("degraded"):
            warnings.append(f"{element['id']}: layout degraded ({lay.get('reason')})")
        zoom = lay.get("box") or [0, 0, 1, 1]
        occupied.append({"id": element["id"], "box": list(zoom),
                         "window": dict(element["window"])})
        element["provenance"]["effect_params_sha256"] = _sha256_text(canonical_json(
            {"type": effect, "window": element["window"], "box": zoom,
             "style": element["style"], "motion": element["motion"],
             "content": content}))
        elements.append(element)

    elements, trailing = _resolve_overlaps(elements, fps, idx)
    warnings.extend(trailing)

    tracks: list[dict] = []
    for tid, z, wanted in (("t_card", 30, ("card",)), ("t_content", 20, ("keyword", "number", "list"))):
        ids = [e["id"] for e in elements if e["type"] in wanted]
        if ids:
            tracks.append({"id": tid, "z": z, "elements": ids})

    doc = {
        "schema": SCHEMA,
        "ir_version": IR_VERSION,
        "source": {
            "path": str(source.resolve()),
            "sha256": src_sha,
            "width": int(src_info.width),
            "height": int(src_info.height),
            "fps": float(fps),
            "duration_s": round(duration, 6),
            "n_frames": n_frames,
            "frame_source": frame_source,
            "duration_fps_frames": int(round(duration * fps)),
            "timebase": {"num": 1, "den": int(round(fps)) if abs(fps - round(fps)) < 1e-9
                         else 1000},
        },
        "canvas": {"width": width, "height": height, "fps": float(fps)},
        "style": str((plan or {}).get("style") or "smart_balanced"),
        "semantic": {
            "transcript_schema": transcript.get("schema"),
            "transcript_sha256": _sha256_text(canonical_json(transcript or {})),
            "anchors_schema": anchors_doc.get("schema"),
            "anchors_sha256": _sha256_text(canonical_json(anchors_doc or {})),
            "decisions_sha256": _sha256_text(canonical_json((plan or {}).get("decisions") or [])),
            "anchors": deepcopy(anchors_doc.get("anchors") or []),
            "decisions": deepcopy((plan or {}).get("decisions") or []),
            "transcript_text": _transcript_text(transcript or {}),
            "face_safe": {"bbox_union": face_union, "margin_px": face_margin,
                          "source": "vision.face_tracks[].bbox_union union"},
        },
        "tracks": tracks,
        "elements": elements,
        "collision_ok": [],
        "output": {"width": width, "height": height, "fps": float(fps),
                   "video_codec": "h264", "pix_fmt": "yuv420p",
                   "audio_codec": "aac", "audio_bitrate": "192k",
                   "loudness_target_db": None},
        "provenance": {
            "generated_utc": _now_utc(),
            "tool_versions": media.tool_versions(),
            "host": "macOS-arm64",
            "policy": "schema-validated declarative primitives only; no code execution",
            "frame_count": {"n_frames": n_frames, "source": frame_source,
                            "container_duration_s": round(duration, 6),
                            "duration_fps_frames": int(round(duration * fps))},
            "build_warnings": warnings,
            "skipped_decisions": skipped,
            "n_elements": len(elements),
        },
    }
    return seal(doc)


def _resolve_overlaps(elements: list[dict], fps: float,
                      idx: dict[str, dict] | None = None) -> tuple[list[dict], list[str]]:
    """Trim a held tail (never an onset) when two windows would overlap.

    Keeping onsets exact matters for the timing metric; the only slack we use is
    the post-anchor hold. If even that is not available the later element is
    dropped and reported — never left silently overlapping.
    """
    warnings: list[str] = []
    out: list[dict] = []
    last_end = -1
    for el in sorted(elements, key=lambda e: (e["window"]["start_frame"], e["id"])):
        sf, ef = el["window"]["start_frame"], el["window"]["end_frame"]
        if sf < last_end:
            trimmed = sf - 1
            prev_refs = [str(r) for r in (out[-1].get("anchor_refs") or [])]
            anchor_end = max((float((idx or {}).get(r, {}).get("end_s", 0.0))
                              for r in prev_refs), default=0.0)
            floor_frame = max(int(round(anchor_end * fps)),
                              out[-1]["window"]["start_frame"] + 2)
            if trimmed >= floor_frame:
                warnings.append(f"{out[-1]['id']}: hold trimmed to avoid overlap with {el['id']}")
                out[-1]["window"]["end_frame"] = trimmed
                out[-1]["window"]["end_s"] = round(trimmed / fps, 6)
                out[-1]["window"]["hold_s"] = round(
                    max(0.0, out[-1]["window"]["end_s"] - anchor_end), 6)
                last_end = ef
                out.append(el)
                continue
            warnings.append(f"{el['id']}: dropped — window would overlap {out[-1]['id']} "
                            "and the hold is too short to trim")
            continue
        out.append(el)
        last_end = ef
    return out, warnings


# --------------------------------------------------------------------------- validate
def validate(ir: dict) -> list[str]:
    """Return a list of human-readable invariant violations (empty = valid)."""
    errs: list[str] = []
    if not isinstance(ir, dict):
        return ["IR is not an object"]
    if ir.get("schema") != SCHEMA:
        errs.append(f"schema must be {SCHEMA!r}, got {ir.get('schema')!r}")
    if str(ir.get("ir_version")) != IR_VERSION:
        errs.append(f"ir_version must be {IR_VERSION!r}, got {ir.get('ir_version')!r}")

    source = ir.get("source") or {}
    canvas = ir.get("canvas") or {}
    try:
        fps = float(source.get("fps"))
        n_frames = int(source.get("n_frames"))
        duration = float(source.get("duration_s"))
    except (TypeError, ValueError):
        return errs + ["source.fps/n_frames/duration_s missing or not numeric"]
    if fps <= 0:
        errs.append(f"source.fps must be > 0, got {fps}")
    if n_frames <= 0:
        errs.append(f"source.n_frames must be > 0, got {n_frames}")
    frame_source = str(source.get("frame_source") or "")
    derived = round(duration * fps)
    frame_tol = max(3.0, 0.02 * fps) if frame_source in ("nb_frames", "transcript") else 1.0
    if abs(n_frames - derived) > frame_tol:
        errs.append(f"source.n_frames={n_frames} inconsistent with duration_s*fps={derived} "
                    f"(frame_source={frame_source or 'unknown'}, tolerance {frame_tol:g})")
    if frame_source and frame_source not in ("nb_frames", "transcript", "duration*fps"):
        errs.append(f"source.frame_source must be 'nb_frames', 'transcript' or 'duration*fps', "
                    f"got {frame_source!r}")
    sha = str(source.get("sha256") or "")
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        errs.append("source.sha256 must be a 64-char lowercase hex digest")
    try:
        cw, ch = int(canvas.get("width")), int(canvas.get("height"))
        cfps = float(canvas.get("fps"))
    except (TypeError, ValueError):
        return errs + ["canvas.width/height/fps missing or not numeric"]
    if cw <= 0 or ch <= 0:
        errs.append(f"canvas must be positive, got {cw}x{ch}")
    if abs(cfps - fps) > 1e-6:
        errs.append(f"canvas.fps={cfps} != source.fps={fps}")

    anchors = {str(a.get("id")): a for a in (ir.get("semantic") or {}).get("anchors") or []
               if isinstance(a, dict) and a.get("id")}
    transcript_norm = _norm_text((ir.get("semantic") or {}).get("transcript_text") or "")
    anchor_norm = _norm_text("".join(str(a.get("text") or "") for a in anchors.values()))
    face_safe = (ir.get("semantic") or {}).get("face_safe") or {}
    face_box = face_safe.get("bbox_union")
    face_area = (float(face_box[2]) * float(face_box[3])) if face_box else 0.0

    elements = ir.get("elements")
    if not isinstance(elements, list):
        return errs + ["elements must be a list"]
    seen: set[str] = set()
    frames_by_id: dict[str, tuple[int, int]] = {}
    for i, el in enumerate(elements):
        if not isinstance(el, dict):
            errs.append(f"elements[{i}] is not an object")
            continue
        eid = str(el.get("id") or f"elements[{i}]")
        if not eid:
            errs.append(f"elements[{i}] has no id")
        if eid in seen:
            errs.append(f"{eid}: duplicate element id")
        seen.add(eid)
        etype = el.get("type")
        if etype not in ELEMENT_TYPES:
            errs.append(f"{eid}: unknown type {etype!r}")

        # -- invariant 1: frames
        win = el.get("window") or {}
        try:
            s_s, e_s = float(win.get("start_s")), float(win.get("end_s"))
            sf, ef = int(win.get("start_frame")), int(win.get("end_frame"))
        except (TypeError, ValueError):
            errs.append(f"{eid}: window needs start_s/end_s/start_frame/end_frame")
            continue
        if sf != int(round(s_s * fps)):
            errs.append(f"{eid}: start_frame={sf} != round(start_s*fps)={round(s_s * fps)}")
        if ef != int(round(e_s * fps)):
            errs.append(f"{eid}: end_frame={ef} != round(end_s*fps)={round(e_s * fps)}")
        if not (0 <= sf < ef <= n_frames):
            errs.append(f"{eid}: window frames [{sf},{ef}) outside [0,{n_frames})")
        frames_by_id[eid] = (sf, ef)

        # -- invariant 2: anchors + tolerance
        refs = [str(r) for r in (el.get("anchor_refs") or []) if r] or \
               ([str(el["anchor_ref"])] if el.get("anchor_ref") else [])
        for r in refs:
            if r not in anchors:
                errs.append(f"{eid}: anchor_ref {r!r} not found in semantic.anchors")
        if etype != "caption":
            if not refs:
                errs.append(f"{eid}: missing anchor_ref")
            elif all(r in anchors for r in refs):
                lo = min(float(anchors[r].get("start_s", 0.0)) for r in refs)
                hi = max(float(anchors[r].get("end_s", 0.0)) for r in refs)
                grace = 0.5 / fps + FPS_EPS
                if s_s < lo - ANCHOR_TOLERANCE_S - grace:
                    errs.append(f"{eid}: start {s_s:.3f}s is before anchor start {lo:.3f}s "
                                f"-{ANCHOR_TOLERANCE_S}s tolerance")
                if e_s > hi + HOLD_TOLERANCE_S + grace:
                    errs.append(f"{eid}: end {e_s:.3f}s trails anchor end {hi:.3f}s by "
                                f"{e_s - hi:.3f}s (> {HOLD_TOLERANCE_S}s hold)")
                elif e_s > hi + 1e-6:
                    hold = win.get("hold_s")
                    expected = e_s - hi
                    if hold is None:
                        errs.append(f"{eid}: window trails the anchor by {expected:.3f}s but "
                                    "window.hold_s is not declared")
                    else:
                        try:
                            hv = float(hold)
                        except (TypeError, ValueError):
                            hv = -1.0
                        if abs(hv - expected) > grace:
                            errs.append(f"{eid}: window.hold_s={hold} does not match "
                                        f"end_s-anchor_end={expected:.3f}s")
                        if hv > HOLD_TOLERANCE_S + grace:
                            errs.append(f"{eid}: window.hold_s={hv}s exceeds "
                                        f"{HOLD_TOLERANCE_S}s")

        # -- invariant 3: box on canvas
        lay = el.get("layout") or {}
        box = lay.get("box")
        ok_box = isinstance(box, (list, tuple)) and len(box) >= 4
        if not ok_box:
            errs.append(f"{eid}: layout.box missing")
        else:
            try:
                x, y, bw, bh = (int(v) for v in box[:4])
            except (TypeError, ValueError):
                errs.append(f"{eid}: layout.box must be 4 integers, got {box!r}")
                x = y = bw = bh = -1
            if bw <= 0 or bh <= 0:
                errs.append(f"{eid}: layout.box has non-positive size {bw}x{bh}")
            if x < 0 or y < 0 or x + bw > cw or y + bh > ch:
                errs.append(f"{eid}: layout.box {[x, y, bw, bh]} is not fully inside "
                            f"canvas {cw}x{ch}")
            # -- invariant 4: face safety
            if lay.get("avoid_face") and face_box and face_area > 0 and not lay.get("allow_face_overlap"):
                from .effects.base import rect_inter_area
                inter = rect_inter_area([x, y, bw, bh], face_box)
                if inter / face_area > FACE_OVERLAP_RATIO:
                    errs.append(f"{eid}: box {[x, y, bw, bh]} covers "
                                f"{100 * inter / face_area:.1f}% of the face-safe box "
                                f"{list(face_box)} (> {100 * FACE_OVERLAP_RATIO:.0f}%)")
            if lay.get("allow_face_overlap") and not (lay.get("allow_face_reason")
                                                      or lay.get("reason")):
                errs.append(f"{eid}: allow_face_overlap needs a recorded reason")

        # -- invariant 8: text provenance
        text = _norm_text((el.get("content") or {}).get("text") or "")
        if not text:
            errs.append(f"{eid}: content.text is empty")
        elif text not in anchor_norm and (not transcript_norm or text not in transcript_norm):
            errs.append(f"{eid}: content.text {text[:40]!r} not found in anchors or transcript")
        try:
            if int((el.get("style") or {}).get("font_size_px") or 0) <= 0:
                errs.append(f"{eid}: style.font_size_px must be > 0")
        except (TypeError, ValueError):
            errs.append(f"{eid}: style.font_size_px is not numeric")

    # -- invariant 5: tracks
    tracks = ir.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        errs.append("tracks must be a non-empty list")
        tracks = []
    owner: dict[str, int] = {}
    for t in tracks:
        tid = t.get("id") if isinstance(t, dict) else None
        if not tid:
            errs.append("track without id")
            continue
        try:
            int(t.get("z"))
        except (TypeError, ValueError):
            errs.append(f"track {tid}: z must be an integer")
        tids = t.get("elements") or []
        if not isinstance(tids, list):
            errs.append(f"track {tid}: elements must be a list")
            continue
        for eid in tids:
            eid = str(eid)
            if eid not in seen:
                errs.append(f"track {tid}: references unknown element {eid!r}")
            owner[eid] = owner.get(eid, 0) + 1
    for eid in sorted(seen):
        n = owner.get(eid, 0)
        if n == 0:
            errs.append(f"{eid}: belongs to no track")
        elif n > 1:
            errs.append(f"{eid}: belongs to {n} tracks (must be exactly one)")

    # -- invariant 6: declared simultaneous visibility
    allowed = {tuple(sorted((str(p[0]), str(p[1]))))
               for p in (ir.get("collision_ok") or []) if isinstance(p, (list, tuple)) and len(p) >= 2}
    ids = [e for e in frames_by_id]
    for a_i in range(len(ids)):
        for b_i in range(a_i + 1, len(ids)):
            a, b = ids[a_i], ids[b_i]
            (asf, aef), (bsf, bef) = frames_by_id[a], frames_by_id[b]
            if asf < bef and bsf < aef and (a, b) not in allowed:
                errs.append(f"{a} and {b}: windows overlap in frames "
                            f"[{max(asf, bsf)},{min(aef, bef)}) but the pair is not "
                            "declared in collision_ok")

    # -- invariant 7: hash
    prov = ir.get("provenance") or {}
    stored = prov.get("ir_sha256")
    actual = compute_ir_hash(ir)
    if not stored:
        errs.append("provenance.ir_sha256 is missing")
    elif str(stored) != actual:
        errs.append(f"provenance.ir_sha256 mismatch (stored {str(stored)[:16]}…, "
                    f"computed {actual[:16]}…)")
    return errs
