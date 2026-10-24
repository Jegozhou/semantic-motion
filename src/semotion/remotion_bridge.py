"""Remotion rendering backend.

Turns a Motion IR into a Remotion props document, stages the source video, renders
with the Remotion CLI (headless Chrome) and returns the same manifest shape the
Pillow/ffmpeg backend produces. The IR stays the source of truth; props contain
data only — no LLM-generated code is ever executed (spec §3.6).
"""
from __future__ import annotations

import colorsys
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import media

ROOT = Path(__file__).resolve().parents[2]
REMOTION_DIR = ROOT / "remotion"
PUBLIC_DIR = REMOTION_DIR / "public"
CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)
SCHEMA = "semotion.remotion-props/1"


# --------------------------------------------------------------------- helpers

def _hex_to_rgba(color: str, alpha: float) -> str:
    color = (color or "#F2B233").strip()
    if color.startswith("rgba") or color.startswith("rgb"):
        return color
    color = color.lstrip("#")
    if len(color) == 3:
        color = "".join(c * 2 for c in color)
    r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _chrome() -> str | None:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    return None


def _frame_of(seconds: float, fps: float) -> int:
    return int(round(seconds * fps))


def _numeric_parts(numeric: dict | None, text: str) -> dict:
    """Split the numeric payload into prefix/value/suffix for the React component."""
    if not isinstance(numeric, dict):
        return {"text": text, "value": None, "prefix": "", "suffix": "", "decimals": 0}
    final = str(numeric.get("final") or "")
    value = float(numeric.get("value") or 0.0)
    decimals = int(numeric.get("decimals") or 0)
    label = str(numeric.get("label") or "")
    prefix, suffix = "", ""
    if label and final and label[:1] and final.startswith(label[:1]) and not final[0].isdigit():
        prefix = ""
    # take unit tail from the final string after the number
    body = final
    for token in (str(int(value)), f"{value:.{decimals}f}"):
        if token and token in body:
            suffix = body[body.index(token) + len(token):]
            break
    return {"text": text, "value": value, "prefix": prefix, "suffix": suffix,
            "decimals": decimals}


def _style_for(element: dict, canvas: dict) -> dict:
    style = element.get("style") or {}
    font_px = int(style.get("font_size_px") or max(48, int(canvas["height"] * 0.045)))
    accent = style.get("accent") or "#F2B233"
    fg = style.get("color") or "#F5F7FA"
    bg_raw = str(style.get("bg") or "rgba(12,16,28,0.88)")
    bg = bg_raw if bg_raw.startswith("rgba") else _hex_to_rgba(bg_raw, 0.88)
    return {
        "accent": accent,
        "fg": fg,
        "bg": bg,
        "fontPx": font_px,
        "radius": int(style.get("radius_px") or max(16, int(font_px * 0.32))),
        "align": "left" if element.get("type") == "list" else "center",
        "weight": 700,
    }


def _em_width(text: str) -> float:
    """Approximate text width in em (CJK ≈ 1em, Latin ≈ 0.55em)."""
    return sum(1.0 if ord(ch) > 127 else 0.55 for ch in str(text or ""))


def _fit_font(text: str, box_w: int, base: int, *, pad: int = 104, floor: int = 36) -> int:
    """Shrink the font so the copy fits the declared box on one line where possible.

    A card rendered at 96px into a 750px box breaks a 9-character sentence into
    "…没" + "错"; fitting the size keeps the line intact and stays inside the box
    (the containment metric already enforces the box).
    """
    em = _em_width(text)
    if em <= 0 or box_w <= 0:
        return base
    fit = int((box_w - pad) / em)
    return max(floor, min(base, fit))


def _motion_for(element: dict, fps: float) -> dict:
    motion = element.get("motion") or {}
    enter = float((motion.get("enter") or {}).get("dur_s") or 0.28)
    exit_ = float((motion.get("exit") or {}).get("dur_s") or 0.18)
    return {"enterDurFrames": max(2, int(round(enter * fps))),
            "exitDurFrames": max(1, int(round(exit_ * fps)))}


def _list_items(element: dict, fps: float) -> list[dict]:
    """Row reveal frames.

    IR rows carry ``at_s`` **relative to the element start** (``text_at_s`` for the
    text), not absolute timestamps; using them as absolute pinned every row to the
    first frame and the whole list appeared at once.
    """
    content = element.get("content") or {}
    rows = content.get("rows") or []
    items = content.get("items") or []
    start_frame = int(element["window"]["start_frame"])
    out = []
    for i, text in enumerate(items):
        rel = 0.0
        if i < len(rows) and isinstance(rows[i], dict):
            row = rows[i]
            if row.get("start_s") is not None:
                rel = float(row["start_s"]) - float(element["window"]["start_s"])
            elif row.get("text_at_s") is not None:
                rel = float(row["text_at_s"])
            elif row.get("at_s") is not None:
                rel = float(row["at_s"])
        out.append({"text": str(text),
                    "startFrame": start_frame + max(0, int(round(rel * fps)))})
    return out


# ----------------------------------------------------------------------- props

def _norm(text: str) -> str:
    return "".join(ch for ch in str(text) if ch.isalnum()).lower()


def _suppress_duplicate_captions(elements: list[dict], captions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Drop caption cues that repeat an on-screen element word-for-word.

    Showing «其实是节奏» as a keyword card *and* as the subtitle at the same moment
    is visual noise. The element wins; the cue is removed (and recorded, not silently
    dropped).
    """
    kept, dropped = [], []
    for cue in captions:
        cue_text = _norm("".join(cue.get("lines") or []))
        clash = None
        for el in elements:
            if el["endFrame"] <= cue["startFrame"] or el["startFrame"] >= cue["endFrame"]:
                continue
            content = el.get("content") or {}
            # list elements show their rows, not their marker text
            el_text = _norm((content.get("text") or "")
                            + "".join(str(it.get("text") or "") for it in (content.get("items") or [])))
            # Suppress only when the element covers most of the subtitle: a keyword
            # ("痛点") inside a full sentence ("今天聊一个真实的痛点") is a highlight,
            # not a duplicate, and the sentence must still be readable.
            items_norm = [_norm(str(it.get("text") or "")) for it in (content.get("items") or [])]
            covers = (el_text == cue_text
                      or (el_text in cue_text and len(el_text) >= 0.7 * len(cue_text))
                      or (cue_text in el_text and len(cue_text) >= 0.7 * len(el_text))
                      or any(it and (it == cue_text
                                     or (it in cue_text and len(it) >= 0.7 * len(cue_text)))
                             for it in items_norm))
            if el_text and cue_text and covers:
                clash = el["id"]
                break
        if clash:
            dropped.append({"cue": [cue["startFrame"], cue["endFrame"]],
                            "lines": cue.get("lines"), "replaced_by": clash})
        else:
            kept.append(cue)
    return kept, dropped


def props(ir: dict, *, captions: list[dict] | None = None,
          caption_safe_bottom_px: int = 0, hidden: bool = False) -> dict:
    canvas = ir["canvas"]
    fps = float(canvas["fps"])
    elements = []
    for el in ir.get("elements", []):
        win = el["window"]
        content = el.get("content") or {}
        effect = str(el.get("type") or "keyword")
        base = {
            "id": el["id"],
            "effect": effect if effect in ("keyword", "number", "list", "card") else "keyword",
            "startFrame": int(win["start_frame"]),
            "endFrame": int(win["end_frame"]),
            "box": [int(v) for v in el["layout"]["box"]],
            "style": _style_for(el, canvas),
            "motion": _motion_for(el, fps),
        }
        if effect == "number":
            base["content"] = _numeric_parts(content.get("numeric"), str(content.get("text") or ""))
            shown = (base["content"].get("prefix") or "") + str(int(base["content"].get("value") or 0)) \
                + (base["content"].get("suffix") or "")
            base["style"]["fontPx"] = _fit_font(shown, base["box"][2], base["style"]["fontPx"])
        elif effect == "list":
            items = _list_items(el, fps)
            base["content"] = {"text": str(content.get("text") or ""), "items": items}
            longest = max((it["text"] for it in items), key=len, default="")
            base["style"]["fontPx"] = _fit_font(longest, base["box"][2], base["style"]["fontPx"],
                                                pad=150)
        else:
            text = str(content.get("text") or "")
            base["content"] = {"text": text}
            base["style"]["fontPx"] = _fit_font(text, base["box"][2], base["style"]["fontPx"])
        elements.append(base)

    kept_captions, dropped_captions = _suppress_duplicate_captions(elements, captions or [])
    return {
        "schema": SCHEMA,
        "canvas": {"width": int(canvas["width"]), "height": int(canvas["height"]),
                   "fps": fps, "durationInFrames": int(ir["source"]["n_frames"])},
        "source": {"file": "source.mp4", "audio": not hidden, "hidden": hidden},
        "captions": kept_captions,
        "captionsSuppressed": dropped_captions,
        "captionSafeBottomPx": int(caption_safe_bottom_px),
        "elements": elements,
    }


# ---------------------------------------------------------------------- render

def render(ir: dict, source: Path, out: Path, *, preview: bool = False,
           captions: list[dict] | None = None, hidden: bool = False,
           chrome: str | None = None, log_dir: Path | None = None) -> dict:
    source, out = Path(source), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir or (ROOT / "artifacts/tmp/remotion_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%H%M%S")
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

    staged = PUBLIC_DIR / "source.mp4"
    if not staged.exists() or media.sha256_file(staged) != media.sha256_file(source):
        shutil.copy2(source, staged)

    doc = props(ir, captions=captions, hidden=hidden)
    props_path = PUBLIC_DIR / "props.json"
    props_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    props_artifact = (out.parent / "remotion_props.json" if out.stem.endswith("_motion")
                      else out.parent / f"{out.stem}.props.json")
    props_artifact.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")

    chrome = chrome or _chrome()
    cmd = [str(ROOT / "remotion/node_modules/.bin/remotion"), "render",
           "src/index.ts", "Main", str(out.resolve()),
           f"--props={props_path.relative_to(REMOTION_DIR)}",
           "--log=error", "--concurrency=2", "--codec=h264", "--crf=18",
           "--pixel-format=yuv420p", "--audio-codec=aac",
           f"--timeout=60000"]
    if chrome:
        cmd.append(f"--browser-executable={chrome}")
    if preview:
        cmd.append("--scale=0.5")
    log_path = log_dir / f"remotion_{stamp}.log"
    proc = subprocess.run(cmd, cwd=REMOTION_DIR, capture_output=True, text=True, timeout=3600)
    log_path.write_text("$ " + " ".join(cmd) + "\n\n" + (proc.stdout or "")
                        + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else ""),
                        encoding="utf-8")
    if proc.returncode != 0 or not out.exists():
        raise media.MediaError("E_RENDER_FAILED", f"remotion exited {proc.returncode}",
                               cmd=cmd, stderr_tail=(proc.stderr or proc.stdout)[-3000:])

    mi = media.info(out)
    src_mi = media.info(source)
    # An overlay-only pass intentionally has no audio: never treat that as a failure.
    has_audio = mi.n_audio_streams > 0
    audio = _audio_stats(source), (_audio_stats(out) if has_audio else
                                   {"n_samples": 0, "duration_s": 0.0, "rms_db": -120.0})
    manifest = {
        "schema": "semotion.render/1",
        "backend": "remotion",
        "ir_sha256": (ir.get("provenance") or {}).get("ir_sha256"),
        "output": {
            "path": str(out.resolve()), "sha256": media.sha256_file(out),
            "size_bytes": out.stat().st_size, "duration_s": round(mi.duration, 6),
            "n_frames": mi.frame_count, "width": mi.width, "height": mi.height,
            "fps": mi.fps, "video_codec": mi.video_codec, "audio_codec": mi.audio_codec,
        },
        "source": {"path": str(source.resolve()), "sha256": media.sha256_file(source),
                   "duration_s": round(src_mi.duration, 6), "n_frames": src_mi.frame_count},
        "audio": ({
            "source": audio[0], "rendered": audio[1],
            "duration_delta_s": round(abs(audio[0]["duration_s"] - audio[1]["duration_s"]), 6),
            "rms_delta_db": round(abs(audio[0]["rms_db"] - audio[1]["rms_db"]), 3),
            "no_audio_by_design": False,
        } if has_audio else {
            "source": audio[0], "rendered": None,
            "no_audio_by_design": True,
            "note": "overlay-only verification pass renders without the source audio",
        }),
        "overlay": {"backend": "remotion", "elements": len(doc["elements"]),
                    "captions": len(doc["captions"] or [])},
        "commands": [" ".join(cmd)],
        "tool_versions": {
            **media.tool_versions(),
            "node": subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip(),
            "remotion": _remotion_version(),
            "chrome": chrome or "not found",
        },
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    # Never overwrite the deliverable manifest with a verification pass when both are
    # rendered into the same directory.
    manifest_path = out.parent / (f"{out.stem}.manifest.json" if out.stem != "motion"
                                  else "render_manifest.json")
    if out.stem.endswith("_motion"):
        manifest_path = out.parent / "render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return manifest


def _remotion_version() -> str:
    try:
        pkg = json.loads((REMOTION_DIR / "node_modules/remotion/package.json").read_text())
        return str(pkg.get("version"))
    except Exception:
        return "unknown"


def _audio_stats(path: Path) -> dict[str, float]:
    """Decoded PCM duration/RMS (same method as verify.py, kept local to avoid cycles)."""
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".s16le", delete=False) as tf:
        raw = Path(tf.name)
    try:
        media.run([media.FFMPEG, "-v", "error", "-y", "-i", str(path), "-vn",
                   "-ac", "2", "-ar", "48000", "-f", "s16le", str(raw)])
        data = np.frombuffer(raw.read_bytes(), dtype=np.int16).astype(np.float32) / 32768.0
        if data.size == 0:
            return {"n_samples": 0, "duration_s": 0.0, "rms_db": -120.0}
        rms = float(np.sqrt(np.mean(data ** 2)))
        return {"n_samples": int(data.size), "duration_s": round(data.size / 2 / 48000, 6),
                "rms_db": round(20 * np.log10(max(rms, 1e-9)), 3)}
    finally:
        raw.unlink(missing_ok=True)
