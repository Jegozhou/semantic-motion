#!/usr/bin/env python3
"""Verify numeric overlays settle on the semantically correct value.

Method (deterministic, no OCR guesswork)
----------------------------------------
For every `number` element in the Motion IR:

1. extract the rendered frame where the count-up has settled (and the matching
   source frame, frame-exact via `select=eq(n,..)`);
2. re-render the *expected* element tile from the IR itself with the same
   deterministic rasteriser (`raster.element_spec` + `effects.number.draw`);
3. composite that tile over the source crop (the panel is semi-transparent, so a
   naive crop comparison would be wrong) and compare with the actual rendered crop;
4. repeat with a deliberately wrong value (value/2) and require the expected tile
   to match clearly better than the decoy — that guards against a vacuous pass.

Verdicts: PASS only when the expected composite matches within tolerance *and*
the decoy is clearly worse. Anything else is FAIL and never counted as a pass.
Exit code 0 on PASS/NOT RUN, 1 on FAIL.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semotion import effects, media, raster  # noqa: E402

# Tolerances are set from measured behaviour: an H.264 CRF-18 render differs from a
# fresh raster by a few units of antialiasing noise (mean ≈2.2, p99 ≈19), while a
# wrong value produced by the decoy differs by an order of magnitude (p99 ≈138).
TOL_MEAN = 3.5        # mean |Δ| over the box (0..255)
TOL_P99 = 40          # 99th percentile |Δ| for the correct value
DECOY_P99_MIN = 80    # the half-value decoy must be clearly worse
DECOY_MEAN_MULT = 1.6


def _frame_at(video: Path, frame: int, dst: Path) -> None:
    subprocess.run([media.FFMPEG, "-v", "error", "-y", "-i", str(video),
                    "-vf", f"select=eq(n\\,{frame})", "-vsync", "0",
                    "-frames:v", "1", str(dst)], check=True)


def _crop(img: Image.Image, box: list[int]) -> Image.Image:
    x, y, w, h = (int(v) for v in box)
    return img.crop((x, y, x + w, y + h)).convert("RGBA")


def _alpha_at(element: dict, t_rel: float, window_s: float) -> float:
    """Total entrance*exit opacity of the element at this relative time."""
    motion = element.get("motion") or {}
    enter_dur = max(1e-3, float((motion.get("enter") or {}).get("dur_s") or 0.26))
    eased = raster.base.ease(str((motion.get("enter") or {}).get("ease") or "out_cubic"),
                             min(1.0, max(0.0, t_rel / enter_dur)))
    alpha_enter, _ = raster._enter(eased, motion)
    return alpha_enter * raster._exit_alpha(t_rel, window_s, motion)


def _best_frame(el: dict, fps: float) -> tuple[int, float]:
    """Pick the most opaque frame after the count-up settled (strongest evidence)."""
    win = el["window"]
    window_s = (float(win["end_s"]) - float(win["start_s"])) or 1.0
    start, end = int(win["start_frame"]), int(win["end_frame"])
    lo = start + int(round(0.30 * fps))
    hi = max(lo + 1, end - 1)
    best_f, best_a = lo, -1.0
    for f in range(lo, hi + 1):
        a = _alpha_at(el, f / fps - float(win["start_s"]), window_s)
        if a > best_a + 1e-6:
            best_f, best_a = f, a
    return min(best_f, end - 1), window_s


def _tile(element: dict, canvas: dict, *, value: float | None, t_rel: float,
          window_s: float) -> Image.Image:
    """Re-render the tile exactly as raster.py would at this relative time.

    effects.draw paints the settled artwork; the entrance/exit alpha and scale are
    applied by raster.py around it, so they are applied here too — otherwise a
    sample taken while the element fades would compare against a full-opacity tile.
    """
    el = json.loads(json.dumps(element))          # deep copy
    if value is not None:
        numeric = el["content"].setdefault("numeric", {})
        numeric["value"] = float(value)
    spec = raster.element_spec(el, canvas)
    tile = effects.draw(el, canvas, spec, t_rel=t_rel, window_s=window_s)

    motion = el.get("motion") or {}
    enter_dur = max(1e-3, float((motion.get("enter") or {}).get("dur_s") or 0.26))
    eased = raster.base.ease(str((motion.get("enter") or {}).get("ease") or "out_cubic"),
                             min(1.0, max(0.0, t_rel / enter_dur)))
    alpha_enter, scale = raster._enter(eased, motion)
    alpha_exit = raster._exit_alpha(t_rel, window_s, motion)
    alpha = alpha_enter * alpha_exit
    if abs(scale - 1.0) > 1e-6:
        w, h = tile.size
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        scaled = tile.resize((nw, nh), Image.LANCZOS)
        canvas_tile = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        canvas_tile.paste(scaled, ((w - nw) // 2, (h - nh) // 2))
        tile = canvas_tile
    if alpha < 0.999:
        arr = np.asarray(tile, dtype=np.float32).copy()
        arr[..., 3] *= alpha
        tile = Image.fromarray(arr.astype(np.uint8), "RGBA")
    return tile


def _compare(actual: np.ndarray, source: np.ndarray, tile: Image.Image) -> dict:
    tile_np = np.asarray(tile.convert("RGBA"), dtype=np.float32)
    alpha = tile_np[..., 3:4] / 255.0
    sim = tile_np[..., :3] * alpha + source[..., :3] * (1.0 - alpha)
    d = np.abs(sim - actual[..., :3]).mean(axis=2)
    return {
        "mean": round(float(d.mean()), 4),
        "p99": round(float(np.percentile(d, 99)), 4),
        "max": round(float(d.max()), 4),
        "within_p99": round(float((d <= TOL_P99).mean()), 4),
    }


def check(ir_path: Path, video: Path, out: Path) -> dict:
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    canvas = {"width": int(ir["canvas"]["width"]), "height": int(ir["canvas"]["height"]),
              "fps": float(ir["canvas"]["fps"])}
    source = Path(ir["source"]["path"])
    fps = float(ir["source"]["fps"])
    results = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for el in ir.get("elements", []):
            if el.get("type") != "number":
                continue
            numeric = (el.get("content") or {}).get("numeric")
            if not isinstance(numeric, dict) or "value" not in numeric:
                results.append({"element_id": el["id"], "status": "NO_NUMERIC_PAYLOAD",
                                "text": (el.get("content") or {}).get("text")})
                continue
            expected = float(numeric["value"])
            win = el["window"]
            frame, window_s = _best_frame(el, fps)
            t_rel = frame / fps - float(win["start_s"])

            out_png, src_png = td / "out.png", td / "src.png"
            _frame_at(video, frame, out_png)
            _frame_at(source, frame, src_png)
            actual = np.asarray(_crop(Image.open(out_png), el["layout"]["box"]), dtype=np.float32)
            src_np = np.asarray(_crop(Image.open(src_png), el["layout"]["box"]), dtype=np.float32)

            exp_tile = _tile(el, canvas, value=None, t_rel=t_rel, window_s=window_s)
            decoy_tile = _tile(el, canvas, value=max(0.0, expected * 0.5), t_rel=t_rel,
                               window_s=window_s)
            exp = _compare(actual, src_np, exp_tile)
            dec = _compare(actual, src_np, decoy_tile)

            ok = (exp["mean"] <= TOL_MEAN and exp["p99"] <= TOL_P99
                  and dec["p99"] >= DECOY_P99_MIN
                  and dec["mean"] >= exp["mean"] * DECOY_MEAN_MULT)
            results.append({
                "element_id": el["id"], "text": (el.get("content") or {}).get("text"),
                "expected_value": expected, "numeric_final": numeric.get("final"),
                "frame": frame, "t_rel_s": round(t_rel, 3), "alpha_at_frame": round(
                    _alpha_at(el, t_rel, window_s), 4), "box": el["layout"]["box"],
                "expected_match": exp, "decoy_half_value_match": dec,
                "status": "PASS" if ok else "FAIL",
            })
    mismatches = [r for r in results if r["status"] == "FAIL"]
    passes = [r for r in results if r["status"] == "PASS"]
    verdict = "NOT RUN" if not results else ("FAIL" if mismatches else
                                             ("PASS" if passes else "NOT RUN"))
    doc = {"schema": "semotion.number_check/2", "video": str(video), "ir": str(ir_path),
           "checks": results, "verdict": verdict,
           "readable": len(passes), "mismatches": len(mismatches),
           "method": ("deterministic re-render comparison: expected tile composited over the "
                      "source crop vs the actual rendered crop, with a decoy value proving "
                      "the test can fail"),
           "tolerance": {"mean_abs": TOL_MEAN, "p99_abs": TOL_P99,
                         "decoy_p99_min": DECOY_P99_MIN, "decoy_mean_mult": DECOY_MEAN_MULT}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ir")
    ap.add_argument("video")
    ap.add_argument("out")
    a = ap.parse_args()
    doc = check(Path(a.ir), Path(a.video), Path(a.out))
    for c in doc["checks"]:
        if c["status"] == "PASS":
            print(f"  {c['element_id']}: PASS  value={c['expected_value']} "
                  f"mean={c['expected_match']['mean']:.2f} "
                  f"decoy_mean={c['decoy_half_value_match']['mean']:.2f}")
        else:
            print(f"  {c['element_id']}: {c['status']}  value={c.get('expected_value')} "
                  f"{json.dumps(c.get('expected_match'), ensure_ascii=False)}")
    print(f"{doc['verdict']} -> {a.out}")
    return 0 if doc["verdict"] in ("PASS", "NOT RUN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
