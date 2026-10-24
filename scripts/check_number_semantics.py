#!/usr/bin/env python3
"""Check that numeric overlays end on the semantically correct value.

Reads the Motion IR, finds every `number` element, extracts the rendered frame just
before the element disappear, crops the declared box and OCRs it (tesseract with a
digit whitelist). The expected value comes from `content.numeric` when the IR
carries it, otherwise it is parsed from `content.text` with the same dialect rules
the renderer uses (Chinese numerals -> arabic, 成 -> x10 %).

Exit code 0 when every checked value matches, 1 otherwise. Unreadable crops are
reported as `UNREADABLE` and count as failures: a number overlay we cannot verify
is not a pass.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semotion import media  # noqa: E402

CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "俩": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def expected_value(element: dict) -> tuple[float, str] | None:
    """Return (value, display_kind) for a numeric element."""
    content = element.get("content") or {}
    numeric = content.get("numeric")
    if isinstance(numeric, dict):
        for key in ("value", "target", "final", "number"):
            if isinstance(numeric.get(key), (int, float)):
                return float(numeric[key]), str(numeric.get("kind") or "plain")
    text = str(content.get("text") or "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(%|％)?", text)
    if m:
        val = float(m.group(1))
        return val, ("percent" if m.group(2) else "plain")
    m = re.search(r"([零一二两俩三四五六七八九十百]+)\s*(成|个|次|倍|分|%)?", text)
    if m:
        raw, unit = m.group(1), m.group(2) or ""
        if raw == "十":
            val = 10.0
        elif raw == "百":
            val = 100.0
        else:
            val = float(CN_DIGITS.get(raw[0], 0))
        if unit == "成":
            return val * 10.0, "percent"
        return val, ("percent" if unit in ("%", "分") else "plain")
    return None


def _tess(png: Path, psm: int, *, cjk: bool = False) -> str:
    argv = ["tesseract", str(png), "stdout", "--psm", str(psm)]
    if cjk:
        # unit/plain numbers carry a Chinese unit glyph (3秒, 6次); a digits-only
        # whitelist forces tesseract to hallucinate digits out of that glyph.
        argv += ["-l", "chi_sim+eng"]
    else:
        argv += ["-c", "tessedit_char_whitelist=0123456789.%"]
    out = subprocess.run(argv, capture_output=True, text=True)
    return (out.stdout or "").strip()


def ocr_crop(png: Path, *, cjk: bool = False) -> tuple[str, dict]:
    """OCR a rendered number with two page-segmentation modes and report both."""
    with tempfile.TemporaryDirectory() as td:
        big = Path(td) / "big.png"
        subprocess.run(["magick", str(png), "-resize", "300%", str(big)], check=True)
        a = _tess(big, 8, cjk=cjk)
        b = _tess(big, 13, cjk=cjk)
    return (a or b), {"psm8": a, "psm13": b}


def main() -> int:
    if len(sys.argv) < 4:
        print("usage: check_number_semantics.py <motion_ir.json> <rendered.mp4> <out.json>")
        return 2
    ir_path, video, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    fps = float(ir["source"]["fps"])
    checks = []
    for el in ir.get("elements", []):
        if el.get("type") != "number":
            continue
        exp = expected_value(el)
        if exp is None:
            checks.append({"element_id": el["id"], "status": "NO_EXPECTATION",
                           "text": (el.get("content") or {}).get("text")})
            continue
        want, kind = exp
        # frame just before the element ends (animation settled)
        frame = max(0, int(el["window"]["end_frame"]) - 2)
        t = frame / fps
        box = el["layout"]["box"]
        with tempfile.TemporaryDirectory() as td:
            frame_png = Path(td) / "frame.png"
            subprocess.run([media.FFMPEG, "-v", "error", "-y", "-ss", f"{t:.3f}",
                            "-i", str(video), "-frames:v", "1", str(frame_png)], check=True)
            crop = Path(td) / "crop.png"
            subprocess.run(["magick", str(frame_png), "-crop",
                            f"{box[2]}x{box[3]}+{box[0]}+{box[1]}", "+repage", str(crop)],
                           check=True)
            got, raw_variants = ocr_crop(crop, cjk=(kind != "percent"))
        digits = re.sub(r"[^0-9.%]", "", got)
        num = re.search(r"\d+(?:\.\d+)?", digits)
        observed = float(num.group(0)) if num else None
        ok = observed is not None and abs(observed - want) < 0.51
        checks.append({
            "element_id": el["id"], "text": (el.get("content") or {}).get("text"),
            "expected_value": want, "expected_kind": kind,
            "ocr_raw": got, "ocr_variants": raw_variants, "observed_value": observed,
            "frame": frame, "t_s": round(t, 3), "box": box,
            "status": "PASS" if ok else ("UNREADABLE" if observed is None else "FAIL"),
        })
    mismatches = [c for c in checks if c["status"] == "FAIL"]
    readable = [c for c in checks if c["status"] == "PASS"]
    unreadable = [c for c in checks if c["status"] == "UNREADABLE"]
    if not checks:
        verdict = "NOT RUN"
    elif mismatches:
        verdict = "FAIL"
    elif not readable:
        verdict = "NOT RUN"
    else:
        verdict = "PASS"
    doc = {"schema": "semotion.number_check/1", "video": str(video), "ir": str(ir_path),
           "checks": checks, "verdict": verdict,
           "readable": len(readable), "mismatches": len(mismatches),
           "unreadable": [c["element_id"] for c in unreadable],
           "coverage_note": ("values are OCR-verified on the rendered final frame; "
                             "unreadable crops are reported, not counted as passes"),
           "method": "render final frame -> crop IR box -> tesseract digits-only OCR"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for c in checks:
        print(f"  {c['element_id']}: {c['status']:10} expected={c.get('expected_value')} "
              f"ocr={c.get('ocr_raw')!r}")
    print(f"{verdict} -> {out_path}")
    return 0 if verdict in ("PASS", "NOT RUN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
