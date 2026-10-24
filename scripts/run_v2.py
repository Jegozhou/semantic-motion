#!/usr/bin/env python
"""Run the v2 design + render pipeline on an existing analysis bundle.

Example:
  PYTHONPATH=src .venv/bin/python scripts/run_v2.py \
    --video inputs/zhengqinwen_25s.mp4 \
    --anchors artifacts/dl/anchors_llm.json \
    --transcript artifacts/dl/run/transcript.json \
    --vision artifacts/dl/run/vision.json \
    --decisions artifacts/llm/download_zhengqinwen.decisions.json \
    --out artifacts/v2/demo_zhengqinwen --draft
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semotion import design2, media, render2, theme2  # noqa: E402


def load(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"  ! missing {p}")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--transcript")
    ap.add_argument("--vision")
    ap.add_argument("--perception")
    ap.add_argument("--decisions")
    ap.add_argument("--captions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--no-overlay-pass", action="store_true")
    ap.add_argument("--language", default="zh")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = media.info(args.video)
    source = {
        "path": str(Path(args.video).resolve()),
        "sha256": media.sha256_file(args.video),
        "width": info.width, "height": info.height, "fps": info.fps,
        "duration_s": info.duration, "frames": info.frame_count,
    }
    canvas = {"width": info.width, "height": info.height, "fps": info.fps}
    anchors_doc = load(args.anchors) or {}
    anchors = anchors_doc.get("anchors") or anchors_doc if isinstance(anchors_doc, list) else []
    if isinstance(anchors_doc, dict):
        anchors = anchors_doc.get("anchors") or []

    print(f"video   {info.width}x{info.height} @ {info.fps:.2f}  {info.duration:.2f}s")
    t0 = time.time()
    frames = render2.grab_frames(args.video, [info.duration * f for f in (0.1, 0.3, 0.5, 0.7, 0.9)])
    ir = design2.design(
        source=source, canvas=canvas, anchors=anchors,
        transcript=load(args.transcript), perception=load(args.perception),
        vision=load(args.vision), decisions=load(args.decisions),
        frames=frames,
        cfg=design2.DesignConfig(language=args.language),
    )
    (out / "motion_ir.json").write_text(json.dumps(ir, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"design  {len(ir['elements'])} elements in {time.time()-t0:.1f}s  "
          f"coverage={ir['layout_audit']['coverage']:.2f}")
    for e in ir["elements"]:
        print(f"  {e['type']:9s} {e['window']['start_s']:6.2f}-{e['window']['end_s']:6.2f}  "
              f"{e['layout']['treatment']:12s} box={e['box']}  "
              f"{str((e.get('content') or {}).get('text') or (e.get('content') or {}).get('title') or '')[:30]}")

    t1 = time.time()
    manifest = render2.render(
        ir, args.video, out / "final.mp4",
        captions=load(args.captions), perception=load(args.perception),
        overlay_pass=not args.no_overlay_pass, draft=args.draft,
        log_path=out / "render.log", max_frames=args.max_frames,
        progress=lambda i, n: print(f"\r  render {i}/{n}", end="", flush=True),
    )
    print(f"\nrender  {manifest['output']['frames']} frames in {manifest['elapsed_s']}s "
          f"({manifest['fps_real']} fps)")
    sheet = render2.contact_sheet(ir, out / "final.mp4", out / "contact.png")
    print("contact", sheet)
    print("output ", (out / "final.mp4").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
