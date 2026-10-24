#!/usr/bin/env python3
"""Build human-facing demo assets for one rendered project.

Outputs (into --out):
  * before_after.mp4  — source on the left, animated render on the right
  * storyboard.png    — labelled frames at each effect onset (+ a few empty moments)

Labels are rasterised with Pillow (this ffmpeg has no drawtext).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semotion import media  # noqa: E402

FONT = ROOT / "assets/fonts/NotoSansSC-Regular.otf"
TYPE_CN = {"keyword": "关键词强调", "number": "数字计数", "list": "步骤列表", "card": "信息卡片"}


def frame_at(video: Path, frame: int, dst: Path) -> None:
    subprocess.run([media.FFMPEG, "-v", "error", "-y", "-i", str(video),
                    "-vf", f"select=eq(n\\,{frame})", "-vsync", "0",
                    "-frames:v", "1", str(dst)], check=True)


def before_after(src: Path, rendered: Path, dst: Path) -> None:
    subprocess.run([media.FFMPEG, "-v", "error", "-y", "-i", str(src), "-i", str(rendered),
                    "-filter_complex", "[0:v]scale=540:-2,setpts=PTS-STARTPTS[l];"
                                       "[1:v]scale=540:-2,setpts=PTS-STARTPTS[r];"
                                       "[l][r]hstack=inputs=2[v]",
                    "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-preset", "medium",
                    "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart", str(dst)], check=True)


def storyboard(ir: dict, rendered: Path, out: Path, *, title: str) -> None:
    fps = float(ir["source"]["fps"])
    duration = float(ir["source"]["duration_s"])
    elements = ir["elements"]
    marks: list[tuple[int, str]] = []
    for el in elements:
        w = el["window"]
        f = int((float(w["start_s"]) + float(w["end_s"])) / 2 * fps)
        text = (el.get("content") or {}).get("text") or ""
        marks.append((f, f"{float(w['start_s']):5.2f}s  {TYPE_CN.get(el['type'], el['type'])}"
                         f"「{text}」"))
    # a couple of "no effect" moments for contrast
    busy = {f for f, _ in marks}
    empties = [int(duration * 0.10 * fps), int(duration * 0.62 * fps)]
    marks.extend((f, "此处无动效（普通叙述，保持原片）") for f in empties if all(abs(f - b) > 0.5 * fps for b in busy))
    marks.sort()
    marks = marks[:8]

    tmp = out.parent / "_frames"
    tmp.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for i, (f, label) in enumerate(marks):
        png = tmp / f"f{i}.png"
        frame_at(rendered, f, png)
        img = Image.open(png).convert("RGB")
        W = 360
        img = img.resize((W, int(img.height * W / img.width)), Image.LANCZOS)
        thumbs.append((img, label))

    cols = 4 if len(thumbs) > 4 else len(thumbs)
    rows = (len(thumbs) + cols - 1) // cols
    cell_w = 360
    cell_h = max(t.height for t, _ in thumbs)
    pad, bar = 12, 46
    head = 64
    sheet = Image.new("RGB", (cols * cell_w + (cols + 1) * pad,
                              head + rows * (cell_h + bar + pad) + pad), (18, 20, 26))
    d = ImageDraw.Draw(sheet)
    f_title = ImageFont.truetype(str(FONT), 30)
    f_small = ImageFont.truetype(str(FONT), 19)
    d.text((pad, 18), title, font=f_title, fill=(240, 242, 248))
    for i, (img, label) in enumerate(thumbs):
        r, c = divmod(i, cols)
        x = pad + c * (cell_w + pad)
        y = head + r * (cell_h + bar + pad)
        sheet.paste(img, (x, y))
        d.rectangle([x, y + cell_h, x + cell_w, y + cell_h + bar], fill=(30, 34, 44))
        d.text((x + 8, y + cell_h + 12), label, font=f_small, fill=(242, 178, 51))
    sheet.save(out, "PNG")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", required=True)
    ap.add_argument("--rendered", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="语义动效效果分镜")
    a = ap.parse_args()
    ir = json.loads(Path(a.ir).read_text(encoding="utf-8"))
    src = Path(ir["source"]["path"])
    rendered = Path(a.rendered)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    before_after(src, rendered, out / "before_after.mp4")
    storyboard(ir, rendered, out / "storyboard.png", title=a.title)
    print(f"{out / 'before_after.mp4'}\n{out / 'storyboard.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
