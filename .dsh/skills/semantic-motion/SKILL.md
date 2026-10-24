---
name: semantic-motion
description: |
  Add semantic-driven motion graphics to a talking-head video while keeping the original footage and audio: transcribe with word-level timing, pick the moments worth emphasising, and render keyword / number / step-list / info-card overlays that are time-aligned, face-safe and reproducible. Produces a playable MP4 plus an editable Motion IR and a metrics report.
  Triggers: 给口播视频加动效, 自动字幕动效, 关键词强调/数字动画/步骤列表/信息卡片, 语义动效, semantic motion, talking-head motion graphics, 把口播素材做成动效成片, 需要可编辑工程与质量报告的口播后期.
---

# Semantic Motion

Turn a talking-head video into the same video **plus** restrained, meaningful motion
graphics — or, just as often, leave it alone. The skill never re-shoots or re-voices
your footage: it keeps picture and sound, measures where words actually occur, and
only overlays what the sentence deserves.

## When to use

* "给这段口播加动效 / 自动做关键词强调、数字动画、步骤列表、信息卡片"
* "字幕要和语音精确对齐，别挡住我的脸"
* "给我一份可编辑的动效工程（IR）和质检报告"
* "同样是这条口播，出一版 9:16 和一版 16:9"

Do **not** use it to dub, cut, colour-grade, or generate a new performance.

## Requirements (checked by `hypit`-independent local scripts)

* macOS (arm64 tested) with `ffmpeg`/`ffprobe` (this build must include `overlay`;
  `drawtext`/`subtitles` are NOT required), Python 3.11+ for the project venv,
  `tesseract` optional (only for legacy OCR checks; the shipped numeric check is
  deterministic and does not need it).
* Project checkout at `/Volumes/ZHITAI/SemanticMotion` (or set `SEMOTION_ROOT`).
* Model: local CTranslate2 Whisper snapshot (offline). No network is required at
  render time.

## Quick start

```bash
cd "$SEMOTION_ROOT" && export PYTHONPATH=src

# 1) end-to-end on a fixture (sanity: proves the toolchain works)
.venv/bin/python -m semotion.cli run --fixture fx_a_zh_portrait --outdir out/demo

# 2) your own video
.venv/bin/python -m semotion.cli run --input /path/to/talk.mp4 --outdir out/talk

# 3) or step by step (each step is cacheable and resumable)
.venv/bin/python -m semotion.cli asr run    --video talk.mp4 --out work/transcript.json
.venv/bin/python -m semotion.cli vision analyze --video talk.mp4 --out work/vision.json
.venv/bin/python -m semotion.cli anchors build --transcript work/transcript.json --video talk.mp4 --out work/anchors.json
.venv/bin/python -m semotion.cli plan  --anchors work/anchors.json --vision work/vision.json \
    --transcript work/transcript.json --canvas 1080x1920 --fps 25 --out work/plan.json
.venv/bin/python -m semotion.cli ir build --transcript work/transcript.json --vision work/vision.json \
    --anchors work/anchors.json --plan work/plan.json --source talk.mp4 \
    --width 1080 --height 1920 --fps 25 --out work/motion_ir.json
.venv/bin/python -m semotion.cli render --ir work/motion_ir.json --source talk.mp4 --out out/final.mp4
.venv/bin/python -m semotion.cli verify --ir work/motion_ir.json --render out/render_manifest.json \
    --source talk.mp4 --rendered out/final.mp4 --vision work/vision.json --gate manual --out out/metrics.json
```

## What you get

| File | Meaning |
|---|---|
| `out/final.mp4` | the deliverable; original audio kept, overlays burned in |
| `motion_ir.json` | editable engineering file (schema `semotion.motion-ir/1`) |
| `transcript.json` | word-level ASR (`semotion.transcript/1`) |
| `anchors.json` | semantic anchors with confidence and onset refinement |
| `plan.json` | director decisions, including what was deliberately skipped |
| `render_manifest.json` | exact commands, hashes, tool versions |
| `metrics.json` | timing / position / render verdicts with raw numbers |

## Editing an existing project

Edit `motion_ir.json` directly (text, timing window, colour, box), then:

```bash
.venv/bin/python -m semotion.cli render --ir motion_ir.json --source talk.mp4 --out out/v2.mp4
```

The IR is validated first (8 invariants: frame math, anchor tolerance, on-canvas box,
≤1% face coverage, track ownership, declared collisions, stable hash, text provenance).
Illegal edits are refused with an explicit error instead of rendering something broken.
For an intentional human rewrite of the copy set `content.edited = true`.

## Design defaults (Smart Balanced)

* Only important information triggers motion; **plain speech gets zero overlays**.
* At most ~4 new elements per 10 s, ≥0.9 s between elements, ≤58% runtime covered.
* One element per time span; numbers/steps outrank loose keywords.
* Overlays never straddle a camera cut; if no face-free space exists the element is
  dropped rather than covering the speaker.
* Overlays are static-safe: they do **not** chase the face.

## Honest limits

* **Chinese-first rules.** English footage is supported for numbers/content keywords,
  but step lists and conclusion cards need Chinese patterns — the skill reports
  `NO_ANCHORS` rather than inventing anchors.
* Timing accuracy is validated on synthetic TTS calibration (56 events, P90 ≈ 50 ms).
  It is **not** validated against human-labelled real speech (no such labels exist yet).
* No visual tracking; degradation is conservative (pin overlays, avoid the face union).
* Aesthetics are subjective; the automated checks only cover timing, position,
  reproducibility and numeric correctness.
* Local generation only: no hosted models, no cloud rendering, no custom code execution.

## Verify the installation

```bash
cd "$SEMOTION_ROOT" && export PYTHONPATH=src
.venv/bin/python scripts/env_probe.py artifacts/env/env_probe.json   # toolchain
.venv/bin/python -m pytest tests/ -q                                 # 92 tests
.venv/bin/python scripts/regression.py --quick                       # evidence chain
```
