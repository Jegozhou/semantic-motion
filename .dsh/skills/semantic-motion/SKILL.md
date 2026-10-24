---
name: semantic-motion
description: |
  Turn a raw talking-head video into a finished, publishable video with semantic motion
  graphics: word-accurate transcription, person/pose-aware layout, typographic overlays
  (keywords, numbers, steps, cards, charts, comparisons), burned-in captions, objective
  QA and a deterministic render. Keeps the original picture and voice; every overlay is
  time-aligned to the words that justify it. Ships an editable Motion IR, a metrics report
  and a reproducible render manifest.
  Triggers: 给口播视频加动效, 自动字幕动效, 关键词强调/数字动画/步骤列表/信息卡片, 语义动效,
  口播后期, semantic motion, talking-head motion graphics, 把口播素材做成动效成片,
  需要可编辑工程与质量报告的口播后期, 视频动效自动化.
---

# Semantic Motion

A semantic motion-design pipeline for talking-head video. It does not re-shoot, dub,
cut or colour-grade. It measures where the words actually are, decides which ideas earn
a graphic, and composes that graphic **beside the speaker** — then proves with numbers
that it did not cover the face, the burned-in credits or the captions.

The default style is *Smart Balanced*: type first, one accent colour sampled from the
footage, a directional scrim instead of pasted panels, short eased motion, and plenty of
silence. Most sentences get nothing. That is the point.

## When to use it

* "给这段口播加上关键词/数字/步骤的动效，别挡脸"
* "字幕要和语音对得上，还要能改"
* "同样的素材出一版成片 + 可编辑工程 + 质检报告"
* Any interview, explainer, course clip or vlog where the speaker's words should be
  reinforced visually without editing the performance.

Do **not** use it for dubbing, translation, auto-cutting, colour grading, or generating a
new performance. It does not invent content: overlay text comes from the transcript or
from decisions you author.

## Requirements

* macOS 14+ (Apple Silicon recommended) — the perception layer uses Apple Vision for
  person segmentation, body pose, hands and burned-in-text OCR. No cloud calls.
* `ffmpeg` / `ffprobe` on PATH (no `drawtext`/`libass` needed; text is rasterised by us).
* Python 3.11+ and a project venv with the dependencies in `requirements.lock`
  (Pillow, numpy, faster-whisper, pyobjc-framework-Vision).
* Bundled fonts: Noto Sans SC (Bold/Medium), Inter. No system font is required.
* This repository checkout. Set `SM_ROOT` to it.

```bash
export SM_ROOT="/path/to/Semantic Motion"
cd "$SM_ROOT" && export PYTHONPATH=src
.venv/bin/python scripts/env_probe.py artifacts/env/env_probe.json   # toolchain check
```

## Quick start

```bash
cd "$SM_ROOT" && export PYTHONPATH=src

# 1) draft: fast, low bitrate, same design decisions
.venv/bin/python -m semotion.pipeline run \
    --video /path/to/talk.mp4 --out out/talk --draft

# 2) look at out/talk/storyboard.png and out/talk/final.mp4, fix taste (see Directing)

# 3) full quality (the deliverable)
.venv/bin/python -m semotion.pipeline run \
    --video /path/to/talk.mp4 --out out/talk
```

Output:

| file | what it is |
|---|---|
| `final.mp4` | deliverable; original audio preserved, overlays + captions burned in |
| `final_overlay.mp4` | overlay-only pass — how QA attributes every changed pixel |
| `motion_ir.json` | **editable** Motion IR (`semotion.motion-ir/2`) |
| `captions.json/.srt/.ass` | frame-accurate captions |
| `storyboard.png` | one frame per element, labelled |
| `qa_metrics.json` | objective checks with raw numbers |
| `run_summary.json` | everything needed to reproduce the run |

## The five stages

1. **Speak** — faster-whisper (local, cached) for text; `src/semotion/align.py` aligns
   every character to the audio onset envelope with a monotonic DP. Whisper's Chinese
   word timestamps are not trusted (they lag by ~0.2 s on real speech).
2. **See** — `src/semotion/perception.py` (Apple Vision) extracts a per-frame person
   mask, body pose, hands and OCR boxes for burned-in text, then ranks quiet zones.
3. **Decide** — the agent (or a human) writes `decisions.json`; the deterministic
   fallback in `pipeline.auto_anchors` is deliberately conservative. See *Directing*.
4. **Design** — `design2.py` picks a treatment and a type scale inside measured free
   space; `render2.py` composites deterministically with Pillow + ffmpeg.
5. **Verify** — `verify2.py` measures the rendered frames against the person mask,
   burned-in text, contrast, margins, concurrent collisions and per-element timing;
   `repair.py` can relocate/shrink/drop and re-render.

## Directing (this is what makes the output good)

Read `out/talk/anchors.json` (and `alignment.json` when you need the exact words), then
write `decisions.json`. Do not add motion mechanically to every sentence.

Rules of taste:

* **0–4 elements per 10 s**, ≥0.9 s apart, ≤58 % of runtime covered. Plain narration
  stays plain.
* Earn the graphic: a number, an enumeration, a contrast (“但是/其实/不是…而是”), a
  conclusion (“就是/意味着”), a superlative.
* One idea per element. 8–16 Chinese characters (≤ 2 lines). Never invent digits, names
  or claims that are not in the transcript.
* Numbers must be verbatim: if you show 90 %, the speaker must have said 九成 or 90 %.
* Steps need ≥2 rows and must be spoken in order; use `effect: "list"` with
  `anchor_ids` in speech order.
* Do not animate filler (“大家好”, “然后呢”, “对”). Let the face carry it.

`decisions.json` (pass with `--decisions`):

```json
{
  "schema": "semotion.llm-decisions/1",
  "decision_model": "claude",
  "prompt_summary": "why this clip got these choices",
  "decisions": [
    {"anchor_id": "a01", "anchor_ids": ["a01"], "effect": "keyword",
     "emphasis": "high", "hold_s": 0.35, "confidence": 0.85,
     "reason": "转折句，7 个字，值得停一下",
     "params": {"emphasis": ["梦想"]}},
    {"anchor_id": "a02", "anchor_ids": ["a02","a03","a04"], "effect": "list",
     "reason": "三层递进，按口播顺序逐行揭示", "params": {"kicker": "要点"}},
    {"anchor_id": "a06", "effect": "stat",
     "params": {"unit": "%", "label": "受访者认为节奏最重要"}}
  ],
  "skipped": [{"anchor_id": "a05", "reason": "与 a06 语义连续，两个一起出现会过密"}]
}
```

Effects: `keyword` (display line + rule), `card` (kicker/title/body), `list` (numbered,
row-by-row), `stat` (big number + label; counts up), `barchart` (`params.bars`),
`compare` (`params.left/right`), `quote` (`params.attribution`).

Everything spatial is automatic: never put a box over the speaker, the burned-in
credits or the caption band. If no safe space exists the element is **dropped**, and the
reason is recorded in `layout_audit`.

## Editing a finished render

`motion_ir.json` is the engineering file. Useful fields per element:

* `window.start_s/end_s` — timing (keep `anchor_start_s` intact unless you mean it);
* `content.text/title/rows[]` — copy;
* `motion.enter.dur_s`, `motion.enter.ease` — movement;
* `fit.display_px` / `fit.title_px` — type size (the layout box comes from `box`);
* `backdrop.kind` — `scrim_right|scrim_left|band_top|band_bottom|card`.

Re-render after editing:

```bash
.venv/bin/python -m semotion.render2 --ir out/talk/motion_ir.json \
    --source /path/to/talk.mp4 --out out/talk/final.mp4
```

A *natural language* request (“把第二个关键词往右挪，字号小一点，晚两帧出现”) should be
translated by the agent into those fields — editing the IR is the supported interface.

## QA: what the numbers mean

`qa_metrics.json` reports, per element and overall:

| check | meaning | threshold |
|---|---|---|
| `person_occlusion` | overlay pixels inside the person mask / face | face 0 px; person ≤0.2 % frame |
| `existing_text_collision` | overlay over protected burned-in text | 0 px |
| `legibility` | WCAG contrast of glyph pixels vs the real background | ≥4.5 |
| `layout` | on-canvas, safe margins, collisions | margins ≥2.5 % |
| `timing` | first/last visible frame vs anchor onset | ≤1 frame |
| `displayed_text` | OCR of the render vs the IR copy | no truncation |
| `footprint` | what changed vs the source | attribution only |
| `determinism` | two renders, decoded-frame diff | mean ≤0.05/255 |

A FAIL is information, not a crash: `semotion.repair` proposes deterministic fixes
(relocate to a zero-person-overlap zone, shrink, add scrim, retime, drop) and writes
`repairs.json`. Re-render and re-verify.

## Honest limits

* **Timing truth is synthetic.** The 7 ms P90 figure comes from TTS fixtures with known
  unit boundaries. Real human speech has no labelled corpus here: on real clips we can
  only prove monotonicity, plausible syllable durations and onset-envelope tracking.
* **Apple Vision is macOS-only.** On other platforms perception degrades to face boxes
  and the layout becomes conservative. The renderer itself is portable.
* **Occlusion policy.** Default: never over the person. A last-resort `lower_third` may
  sit over clothing but never over face/hair/hands. If nothing is free, an element is
  dropped rather than pasted on someone.
* **Aesthetics are not machine-verified.** Contrast, timing, margins and coverage are;
  “does it look designed” is a human/agent judgement. Look at the storyboard.
* **No data visualisation from nowhere.** `barchart`/`compare` render numbers you supply;
  they do not infer statistics.
* Charts from external data, 3-D, stock footage, music and voice change are out of scope.

## Verify the installation

```bash
cd "$SM_ROOT" && export PYTHONPATH=src
.venv/bin/python scripts/env_probe.py artifacts/env/env_probe.json
.venv/bin/python -m pytest tests/ -q          # unit + integration
.venv/bin/python -m semotion.pipeline run \
    --video "inputs/zhengqinwen_25s.mp4" --out /tmp/smoke --draft --no-verify
```

Reference renders and their metrics live in `delivery/` and `reports/`.
