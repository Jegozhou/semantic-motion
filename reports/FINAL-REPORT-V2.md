# Final report — Semantic Motion

Status: **v2 delivered** (2026-10-24). This report supersedes `FINAL-REPORT.md` v1 and the
G0–G5 gate reports for the parts they overlap. Everything below is backed by a file in
this repo; numbers without a path are not claims.

## 1. What was asked, and what exists

The brief: a Claude Code skill that takes one raw talking-head clip, understands what is
being said, decides where a visual would help, designs and renders that visual so it is
semantically right, frame-accurate, compositionally safe and professionally restrained —
then delivers a publishable video plus an editable project and an honest quality report.

What exists:

* A native skill (`.dsh/skills/semantic-motion/SKILL.md`, installed into
  `~/.claude/skills` and `~/.dsh/skills` via `scripts/install_skill.py`).
* A complete pipeline (`python -m semotion.pipeline run --video … --out …`) covering
  transcription, character alignment, perception, directing, layout, rendering, captions
  and QA.
* A v2 design/render stack (`theme2.py`, `effects2.py`, `design2.py`, `render2.py`) that
  replaced v1's panel-on-a-face layout.
* Objective verification (`verify2.py`) that measures the rendered pixels against the
  person mask, the burned-in text, contrast, margins and per-frame timing, plus a
  deterministic repair planner (`repair.py`).
* Evidence: masks, metrics, storyboards, two independent full-quality renders, and a
  reference delivery bundle on real footage.

## 2. The reference run (real footage)

Source: Wikimedia Commons, CC BY 3.0, interview with Zheng Qinwen, 1920×920, 25.0 s,
25 fps, 625 frames (`inputs/zhengqinwen_25s.mp4`, sha256 in
`artifacts/v2/final_zq/run_summary.json`). This is real human speech with a burned-in
venue logo, mic tags and end credits — not a synthetic fixture.

Design decisions (authored, `artifacts/llm/download_zhengqinwen.decisions.json`):
card “奥运会是我的梦想没错” → keyword “但是我还有其他的梦想” → three-row list
“每一年我们都有比赛 / 每一年都有更大的困难 / 更大的山坡” → card “停滞在一个位置” →
keyword “没有办法向前进步”. Five elements, 51 % of runtime covered, all in one consistent
right-hand column beside the silhouette.

Deliverable: `delivery/06_v2_real_footage_16x9/final.mp4` (9.7 MB, 25.0 s, 1920×920,
H.264 + AAC, captions burned in), with `motion_ir.json`, captions (SRT/ASS/JSON),
`storyboard.png`, `before_after.mp4`, `qa_metrics.json`, `run_summary.json`.

## 3. Measured quality

QA was run on the rendered MP4, not on the plan. Renders: `artifacts/v2/final_zq/` and a
second full-quality render `artifacts/v2/final_zq_2/`.

| check | measured | threshold |
|---|---|---|
| face occlusion | 0 px | 0 |
| person occlusion | 0.0 % of frame, 0.0 % of person | ≤0.2 % / ≤0.5 % |
| burned-in text collision | 0 px over protected regions | 0 |
| legibility (WCAG) | min 6.43 : 1 | ≥4.5 |
| layout margins | 48 px (2.5 % of width) | ≥2.5 % |
| concurrent collisions | 0 | 0 |
| timing, appearance vs anchor | 0 frames on all 5 elements | ≤1 frame (40 ms) |
| ink on the anchor frame | first ink frame = window frame 0, all 5 elements (8.4k–25.7k ink px) | ≤1 frame |
| displayed text | OCR matches IR copy, no truncation | exact |
| determinism | 625/625 frames, mean \|Δ\| = 0.0 | ≤0.05/255 |

Evidence: `artifacts/v2/final_zq/qa/qa_metrics.json`,
`artifacts/p2/qa/final_zq/qa_metrics.json`, `artifacts/v2/final_zq/qa/contact_sheet.png`.

Caption timing is character-level: `align.py` scores **P50 2.75 / P90 7.56 ms** on 147
known unit boundaries from TTS fixtures, against **P90 ~200 ms** for the previous
word-timestamp pipeline; the shuffled-envelope control scores 98.9 ms, so the metric can
fail. On real speech: 0 monotonicity violations, median syllable 150 ms, 97.6 % of
predicted onsets within 50 ms of a prominent envelope peak
(`artifacts/p2/align/`).

## 4. What changed in v2 (why the old output looked pasted-on)

v1 placed a translucent panel in a face-safe box. On the reference clip that put a
grey slab across the speaker's hair and left the clean wall unused, because layout only
knew the face rectangle. v2:

1. **Sees the silhouette.** Apple Vision person masks (5 Hz) are unioned per element
   window and reduced to band edges, so the free column starts where the *person* ends at
   the height where the type will sit, not where her face box ends.
2. **Composes instead of boxing.** A directional scrim, multiplied by a vertical falloff
   and by a per-row silhouette mask, fades out exactly where she begins. Type is anchored
   to the outer edge where the scrim is strongest. Cards/bands remain available; the
   solver picks by measurement, not by template.
3. **Types properly.** Noto Sans SC Bold/Medium at a scale derived from the canvas
   (8.2 % display), word-aware wrapping (jieba) with forbidden particle line-starts, and
   balanced two-line breaks — 我们/结束/比赛 are never split.
4. **Moves like a cut, not a slideshow.** Ink exists on the anchor frame (pre-roll
   −0.08 s), the settle runs 0.34–0.42 s, list rows land on their own words, numbers
   count with `out_expo`, exits are 0.18 s.
5. **Refuses to lie.** No measured space ⇒ element dropped, with the candidate table and
   reason in `layout_audit`. Text is never truncated; a fit that cannot show the whole
   string is rejected.

## 5. Verification philosophy

Two rules ran the project:

* **A number must come from the artifact being sold.** Timing is measured by decoding
  frames and finding the first changed/inked frame; occlusion by intersecting the overlay
  footprint with the person mask; contrast on the rendered glyph pixels against the frame
  underneath them; determinism by decoding two renders.
* **Every check must be able to fail.** `tests/test_verify2.py` includes negative controls
  (a box deliberately painted over the face must FAIL, then repair must move it and PASS);
  the alignment metric has a shuffled-envelope control; perception has a cache-hit and a
  determinism test; the renderer has a double-render test.

QA found real bugs during integration and they were fixed, not explained away: 2-frame
late appearance, a 27 px margin, silent text truncation in list rows, word-breaking line
wraps, and a scrim that tinted a shoulder.

## 6. Constraints and honest limitations

1. **Timing truth is synthetic.** Real speech has no human-labelled corpus in this repo.
   The 7 ms P90 is TTS-unit truth; on real clips we demonstrate monotonicity, plausible
   syllable durations and envelope-peak tracking only. Do not quote 7 ms as human accuracy.
2. **Apple Vision is macOS-only.** Without it perception degrades to face boxes (OpenCV
   fallback) and layout becomes conservative.
3. **Body pose fails on head-and-shoulders crops** (0/120 frames on the wide fixture);
   masks still carry the silhouette. Hands are sparse (14/125 frames on the reference),
   so they act as a "never cover" guard rather than as a layout input.
4. **OCR is unreliable for Chinese** (conf ≈0.30, boxes under-cover long lines); burned-in
   text detection is a heuristic, not semantics. The reference clip's bottom-centre text
   is a constant set banner, not subtitles.
5. **Aesthetics are not machine-verified.** Contrast, timing, margins, coverage and
   determinism are. "Does it look designed" is a human/agent judgement — look at the
   storyboard. The reference run was reviewed frame by frame.
6. **No invented data.** `barchart`/`compare` render numbers supplied by the director.
   Number animation uses only digits present in the transcript or decisions.
7. **One aspect per run.** The canvas equals the source geometry; there is no automatic
   reframing (cropping a 16:9 interview to 9:16 loses resolution and changes composition —
   deliberately out of scope).
8. **The deterministic director is conservative and occasionally splits phrases at
   acoustic pauses that are not word boundaries.** The intended path is agent-authored
   `decisions.json`; the fallback is a safety net, documented as such.

## 7. How to reproduce

```bash
cd "Semantic Motion" && export PYTHONPATH=src
.venv/bin/python -m pytest tests/ -q                                  # 228+ tests
.venv/bin/python -m semotion.pipeline run \
    --video inputs/zhengqinwen_25s.mp4 --out out/ref \
    --anchors artifacts/dl/anchors_llm.json \
    --decisions artifacts/llm/download_zhengqinwen.decisions.json \
    --transcript artifacts/dl/run/transcript.json
.venv/bin/python -m semotion.verify2 \
    --ir out/ref/motion_ir.json --source inputs/zhengqinwen_25s.mp4 \
    --rendered out/ref/final.mp4 --overlay-only out/ref/final_overlay.mp4 \
    --masks out/ref/perception.json --out-dir out/ref/qa
```

Runtime on the reference machine (M4, 16 GB): perception 18.7 s cold / 0.05 s cached,
alignment 0.1 s, design 2 s, full-quality render 15 s. No network, no paid API.

## 8. Next steps that would actually raise the ceiling

1. A human-labelled real-speech corpus (even 100 manually marked word onsets) to replace
   the synthetic timing claim.
2. Scene-change-aware layout for multi-shot clips (perception computes shots but design2
   currently uses a per-window union, which is conservative across cuts).
3. An explicit anchor "value" model (numbers, comparisons, causal claims) trained on
   editor decisions, instead of the current rule/agent hybrid.
4. Optional 9:16 reframing with a subject-aware camera path, if the user accepts the
   resolution trade-off.
