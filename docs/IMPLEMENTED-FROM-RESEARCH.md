# From research to code — what changed

`docs/BEST-PRACTICES.md` distilled 17 OSS projects and standards into 20 ranked
recommendations. This file tracks which ones actually landed, with the evidence. Status
values: **done** (shipped + tested), **partial**, **deferred** (with reason).

## Implemented

| id | change | where | evidence |
|---|---|---|---|
| R1 | Aspect-aware caption type scale: line height 7.5 % H landscape / 4.2 % H portrait (BBC 9.2) | `render2.CaptionTrack`, `policy.captions.line_height_ratio` | landscape ink measured **45 px = 4.89 % H** (pitch 68.6 px = 7.45 % H); portrait ≈ 55 px font |
| R2 | Aspect-aware caption safe area: 5.2 % H landscape, **13 % H** portrait; obstacle lift capped at +6 % H | `render2.render`, `policy.captions.bottom_ratio` / `max_lift_ratio` | landscape bottom margin 12.07 % H (lifted above burned-in credits, within floor+lift) |
| R3 | Type-aware column feasibility: a side column must hold ≥3.2 display glyphs, not just 19 % of width | `design2.Designer.candidates` | portrait IR no longer accepts a 116 px column for 52 px glyphs |
| R4 | Quietness measured inside the element's window (5 Hz grey frames), not from five clip-wide frames | `design2.Designer._quietness` / `_scene_frames` | same box now scores differently per window; frames decoded once (≈6 MB for 25 s) |
| R6 | Layout audit: every element carries its scored candidate table (chosen + rejected + reason) | `design2.Designer._element` | `motion_ir.json → elements[].layout.candidates` (8 rows for the reference clip) |
| R7 | Decision params are honoured, not silently dropped: `params.enter` → `motion.enter.style`, `params.accent` → per-element palette, with requested vs applied recorded | `design2.Designer._element`, `render2` per-element palette + scale transform | reference IR: `scale_in→scale`, `spring_pop→scale`, `reveal→wipe`, `#F2B233` applied to 4 elements |
| R9 | Enter duration can scale with travel distance (policy envelope) — `motion.enter_s` is now a policy value | `policies/smart_balanced.json` | policy field; per-element auto-tuning deferred |
| R13 | Draft/full quality is explicit in the render manifest and QA can refuse a draft | `render2` manifest `encode.draft` | manifest field already emitted |
| R14/R15 | Caption break rules: word-aware wrapping (jieba), forbidden particle line starts, segment-respecting cues | `theme2.wrap_text`, `captions.py` | 0 mid-word cue starts, 0 particle-led lines on the reference clip |
| R16 | Non-text graphics contrast (WCAG 1.4.11, 3.0) reported separately | `verify2` `non_text_graphics` | PASS (advisory) |
| R17 | Zone consistency across consecutive elements (one column per passage) | `design2.Designer._score` consistency term | reference IR: all 5 elements in `col_right` |
| R18 | Alignment confidence surfaced to the director | `director.build_brief` (`timing_confidence`), CTC `chars[].conf` | candidates carry mean char posterior |
| R20 | Determinism is a first-class gate (two renders, decoded-frame diff) | `verify2` determinism + `final_ink.mp4` | 625/625 frames, mean \|Δ\| = 0.0 |

## Model-directed decisions (the "don't hard-code judgement" part)

| judgement | before | now |
|---|---|---|
| which words earn a graphic | fixed token lists + scores (0.90/0.84/0.78…) in `auto_anchors` | **`director` brief → model/host agent decides**; rules remain a labelled fallback |
| how ideas group (list vs separate cards) | merged by fixed gap rules | model chooses `anchor_ids` groups; validation enforces order/cardinality |
| emphasis / wording / kicker | hard-coded per type | `params.emphasis`/`kicker`/`label` validated against the transcript text |
| pacing / hold | constant 0.45 s | model may set `hold_s` within the policy envelope; policy holds the hard bounds |
| layout | fixed preference table | still deterministic (silhouette + measurements), but **policy-driven** and auditable, with a scored candidate table the model can veto later |

Numbers that used to be constants now live in `policies/smart_balanced.json` with
provenance links, and the resolved policy hash is recorded in the IR.

## Alignment

The energy-DP aligner is no longer the only option. `semotion.forced` implements a real
acoustic CTC alignment (XLSR-53 character CTC for zh; letter CTC for en), and the pipeline
now defaults to `--aligner auto`: use CTC when the model is cached, otherwise the envelope
DP.

* Synthetic TTS truth (n=147): energy DP P90 **7.56 ms** vs CTC **72.81 ms** — but that
  truth is a splice point (an energy jump), so it structurally favours the envelope method.
* Real speech (no human labels): CTC agrees with Whisper word starts at median **50 ms**
  (83.7 % within 100 ms) vs energy **150 ms** (34.7 %); CTC posteriors median 0.966.
* Verdict: CTC recommended for real human speech; envelope DP for missing models, very
  short clips, non-speech and synthetic regression. English CTC is implemented but
  unmeasured.

## Deferred (and why)

| id | change | why not yet |
|---|---|---|
| R5 | caption typography QA gate | implemented by the QA owner; thresholds in `policy.qa`, measured on the rendered overlay pass |
| R8 | director may promote *rejected* candidates | brief already includes context; adding the rejected pool needs a recall benchmark we do not have |
| R10 | Material emphasized-decelerate for large elements | needs a velocity measurement on rendered frames first |
| R11 | keyframe list in the IR (per-property easing) | real refactor; current enter/exit covers the shipped vocabulary |
| R12 | content-hash frame cache | render is 15–20 s for 25 s of video; optimisation without a user-visible win yet |
| R19 | protected regions by OCR kind instead of size heuristics | needs a hand-labelled frame set to prove it is better |

## What we deliberately did not copy

3-D/shaders (no measurable benefit for typographic overlays), particle/spring overshoot
(reads cheap on type and breaks "ink on the anchor frame"), Lottie/AE import (opaque to QA
attribution), browser rendering as the default (dependency + licence weight; kept optional),
virtual camera moves (the product keeps the original framing), TTS-reference alignment
(no confidence output), karaoke word-by-word captions as the default look (different genre).
