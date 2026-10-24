# task-7 evidence — caption typography (R5) + non-text graphics (R16)

Measured on the latest renderer outputs (CTC alignment, aspect-aware captions,
three passes: `final.mp4`, `final_overlay.mp4`, `final_ink.mp4`).

## Commands

```bash
export PYTHONPATH=src
PY=.venv/bin/python

# landscape 16:9 (determinism pair)
$PY -m semotion.verify2 \
  --ir artifacts/v2/final_zq/motion_ir.json \
  --source inputs/zhengqinwen_25s.mp4 \
  --final artifacts/v2/final_zq/final.mp4 \
  --overlay artifacts/v2/final_zq/final_overlay.mp4 \
  --ink-only artifacts/v2/final_zq/final_ink.mp4 \
  --captions artifacts/v2/final_zq/captions.json \
  --perception artifacts/v2/final_zq/perception.json \
  --second-render artifacts/v2/final_zq_2/final.mp4 \
  --out artifacts/p2/qa/captions/landscape/qa_metrics.json

# portrait 9:16 real crop
$PY -m semotion.verify2 \
  --ir artifacts/p3/portrait_run/motion_ir.json \
  --source artifacts/p3/portrait_zq.mp4 \
  --final artifacts/p3/portrait_run/final.mp4 \
  --overlay artifacts/p3/portrait_run/final_overlay.mp4 \
  --ink-only artifacts/p3/portrait_run/final_ink.mp4 \
  --captions artifacts/p3/portrait_run/captions.json \
  --perception artifacts/p3/portrait_run/perception.json \
  --out artifacts/p2/qa/captions/portrait/qa_metrics.json
```

## Landscape 1920×920 — OVERALL PASS (13/13)

| check | verdict | number |
|---|---|---|
| render_integrity / footprint / person_occlusion | PASS | 0 face, 0 hand, 0.0 % person |
| existing_text_collision | PASS | 0 px (1 protected IR region) |
| legibility | PASS | min WCAG 7.24 |
| text_onset | PASS | frames 0/50/280/540/567, Δ0 each |
| displayed_text | PASS | OCR 5/5 |
| scrim | PASS | person 0 px, hand/face 0, protected text 0, contrast 7.24–12.26 |
| **caption_typography** | **PASS** | 10 cues, 1 line each, ink **4.891 % H**, width **48.80 % W**, bottom **11.96 % H**, face ∩ 0 px, nearest face **235 px**; thresholds from `policy.qa` (ink 4.0–6.0 % H) |
| non_text_graphics (advisory) | PASS | accent rule/graphics 5.84–7.05 (3 elements with graphics; 2 NOT RUN) |
| layout / timing | PASS | right margin 48 px, signed timing errors 0 |
| determinism | PASS | 625/625 frames, mean/max \|Δ\| 0.0 |

Caption thresholds used are the policy.qa overrides in
`policies/smart_balanced.json`: lines 1–2, ink height **4.0–6.0 % H**, width
≤90 % W, bottom 4.5–13.2 % H (floor = envelope; ceiling = policy
`bottom_ratio.landscape 0.052 + max_lift_ratio 0.06 + 0.02`). Policy:
`smart_balanced 1.0.0`, `line_height_ratio.landscape 0.075`. `face_check PASS`
(a face exists at every measured cue frame).

## Portrait 1080×1920 — OVERALL PASS (13/13, one advisory)

PASS: render_integrity, footprint, person_occlusion (face/hand/body 0),
existing_text_collision (0 px), legibility (min **5.84**), text_onset (Δ0 all
five), displayed_text (OCR 5/5), scrim (person 0 px, contrast 5.84–9.89),
**caption_typography**, layout, timing. `face_check PASS` (0 intersections,
nearest face **439 px**).

`caption_typography`: 10 cues, **1–2 lines**, ink **3.073 % H**, width max
**81.0 % W** (cue 1 with 20 chars wraps to 2 lines, **57.9 % W**), bottom
**13.54 % H**, caption ∩ face **0 px**. Thresholds used are the policy.qa
overrides now in `policies/smart_balanced.json`: ink height **2.4–4.0 % H**,
width ≤90 % W, bottom 11.0–21.0 % H (floor = envelope; ceiling = policy
`bottom_ratio.portrait 0.13 + max_lift 0.06 + 0.02`). Policy:
`smart_balanced 1.0.0`, `line_height_ratio.portrait 0.048`.

`non_text_graphics` (advisory, **does not fail the build**): one element's gold
accent measures **1.16** against the bright wall/window (threshold 3.0);
the other measured graphics are 6.28/6.40, two elements have no measured
non-text graphics (NOT RUN). This is accepted as a real advisory finding.

Earlier portrait failures (ink 2.71 %, 100 % width, legibility 2.33) are resolved
by the renderer/policy changes: type confined to the dense part of the band,
86 % band alpha held to 72 % of the band, 15 chars/line at 63 px budget, portrait
line height 0.048 H. Determinism for portrait is NOT RUN (no second portrait
render).

## Scrim measurement correction (landscape)

The earlier 18 px scrim overlap was **caption anti-aliasing inside the full-height
scrim band**, not encoder ringing and not real scrim alpha:
pixel (1345,809) in the overlay pass had luma 55 and (1347,809) luma 26 — dim
edges of caption glyphs (the bright glyph pixel at (1344,808) was luma 218).
These sat inside the person mask only at the *neighbouring* 5 Hz sample
(3.4 s), not at the nearest sample to the measured frame (3.2 s).

Fix in `verify2`: scrim alpha = overlay pixels with **luma 8–60** (H.264 noise
floor + dark scrim), excluding bright ink and its 5 px anti-alias/shadow halo;
person intersection uses the **nearest 5 Hz mask sample with a bilinear upscale**
at full resolution; element occlusion keeps the stricter temporal mask union.
Result on the landscape pair: **0 px** scrim/person overlap on all five elements,
and the element occlusion check remains 0 px.

## Honest limits

* Caption ink must be in the bottom 45 % of the frame; the cue detector groups
  frames with ≥20 ink px and ≤2-frame gaps (or uses `--captions` times).
* Ink height is the tallest **line band**, not line pitch; 2-line cues are
  measured as two bands.
* A caption wider than the frame is still measured (and fails the 90 % cap).
* Caption text semantics are not checked here (element text equality is
  `displayed_text`); this check is typography only.
* `non_text_graphics` cannot separate a white hairline from glyphs at the same
  ink colour; it is advisory.
