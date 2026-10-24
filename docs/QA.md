# QA v2 — measured verification of the rendered MP4

This document defines what `src/semotion/verify2.py` measures, the thresholds it
uses and why, what is provable from the delivered artifacts versus what remains a
human judgement, and the honest limits of each check. It also documents
`src/semotion/repair.py`, the deterministic repair planner/applier.

Scope: **the delivered video**, not the plan. Every number below is computed from
decoded frames of the final MP4, the source MP4, the overlay-only pass, the
renderer's ink-only pass, the Motion IR and the perception mask document.

---

## 1. Run it

```bash
# arbitrary render (aliases shown in the lead's shorthand: --final/--overlay/--perception/--out)
PYTHONPATH=src .venv/bin/python -m semotion.verify2 \
  --ir motion_ir.json \
  --source src.mp4 \
  --final final.mp4 \
  --overlay final_overlay.mp4 \
  --ink-only final_ink.mp4 \
  --perception perc.json \
  --out metrics2.json

# full-quality determinism (two renders of the same IR)
PYTHONPATH=src .venv/bin/python -m semotion.verify2 \
  --ir motion_ir.json --source src.mp4 \
  --final run1/final.mp4 --overlay run1/final_overlay.mp4 \
  --ink-only run1/final_ink.mp4 --perception perc.json \
  --second-render run2/final.mp4 \
  --out artifacts/qa/run1

# tests (negative controls included)
PYTHONPATH=src .venv/bin/python -m pytest tests/test_verify2.py -q

# deterministic repair from the QA metrics
PYTHONPATH=src .venv/bin/python -m semotion.repair \
  --ir motion_ir.json --metrics artifacts/qa/run1/qa_metrics.json \
  --masks perc.json --out-dir artifacts/repair
```

`--out` may be a directory (writes `qa_metrics.json` + `contact_sheet.png`) or a
`*.json` file (writes just the metrics there; the contact sheet goes next to it).
Exit code is `0` for `PASS`, `1` otherwise.

Inputs that are optional and what happens without them:

| input | used by | without it |
|---|---|---|
| `--overlay` | footprint, text onset heuristic, displayed text, scrim, person/face/text checks | footprint is a diff vs source (noisier); `displayed_text` → NOT RUN |
| `--ink-only` | `text_onset` | heuristic diagnostic only, verdict NOT RUN |
| `--perception` | person/face/hand masks, quiet zones, burned-in text fallback | person/hand checks → NOT RUN (never a vacuous pass) |
| `--second-render` | determinism | NOT RUN |
| `--ir` `perception.existing_text` | protected burned-in text (preferred over the mask document) | falls back to the mask document's `regions.existing_text` |

---

## 2. Checks, definitions and thresholds

Verdicts are `PASS`, `FAIL` or `NOT RUN`. `NOT RUN` never counts as a pass; the
overall verdict is `FAIL` if any check is `FAIL`, otherwise `PASS` (with the
`NOT RUN` list reported). Every check carries its raw numbers.

### 2.1 `render_integrity`
Compares geometry, fps, frame count, duration and decoded audio statistics of the
final file with the source. Thresholds: identical `width×height`, `|fps| ≤ 1e-3`,
`|frames| ≤ 1`, `|duration| ≤ 1 frame + 10 ms`, audio RMS delta ≤ 0.5 dB (when
both files have audio). **Why:** every pixel check is meaningless if the final
file is not the same geometry/timebase as the source.

### 2.2 `footprint`
Per-pixel overlay mask. With `--overlay`: `max(R,G,B) > 4` on the overlay-only
pass (black background, so dark scrims are included; codec noise is 0–2). Without
it: `max channel |final − source| > T`, where `T = clamp(12, 1.5 × p99.9(noise) + 4, 80)`
and the noise floor is measured on frames outside every element window. Per
element, only pixels inside the IR `layout.box` / v2 element `box` are attributed.
Reported: union pixels, max per-frame pixels, visible frames, footprint bbox,
threshold and noise floor. **Why:** this is the only way to know what the motion
layer actually put on the picture; it is also the input for timing and occlusion.

### 2.3 `person_occlusion`
Footprint ∩ person mask from the perception contract (`semotion.perception/1`:
`frames[].mask` relative to the JSON file, `mask_scale`, bilinear upscale). The
overlay footprint is max-pooled down to the mask's native grid (conservative: a
single full-res pixel counts). Per frame, per element:

* **face overlap must be 0 px** (face mask = face bounding box; conservative);
* **hand overlap must be 0 px** (hand boxes from the contract);
* person overlap ≤ **0.2 % of frame area** and ≤ **0.5 % of the person's pixels**.

Failure is reported with the worst frame, worst element, per-element totals and a
`worst_occlusion_crop.png`. **Why those numbers:** the contract fixes 0.2 %/0.5 %;
they are small enough to catch a card sitting on a shoulder (tens of thousands of
pixels) and tolerant of a 1–2 px anti-aliasing leak at a box edge.

**`lower_third` exception:** for a candidate treatment named `lower_third` the
layout is allowed to sit over clothing. Face and hand overlap must still be 0;
body overlap is reported (`max_person_frame_fraction`, `max_person_fraction`) but
does not fail the check (`policy: lower_third`). Same rule for its scrim.

### 2.4 `existing_text_collision`
Footprint ∩ protected burned-in text must be **0 px** for every frame of the
element window. Protected regions come from
`ir.perception.existing_text` when present — design2 already filters venue
signage, keeps captions/logos — and otherwise from the mask document's
`regions.existing_text`. `0 regions` from an IR-filtered list is a measured pass;
no protected-text source at all is `NOT RUN`. **Why:** covering a burned-in
caption is a hard delivery error; dimming background signage is a design choice,
so the filtered list is the contract.

### 2.5 `legibility` (WCAG 2.x)
Glyph pixels = overlay pixels close to the declared ink colour
(`style.color`, else `ir.theme.ink`, else `#F5F7FA`); if colour matching swallows
most of the overlay, an Otsu luminance split separates text from its background.
Background = rendered pixels around the glyphs (scrim included), so the number is
what a viewer sees. Contrast is computed per 24 px block and the **minimum block
with ≥ 12 glyph pixels** must be ≥ **4.5**. The contrast against the raw source
under the glyphs (`contrast_vs_source`) is also reported — it shows how much the
scrim is doing. A solid block/image element with no ink-coloured pixels is
`NOT RUN`, not a fake fail. **Why 4.5:** WCAG AA for normal text; measuring the
rendered local background (not the declared scrim colour) is what makes it
objective.

### 2.6 `text_onset` (strict, first word)
Uses the renderer's **ink-only pass** (`final_ink.mp4`: glyphs on black, no scrim,
no captions): in the element box, count pixels with **luma > 12**; first frame in
`[window_start − 2, window_end)` with **count > 32** is `text_onset_frame`.
`text_onset_frame > window_start + 1` → **FAIL** (1-frame tolerance). Early ink
(`< window_start − 1`) is a warning, not a failure. Reported per element:
`text_onset_frame`, `ink_onset_delta_frames`, `ink_pixels_at_onset`,
`max_ink_pixels`, gate, method. **Why:** the product promise is "the first word is
visible on the anchor frame"; footprint timing passes as soon as the scrim fades
in, so the ink pass is the only deterministic way to test the promise. Without
`--ink-only` the overlay-pass colour/luminance heuristic is reported but the
verdict is `NOT RUN`.

### 2.7 `displayed_text` (text equality / truncation)
OCR of the overlay-only pass with Apple Vision (accurate level, zh-Hans + en-US,
traditional→simplified folding). Reference string is the IR text:
`content.text`, else `content.title`, else `content.rows[i].text` joined. The
element is sampled at **50 / 70 / 85 %** of its window and the best frame must
contain the whole normalised IR text as a **subsequence** (whitespace/punctuation
ignored); an ellipsis (`…`, `...`) is an explicit truncation failure. Multiple
samples are required because v2 list rows reveal progressively — mid-hold alone
legitimately shows a prefix. Reported: OCR text, confidence, matched chars/recall,
frame, per-frame attempts, problems. **Why:** OCR is the only way to prove the
*delivered glyphs* are the IR words; it can misread, so `ocr_text` is kept for
inspection and an honest FAIL is a measured discrepancy, not proof of a bug.

### 2.8 `scrim` (backdrop alpha regions)
For elements whose `backdrop.kind` starts with `scrim` or `band`:

* rendered footprint inside the projected backdrop box ∩ person mask must be
  **0 px** (hard); hands/face must be 0 px; for `lower_third`, body overlap is
  allowed and reported;
* the element text's local contrast, measured on the rendered frame (legibility),
  must be ≥ **4.5** — not the scrim's declared colour;
* protected burned-in text under the scrim must be **0 px**.

No scrim/band elements → `NOT RUN` with `elements: 0`. **Why:** a full-height
scrim can silently cover a person or a caption even when the text box is safe.

### 2.9 `layout`
* element box fully on canvas and ≥ **32 px** safe margin from every edge
  (`regions.safe_margin_px` when supplied by the contract);
* rendered **footprint** must keep **≥ 2.5 % of canvas width/height** from every
  edge (lead rule; catches raster overflow beyond the declared box);
* concurrently visible elements: rendered footprints may overlap at most
  **0.5 % of the smaller box** per frame (denominator = IR box area); a static
  box-intersection check > 1 % is also reported;
* font-size floor: `fit.min_font_px` is **authoritative** when present (FAIL below
  it). Otherwise a v1 `style.font_size_px` keeps the legacy module floor (hard),
  and v2 `fit` typography gets a **warning** at `0.045 × min(canvas W,H)` — the
  module's multi-size typography (card body 26 px at 920 px high) is legitimate
  and a 59 px hard floor would be wrong there.

Reported per element: box, margins, on-canvas, footprint bbox/margins/verdict,
font size, floor and floor source, problems.

### 2.10 `timing`
Footprint appearance = first frame whose in-box footprint exceeds
`max(0.5 % of box area, 3 × pre-window noise)`. Disappearance = last visible + 1.
Compared with the IR window start/end and the anchor onset; **tolerance 1 frame**;
signed errors in frames and ms are reported per element. This measures *overlay
presence* (scrim included); the stricter word-level promise is `text_onset`.

### 2.11 `determinism`
Decodes both renders and compares per-frame pixel differences with **explicit**
tolerances: mean |Δ| ≤ **0.05** and max |Δ| ≤ **32** (0–255 scale), plus the raw
mean/max/worst frame. Byte-identical files are never required. Prior full-quality
x264 double runs decode with mean |Δ| 3.5e-05 (and here 0.0); the max tolerance
allows isolated codec/AA noise but not a different picture.

---

## 3. Metrics JSON and contact sheet

`qa_metrics.json` (`semotion.verify2/1`) contains `checks` (one object per check
with `verdict` and raw numbers), `verdicts`, `failing_elements` (per element, per
check), `elements` (footprint, occlusion, text, legibility, text_onset,
displayed_text, scrim, layout, timing), `inputs` (paths + IR hash), `thresholds`,
`notes`, `tool_versions`, and `contact_sheet` / `worst_crop` paths.

`contact_sheet.png` shows sampled frames (window start/middle/end per element and
the worst occlusion frame) with: element box in green/red by verdict, person-mask
contour in green, face box in cyan, and verdict labels. `worst_occlusion_crop.png`
magnifies the worst person-overlap frame. Both are meant to be opened and looked
at; the numbers do not replace the eyes.

---

## 4. Deterministic repair (`repair.py`)

Input: IR + metrics + perception document. Output: `motion_ir.repaired.json` and
`repairs.json` (`semotion.repair/1`) with `noop`, per-repair `checks_failed`,
`action`, `reason`, `before`, `after`, `evidence`, plus `dropped`, `unfixable`,
`summary`, `validation_errors`.

Ladder, fixed order per element:

1. **retime** to the anchor onset when measured appearance/exit is outside the
   1-frame tolerance (window length preserved, hold recomputed);
2. **relocate** to the best quiet zone with zero person/face/text overlap; shrink
   towards the module floor only if needed; v2 scrim/band backdrops are
   recomputed for the new box and must be person-free (opposite side is tried);
3. **scrim**: raise the background alpha to ≥ 0.88 when measured contrast < 4.5
   and the placement is otherwise safe (a scrim never fixes person overlap);
4. **drop** the element when no safe placement exists — recorded with the reason.

Guarantees: deterministic (no clock/randomness, sorted candidates), semantics
preserved (text/type/anchors unchanged; only box, font size down to the floor,
background alpha, timing), and **idempotent**: a second run on the repaired IR —
even with the same stale metrics — produces no changes (`noop: true`, identical
IR bytes). v2 IRs (`element.box`, no tracks) are supported; the v1-only
`ir.validate` seal check is skipped for that shape and noted.

---

## 5. Objective vs subjective

**Provable from the artifacts (the numbers above):** geometry/timebase/audio
identity; what pixels changed; person/face/hand pixel counts under the mask;
burned-in-text overlap; rendered glyph contrast per block; first ink frame;
delivered text vs IR text (OCR); scrim coverage of the person; margins/collision;
frame count/timing in frames; decoded-frame reproducibility.

**Not provable / still subjective:** aesthetic quality and composition; whether
the pacing feels right; mask segmentation quality itself (verify2 consumes the
producer's mask and says so); whether an OCR misread is a font/rendering problem
(inspect `ocr_text`); motion smoothness/easing taste; caption typography and
readability for a specific audience; "the overlay looks pasted on" beyond the
measured silhouette/contrast proxies; audio aesthetics.

---

## 6. Honest limits

1. **Mask quality is external.** `apple-vision` masks are good on this footage,
   but hair/soft edges and motion blur can be wrong. verify2 does not validate
   segmentation; it reports `backend`, grid and sample rate with the numbers.
   Person overlap is computed on the mask's native grid and max-pooled
   (conservative, not sub-pixel exact).
2. **Face "mask" is the face bounding box** — conservative: it may include
   background beside the face, so the strict 0 px rule can reject a box that
   merely touches the face box's background corner.
3. **Contrast is a proxy for readability.** WCAG ratio on mean block colours
   does not model font weight, stroke width, motion or viewer distance. Glyph
   detection is colour-based; unusual ink colours/scrims can misclassify (a
   solid block is NOT RUN, not PASS). Decorations near text can lower the block
   minimum; that is reported as an offending element for inspection.
4. **`text_onset` needs `final_ink.mp4`.** Without it the heuristic is diagnostic
   only. Ink threshold (luma > 12, > 32 px) is tuned for the delivered passes; a
   different renderer/profile must be re-tuned and documented.
5. **`displayed_text` depends on OCR.** Vision is deterministic for fixed input
   bytes but can misread low-contrast/small text; multiple frames and t2s folding
   reduce false negatives, and `ocr_text` is always included. It checks element
   text, not burned-in captions.
6. **Footprint diff without an overlay pass** is sensitive to camera motion,
   re-encoding and dark scrims over dark backgrounds (adaptive threshold clamped
   at 80). Pass `--overlay` for attribution-quality numbers.
7. **Timing footprint includes the scrim**, so it passes as soon as the backdrop
   appears; `text_onset` exists precisely to cover the word-level promise.
8. **Determinism is decoded-pixel identity within tolerance**, not byte identity;
   it depends on encoder/OS/build. Two renders must be supplied.
9. **Existing-text protection depends on the perception OCR/filter.** Small
   captions can be missed; venue signage is intentionally excluded by design2's
   filtered list. 0 protected regions is reported as a pass with that source.
10. **No subjective gate.** The contact sheet exists so a human can reject a
    PASS that looks bad. QA v2 cannot judge taste.
11. **No network, no model calls, no wall-clock dependence** in the metrics JSON;
    the same inputs produce the same JSON (paths/hashes included).

---

## 7. Evidence (this cycle)

| run | verdict | key numbers | evidence |
|---|---|---|---|
| demo, deliberately bad overlay over the face | **FAIL** | face overlap **6,719,704 px**, 9.34 % of frame / 31.08 % of person, worst frame 61 | `artifacts/p2/qa/qa_bad/qa_metrics.json`, `worst_occlusion_crop.png` |
| same clip after `repair.py` relocation | **PASS** | face 0 px, person 0.0 %, contrast 13.85, text onset frame 30/30 (+0), determinism mean/max 0.0 over 120 frames | `artifacts/p2/qa/qa_fixed/`, `artifacts/p2/qa/repair/repairs.json` |
| v2 draft render (`demo_zq5`) | **PASS** | person 0.0, contrast 5.45, text onset all +0, displayed text 5/5, scrim person 0, layout 48 px margin | `artifacts/p2/qa/zq5/` |
| v2 full-quality pair (`final_zq` / `final_zq_2`) | **PASS** | 11/11 checks; face/hand/person 0; contrast min 7.28; text onset all +0; OCR 5/5; scrim 0 px; determinism 625 frames mean/max 0.0 | `artifacts/p2/qa/final_zq/` |

Demo reproduction: `artifacts/p2/qa/run_demo.sh` (or step-by-step commands in
`artifacts/p2/qa/DEMO-REPORT.md`).
