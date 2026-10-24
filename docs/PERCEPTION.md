# Perception v2 — person mask, pose, hands and burned-in text

Implementation notes for `semotion.perception/1` (contract:
[`PERCEPTION-CONTRACT.md`](PERCEPTION-CONTRACT.md)); producer:
`src/semotion/perception.py`; evidence: `artifacts/p2/perception/`.

Status: **implemented and measured** on Apple Vision (macOS 26.2, Apple M4,
Python 3.12, pyobjc-framework-Vision 12.2.2). All numbers below were produced
by the commands in [§9](#9-reproduce) and are reproducible offline.

## TL;DR (measured)

| | `inputs/zhengqinwen_25s.mp4` | `tests/fixtures/clips/fx_pd_real_16x9_a/…` |
|---|---|---|
| source | 1920×920, 25 fps, 25.0 s, 625 frames | 1280×720, 30 fps, 24.0 s, 720 frames |
| samples @ 5 Hz | 125 | 120 |
| **cold runtime** | **18.7 s** (budget 90 s; 17.1–19.9 s across runs) | **6.6 s** |
| runtime phases | decode 0.41 / Vision 17.03 / zones 0.03 / masks 0.24 / debug sheet 0.12 s | 0.38 / 5.60 / 0.04 / 0.22 / 0.12 s |
| person-mask coverage | mean 0.1879 (19 %); on hand frames 0.1922, hand boxes 58 % covered | mean 0.2975 |
| frames with zero person mask | 3 — frames 610/615/620, the black tail (24.4–24.8 s) | 0 |
| pose available | 122/125 samples | **0/120** (head-and-shoulders crop, see limits) |
| hands detected | 14/125 samples | 0/120 (hands below frame) |
| burned-in text regions | 11 (sub/side kinds below) | 0 |
| quiet zones | **12, all `person_overlap == 0`**, 8 of them ≥ 400×200 px | **10, all `person_overlap == 0`** |
| validation | OK | OK |
| determinism (2 forced runs) | **PASS** — JSON, 125 mask PNGs and the debug sheet byte-identical (18.7 s / 18.2 s) | **PASS** (6.6 s / 5.5 s) |
| cache hit | 0.061 s, byte-identical JSON/masks/debug | covered by tests |

The clip is the one used by the v1 pipeline: its original subtitle bar was
cropped away (PROGRESS.md §1.10), so what remains at the bottom centre is a
**constant scene banner** (black text on the red Paris-2024 backdrop), not a
changing subtitle. The detector reports it as `other`, and the top-left
“中新视频” logo plus the gold “HOUSE” logo as `logo` — see §3 for the full
table. This is a finding of the perception pass, not a classification error to
hide.

## 1. How it works

```
video ──ffmpeg select+scale──▶ raw RGB frames @1280 long side, 5 Hz (memmap, 1 frame in RAM)
   │
   ├─ per frame: ONE VNImageRequestHandler + ONE performRequests_error_ with
   │     VNDetectFaceRectangles, VNDetectHumanBodyPose,
   │     VNGeneratePersonSegmentation (quality level configurable),
   │     VNRecognizeText (zh-Hans + en-US, accurate), VNDetectHumanHandPose
   │
   ├─ mask 512×384 model buffer ──bilinear──▶ frame grid (480×230 for 1920×920)
   │     values thresholded to 0/255, written as 8-bit L PNG (1/mask_scale)
   │
   ├─ accumulate: clip-wide person union, per-pixel edge density (Sobel),
   │     mean luma, per-sample face/pose/hands/OCR
   │
   ├─ group OCR over time ──▶ existing_text regions (subtitle / logo / other)
   ├─ quiet-zone solver on [union person mask, text mask, edge grid, luma]
   ├─ legacy layer: faces, face_tracks, shots (v1 cut rule), safe_zones
   │
   └─ out: perception.json + masks/NNNN.png + <out>.debug.png  (cached)
```

**One pass per frame.** All five requests share one handler/image so the pixels
are copied into Vision once. Measured on the 25 s clip (first 30 samples,
1280×614 decode):

| request | mean ms/frame |
|---|---|
| face rectangles | 8.7 |
| body pose | 8.6 |
| person segmentation (balanced) | 12.8 |
| **text recognition (accurate, zh+en)** | **119.7** |
| hand pose | 5.1 |
| all five in one `performRequests` call | **127.9** |

OCR dominates (94 % of Vision time). At 5 Hz that is ~16 s for 125 samples,
which is why the whole 25 s clip fits in ~19 s. The first frame of a process
also pays model warm-up (~0.5 s here with segmentation already warm); a second
`analyze()` call in the same process is faster (the test suite reuses it).

### 1.1 Geometry — the details that silently misplace boxes

* Vision observation boxes/joints are **normalized with a bottom-left origin**.
  Every value is converted exactly once to pixel `[x, y, w, h]`, top-left
  origin, then clamped to the frame.
* The segmentation request does **not** return a frame-sized mask. The buffer
  size is model-native and depends on the quality level: balanced gives
  **512×384** for a 1920×920 frame and **384×512** for a 1080×1920 portrait.
  It is a plain resize of the whole frame (verified by compositing the mask back
  over the source: hair, shoulders and raised hand align; see
  `artifacts/p2/perception/mask_check.png`), so it is resampled to the frame
  grid before any geometry is derived. Do not assume `mask.shape == (h/4, w/4)`
  — always upscale by the frame aspect, as documented below.
* `nose` is Vision's `head_joint`: the 19-joint body model has no nose joint.
  This is recorded in `backend.notes`.
* Hand `chirality` is often `VNChiralityUnknown`; when it is unknown the side is
  inferred from the nearest confident pose wrist, otherwise it is `"unknown"`.
  On the test clip Vision reports `left` for all 14 hand observations (the
  vendor classifier's own label in image space).

### 1.2 Segmentation quality level (measured choice)

`VNGeneratePersonSegmentationRequest` quality levels in pyobjc are
**Accurate = 0, Balanced = 1, Fast = 2**. Measured on frame 0 of the test clip:

| level | mask buffer | cost/frame | coverage | IoU vs accurate |
|---|---|---|---|---|
| accurate | 2016×1512 | 193 ms | 0.194 | 1.00 |
| balanced (default) | 512×384 | 12 ms | 0.195 | **0.99** (0.19 % of frame differs) |
| fast | 256×192 | 906 ms* | 0.233 (over-segments) | — |

\*fast is slow on the first call; it is not used. Balanced is the default
because after the mandatory 4× downscale the two masks are visually identical
while accurate is ~16× more expensive. `--seg-quality accurate` is available
when a render needs the extra edge fidelity.

### 1.3 Sampling and cache

* Sample rate 5 Hz (`--fps`), realised as `step = ceil(fps / sample_hz)` source
  frames, so the effective rate is always frame-aligned (25 fps → every 5th
  frame; 30 fps → every 6th). `frame` in the JSON is the **source** frame index
  and `t_s = frame / fps`.
* Cache key = `sha256(video)` + a canonical JSON of every option and tuning
  constant (`Options.as_dict()`), stored under
  `artifacts/cache/perception/<key>/`. A hit copies masks + debug sheet into the
  requested locations and writes the same JSON (0.061 s vs 19.1 s measured).
  `--force` bypasses the read but still refreshes the cache. Measured cold 18.7 s vs hit 0.054 s on the 25 s clip.
* Decoding writes one temporary raw file under `artifacts/cache/tmp/` and reads
  it through a read-only memmap, so a long clip never holds more than one frame
  in RAM and the small system disk is untouched.

## 2. Quiet-zone solver

`quiet_zones()` is a deterministic, integral-image solver over the mask grid.

**Candidate generation.** Ten box sizes on a 2 %-of-frame position grid, including
both edges so the frame borders are always reachable: (0.46, 0.34), (0.36, 0.24),
(0.28, 0.16), (0.22, 0.11), (0.22, 0.24), (0.28, 0.34), (0.46, 0.16),
(0.36, 0.11), (0.26, 0.20), (0.34, 0.14) × (W, H). Sizes below the contract
floor (0.20 W × 0.10 H) are never generated.

**Hard gates** (a candidate is rejected, not down-ranked):

1. `person_overlap ≤ 0.005` on the clip-wide **union** person mask;
2. `text_overlap ≤ 0.01` on the padded existing-text raster;
3. the box keeps an 8 px buffer from every person pixel (dilated union mask is
   used as a hard gate; reported `person_overlap` is measured on the *undilated*
   mask, so a qualified zone always reports exactly 0.0);
4. no near-duplicate of an already accepted zone (IoU ≥ 0.45 or ≥ 50 % of the
   smaller box covered).

**Score** (higher = quieter; all components in `score_parts`):

```
score = 0.52·quiet + 0.22·capacity + 0.12·placement + 0.14·clear
quiet     = 1 − min(1, edge_density / 0.18)          (Sobel mean, 0..1)
capacity  = min(1, sqrt(area / (0.30W · 0.26H)))      (big enough for a card)
placement = 1 − |dist(centre, face_centre)/diag − 0.30| / 0.42
clear     = 1 if the box misses the person_safe rectangle entirely, else 0
```

* `edge_density` is the mean of `(|Sobel_x| + |Sobel_y|) / 2040` over the
  box, averaged over all samples (0 ≈ flat wall, 0.18+ ≈ busy foliage/books).
* `contrast_luma` is reported as the contract's literal quantity: mean
  `|luma − 0.5|` over the box, averaged over samples (informational).
* `clear` exists because `design2` rejects any candidate whose intersection
  with the coarse `person_safe` **rectangle** is > 0. On the test clip the
  person union is `[524, 152, 892, 768]` and its 32 px margin covers the top
  centre, so boxes there score lower even though their pixel overlap is 0.

**Output.** 3–12 boxes (0–12 when a person fills the frame — honesty beats
invented safe space; the contract's floor is enforced by `validate()` only when
the backend reports a person mask). Measured on the 25 s clip:

* 12/12 zones with `person_overlap == 0` and `text_overlap == 0`;
* 8 of 12 are at least 400×200 px (the practical card size): `Z1 422×221`,
  `Z3`, `Z5`, `Z6`, `Z7`, `Z8`, `Z9`, `Z11`;
* best zone `[1482, 288, 422, 221]` score 0.907 (`side`, right of the face,
  under the gold logo, edge 0.017);
* largest zone `[684, 18, 691, 101]` score 0.885 (`background`, top centre).

`kind` is positional: `side` when the centre is outside the middle 40 % of the
width and the box is ≥ 0.22 H tall; `lower_third` when the centre is below 0.60 H;
`background` otherwise.

How consumers should use it: take `quiet_zones` first, reject anything
intersecting `person_safe`/`existing_text` (the contract's rule 2 — zones are
already mask-clean but the layout engine must still apply its own safety rule),
then fall back to the fixed grid. Per-shot zone lists are **not** provided:
zones are clip-global and therefore conservative for multi-shot clips.

## 3. Existing burned-in text

OCR runs on every sample (`VNRecognizeTextRequest`, recognition level
*accurate*, languages `zh-Hans`, `en-US`, automatic language detection).

**Grouping.** Detections are linked into lineages by geometry, because a real
subtitle changes its words while staying in place:

* IoU ≥ 0.25 with the lineage's last box → link;
* same baseline, similar height (0.5–2×), close and compact (x gap ≤ 0.5×height,
  union width ≤ 1.5×the wider box) → link (a line split by OCR);
* lines survive a 1-sample dropout (`TEXT_MAX_GAP_SAMPLES = 2`).

A lineage is reported only when it is stable across **≥ 3 consecutive samples**
or present on **≥ 40 % of the clip** (and has ≥ 3 observations). Same-element
boxes seen in disjoint ranges are merged *before* classification when their
IoU ≥ 0.2 and the time gap ≤ 1 s (OCR dropouts), so persistence and text
similarity are measured over the whole appearance. Adjacent-but-different
elements are never unioned (that would invent an exclusion area between a mic
tag and a banner).

**Classification** (documented heuristic, not a learned classifier):

* `subtitle` — lower-third, horizontally centred box (centre y ≥ 0.72 H,
  height 1.5–12 % H, centre x 0.22–0.78 W) whose **words change** over time:
  mean character similarity to the representative text < 0.50.
* `logo` — text anchored to a frame edge/corner (or the bottom band) and
  persistent ≥ 50 % of the clip; or a constant, off-centre lower-third tag
  (microphone logo / watermark).
* `other` — constant **centred** lower-third text (a scene banner/chyron in the
  shot), end credits, and anything else persistent.

Measured on `zhengqinwen_25s.mp4` (all boxes are pixel `[x, y, w, h]`, top-left
origin; `conf` is the mean OCR confidence):

| kind | time range | bbox | conf | text |
|---|---|---|---|---|
| logo | 0.00–24.40 | [90, 55, 267, 87] | 0.32 | `视频` (中新视频 logo, top-left) |
| logo | 0.00–24.40 | [1115, 762, 105, 158] | 0.32 | `CGTA` → CGTN mic tag |
| other | 0.00–13.00 | [885, 842, 214, 73] | 0.35 | `走的巴黎纪念` (constant scene banner) |
| logo | 0.00–24.40 | [708, 865, 159, 44] | 0.79 | `新京报` mic tag |
| logo | 0.60–24.40 | [1394, 90, 524, 169] | 1.00 | `HOUSE` gold logo, top-right |
| other | 3.60–6.20 | [1410, 0, 507, 72] | 0.41 | `АІ ПЕСІ` (top line of the gold logo) |
| other | 17.40–21.60 | [1410, 0, 508, 76] | 0.30 | `НІЛЬС` (same top line, OCR dropout) |
| other | 19.20–23.80 | [896, 851, 203, 60] | 0.32 | `走的巴黎纪` (banner again after an OCR gap) |
| other | 19.80–24.20 | [123, 679, 234, 50] | 0.50 | `迟瀚宇邢翀` end credit |
| other | 19.80–24.20 | [123, 748, 237, 56] | 0.43 | `制作 刘世炯` end credit |
| other | 19.80–24.20 | [123, 812, 411, 51] | 0.85 | `版权所有，未经许可请勿转载` |

`fx_pd_real_16x9_a` has no burned-in text (one spurious OCR detection in one
sample, dropped by the stability gate), so `existing_text == []`.

**Text quality.** Locations are reliable; **content often is not.** The
Chinese banner/logo lines are stylized and small in the source, and the
recognised strings vary between samples at confidence ≈ 0.30 (`走的巴黎纪念` /
`徒的巴黎纪：` / `哇的巴黎纪3` …). The reported `text` is the most frequent
fragment, useful for the debug sheet and for human sanity checks, not for
captioning. A long subtitle line is usually only partly covered by the OCR box
(this clip: ~200 px of a ~430 px visible line), so a padding of
`0.006 × max(W, H)` is added when rasterising the text mask and consumers should
still keep the bottom ~8 % of the frame clear.

## 4. Debug sheet

`<out>.debug.png` is a 2×2 sheet of four samples spread over the clip (samples
with a person are preferred so a black tail frame does not waste a cell):
`S1 sample=0 frame=0 t=0.00s person=19.5% face=Y hands=1 zones=12`.

Per cell:

* source frame with the **person mask tinted green** (per-frame mask, 0/255 L,
  upscaled bilinearly);
* **yellow** face box + confidence;
* **cyan** pose skeleton (neck/shoulders/elbows/wrists/hips + wrists labelled);
* **magenta** hand boxes with side + confidence;
* **orange** existing-text boxes, only those present at that sample time, with
  kind + time range (the full table is in the footer);
* **blue** quiet zones: all of them outlined, the top 6 labelled
  `Z# score WxH` (rank, score, pixel size).

The footer carries the legend, the score formula, the gate rule and the full
`existing_text` table (kind, time range, bbox, recognised text).

Examples: `artifacts/p2/perception/zhengqinwen_25s/perception.json.debug.png`,
`artifacts/p2/perception/fx_pd_real_16x9_a/perception.json.debug.png`.
`artifacts/p2/perception/mask_check.png` verifies the mask PNG geometry by
upscaling `masks/0040.png` back over the source frame.

## 5. Determinism and cache (measured)

* JSON is written with `sort_keys=True`; times/coverage are rounded to 6 dp,
  confidences/scores/overlaps to 4 dp. No wall-clock value is inside the
  document except `provenance.created_utc` (metadata, excluded from the
  determinism comparison) and `source.path`.
* Two forced runs on each clip produced **identical normalised JSON**,
  identical mask PNG hashes and identical debug-sheet bytes
  (`determinism.json` in each evidence directory; `verdict: PASS`).
* Cache key covers video sha256 + options + tuning constants (sample rate,
  decode size, mask scale, segmentation quality, OCR languages, zone sizes,
  thresholds, weights, text rules). A cache hit returns byte-identical
  artifacts in 0.061 s (`artifacts/p2/perception/cache_hit.json`).

Per-frame values from Vision are deterministic on the same machine/OS/revision;
`backend.revisions` records the model revisions so a future OS change is
visible in the document.

## 6. Tests

`tests/test_perception.py` — **31 tests, 3.4 s** (`PYTHONPATH=src`). Coverage:

* mask scaling/thresholding, odd/empty masks, 1/4 vs 1/2 scale;
* `grid_bbox` mapping back to pixels, clamping, `grow_box`;
* zone qualification under person/text overlap (re-checked cell-by-cell),
  empty result when the person fills the frame, score/kind/NMS behaviour,
  determinism of two calls;
* text lineage linking, 1-sample dropout bridging, stability gate, subtitle vs
  constant banner vs logo vs credits, empty/short input;
* cache key stability/sensitivity, `SEMOTION_CACHE_ROOT` override;
* `validate()` accepts a minimal document and rejects 10 targeted corruptions;
* integration on a generated 0.8 s clip: full document validates, legacy
  `semotion.vision/1` validation passes, mask PNGs are 8-bit L at 1/4 scale,
  debug sheet written, JSON sorted; cache hit is identical; forced rerun is
  byte-identical; CLI `--validate-only` returns 0.

Full repository suite after this change: **228 passed, 24 subtests passed,
37.2 s** (at the time of writing). The suite is shared with other in-flight
tasks, so the count grows independently; the 31 perception tests are the
relevant ones.

## 7. Interfaces, compatibility and contract deviations

The document carries the full legacy `semotion.vision/1` field set
(`source`, `faces`, `face_tracks`, `shots`, `safe_zones`,
`provenance.scanned_fps`), so v1 consumers keep working; the only difference is
`schema`, which is intentionally `semotion.perception/1`.

Small, deliberate additions beyond the frozen contract (all additive, no
required field changed):

* `regions.quiet_zones[].score_parts` — the four score components, for audit;
* `provenance` — `cache_key`, `mask_failures` (frames whose mask is empty),
  `face_without_mask`, `pose_failures_count`, `hand_samples`,
  `text_detections`, `zone_count`, `shot_detection` (v1 cut rule);
* `source`/frame `mask` paths are **relative to the output JSON** (e.g.
  `masks/0000.png`) as the contract example shows;
* `capabilities.scene_cuts` is `true` because `shots` is populated (by the v1
  pixel-diff rule, not by Vision);
* a fallback backend exists (`opencv-yunet-fallback`): face boxes only,
  `capabilities.person_mask/body_pose/hands/existing_text = false`,
  `quiet_zones = []` rather than guessed zones. It is used only if pyobjc
  Vision cannot be imported; it is not exercised on this machine.

Consumers: `validate(doc)` is a dependency-free manual schema validator
(`python -m semotion.perception --validate-only FILE`).

### 7.1 Field → consumer map (design2 / verify2)

Written from the producer side after integrating with both consumers. Paths in
`frames[].mask` are resolved **relative to the perception JSON's parent**
(`Path(doc_path).parent / frame["mask"]`); `frames[].mask_scale` is authoritative
when upscaling a mask to source pixels (bilinear, person = grey ≥ 128).

| field | design2 (layout) | verify2 (QA) |
|---|---|---|
| `source.{width,height,fps,duration_s}` | canvas + frame mapping | output frame mapping |
| `frames[].{frame,t_s}` | window selection: samples within `t0−0.12 … t1+0.12 s` | brackets a rendered frame with the two nearest samples by time |
| `frames[].person_bbox` | per-window person union (`person_union_for`), falling back to `regions.person_union` | not used directly (mask is finer) |
| `frames[].face.bbox`, `regions.face_union` | per-window face union (`face_union_for`) | `faces` list per bracket; face overlap must be exactly 0 |
| `frames[].hands[].bbox`, `regions.hands_union` | "never cover" guard, only while hands exist in the window (`hands_union_for`) | per-bracket hand rects for the manifest (the person mask remains the gate) |
| `frames[].mask` + `mask_scale` | window silhouette union (`_union_mask`) for quietness/overlap checks | upsample mask to rendered size, then measure overlay alpha inside the person mask (requirement ≤0.002 of frame area and ≤0.5 % of person pixels) |
| `frames[].mask_coverage` | skip/clamp zones on empty frames (black tail) | mask-quality reporting; 0 coverage frames are visible in the audit |
| `regions.existing_text` | filtered through `_is_obstacle` (subtitle/caption always; any line ≤5.5 % H; logo ≤16 % H only in the top/bottom 16 % bands), then the **filtered list is written to `ir.perception.existing_text`** | prefers `ir.perception.existing_text`; falls back to this document only when `capabilities.existing_text` is true |
| `regions.quiet_zones` | candidate boxes + `score` (+ optional `score_parts`); mask re-check remains the consumer's job | context only; not a gate |
| `regions.person_safe` / `face_safe` | coarse rule-2 rejection (bbox intersection), which is why zones also carry `score_parts.clear` | face rule is exact (`face.bbox`), the rectangle is not used |
| `regions.person_coverage_curve` | dynamic sanity checks / debug | audit |
| `regions.shots` | cut guards (overlay must not cross a cut) | cut reporting |
| `capabilities.*` | feature detection: no `person_mask` ⇒ v1 fallback layout | refuses to claim mask-based verdicts unless `person_mask` is true |
| `backend`, `provenance.mask_failures`, `pose_failures_count` | audit/diagnostics | provenance written into the metrics report |
| `faces`, `face_tracks`, `shots`, `safe_zones` (legacy) | v1 fallback path when the perception doc is absent | legacy comparison |

Two producer-side notes that matter to both consumers:

* `existing_text` boxes are **un-padded OCR unions**. Padding exists only inside
  the zone solver's text raster; a consumer that needs margin must add its own.
* `kind` is informational for layout policy (design2 deliberately treats large
  signage such as the "HOUSE" wall logo as dimmable scenery), but *every* entry
  is an exclusion zone at the perception layer. `verify2` follows the IR's
  already-filtered list, so the scrim decision stays in one place.

## 8. Honest limits

* **Designed artwork / synthetic clips** produce `mask_coverage = 0` and
  `person_bbox = null`: Apple's person segmenter only fires on real people. The
  synth fixtures (`fx_a_zh_portrait`) are illustrations, so they get
  edge/contrast-only zones with trivially zero `person_overlap` — treat those
  zones as *unverified* for person occupancy. The real-face fixtures behave as
  expected.
* **Close-up crops have no body pose.** `fx_pd_real_16x9_a` is a
  head-and-shoulders shot and Vision returned no pose observation on
  **120/120** samples (`pose = null`, reported in provenance). The mask still
  gives the silhouette, which is what layout needs; shoulder/wrist geometry is
  simply absent. Do not assume pose exists when `capabilities.body_pose` is true.
* **Hands are sparse and the mask is the safety net.** On the 25 s clip 14/125 samples had a
  ≥ 3-joint, ≥ 0.25-confidence hand. Small/distant hands at the frame edge are
  missed; `hands_union` is then absent/partial. On the detected hand frames the
  person mask covers a mean **58 %** of the padded hand box (30–91 %): the rest
  is the 15 % padding, finger gaps and the microphone the hand holds, which the
  segmenter correctly does **not** classify as person. The hand boxes include a
  vendor `side` label that is frequently `left` (chirality is often unknown).
* **OCR content is unreliable for stylized Chinese text** (confidence ≈ 0.30)
  and the recognized box under-covers long lines; kind classification is a
  positional/text-stability heuristic, not semantics. In particular a constant
  lower-third *scene banner* and a hard-burned subtitle that happens to repeat
  the same words are indistinguishable from OCR alone — both are honest
  exclusion zones, but the `kind` label may be `other` in that case.
* **5 Hz sampling misses sub-200 ms gestures** and any overlay-region motion
  between samples; zones are built from the union of the sampled masks, so a
  hand that appears only between samples can still be inside a "quiet" zone.
  Raise `--fps` for gesture-heavy footage (cost is linear: OCR dominates).
* **A frame filled by a person yields fewer than 3 zones.** The solver reports
  the honest shortfall (0–2) instead of inventing safe space, and `validate()`
  then flags the document because the contract asks for ≥3 — that is a signal
  for the layout engine to use the fixed fallback grid, not a silent pass.
* **Multi-shot clips share one clip-global zone list.** Different framings in
  one clip make the union conservative (fewer zones). Per-shot zoning is future
  work; `design2`'s fixed fallback grid remains the escape hatch.
* **Black/blank tail frames**: the 25 s clip has 625 frames but only ~610 with
  content (3 sampled black frames at 24.4–24.8 s). They are reported as mask
  failures (`provenance.mask_failures`) and counted in the zero-coverage
  metrics; they barely affect zones because the person union ignores them.
* **Performance is OCR-bound and machine-specific.** 19 s for 25 s / 125
  samples on M4; the 90 s budget leaves ~4.7× headroom. `--seg-quality
  accurate` adds ~22 s on this clip; a bigger `--det-max-side` increases OCR
  cost roughly linearly.
* **Apple Vision is macOS-only.** The fallback covers face boxes only; person
  masks, pose and OCR need Apple frameworks. Everything is offline; no network
  call is made.

## 9. Reproduce

```bash
cd "/Volumes/ZHITAI/Semantic Motion"
export PYTHONPATH=src

# 1) production CLI (contract invocation)
.venv/bin/python -m semotion.perception \
    --video inputs/zhengqinwen_25s.mp4 \
    --out artifacts/p2/perception/zhengqinwen_25s/perception.json \
    --mask-dir artifacts/p2/perception/zhengqinwen_25s/masks
.venv/bin/python -m semotion.perception \
    --video tests/fixtures/clips/fx_pd_real_16x9_a/fx_pd_real_16x9_a.mp4 \
    --out artifacts/p2/perception/fx_pd_real_16x9_a/perception.json \
    --mask-dir artifacts/p2/perception/fx_pd_real_16x9_a/masks

# 2) evidence: report + determinism (runs the analysis twice, cache bypassed)
.venv/bin/python scripts/perception_probe.py \
    --video inputs/zhengqinwen_25s.mp4 \
    --outdir artifacts/p2/perception/zhengqinwen_25s --determinism --force
.venv/bin/python scripts/perception_probe.py \
    --video tests/fixtures/clips/fx_pd_real_16x9_a/fx_pd_real_16x9_a.mp4 \
    --outdir artifacts/p2/perception/fx_pd_real_16x9_a --determinism --force

# 3) schema check of any document
.venv/bin/python -m semotion.perception --validate-only \
    artifacts/p2/perception/zhengqinwen_25s/perception.json

# 4) tests
.venv/bin/python -m pytest tests/test_perception.py -q
.venv/bin/python -m pytest tests/ -q
```

Evidence layout (`artifacts/p2/perception/`):

```
timings.json                      per-request Vision timings (first 30 samples)
cache_hit.json                    cold vs cache-hit identity (0.061 s)
mask_check.png                    mask PNG upscaled back over frame 40
zhengqinwen_25s/perception.json   semotion.perception/1 document (125 samples)
zhengqinwen_25s/masks/0000..0124.png
zhengqinwen_25s/perception.json.debug.png
zhengqinwen_25s/probe_report.json runtime, coverage, zones, text table
zhengqinwen_25s/determinism.json  PASS + run1/run2 runtimes
zhengqinwen_25s/_determinism_run2/ second run artifacts
fx_pd_real_16x9_a/…               same layout (120 samples)
```
