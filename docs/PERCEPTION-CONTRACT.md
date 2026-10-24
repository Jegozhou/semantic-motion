# Perception contract — `semotion.perception/1` (v2 stack)

Status: **frozen for implementation**. Owner: perception module (`src/semotion/perception.py`).
Consumers: `design2` (layout), `verify2` (QA), `render2` (mask-aware compositing).

The goal of this layer is to let the layout engine know *what the frame already
contains* — the person's silhouette, face, hands/gesture area, and any burned-in
text/logo — so an overlay is never pasted over a human. Face boxes alone are not
enough: hair, shoulders and raised hands are the regions that make an overlay look
pasted on.

## Producer

```bash
.venv/bin/python -m semotion.perception --video INPUT.mp4 \
    --out artifacts/perc/INPUT.json --mask-dir artifacts/perc/INPUT_masks [--fps 5] [--force]
```

* Deterministic: same input bytes + same options ⇒ identical JSON (sorted keys, no
  wall-clock, no random). Masks are PNG; their hashes go in the JSON.
* Cached by `sha256(video) + options`; `--force` re-runs.
* Must run offline. Apple Vision (pyobjc) is the reference backend; fall back to
  OpenCV HOG/YuNet face only if Vision is unavailable (then `backend` says so and
  `capabilities.person_mask=false`).
* Target runtime ≤ 90 s for a 25 s 1080p clip on M4.

## Document

```jsonc
{
  "schema": "semotion.perception/1",
  "ir_compatible": "semotion.vision/1",        // also emit legacy fields
  "source": {"path": "...", "sha256": "...", "width": 1920, "height": 920,
             "fps": 25.0, "n_frames": 625, "duration_s": 25.0},
  "backend": {"name": "apple-vision", "os": "macOS 26.2", "requests": ["face","pose","segmentation","ocr"]},
  "capabilities": {"person_mask": true, "body_pose": true, "hands": true,
                   "existing_text": true, "scene_cuts": false},
  "sample_hz": 5.0,
  "frames": [                                   // sampled at sample_hz, t = frame/fps
    {"frame": 0, "t_s": 0.0,
     "person_bbox": [820, 90, 700, 830],        // pixel xywh, null if no person
     "mask": "INPUT_masks/0000.png",            // 8-bit L, 1/4 linear scale
     "mask_scale": 4.0,
     "mask_coverage": 0.42,                     // fraction of frame covered
     "pose": {"neck": [x, y], "left_shoulder": [x, y], "right_shoulder": [x, y],
              "left_wrist": [x, y], "right_wrist": [x, y], "left_hip": [x, y],
              "right_hip": [x, y], "left_elbow": [x, y], "right_elbow": [x, y],
              "nose": [x, y]},                  // pixel coords, null for low confidence
     "hands": [{"side": "right", "bbox": [x, y, w, h], "conf": 0.81}],
     "face": {"bbox": [x, y, w, h], "conf": 0.93}   // null when absent
    }
  ],
  "regions": {
    "frame_size": [1920, 920],
    "face_union": [x, y, w, h] | null,
    "person_union": [x, y, w, h] | null,
    "hands_union": [x, y, w, h] | null,
    "face_safe": [x, y, w, h] | null,           // face_union grown by safe_margin
    "person_safe": [x, y, w, h] | null,         // person_union grown by safe_margin
    "safe_margin_px": 32,
    "existing_text": [                          // burned-in subtitles / logos / watermarks
      {"t0": 0.0, "t1": 25.0, "bbox": [x, y, w, h], "text": "…", "conf": 0.8,
       "kind": "logo" | "subtitle" | "other"}
    ],
    "quiet_zones": [                            // ranked candidate overlay areas (pixel xywh)
      {"bbox": [x, y, w, h], "kind": "background" | "lower_third" | "side",
       "score": 0.0..1.0,                      // higher = quieter / better for type
       "person_overlap": 0.0,                  // fraction of zone covered by person mask
       "text_overlap": 0.0,                    // fraction covered by existing_text
       "edge_density": 0.0,                    // 0..1, lower is quieter
       "contrast_luma": 0.0}                   // mean |luma - mid| proxy for busyness
    ],
    "person_coverage_curve": [[t_s, coverage_frac], ...],  // sampled, for dynamic checks
    "shots": [{"start_s": 0.0, "end_s": 25.0}]             // optional; reuse vision shots
  }
}
```

Notes:

* Boxes are pixel `[x, y, w, h]` in source coordinates, integer, clamped to the frame.
* `mask` files are **downscaled by `mask_scale`** (default 4) and normalized: 0 = background,
  255 = person. Use bilinear resize when upscaling; mask indexes align to `(x/scale, y/scale)`.
* `quiet_zones` must be derived from the mask, not guesses: a zone qualifies only if
  `person_overlap <= 0.005` and `text_overlap <= 0.01`; zones must be ≥ 0.20 W × 0.10 H.
  Provide at least 3, at most 12, sorted by score descending.
* `existing_text` must include burned-in subtitles: OCR with `VNRecognizeTextRequest`
  (zh-Hans + en-US, `.accurate`), group detections that persist into a time range.
  A detection is a subtitle/logo when it is stable across ≥ 3 consecutive samples,
  or present on ≥ 40 % of the clip.
* Legacy compatibility: also write the `semotion.vision/1` fields (`source`, `faces`,
  `face_tracks`, `shots`, `safe_zones`) so v1 consumers keep working.

## How consumers use it

### `design2` (layout)
1. Candidate zones = `quiet_zones` + a fixed fallback grid (top / middle / bottom bands).
2. Reject any candidate whose intersection with `person_safe` or `existing_text` is > 0.
3. Score = zone score + capacity for the text at the chosen type scale − distance penalty
   from the face (avoid "dangling far corner" composition) − overlap with concurrent elements.
4. If nothing fits at the required type scale, degrade in this order: alternative zone →
   reduce size (down to the module floor) → add scrim (does not fix person overlap) → drop.
5. Record the decision and the candidate table in `motion_ir.layout_audit`.

### `verify2` (QA)
* Re-run person segmentation / load masks, and for every rendered frame measure the alpha
  of overlay pixels inside the person mask. Requirement: `person_pixel_overlap <= 0.002`
  of frame area **and** `<= 0.5 %` of the person's pixels; face overlap must be exactly 0.
* `existing_text` overlap must be 0 for the element's whole window.
* Contrast is measured from actual rendered pixels: glyph pixels vs local background
  (frame without overlay) → WCAG ratio ≥ 4.5.
* "Rendered onset": first frame where overlay alpha inside the element box exceeds
  `0.5 %`, compared to the anchor onset; tolerance = 1 output frame.

## Acceptance for the perception task

* JSON validates against the schema above (a `jsonschema`-free manual validator is fine).
* `python -m semotion.perception` on `inputs/zhengqinwen_25s.mp4` completes ≤ 90 s and
  produces ≥ 3 quiet zones with `person_overlap == 0`.
* A visual debug sheet (`<out>.debug.png`) shows masks + zones + existing text drawn on
  four sampled frames; it must be inspectable with `read_image`.
* Unit tests cover: mask scaling, box clamping, zone rejection under overlap, cache hit,
  deterministic JSON. `pytest tests/test_perception.py -q` green.
* Honest limits recorded in `docs/PERCEPTION.md` (e.g. segmentation quality on motion
  blur, OCR misses small burned-in text).
