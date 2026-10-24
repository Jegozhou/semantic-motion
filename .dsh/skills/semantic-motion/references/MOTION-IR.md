# Motion IR (`semotion.motion-ir/2`) — editing reference

The IR is the editable project. Every visual is one entry in `elements`; the renderer
recomputes nothing except the pixels, so what you see is what the numbers say.

```jsonc
{
  "schema": "semotion.motion-ir/2",
  "ir_version": "2.1.0",
  "canvas": {"width": 1920, "height": 920, "fps": 25.0},
  "theme": {"accent": "#EBC898", "ink": "#F7F8FA", "scrim": "#070A10"},
  "semantic": {"language": "zh", "anchors": [ ... ]},
  "perception": {"person_union": [x,y,w,h], "existing_text": [ ... ]},
  "elements": [{
    "id": "el_1_c8961792",
    "type": "card",                    // keyword | card | list | stat | barchart | compare | quote
    "anchor_refs": ["a00"],
    "window": {"start_s": 0.0, "end_s": 2.02, "start_frame": 0, "end_frame": 50,
               "anchor_start_s": 0.0, "anchor_end_s": 2.02},
    "box": [1542, 325, 330, 194],      // final resting box of the text block (pixels)
    "backdrop": {"kind": "scrim_right", "box": [1434, 0, 486, 920], "alpha": 0.86,
                 "profile": [[y, x], ...]},   // per-row silhouette edge (renderer clips to it)
    "content": {"title": "...", "body": "", "kicker": "", "emphasis": []},
    "fit": {"kind": "card", "width": 330, "height": 194, "font_size_px": 53,
            "min_font_px": 30, "title_px": 53, "body_px": 28,
            "lines": ["奥运会是", "我的梦想没错"], "line_pitch": 70, "rule": true},
    "motion": {"enter": {"style": "wipe", "dur_s": 0.42, "ease": "out_expo", "offset_y": 12},
               "exit": {"dur_s": 0.18, "ease": "in_quad", "offset_y": -6}},
    "layout": {"treatment": "scrim_right", "zone": "col_right",
               "candidates": [ {...}, ... ]},   // why this placement won
    "provenance": {"decision_id": "a00", "reason": "..."}
  }]
}
```

## Safe edits

| goal | edit |
|---|---|
| change copy | `content.text` / `content.title` / `content.rows[i].text`; then re-render |
| move | `box[0..1]` (keep the size); the layout solver is not re-run on render |
| larger/smaller | `fit.title_px` / `fit.display_px` **and** `box`/`fit.width/height` consistently |
| timing | `window.start_s/end_s`; frames are recomputed on render |
| motion | `motion.enter.dur_s`, `motion.enter.ease` (`out_expo|out_quint|out_cubic|standard`), `offset_y` |
| backdrop | `backdrop.kind` = `scrim_right|scrim_left|band_top|band_bottom|card`, `alpha` |

Do not change `id`, `schema`, or `backdrop.profile` by hand — the profile is the measured
silhouette; if you move the backdrop, recompute it by re-running the pipeline.

## Re-render

```bash
.venv/bin/python -m semotion.render2 --ir out/talk/motion_ir.json \
    --source talk.mp4 --out out/talk/v2.mp4
```

## Invariants the renderer/QA assume

* `box` is inside the canvas with ≥2.5 % margin.
* `box` does not intersect the person union or protected burned-in text in its window.
* `window.start_s` equals the anchor onset (QA allows ±1 frame).
* text drawn equals `content` (no ellipsis unless the content ends with one).
