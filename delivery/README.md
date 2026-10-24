# Delivery bundle

Reference renders produced by the Semantic Motion pipeline. Each folder contains the
MP4, its Motion IR, captions, metrics and storyboard.

| folder | source | what it proves |
|---|---|---|
| `06_v2_real_footage_16x9/` | Wikimedia Commons interview (CC BY 3.0), real human speech, 1920×920, 25 s | **the v2.1 deliverable**: silhouette-aware scrim column, ink on the anchor frame, 0 px person/face/credit overlap, min contrast 7.24:1, captions at BBC scale (ink 4.89 % H), 13/13 QA PASS, double-render determinism 0.0 |
| `07_v2_portrait_9x16/` | same interview, real 1080×1920 crop | portrait adaptation: band treatment, caption ink 3.07 % H, 2-line cue at 15 chars/line, 12 PASS + 1 advisory (`non_text_graphics` on a bright wall) |
| `01…05_*` | synthetic TTS fixtures and public-domain clips | v1 gates: effect matrix, cut safety, degradation scenarios |
| `demo_*` | earlier v1/Remotion experiments | history — superseded by `06` for visual quality |

`06` is the one to watch. `before_after.mp4` is a side-by-side of the untouched source and
the finished render.
