# QA codes

`sm validate` runs the compiler + pre-render QA; `sm build` additionally repairs, renders and runs
post-render QA. Reports: `qa/report.md` (human) and `qa/report.json` (machine). Levels: **error**
(must fix — build fails unless repaired), **warn** (look at it), **info** (what the engine did).

## Compile / timing
| code | meaning | fix |
|---|---|---|
| `ANCHOR_NOT_FOUND` (error) | the phrase is not in that sentence | copy the phrase from `brief.md` as spoken; add `corrections` for misheard names |
| `ANCHOR_FUZZY` | matched approximately (shows what) | fine if the match is right; otherwise use a more specific phrase |
| `NO_ANCHOR` (error) | element has no `at` and no item anchors | add `at` |
| `EMPTY_CARD` | card visible > 1.2 s before its first item is spoken | anchor `at` on the first item (auto-repaired) |
| `MAX_HOLD` / `LONG_HOLD` | held to the type's max / explicitly longer | shorten `until` or accept |
| `SHORT_READ` | visible for less than its reading time | shorten text or extend `until` |
| `ENTER_ON_CUT` / `END_AT_CUT` | entrance aligned to / exit at a shot cut | info |
| `TRIM_FOR_NEXT` / `DELAYED` / `CUT_SHORT` / `DROPPED` | conflict resolution between primaries | raise `importance`, move an anchor, or remove one |
| `SPOKEN_START` / `SPOKEN_FALLBACK` | spoken reveal started with the first displayed word / text not found in speech | info |
| `CAMERA_TOO_SHORT` | camera move would be < 0.8 s before a cut; skipped | anchor it earlier inside the shot |
| `BAD_CAMERA` (error) | unknown move | `punch_in`, `push_in`, `ease_in`, `reset` |
| `CAMERA_ADJUSTED` (info) | zoom centre (or zoom amount) moved so burned-in subtitles / name tags and the face stay inside the frame | nothing; set `focus: [x, y]` to override |
| `CAMERA_CROPS_TEXT` | no zoom ≥ 1.02 keeps both the burned-in text and the face in frame (e.g. full-width subtitles in a 9:16 crop); skipped | drop the move (`enabled: false`) or use a graphic instead |
| `CAPTIONS_SKIPPED` | burned-in subtitles detected | `captions.force: true` only if the user insists |

## Layout
| code | meaning | fix |
|---|---|---|
| `NO_SPACE` (error) | no face-safe box for the lifetime at any size | shorter text, smaller size, shorter `until`, another moment, a lighter element (auto-repair tries smaller sizes) |
| `SHRUNK` | placed below normal size | consider shorter text |
| `CLIPPED_TO_SHOT` | no room over the whole span (framing changes at a cut) → shown only in the shot(s) with room | usually fine; otherwise re-anchor inside the roomy shot |
| `AUTO_BACKING` | glass added because the background is bright/busy | info |
| `BOX_ON_FACE` | explicit `placement.box` overlaps the face | remove `box` (auto-repaired) |

## Pre-render QA (per element, per analysis sample)
| code | meaning |
|---|---|
| `FACE_OCCLUDED` (error) | box intersects a detected face box in some sampled frame (camera zoom applied) |
| `HEAD_OCCLUDED` (error) | box covers the head/hair segmentation |
| `TEXT_OCCLUDED` (error) | box covers burned-in text (subtitles, logos, name tags) |
| `OUTSIDE_SAFE` (error) | box leaves the platform safe area |
| `BODY_OVERLAP` | box mostly over the body (warn in 16:9, info in 9:16 where the chest band is normal) |
| `LOW_CONTRAST` | unbacked text over a bright background (auto-repair: glass) |
| `COLLISION` (error) | two boxes overlap in time and space |
| `TOO_DENSE` / `SPARSE` | elements per minute outside the style range |
| `LOW_CLEAN_RATIO` | too little graphics-free time |

## Post-render QA (on the rendered file)
| check | pass condition |
|---|---|
| frame count & duration | equal to the source (no dropped/duplicated frames) — `FRAME_COUNT` error |
| audio | stream copied bit-exact (md5 of decoded PCM) — or, if it had to be re-encoded, correlation ≥ 0.98 and lag ≤ 22 ms (`AUDIO_CHANGED`) |
| A/V offset | same start offset as the source |
| pass-through fidelity | PSNR on frames without graphics/camera vs the source (typically 45–48 dB at CRF 18; < 32 dB is `PASSTHROUGH_DEGRADED`) |
| graphics present | pixel difference inside every element box at its reveal frame (`GRAPHIC_MISSING`) |
| determinism (`--determinism`) | rendering the same frames twice gives identical frame hashes |

`qa/contact_sheet.jpg` is made from the **rendered file** (one tile per element at its fully-built
moment, labelled with anchor and times); `qa/timeline.png` shows speech, pauses, cuts and element
spans. Look at both before delivering.

## Auto-repair (sm build)
Up to 3 rounds: remove explicit boxes that hit the face/text/safe area, shrink `NO_SPACE` elements
one size step (disable at `xs`), add glass for `LOW_CONTRAST`, anchor `EMPTY_CARD` on the first item.
Every repair is written to the report; the plan in `plan.json` is updated and snapshotted.
