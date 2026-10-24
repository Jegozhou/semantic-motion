# plan.json reference

`<video>.smproj/plan.json` is the single source of truth for a project's graphics. It is written
by the director (you), compiled by the engine into `build/timeline.json` (absolute times + pixel
boxes), and snapshotted on every build (`plans/plan.NNN.json`) so revisions can be diffed/undone.

```jsonc
{
  "version": 1,
  "style": "balanced",            // balanced (default, 智能均衡) | minimal | bold
  "accent": "auto",               // auto (picked against footage colours) | amber | mint | sky | coral | violet | "#RRGGBB"
  "platform": "auto",             // auto | generic | youtube | bilibili | douyin | tiktok | reels | shorts | xiaohongshu | wechat
  "font": "auto",                 // optional family name; default CJK-first stack (PingFang SC / Noto Sans SC …)
  "safe_area": [0.05,0.07,0.95,0.93], // optional override (x0,y0,x1,y1 fractions); default from platform
  "style_overrides": {},          // optional deep-merge into the style tokens (see engine/semantic_motion/style.py)
  "corrections": {"百老会": "百老汇"}, // ASR fixes: used for anchor matching AND applied to the brief
  "captions": {"enabled": false},  // optional word-synced captions (skipped automatically if subtitles are burned in)
  "elements": [ … ]
}
```

## Element envelope (all types)

| field | meaning |
|---|---|
| `id` | short unique name (`"stat_users"`); used in reports, `sm preview --elements`, revisions |
| `type` | `keyword stat list compare chart flow quote title lower_third callout custom camera captions` |
| `at` | **anchor** where the graphic *lands* (is fully readable) — see Anchors |
| `until` | how long it stays — see Timing |
| `content` | type-specific fields (`reference/elements.md`). Flat fields (`"text"` at top level) are accepted |
| `placement` | `{"region": …, "size": …, "align": …, "x": …, "y": …, "box": […]}` — see Placement |
| `backing` | `auto` (default) \| `none` \| `glass` (frosted plate) \| `plate` (solid dark) |
| `importance` | 1–3 (default 2). Used when two graphics compete for the same moment: lower one is trimmed/delayed/dropped |
| `motion` | `{"lead": 0.15, "enter": 0.46, "exit": 0.30, "reveal": "word"\|"spoken"\|"whole", "energy": 0.5}` |
| `enabled` | `false` keeps the element in the plan but skips it |
| `across_cuts` | `true` lets a graphic survive a hard scene cut (default: ends at the cut) |
| `no_snap` | `true` disables snapping the exit into a nearby pause |
| `intent` | free text: why this graphic exists (kept for revisions and the report) |

## Anchors

An anchor names a moment in the speech. The landing frame is the **onset of the first syllable** of
the phrase (word timestamps are refined against the audio onset envelope; typical error ≈ 1–2 frames).

| syntax | meaning |
|---|---|
| `"3:两千多张"` | phrase in sentence 3 (sentence ids from `brief.md` / `sm transcript`) — **preferred** |
| `"两千多张"` | first occurrence anywhere (ambiguous: avoid when the phrase repeats) |
| `"3:两千多张@end"` | end of the phrase (`@start` default, `@mid`) |
| `"3:两千多张+0.2"` / `"-0.15"` suffix | offset in seconds |
| `"s3"` | start of sentence 3 (`"s3@end"` its end) |
| `"w0042"` | a word id (`sm transcript --words`) |
| `"12.40s"` / `12.4` | absolute time (avoid — not semantic) |
| `{"s":3, "phrase":"张", "n":2}` | dict form; `n` = n-th occurrence |

Matching is normalised (case, punctuation, full/half width, simplified/traditional when opencc is
installed), fuzzy (reports `ANCHOR_FUZZY` with the matched text), honours `corrections`, and may
spill one sentence either way if the ASR split sentences differently. `ANCHOR_NOT_FOUND` is an error:
copy the phrase from `brief.md` exactly as spoken (including ASR misspellings, or add a correction).

Item anchors: list `items[]`, flow `steps[]`, chart `data[]`, compare `left/right` accept their own
`"at"`; each item then appears exactly when it is spoken. If `at` of the element is omitted, the first
item anchor is used.

## Timing

- Entrance starts `land_frac × enter` (≈0.19 s) **before** the anchor so it is readable on the word.
  `motion.lead` moves the landing earlier (anticipation, e.g. 0.1–0.25 s for big reveals).
- `until`:
  - `"auto"` (default): to the end of the anchor's sentence (and all item anchors), then a 0.4 s tail.
    Keywords anchored on a phrase hold through their clause, not a whole long sentence.
  - `"sentence"`: end of the sentence. `"phrase"`: end of the anchored phrase.
  - an anchor (`"5:折扣@end"` is implied: `until` anchors default to the phrase **end**).
  - `{"hold": 2.5}`: seconds after landing.
  - `"next"`: until the next primary graphic replaces it (capped by the type's max hold).
- Minimum visible time = reading time (CJK 6.5 chars/s, Latin 3.2 words/s + 0.7 s); a short phrase
  never flashes. Each type has a max hold (keyword 4.8 s … list/flow 14 s); exceeding it explicitly
  gives `LONG_HOLD`.
- Exits snap into a nearby pause when possible (graphics leave while the speaker breathes).
- Shot cuts: entrances within 0.25 s after a cut are aligned to the cut; at a **scene** cut a graphic
  ends (unless `across_cuts`). Jump cuts inside the same setup (same framing) do not end graphics.
- Conflicts: only one primary graphic at a time (`density.max_primary`), max 2 visible in total.
  The less important one is trimmed, delayed (≤0.6 s) or dropped (`CONFLICT_*` in the report).

## Placement

The layout solver builds a cost field for each element's lifetime from per-frame face, head, body,
hands, burned-in text and the platform safe area. Face/head/burned text are **hard** obstacles.
It prefers negative space on the speaker's open side, keeps consistent sides between consecutive
graphics, and tests several widths/aspect variants (e.g. flows go vertical in narrow space).

| field | values |
|---|---|
| `region` | `auto` (default by type) · `left` · `right` · `top` · `bottom` · `center` · `top_left` · `top_right` · `bottom_left` · `bottom_right` · `beside_face` · `above_head` · `any` |
| `size` | `xs` `s` `m` (default) `l` `xl` |
| `align` | `left` `center` `right` (default: left, centered in portrait center column) |
| `x`, `y` | preferred centre (0–1) — a soft preference, still face-safe |
| `box` | `[x, y, w]` fractions — hard override (top-left + width). Avoid; `BOX_ON_FACE` warns |

## Camera

```json
{"id": "cam1", "type": "camera", "move": "punch_in", "at": "4:最重要", "until": "sentence", "zoom": 1.10}
```
`move`: `punch_in` (instant cut-zoom, classic jump-cut emphasis), `push_in` (slow push over the span),
`ease_in` (0.5 s ease to zoom), `reset`. `focus`: `"face"` (default, tracked median) or `[x, y]`.
Moves never straddle a scene cut; moves shorter than 0.8 s are skipped (`CAMERA_TOO_SHORT`).
Moves never crop burned-in subtitles or name tags: the zoom centre is shifted (or the zoom reduced) to keep them
and the face in frame (`CAMERA_ADJUSTED`), or the move is skipped (`CAMERA_CROPS_TEXT`).
Graphics are composited after the camera move, and the layout uses the zoomed body masks.
Use at most one camera move per ~15 s; they are strong.

## Captions

`"captions": {"enabled": true, "highlight": "spoken", "backing": "none", "size_mul": 1.0, "force": false}`
— word-synced captions with the current word highlighted. Skipped with `CAPTIONS_SKIPPED` when the
video already has burned-in subtitles unless `force`.
