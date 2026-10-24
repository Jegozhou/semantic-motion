# Design guide — the "balanced" (智能均衡) house style

The goal is the look of a careful human motion designer: modern, calm, legible, and always in
service of what is being said. When in doubt, leave it out.

## 1. Directing: what earns a graphic
Read the whole brief first and find the **spine** of the talk (the 3–6 ideas a viewer should remember).
Graphics go on the spine, not on every sentence.

| what is said | good form | avoid |
|---|---|---|
| A number that matters ("两千多张", "40%") | `stat` landing on the number | stats for incidental numbers (dates in passing, "one of them") |
| An enumeration ("第一…第二…", "three things") | `list` with one `at` per item | a list shown before items are spoken |
| A process / sequence / cause→effect | `flow` with item anchors | arrows between unrelated words |
| A contrast (但是/而/however; before/after) | `compare` or two keywords on opposite sides | compare when only one side is spoken |
| A trend or ranking with several numbers | `chart` | charts with made-up or unspoken data |
| A thesis / definition / key term | `keyword` (2–8 CJK chars, 1–4 words) | full sentences as keywords — paraphrase walls |
| A memorable line | `quote` with `reveal: spoken` | quotes longer than ~20 CJK chars / 14 words |
| A structural turn ("接下来说第二点") | `title` | titles every 20 s |
| The speaker's identity | `lower_third`, once, early | if a name tag is already burned in |
| A thing in the frame | `callout` at `hand`/`face`/[x,y] | callouts on nothing |
| A concrete idea with its own shape (timeline, multiplier, progress, map) | `custom` | custom when a standard element says it |
| Emphasis / energy shift | camera `punch_in` (one per ~15 s) | constant zooms; zooms crossing cuts |

**Never invent facts.** On-screen numbers and names must be what was said (after `corrections`).
If the speaker is vague ("很多人"), don't put a number on screen.

## 2. Density and rhythm (balanced)
- One primary graphic at a time; at most two things on screen (primary + lower third/callout).
- 4–9 graphics per minute of speech; ≥ 35–40 % of the runtime with no graphics at all.
- Alternate weight: big moments (stat, list, chart) separated by light ones (keyword) or by nothing.
- Let a graphic breathe: it arrives on the word, holds while the idea is being discussed, and leaves
  in a pause. Don't stack graphics back-to-back with no gap unless they are one idea in steps.
- Openings: the first 2–3 s are usually better clean (the face establishes trust) unless the hook is a number.
- Endings: one closing keyword or nothing; avoid a graphic covering the final expression.

## 3. Timing feel
- Land on the onset of the key syllable (the engine does this); for big reveals use `motion.lead: 0.1–0.2`
  so the graphic is already settling as the word hits.
- Item-by-item builds (`list`, `flow`, `chart`) should follow the speech exactly — that is what makes
  graphics feel "listened to".
- Exits snap into pauses. If a graphic disappears mid-word, set `until` to a later phrase.
- Spoken reveal (`reveal: spoken`) is for lines the speaker is saying *now*; for labels use `word`.

## 4. Space and the person
- The face, head and hair are hard obstacles; hands and body are soft (allowed, discouraged in 16:9).
- 16:9: graphics live in the negative space on the speaker's open side, top-aligned with the eyes
  or the chest; consecutive graphics keep the same side and left edge (the solver does this).
- 9:16: the chest band under the chin and the band above the head are the usual homes; keep within
  the platform safe area (the right rail and bottom caption area on Douyin/TikTok/Reels are reserved).
  Tight close-ups can leave no room: prefer compact keywords over cards, and let the engine clip a
  graphic to the shot that has room (`CLIPPED_TO_SHOT`).
- Burned-in subtitles/logos are hard obstacles; don't add captions on top of them.

## 5. Typography and colour
- One accent colour (auto-picked against the footage's dominant hues), white text, soft shadows,
  frosted-glass cards (dark, 50 % plate + blur) only when the background needs it.
- Hierarchy: one hero element per graphic (the number, the keyword); labels are 35–45 % of hero size.
- Highlight one key part per graphic in the accent — not everything.
- CJK-first font stack (PingFang SC / Noto Sans SC / Source Han Sans) with Latin fallback.

## 6. Styles
- `balanced` (default): as above.
- `minimal`: smaller type, slower softer motion, 3–6 per minute, ≥ 55 % clean. For interviews, serious topics.
- `bold`: bigger type, snappier motion with slight overshoot, 8–14 per minute. For short-form hype content.
  Users ask for these in words ("更克制一点", "more punchy") — see `revision-guide.md`.

## 7. Self-check before building (motion critic)
For each graphic ask: Is it on the spine? Does the text match the speech? Is it the lightest form
that works? Does it land on the word and leave in a pause? Is the face clear? Would the video be
worse without it? If you cannot answer yes, cut it.
