# Element library

All elements share the envelope in `plan-schema.md` (`id`, `type`, `at`, `until`, `placement`,
`backing`, `importance`, `motion`, `intent`). Only `content` differs. Text supports
`"highlight"`: a substring (or list of substrings) rendered in the accent colour.
Sizes are in `u` = 1 % of the short frame side, so designs scale from 720p to 4K and 9:16.

Choose the **lightest** element that carries the meaning. Keywords and stats are the workhorses;
cards (list/compare/chart/flow/quote) are for moments where structure is the message.

---

## keyword — a phrase that matters
```json
{"type": "keyword", "at": "4:8月17日", "until": "4:之前",
 "content": {"text": "8月17日前", "highlight": "8月17日", "sub": "截止日期", "marker": "underline"},
 "motion": {"reveal": "word"}}
```
- `text` (≤ 8 CJK chars / ≤ 4 words ideal, 2 lines max), `sub` small second line, `highlight`.
- `marker`: `underline` (default when highlighted) · `bar` (accent block behind highlight) · `none`.
- `motion.reveal`: `word` (staggered, default) · `spoken` (each word appears exactly when spoken —
  great for quotes the speaker is saying right now; if the text starts before the anchored phrase the
  element starts with the first spoken word) · `whole`.
- Unbacked by default (clean type with soft shadow); the compiler adds glass automatically when the
  background under the box is bright or busy.

## stat — a number
```json
{"type": "stat", "at": "3:两千多张", "content": {"value": "2,000+", "unit": "张", "label": "百老汇门票", "sub": "免费送出"}}
```
- `value`: number or string; strings are parsed: `"2,000+"`, `"3.5万"`, `"40%"`, `"$1.2B"`, `"3x"` →
  value + prefix/suffix (+ separator / decimals). Explicit `prefix`, `suffix`, `unit`, `decimals`,
  `separator` override.
- Counts up from `from` (default 0) across the spoken number (`count: false` to disable).
- Suffix symbols (`+ % x 倍`) are attached in accent at 62 % size; `unit` follows in white.

## list — enumeration that grows as it is spoken
```json
{"type": "list", "at": "5:Theatre", "until": "7:折扣",
 "content": {"title": "TDF 会员 · 免费", "style": "numbered",
   "items": [{"text": "…", "at": "5:Theatre"}, {"text": "为期 2 个月", "highlight": "2 个月", "at": "6:两个月"}]}}
```
- `style`: `numbered` · `bullet` · `check`. `focus: true` (default) dims earlier items when a new one lands.
- The card grows with each item (no empty card waiting for content). ≤ 4 items, ≤ 14 CJK chars each.

## compare — A vs B / before → after
```json
{"type": "compare", "at": "9:以前",
 "content": {"mode": "vs", "winner": "right",
   "left":  {"title": "以前", "value": "3 天", "points": ["人工审核"], "at": "9:以前"},
   "right": {"title": "现在", "value": "10 分钟", "points": ["自动审核"], "at": "9:现在"}}}
```
- `mode`: `vs` (divider with "VS") · `arrow` (before → after). `winner` side gets the accent.

## chart — trend, ranking, proportion
```json
{"type": "chart", "at": "12:增长", "content": {"kind": "bar", "title": "月活 (万)", "unit": "万",
  "data": [{"label": "2022", "value": 120}, {"label": "2023", "value": 260, "at": "12:二百六十万"}, {"label": "2024", "value": 410, "highlight": true}]}}
```
- `kind`: `bar` · `hbar` (ranking, long labels) · `line` (trend; draws on) · `ring` (single share:
  `{"kind": "ring", "value": 68, "label": "用户满意度", "unit": "%"}`).
- Values animate from 0; each datum may have its own `at`. Keep ≤ 6 data points.

## flow — process, sequence, causality
```json
{"type": "flow", "at": "4:访问", "content": {"steps": [{"text": "访问网站", "at": "4:访问"}, {"text": "了解规则", "at": "4:了解"}, {"text": "参加抽签", "at": "4:抽签"}]}}
```
- Horizontal chips with drawing arrows; the layout solver switches to a vertical stack when the free
  space is tall and narrow. `direction`: `auto` · `horizontal` · `vertical`. ≤ 5 steps.

## quote — a sentence worth keeping
```json
{"type": "quote", "at": "14:真正的", "content": {"text": "真正的护城河是用户的信任", "highlight": "信任", "by": "— 张三"}, "motion": {"reveal": "spoken"}}
```

## title — section / chapter card
```json
{"type": "title", "at": "s6", "content": {"kicker": "PART 2", "text": "为什么会失败", "highlight": "失败", "sub": "三个常见误区"}}
```
- Use only at real structural turns (≤ 1 per ~45 s). Prefers the top of frame / above the head.

## lower_third — who is speaking
```json
{"type": "lower_third", "at": "1:我叫", "until": {"hold": 3.5}, "content": {"name": "Zohran Mamdani", "role": "纽约市长"}}
```
- Once per speaker, early, when they introduce themselves. Skip if the video already has a name tag.

## callout — point at something in frame
```json
{"type": "callout", "at": "7:这块手表", "content": {"text": "Casio A158W", "sub": "¥199", "target": "hand"}}
```
- `target`: `face` · `hand` (tracked median position during the element) · `[x, y]` (0–1 frame coords).
  Draws a target dot + ring, a leader line and a label placed in free space next to the target.

## custom — anything else (procedural animation)
A box of `content.size` = `[w, h]` in `u`, containing primitive **layers** drawn in order. Use for
one-off ideas: a timeline that draws, a multiplier popping, a progress bar, a map-pin drop, a
simple icon-like diagram, a counter riding a line…

```json
{"type": "custom", "at": "6:20 years", "backing": "glass",
 "content": {"size": [48, 18.5], "layers": [
   {"shape": "line", "points": [[7, 12], [41, 12]], "stroke": "text@0.22", "stroke_w": 0.3, "from": "land"},
   {"shape": "line", "points": [[7, 12], [41, 12]], "stroke": "accent", "stroke_w": 0.6,
    "keys": {"trim": [["6:first started", 0], ["6:first started+1.5", 1, "inout"]]}},
   {"shape": "circle", "cy": 12, "r": 1.2, "fill": "accent", "keys": {"cx": [["6:first started", 7], ["6:first started+1.5", 41, "inout"]]}},
   {"shape": "text", "text": "", "count": {"from": 2001, "to": 2021}, "y": 4.4, "size": 3.8, "weight": 800, "fill": "accent", "align": "center",
    "keys": {"x": [["6:first started", 7], ["6:first started+1.5", 41, "inout"]], "progress": [["6:first started", 0], ["6:first started+1.5", 1, "inout"]]}},
   {"shape": "text", "text": "English Wikipedia", "x": 4.6, "y": 14.6, "size": 2.2, "fill": "text2", "from": "6:English Wikipedia"}]}}
```

Layer shapes and their properties (all lengths in `u`, relative to the box's top-left):

| shape | properties |
|---|---|
| `rect`, `rrect` | `x y w h r`, `fill`, `stroke`, `stroke_w`, `glass: true` (frosted) |
| `circle` | `cx cy r`, `fill`, `stroke` |
| `arc` | `cx cy r start sweep` (degrees), `stroke`, `stroke_w`, `track: true` (dim full ring behind) |
| `line`, `polyline`, `arrow` | `points [[x,y],…]`, `stroke`, `stroke_w`, `trim` 0–1 (draw-on), `head` (arrow size) |
| `path` | `d` (SVG path data: M L H V C S Q T Z, abs + rel), `fill` / `stroke`, `trim` |
| `text` | `text`, `x y`, `size`, `weight`, `align` (left/center/right), `valign`, `max_w`, `highlight`, `reveal` (`whole`/`word`), `count: {from, to, decimals, prefix, suffix, separator}` driven by `progress` 0–1 |

Common to all layers: `opacity`, `scale`, `rotate`, `dx`, `dy`, `blur`, and
- `from`: when the layer fades/rises in (default: with the element).
- `keys`: `{"prop": [[time, value, easing], …]}` keyframes for any numeric property
  (`x y w h r cx cy opacity scale rotate trim sweep start dx dy blur progress stroke_w size`).
  Easing (applies to the segment arriving at that key): `out` (default) · `in` · `inout` · `linear` ·
  `spring` · `hold` · or a cubic-bezier `[x1, y1, x2, y2]`.

**Times** (in `from` and keys): a number = seconds after the element starts · `"land"`, `"land+0.3"`,
`"out-0.2"`, `"in"`, `"end"` · `"@12.34"` absolute · **a spoken anchor** like `"6:first started"` or
`"6:first started+1.5"` (resolved to the word onset; `from` anchors are shifted so the layer is
readable on the word). Spoken anchors also extend the element's lifetime to cover them.

**Colours**: `text`, `text2`, `muted`, `accent`, `plate`, `positive`, `negative`, or `#RRGGBB`; append
`@alpha` (`"text@0.22"`). The element as a whole still gets the house entrance/exit
(`content.auto_motion: false` to animate everything yourself).

## camera — punch-in / push-in
See `plan-schema.md` → Camera. Not a graphic: it scales the original footage around the face.

## captions — optional word-synced subtitles
See `plan-schema.md` → Captions. Off by default; never on top of burned-in subtitles.
