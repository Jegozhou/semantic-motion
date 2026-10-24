# Design system — Semantic Motion v2

This is the "why" behind the look. The renderer implements it; the numbers are in
`theme2.py` and `design2.py`.

## 1. Intent

A talking-head video is a face and a voice. Motion graphics must read as **art
direction applied to the footage**, not as stickers laid on top. Three rules drive
every choice:

1. **Type first.** The graphic is a typographic composition. Panels, icons and rules
   exist only to make type legible in that particular frame.
2. **The speaker wins.** Nothing is ever placed over the face, hair, hands or the
   burned-in credits. If there is no honest space, the element is dropped.
3. **Restraint.** Most sentences get nothing. 0–4 elements per 10 s, ≤58 % runtime
   covered, one idea per element.

## 2. Visual language

### Colour
One accent is sampled from the footage itself: saturated mid-tone pixels are
histogrammed by hue, smoothed, and a warm hue is preferred when it is close to the
peak (brand-friendly, flattering on skin). Saturation and value are clamped so the
accent never looks neon. Ink is a warm white (`#F7F8FA`); scrim is a very dark
blue-black (`#070A10`). The result: the graphic belongs to the shot.

### Treatments
Spatial composition is chosen from what the frame allows, in this order:

| treatment | when | look |
|---|---|---|
| `scrim_right` / `scrim_left` | a side column ≥19 % of width beside the silhouette | directional gradient that fades out along the person's per-row silhouette; type anchored to the outer edge where the gradient is strongest |
| `band_top` | headroom ≥11.5 % of height | horizontal gradient from the top edge |
| `band_bottom` | person ends above the caption band | gradient from the bottom |
| `card` | a measured quiet zone big enough for the type | blurred glass panel, 20 px radius, 1 px hairline |
| `lower_third` | nothing else fits | last resort over clothing; never over face/hands |

The scrim is **not a rectangle**. It is a 2-D gradient: the horizontal ramp is
multiplied by a soft vertical bell, and then multiplied by a per-row mask derived
from the person silhouette, so it fades to zero exactly where the person begins.
That is why a moving shoulder never gets tinted and why the edge never looks pasted.

### Typography
* Noto Sans SC (Bold display, Medium body) + Inter for Latin/numerals.
* Scale relative to `reference_height = min(H, 0.62 W)`: display 8.2 %, title
  6.6 %, body 3.1 %, meta 2.2 %, stat 15.8 %. A 1920×920 frame gets ~75 px display.
* Two lines for headlines; three only if the phrase is long. Words are the wrap
  unit (jieba), so lines never break inside 我们/结束/比赛. Particles (的/了/着/和…)
  may not open a line.
* Accent rule is 3 px, 1.15× the display size, drawn with a 0.3 s scale-x wipe.

### Motion
Short, eased, purposeful, once:

* enter: 0.42 s `out_expo` rise of 12 px + wipe/fade per line, staggered 75 ms;
* ink is already measurable on the anchor frame (pre-roll −0.08 s), so the word and
  the graphic land together;
* list rows reveal on their own word onset (no fixed stagger);
* numbers count with `out_expo` over 0.6 s, thousands separators preserved;
* exit: 0.18 s fade + 6 px up;
* the scrim appears with a 0.34 s opacity settle (lighting, not a panel sliding in).

Nothing bounces, spins, or loops. "Hold" is static by design.

## 3. Layout solver

1. **Per-window silhouette.** For each element's time window, the person masks are
   unioned (Apple Vision, 5 Hz) and reduced to band edges; the scrim column starts
   at the widest edge inside the type band plus the safe margin. A full-clip union
   would be needlessly conservative.
2. **Candidates** are generated from band edges, headroom, bottom gap and measured
   quiet zones. Each candidate carries its available type area, backdrop geometry
   and a quietness score (edge density of the region across sampled frames).
3. **Hard constraints** reject a candidate: intersects the person union, intersects
   a protected burned-in region (caption/credit/small logo), intersects a still
   visible element, does not fit the canvas, exceeds the density budget.
4. **Scoring** = 0.40 × quietness + treatment preference for the element type +
   column consistency with the previous element + type-scale fidelity + distance
   from the face − torso penalty.
5. If nothing fits, the element is dropped with a reason in `layout_audit` — never
   squeezed to an unreadable size.

## 4. What this is not

It is not a template library. There are eight element types, but their geometry,
type scale, colour, timing and motion parameters are all derived per frame from the
footage and the words. The same IR run on another face produces another composition.
