# Best practices — what to borrow from OSS engines, forced alignment and broadcast standards

**Status:** research distilled for v2 → v3. Date 2026-10-25.
**Scope:** this document changes nothing; it names the change, the symbol it touches and the
effect we should be able to measure afterwards. Every number below is either quoted from a
primary source (with a link) or computed from this checkout (with the file:line and the
command in `artifacts/research/NOTES-raw.md`).

**Method.** Ten repositories were shallow-cloned at a pinned commit and read at the file level,
not summarised from their READMEs (`/tmp/sm-research/vendor`, commit SHAs in the notes file).
Six standards pages were fetched and stripped to text. Two were client-rendered and had to be
read through a text renderer; two dead ends are recorded honestly in §1.17 and §7. Raw evidence,
including the numbers in §5 that describe *our own* output, is in
[`artifacts/research/NOTES-raw.md`](../artifacts/research/NOTES-raw.md).

A note on what "best practice" means here. This project's differentiator is **restraint +
measurability**, and the honest current state is: 11/11 QA PASS on one 25 s clip
(`reports/FINAL-REPORT-V2.md`), with caption typography explicitly listed as human judgement
(`docs/QA.md:252`). So the recommendations below are biased towards *closing measured gaps in our
own pipeline* rather than importing features. Where an OSS project solves a problem we do not
have, it is in §4 ("do not copy") instead.

---

## 1. Sources

### 1.1 movis — a Python motion-graphics engine (`64a9618`)

**What it is.** A "video editing as code" engine whose selling points are exactly the ones a
deterministic renderer needs: layer transforms with sub-pixel precision, a keyframe/easing
animation engine, nested compositions, 1/2 and 1/4 quality drafts, and a frame cache
([README](https://github.com/rezoo/movis#-main-features)).

**Technique worth borrowing — per-segment easing on a keyframe list.** `Motion.append(keyframe,
value, easing)` stores a *callable per segment*, and evaluation bisects the keyframe list,
normalises time inside the segment, applies **that segment's** easing, then lerps
([`movis/motion.py#L150-L175`](https://github.com/rezoo/movis/blob/main/movis/motion.py)).
Values are `float64` numpy arrays (`transform_to_numpy`), so position/scale are sub-pixel; one
`Motion` per attribute (`movis/transform.py`) means a layer's animation is *data*, not code.

**Technique worth borrowing — a bounded frame cache + draft level.**
`Composition` holds `Cache(size_limit=1024**3)` keyed by `(CacheType, layer name, key_body)` and
invalidates on structural edits
([`movis/layer/composition.py#L65`](https://github.com/rezoo/movis/blob/main/movis/layer/composition.py));
`level=2` renders the whole composition at half resolution for drafts
([`docs/overview.rst#L206`](https://github.com/rezoo/movis/blob/main/docs/overview.rst)).

**What we should do.** (a) Adopt the keyframe model as the IR's motion representation instead of
the current fixed `{enter, exit, hold}` triple — see R11. (b) Our renderer re-composites every
element and backdrop on every pass; the caption tiles are cached
(`render2.CaptionTrack._tile_cache`) but elements are not. Add a content-hash keyed cache with a
size limit — see R12. (c) Make draft resolution first-class rather than a flag on the encoder —
see R13.

### 1.2 clipify — a Claude Code skill where the prompt *is* the product (`5f6b75e`)

**What it is.** A skill that finds clip-worthy moments in a long video and cuts them vertical,
with word-by-word captions ([SKILL.md](https://github.com/louisedesadeleer/clipify/blob/main/SKILL.md)).
Two of its design choices matter more than its code.

**Technique worth borrowing — a *signal list* plus a *candidate schema* handed to the host agent.**
Step 1A does not ask the model "find good moments"; it enumerates the signals (punchlines,
reversals, awkward pauses, self-roast one-liners, audio peaks), fixes the output schema
`[start, end, why-it's-funny, suggested title]`, and fixes the target duration ("aim for 10–25 s")
([SKILL.md#L60-L73](https://github.com/louisedesadeleer/clipify/blob/main/SKILL.md)). Step 1.5
then makes the agent *look at a frame per candidate* and drop it if the subject is off-frame —
"cheap insurance" against audio-only confidence
([SKILL.md#L110-L127](https://github.com/louisedesadeleer/clipify/blob/main/SKILL.md)). That
verify-then-trust loop is the single most transferable idea in this repository, and it is what
our pipeline lacks: no model in our stack has ever seen a frame.

**Technique worth borrowing — motion energy in a mouth ROI instead of face ID.** No face model:
two hand-drawn ROIs, per-frame frame-differencing energy via `tblend=all_mode=difference`, and
"whoever's rectangle moves more is speaking", with **minimum dwell 1.0 s** so interjections merge
into the previous speaker
([SKILL.md#L150-L175](https://github.com/louisedesadeleer/clipify/blob/main/SKILL.md),
[`scripts/analyze.py`](https://github.com/louisedesadeleer/clipify/blob/main/scripts/analyze.py)).
The dead-zone-with-minimum-dwell pattern reappears in §1.4; it is the standard fix for
flickering.

**What we should do.** Import the *contract* shape (enumerated signals + fixed schema + explicit
duration target + a visual verification step) into `llm_director.build_prompt`, and give the
director the ability to add moments, not only style the ones our rules already found — see R7,
R8 and §6.1.

### 1.3 shortcast — an auto-cut/caption product in Swift (`618a507`)

**What it is.** A macOS app that cuts long video into vertical shorts, burns an AI-written *hook*
line, and writes per-platform post copy
([README](https://github.com/mutonby/shortcast#readme)).

**Technique worth borrowing — the hook band geometry.** `VideoOverlayRenderer.swift` centre-aligns
a band at **8.5 % of height** from the top, with hook type at **5.8 % of width**, padding
`0.55 × fontSize`, and a fade
([`Shortcast/Services/VideoOverlayRenderer.swift#L91-L116`](https://github.com/mutonby/shortcast/blob/main/Shortcast/Services/VideoOverlayRenderer.swift)).
Those are the only placement numbers in the repo, and they corroborate the "keep the top ~10 % and
bottom ~13 % of a 9:16 frame free" convention.

**Honest limitation.** There is **no burned-in caption engine** in this repository — no subtitle
typography, no line-breaking, no safe-area constants for captions. Its caption contribution is
platform *copy* rules (`social-content-coach.md`: TikTok hook ≤ 90 chars, 1–2 line body, 3–6
hashtags; IG 2–4 paragraphs, 20–30 hashtags; Shorts title 40–60 chars). I expected caption layout
here and it is not here; the recommendation below uses it only for the hook/top safe area.

**What we should do.** Nothing structural. Cite the 8.5 %/5.8 % pair when defining the portrait
top safe area (R2, §5.4), and adopt the platform copy contracts as *labels* on exported caption
variants if we ever ship per-platform text.

### 1.4 auto-vertical-reframe — subject-aware reframing with a real camera model (`1556b3c`)

**What it is.** YOLO11 segmentation + pose + face + saliency + PySceneDetect, a ranked subject
tracker, and a virtual camera (pan/zoom) with per-preset tuning
([README](https://github.com/KazKozDev/auto-vertical-reframe#readme)).

**Technique worth borrowing — dead zone before smoothing, then critical damping.**
The observation is snapped to the current camera position while inside a dead zone
(**28 px x, 21 px y**, and zoom frozen while `|Δzoom| < 0.045`), then a critically damped spring
(`_smooth_damp`, `omega = 2/smooth_time`, rational `exp` approximation, returns value *and*
velocity) integrates toward it; the filter's alpha is scaled by detection confidence
(`obs_weight = clamp(confidence,0.15,1)`)
([`auto_reframe.py#L1801-L1815`](https://github.com/KazKozDev/auto-vertical-reframe/blob/main/src/verthor/auto_reframe.py),
[`#L1893`](https://github.com/KazKozDev/auto-vertical-reframe/blob/main/src/verthor/auto_reframe.py)).
The `movie` preset goes further: **one static crop per shot**, hard cut on scene or subject change,
centre-lock on title cards and wide shots. Scene cuts come from PySceneDetect's `AdaptiveDetector`
(`--scene-threshold 3.0`, `--min-scene-len 15`).

**What we should do.** We deliberately do **not** move the camera (the original picture is the
product), so the pan/zoom part is "do not copy" (§4). But the *decision* pattern transfers
directly: hysteresis for stable placement, scene/segment-scoped commitment instead of per-element
re-deciding. Our layout already has a consistency term (`design2.Designer._score` adds `+0.10` for
the same zone within 9 s); make the unit a *shot/segment* rather than a 9 s window, and refuse
micro-moves inside a dead zone — see R17, R19.

### 1.5 whisperX — ASR + word-level forced alignment (`5d3678b`)

**What it is.** Whisper transcription with VAD chunking, then wav2vec2 CTC alignment to get
word- and character-level timestamps.

**Technique worth borrowing — confidence from the CTC path, aggregated per word.** Alignment
builds a wildcard column (`max non-blank score per frame`) so unmatched audio cannot derail the
path, then per-character `score` is the path score and the **word score is the mean of its
character scores** ([`whisperx/alignment.py#L316-L402`](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py)).
`return_char_alignments=True` is how CJK gets character timings, which is exactly what we need
for a language with no spaces.

**What we should do.** We already have character alignment, but anchor provenance carries only a
*rule* score, not an *alignment* score. Surface the per-character confidence into the anchor so
timing-sensitive elements (a number, a name) can refuse low-confidence windows, and so QA can say
"this overlay is anchored on 0.61-confidence text" — see R18.

### 1.6 ctc-forced-aligner — CTC segmentation with fixed overlapping windows (`64293cc`)

**What it is.** Wav2Vec2 CTC forced alignment with selectable granularity (sentence/word/char),
built for long audio ([README](https://github.com/MahmoudAshraf97/ctc-forced-aligner#readme)).

**Technique worth borrowing — chunk, don't VAD.** The default is `--window_size 30` s with
`--context_size 2` s of overlap, and every span carries a CTC score; `get_alignments` returns
`(segments, scores, blank_token)` and spans are derived from the starred token sequence
([README options table](https://github.com/MahmoudAshraf97/ctc-forced-aligner#readme)).
The lesson for us: VAD failure is a *silent* failure mode (dropped speech), whereas fixed
overlapping windows trade a little accuracy for a bounded, auditable failure.

**What we should do.** Audit `align.py`/`asr.py` for whether a VAD-missed region can silently
shorten a cue, and prefer an overlap-and-reconcile step over trusting a single VAD boundary. This
is a robustness change, not a feature: measure it as "cue boundaries that disagree with the
overlap pass" (should be 0).

### 1.7 aeneas — DTW forced alignment against synthetic speech (`4d200a0`)

**What it is.** MFCC + DTW alignment of text to audio, where the text is first *spoken* by a TTS
engine to create a reference to align against; multilevel (paragraph → sentence → word)
([README](https://github.com/readbeyond/aeneas#readme)).

**Honest negative result.** aeneas solves the no-model case we already have models for, its
README explicitly says word level is not its design target, and it produces **no confidence** —
the one thing a hybrid system needs. The genuinely useful idea is its *multilevel* strategy:
align coarse first, then refine locally inside each coarse span. If we ever need to re-align a
quote the director rewrote, refine inside the cue rather than re-running global alignment.

**What we should do.** Do not adopt. Use the multilevel idea only if `repair.py` needs to
re-time an edited string (R18, optional).

### 1.8 Remotion — React-based programmatic video (`a64a027`)

**What it is.** A React renderer for video: compositions, `<Sequence>`, `interpolate()`,
`spring()`, and a captions package.

**Technique worth borrowing — "pages" of tokens as a first-class caption model.**
`createTikTokStyleCaptions({captions, combineTokensWithinMilliseconds: 1200})` groups word
tokens into pages and returns token-level `fromMs/toMs` inside each page, so the *page* is the
readable unit and the *token* is the timing unit
([docs](https://www.remotion.dev/docs/captions/create-tiktok-style-captions)). A low combine
window gives word-by-word animation; a high one gives many words per page. Our cue model has the
same shape already (`captions._Cue` with `_Atom`s), which is reassuring; what we lack is the
explicit, single number for "how much text may share a page" — ours is spread across four
constants (`target_cue_chars`, `max_cue_chars`, `merge_gap_s`, `pause_break_s`).

**Technique worth borrowing — clamped interpolation.** `interpolate()` requires
`extrapolateLeft/Right`, i.e. **every** interpolation is explicitly clamped at both ends so a frame
before or after the window cannot overshoot
([docs](https://www.remotion.dev/docs/interpolate)). We do this per-effect (opacity clamps,
offset clamps) rather than once at the transform layer.

**What we should do.** Collapse caption page sizing into one derived budget (R14); centralise
clamping in `render2._element_transform` so no effect can leak outside its box (part of R11).
**Note the licence**: Remotion is free for individuals and ≤ 3-person companies and requires a
paid company licence above that
([Licence](https://github.com/remotion-dev/remotion/blob/main/LICENSE.md)); our own
`docs/REMOTION.md` already records this. That is a business risk, not a technical one, but it is
another reason not to make it the default backend.

### 1.9 Revideo — generator-based programmatic video (`b5de67a`)

**What it is.** A TypeScript framework where animation is written as generators —
`yield* waitFor(0.5)` — and rendered headlessly with `renderVideo()`
([README](https://github.com/redotvideo/revideo#readme)). Note: the repo moved from `revideo-dev`
(404) to `redotvideo`.

**Technique worth borrowing — time comes from the frame index, never the wall clock.** Because
the generator is resumed for a *duration*, not a frame rate, the same script renders identically
at any fps and in any number of parallel workers. Our renderer is already frame-indexed (it
iterates `n_frames`), so this is confirmation rather than a change.

**What we should do.** Nothing. The generator style is pleasant to write and impossible to audit
in a JSON IR; our IR *is* the audit surface. Recorded in §4.

### 1.10 Motion Canvas — generators plus an editor (`7b91435`)

**What it is.** "A TypeScript library that uses generators to program animations"
([README](https://github.com/motion-canvas/motion-canvas#readme)), with a GUI timeline.

**Honest limitation.** I skimmed this one (README + package layout) and it added no technique
beyond §1.9. Its value to us would be the *editor*, which is out of scope for a headless skill.
Listed for completeness, not as a source of change.

### 1.11 Material Design 3 — easing and duration tokens

**What it is.** The token table for transition timing
([Easing & duration](https://m3.material.io/styles/motion/easing-and-duration/tokens-specs)).

**Numbers worth borrowing.** Emphasized decelerate `cubic-bezier(0.05, 0.7, 0.1, 1)`; standard
`cubic-bezier(0.2, 0, 0, 1)`; emphasized accelerate `cubic-bezier(0.3, 0, 0.8, 0.15)`. Durations:
short 50–200 ms, medium **250–400 ms**, long 450–600 ms, extra-long 700–1000 ms; "a card expanding
to full screen uses a long 500 ms duration".

**What we should do.** Our enter is **420 ms `out_expo`** (`policy.motion.enter_s`) for *every*
element, and `theme2.ease_out_expo` is `1 − 2^(−10t)` — **half the move is finished in the first
10 % of the duration**, harder than any Material or Carbon curve (≈ `cubic-bezier(0.16, 1, 0.3, 1)`
against Material's emphasized decelerate `(0.05, 0.7, 0.1, 1)`). Neither choice is wrong alone:
the hard curve is why the ink "lands" on the word, and 420 ms is the top of Material's *medium*
band, defensible for a full-width card. Together, for a 12 px move, they are the slowest possible
pairing — a two-glyph keyword waits 0.42 s to travel 12 px. Make duration a function of travelled
distance and area (Carbon's rule) and use emphasized-decelerate only for large areas — R9, R10.

### 1.12 IBM Carbon — productive vs expressive motion

**What it is.** Two motion styles with named curves, duration tokens, and an explicit
duration-vs-distance rule ([Motion](https://carbondesignsystem.com/elements/motion/overview/)).

**Numbers worth borrowing.** Productive standard `cubic-bezier(0.2, 0, 0.38, 0.9)`, expressive
standard `(0.4, 0.14, 0.3, 1)`, entrance `(0, 0, 0.38, 0.9)` / `(0, 0, 0.3, 1)`, exit
`(0.2, 0, 1, 0.9)` / `(0.4, 0.14, 1, 1)`; durations fast-01 **70 ms**, fast-02 **110 ms**,
moderate-01 **150 ms**, moderate-02 **240 ms**, slow-01 **400 ms**, slow-02 **700 ms**.

**The rule worth quoting in a design review:** "Duration should be dynamic based on the size of
the animation; the larger the change in distance (traveled) or size (scaling) of the element, the
longer the animation takes", and the checklist item "do micro-interactions fall within a static
duration ranging from **90–120 ms**?".

**What we should do.** Carbon's approach is the argument for R9: replace the constant
`enter_s = 0.42` with `clamp(0.16 + travelled_px/(0.9·ref_h), 0.18, 0.42)`. Our current 12 px
offset means *every* element travels the same tiny distance and still takes 420 ms — the slowest
possible pairing of a short move with a long duration, which is exactly what Carbon says not to do.

### 1.13 Apple HIG — Motion

**What it is.** Principles, not tokens
([Motion](https://developer.apple.com/design/human-interface-guidelines/motion)): "Add motion
purposefully…", "Aim for brevity and precision in feedback animations", "In apps, generally avoid
adding motion to UI interactions that occur frequently", "Consider using fades when you need to
relocate an object", "avoid displaying motion at the edges of a person's field of view".

**What we should do.** Two concrete adoptions. (a) "Avoid motion at the edges of the field of
view" is an argument *against* our edge-column scrims animating their gradient: keep the scrim's
appearance a pure opacity settle (we already do, 0.34 s — keep it, and never add a slide). (b)
"Use fades to relocate" justifies our exit being a fade+6 px rather than a slide-out, and argues
for dropping any future "slide to new position" transition between zones in favour of
fade-out/fade-in. No numeric change; a constraint on future work.

### 1.14 BBC Subtitle Guidelines — the only standard that specifies 9:16

**What it is.** The editorial + technical subtitle specification the BBC actually delivers
against, including explicit 9:16 guidance
([Subtitle Guidelines](https://www.bbc.co.uk/accessibility/forproducts/guides/subtitles/)).

**Numbers worth borrowing.**
* Authoring line height: **7–8 % of active video height** for 16:9 / 4:3 / 1:1;
  **3.9–4.5 %** for **9:16** ("4.5 % results from adjusting the 8 % value … by multiplying it by
  9/16").
* Lines: **2 max** landscape, **3 max** for 9:16.
* Line width (online): **68 % of width** for 16:9, **90 %** for 4:3 / 1:1 / 9:16. Character
  equivalence given: 37 chars in 75 % of a 16:9 frame ≈ **25 chars in 90 % of a 9:16 frame**.
* Position: outside the central **90 %** vertically is forbidden (landscape); for 9:16 the
  central **75 %** ("in 9:16 videos it is common to position subtitles a little higher up").
* One complete sentence per subtitle; break at punctuation; an explicit list of **prohibited break
  points** (article+noun, preposition+phrase, conjunction+clause, pronoun+verb, parts of a complex
  verb).

**What we should do.** This is the backbone of §5 and R1/R2/R14/R15. Our captions are
**3.15 % of height** (29 px on 920) — below the 7 % landscape floor — and in 9:16 the same formula
yields **1.2 % of height**, roughly a third of the BBC's 9:16 floor.

### 1.15 Netflix Timed Text Style Guide — the readability contract

**What it is.** General and language-specific timed-text requirements
([general](https://partnerhelp.netflixstudios.com/hc/en-us/articles/215758617-Timed-Text-Style-Guide-General-Requirements),
[English](https://partnerhelp.netflixstudios.com/hc/en-us/articles/217350977-English-Timed-Text-Style-Guide)).

**Numbers worth borrowing.** Min duration **5/6 s**, max **7 s** per event; **42 characters per
line**, **max 2 lines**; "text should usually be kept to one line"; prefer a **bottom-heavy
pyramid**, avoid a 1–2 word top line; dual speakers on separate lines with a hyphen. Font size is
specified as 100 % *relative to the ability to fit 42 characters*, i.e. **width-driven**, not
px-driven.

**What we should do.** Our `min_display_s = 0.90` matches the 5/6 s floor; our 2-line cap matches;
our Latin budget is 42 chars — an exact match (nice). What we lack is the *pyramid* rule as a hard
constraint on the second line and the width-driven sizing principle: our per-line budget is a
character count, while Netflix and BBC both derive the limit from the rendered width. See R14/R15.

### 1.16 WCAG 2.2 — contrast

**What it is.** The normative contrast floor
([Understanding 1.4.3](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html)).

**Numbers worth borrowing.** 4.5:1 normal text; **3:1 large text** (≥ 18 pt ≈ 24 px, or ≥ 14 pt
≈ 18.66 px bold); 7:1 / 4.5:1 for AAA (1.4.6); 3:1 for non-text/UI boundaries (1.4.11).

**What we should do.** Our QA uses `MIN_CONTRAST = 4.5` for *all* text
(`verify2.py:81`, `policy.qa.contrast_min`) and the reference run measured **7.28** — stricter
than AA and near AAA. Keep 4.5 as the text floor, but two gaps: (a) the accent rule / hairline /
bars are **non-text graphics** that are never contrast-checked against 1.4.11's 3:1; (b) a large
display keyword legally only needs 3:1, and holding it to 4.5 costs composition freedom (it can
force a scrim that the frame does not need). R16 makes the thresholds explicit per role rather
than global.

### 1.17 TikTok — checked, nothing citable

I expected a normative caption spec. There is none publicly. `ads.tiktok.com/help/article/
tiktok-video-ad-specifications`, `…/video-specifications`, `support.tiktok.com/.../captions` and
the creator portal are all client-rendered shells; the ad spec page contains no typography or
safe-area numbers in its DOM. The widely repeated "keep text out of the bottom 250 px / right
120 px" figures are **not** in any primary source I could verify, so I am not asserting them as
normative.

**What we should do.** Treat "TikTok practice" as *observed convention* for the 9:16 bottom band
(≈13 % of height, which is what clipify's `MarginV=280/1920 = 14.6 %` and the BBC 9:16 central-75 %
rule both imply), and label it as convention in `policy.py` provenance rather than as a standard.
Our per-platform export (`hypit`-style delivery presets in `delivery/`) should carry the number in
its own manifest so a client can change it.

---

## 2. Ranked recommendations

Ordered by *value ÷ effort* for the next iteration. "Effect" is what we should be able to measure;
where a number is an estimate it is marked ≈. Symbols are in this checkout.

| # | Change | Symbol | Source | Expected effect | Effort |
|---|---|---|---|---|---|
| R1 | Caption type scale becomes aspect-aware: landscape **line height 7.5 % of H**, portrait **4.2 % of H** (i.e. `size = line_height / 1.46`) | `render2.CaptionTrack.__init__` (`0.034*ref`), `policy.captions.size_ratio` | BBC 9.2.1 | Font **31 px → 47 px** landscape (5.1 % of H; measured ink 29 → ≈41 px = 4.5 % of H); **23 px → 55 px** at 1080×1920. Both move inside the broadcast envelope. | S |
| R2 | Caption safe area becomes aspect-aware, and the obstacle lift gets a ceiling: floor `0.052 H` landscape / **`0.13 H` portrait**, lift `≤ 0.06 H`, never into the face box | `render2.render()` (the `0.62*H` / `0.035*H` / `0.018*H` block), `policy.captions.bottom_ratio` | BBC 9.3, clipify `MarginV=280`, shortcast 8.5 % top | Portrait captions keep the bottom 13 % clear by construction; **0** caption↔face intersections; measured bottom margin stays ≥ floor on 100 % of cues. | S |
| R3 | Column candidates must be *type-aware*: require `width ≥ 3.2 × display_px + 2·pad` instead of a canvas-independent `>= 180` px and a 19 %-of-W ratio | `design2.Designer.candidates()` | measured: portrait IR has a 116 px column at 52 px/glyph | Portrait demo's 2-glyph lines disappear; **min glyphs per line ≥ 4** on 9:16; 0 keywords needing 3 lines. | S |
| R4 | Quietness must be computed **inside the element's window** (perception samples are 5 Hz and already exist), not from five clip-wide frames | `design2.Designer._quietness` (`self.frames[:5]`), called from `candidates()` | logic bug found while reading | Distinct scores for the same box at different times; elements whose chosen box is busy in-window drop out (expect ≈10–20 % of candidates to change rank on a real clip). | S |
| R5 | A caption typography check in QA: ink height %, lines, ink width %, bottom margin, distance to face box | new check in `verify2` (next to `legibility`, keyed under `checks`), wired into `policy.qa` | BBC + Netflix | The two known caption gaps become **red gates**; regression-proof for R1/R2/R14. | M |
| R6 | Emit the top-k layout candidates and their score terms into the IR instead of `"candidates": []` | `design2.Designer._score`, `Designer._element` | our own audit gap; clipify's "show your work" schema | Every element carries ≥ 2 scored alternatives; the host agent/model can veto a placement; QA can report chosen-vs-best margin (a real confidence number). | S |
| R7 | Close the decision→IR contract: honour `params.enter`, `params.accent`, and `content` — or reject them in `validate_decisions` | `design2.Designer._groups`, `_content_for`, `_element`, `llm_director.validate_decisions` | our own contract gap (§6.1) | **0 silently dropped decision fields**; model-chosen motion appears in the IR for ≥ 1 element. | M |
| R8 | Let the director *add* moments, not only style rule-found ones: pass a candidate pool (phrases + pauses + numbers) and require a `source` per output | `pipeline.auto_anchors`, `llm_director.build_prompt` / `to_plan` | clipify step 1A | On a labelled 25 s clip, recall of human-picked beats +1–2 within the same density budget; false positives unchanged (measured by a 2-person rating of 10 clips). | L |
| R9 | Enter duration from distance/area: `clamp(0.16 + travelled_px/(0.9·ref_h), 0.18, 0.42)`; keep the 12 px offset as the floor | `policy.motion.enter_s`, `design2.Designer._element`, `effects2.paint_element` | Carbon duration-vs-distance, Material medium band | Median enter **420 ms → ≈260 ms**; no element above 420 ms; measured per element from the IR. | S |
| R10 | Use Material's emphasized-decelerate for elements larger than ≈25 % of frame area; reserve `out_expo` for the wipe/accent rule | `theme2.EASES`, `policy.motion.enter_ease` | Material 3, Carbon | Peak ink-bbox velocity on large elements drops (measurable from rendered frames: first-difference of the element box), same perceived snappiness. | S |
| R11 | Replace `{enter, exit, hold}` with a keyframe list (per-property, per-segment easing) in the IR; clamp once in the transform layer | `design2.Designer._element`, `render2._element_transform`, `FX.paint_element` | movis `Motion`, Remotion `extrapolate*` | The IR can express a stagger or a bespoke curve without a code change; 0 out-of-box pixels across all effects (one clamp site, testable). | M |
| R12 | Content-hash keyed frame cache for element ink and backdrops, with a size limit | `render2.render`, `render2._backdrop_image`, `effects2._masked_layer` | movis `Cache(size_limit=…)` | Second render of the same IR is faster (target: draft wall-clock −40 %); outputs still byte-identical (hash asserted). | M |
| R13 | Make draft resolution first-class (`level` 1/2/4) and record it in the manifest | `render2.render` signature + manifest, `pipeline` preview path | movis `level`/`preview_level` | Draft render at 1/2 linear size ≈ 4× fewer pixels: target −60 % wall clock; manifest states the level so QA can refuse to score a draft. | S |
| R14 | Derive the per-line character budget from measured advance width against the 68 %/90 % width caps; store the derived number, not a constant | `captions.DEFAULT_STYLE`, `captions.validate`, `verify2` width check | BBC 3.1, Netflix 42 chars | At 55 px portrait type, 22 CJK chars would be **112 % of W**; derived budget ≈ **17**. 0 cues whose rendered line exceeds the cap. | S |
| R15 | Add the BBC/Netflix break prohibitions we lack (number+unit, negation+verb, verb+particle) and enforce the "bottom-heavy pyramid" (no 1–2 glyph top line) | `theme2.wrap_text`, `captions._layout_lines` | BBC 3.4, Netflix I.10 | 1-glyph orphan lines in `fit.lines` = **0**; caption wrap violations = 0 on a hand-checked 40-cue set. | S |
| R16 | Contrast thresholds per role: text 4.5, large display 3.0 floor (keep 4.5 as default), **non-text graphics 3.0** (accent rule, hairline, bars) | `verify2.MIN_CONTRAST`, `policy.qa` | WCAG 1.4.3 / 1.4.6 / 1.4.11 | Non-text graphics become measurable (currently unchecked); QA reports AA and AAA columns; reference run keeps PASS at 7.28. | S |
| R17 | Commit the placement zone per transcript segment/shot, not per 9 s window; hard-cut to a new zone only at a segment boundary | `design2.Designer._score` (consistency term), `_place`, `llm_director._shot_cuts` | auto-vertical-reframe "one crop per shot" | Zone switches per minute ↓ (measure from IR: `layout.zone` sequence); fewer re-reads of the same words. | S |
| R18 | Surface alignment confidence per anchor (mean char score) and refuse windows below 0.6 for number/name elements | `align.py` output → `pipeline.auto_anchors` → anchor provenance | whisperX word score = mean char score; ctc span score | Timing-sensitive overlays carry a confidence; QA can flag "anchored on low-confidence text"; measured distribution before/after. | M |
| R19 | Classify protected regions by OCR kind, not by size: replace `w < 0.02W / h < 0.012H / h ≤ 0.055H` heuristics with kind + position + confidence | `design2._is_obstacle` | our own QA note about venue signage | Fewer wrongly-protected regions (a venue logo stops blocking a column) and no missed small credits; measured as protected-px area vs a hand-labelled frame set. | M |
| R20 | Golden-frame determinism gate in CI: hash 12 decoded frames (ink-only pass) and compare | `tests/` + `verify2` ink-only output | Remotion/Revideo frame-indexed determinism | Determinism becomes a 5-second test instead of a 25 s double render; catches any accidental wall-clock/random use. | S |

Two cross-cutting notes on the table. First, R1–R4 and R14–R15 are all "numbers that were invented
in code instead of derived from a standard or from the type you are actually setting" — they are
the cheapest quality win available and they are the ones a reviewer can check without watching the
video. Second, R6–R8 and R18 are all "the audit surface is thinner than the decision surface":
today a decision can be dropped between layers with no error, and the IR does not record the
alternatives that were rejected.

---

## 3. Where the judgement currently lives (why §6 is the big section)

For context on the model section below, the decision path is:

```
transcript/chars ──► pipeline.auto_anchors        (rules: token lists + scores)   [text only]
                        │
                        ▼
                llm_director.build_prompt  ──► model ──► decisions.json          [text only]
                        │                                  (effect, emphasis, hold, params)
                        ▼
             design2.Designer (candidates → hard constraints → _score)           [geometry only]
                        │
                        ▼
             effects2.fit_element (type metrics) ──► Motion IR ──► render2      [pixels]
                        │
                        ▼
                     verify2 (measures pixels)
```

Two observations follow, and both are defects rather than preferences:

1. **The model never sees a frame.** It styles moments it did not choose and cannot choose where
   they go. Placement is decided by 0.40·quietness + a preference table, which is why the portrait
   demo produced a 116 px column (`design2.Designer._score` never knows how wide a glyph will be).
2. **The model's output is partially discarded.** `llm_director.build_prompt` documents
   `params.enter` (`fade_up | spring_pop | count_up | stagger_rows`), `params.accent`, and
   `content`; `design2._content_for` reads none of them, and `_element` sets motion from a fixed
   type→style map. The decision doc's own example requests `"enter": "spring_pop"` and
   `"accent": "#F2B233"` (`artifacts/llm/download_zhengqinwen.decisions.json`) — neither reaches
   the render. `validate_decisions` accepts them, so nothing warns.

---

## 4. Do not copy

Each of these is a real feature of a real project above, rejected with a reason and (where
relevant) the cost we would pay.

| Not this | Where it comes from | Why not |
|---|---|---|
| **3-D, WebGL, shaders, Three.js** | Remotion/Revideo examples | Our entire quality claim is a *measured 2-D pixel diff*. 3-D adds a camera model, depth sorting and non-determinism in GPU drivers, and buys nothing for typographic overlays on a talking head. |
| **Particles, physics, springs with overshoot** | Remotion `spring(damping, stiffness, mass)`, many motion tutorials | Apple: "avoid showing objects that oscillate in a sustained way". Overshoot on type over a face reads as cheap, and it breaks our "ink is already measurable on the anchor frame" premise: an overshooting glyph is at the wrong size on the word onset. We want *damping without overshoot* (§1.4's `_smooth_damp` produces that, and only for cameras). |
| **Lottie / Rive / After Effects import** | common in motion pipelines | A Lottie comp is an opaque timeline we cannot measure, anchor to a word, or repair. It would punch a hole in `verify2`'s attribution (every changed pixel must be explainable by the IR). |
| **Browser rendering as the default backend** (Remotion, Revideo, Motion Canvas) | §1.8–1.10 | Determinism and dependencies: a Chromium + Node + pnpm stack (~hundreds of MB) for effects we already render with Pillow, plus Remotion's licence above 3 employees. Keep it as an optional backend (it already exists, `remotion_bridge.py`) but never as the default, and never as the QA reference. |
| **Moving/zooming the camera** (auto-vertical-reframe's whole purpose) | §1.4 | Our promise is "keeps the original picture and voice". A virtual camera changes framing decisions the speaker made (and can crop the burned-in credits our OCR protects). Borrow the *hysteresis*, not the crop. |
| **TTS-reference forced alignment** (aeneas) | §1.7 | It exists to solve alignment without a model; we have wav2vec2/CTC. It also emits no confidence, which is the one output we want. |
| **VAD as the only chunking strategy** | whisperX/ctc contrast | A VAD miss is silent data loss. Prefer overlapping windows with reconciliation (§1.6). |
| **Word-by-word karaoke captions as the default look** | clipify's opus preset (3-word chunks, yellow active word, 100 px Arial Black) | It is a *different genre* (retention-bait) and it conflicts with our reading model: BBC/Netflix specify one sentence per event and 2 lines max, precisely so the eye stops re-reading. Offer it as an explicit `captions.style` variant, never as the default, and never at the cost of the sentence-per-cue rule. |
| **A model-generated renderer / code execution** | LLM-to-React pipelines | `docs/REMOTION.md` already says it: "不执行 LLM 生成的代码：props 只含数据". Keep it. Code generation would destroy both the repair planner and the determinism gate. |
| **Distributed / lambda rendering, thread pools** | Remotion Lambda, `threads` in Revideo | 25 s of 1920×920 renders in seconds-to-a-minute; the scheduler, the cache key and the failure modes cost more than they save. Revisit only if clip length > 5 min becomes normal. |
| **Per-frame ML inference in the render loop** | auto-vertical-reframe (YOLO per frame) | Our perception is a pre-pass at 5 Hz with a mask document; putting a model in the render loop would make render time unpredictable and the output non-reproducible on a different machine. |

---

## 5. Caption typography and safe areas — the numbers we should adopt

### 5.1 What we do today (computed, then measured)

`theme2.reference_height(W,H) = max(640, min(H, 0.62·W))`; `render2.CaptionTrack` sets
`size = 0.034·ref`, `pitch = 1.46·size`, `bottom = 0.052·H` unless an OCR'd obstacle below
`0.62·H` lifts it.

| canvas | ref | font px | pitch px | pitch as %H | measured ink |
|---|---|---|---|---|---|
| 1920×920 (the reference delivery) | 920 | **31** | 46 | 5.0 % | **29 px = 3.15 % H**, bottom margin 14.1 % (lifted by a detected mic tag) |
| 1080×1920 (9:16 here) | 669.6 | **23** | 33 | 1.7 % | not shipped; computed |

### 5.2 The standards side by side

| metric | BBC (16:9) | BBC (9:16) | Netflix | clipify (9:16) | shortcast (9:16) | ours today |
|---|---|---|---|---|---|---|
| line height | 7–8 % H | **3.9–4.5 % H** (4.2 % example) | — (100 %, fit 42 chars) | font 100 px ≈ 5.2 % H → pitch ≈ 6.3 % H | hook 5.8 % **of W** | **5.0 % H** landscape, 1.7 % portrait |
| max lines | 2 | 3 | 2 | 1 (3-word page) | 1–2 | 2 |
| line width cap | 68 % W | 90 % W | fit 42 Latin chars | ~90 % W | band ≈ 86 % W | 24 CJK / 42 Latin **chars**, width unmeasured |
| bottom safe | central 90 % ⇒ ≥5 % H | central 75 % ⇒ **≥12.5 % H** | — | MarginV 280 = **14.6 % H** | top 8.5 % H | 5.2 % H floor, lifted by OCR |
| min duration | — | — | **5/6 s** (0.83 s) | 3 words | — | **0.90 s** ✓ |
| per-cue text | one sentence | one sentence | one sentence, 2 lines max | 3 words | hook | one cue, ≤44 CJK chars |

### 5.3 What already matches (do not churn these)

* `min_display_s = 0.90` ≥ Netflix's 5/6 s. Good.
* 2-line cap for landscape; 24 CJK / 42 Latin char budgets are *close* to Netflix's 42-char Latin
  rule and BBC's 25-char 9:16 equivalence.
* Bottom margin floor 5.2 % H already satisfies BBC's central-90 % rule for landscape.
* Caption contrast is held to 4.5:1 (stricter than the 3:1 WCAG large-text floor) and measured
  7.28 on the reference — keep.
* The dynamic lift away from detected burned-in text (mic tags) is a genuinely good idea that
  neither BBC nor Netflix has to deal with, because they own the whole frame. Keep it, bound it
  (R2).

### 5.4 What to adopt (concrete, in order)

1. **Size.** Define caption size by *line height as a fraction of active height*, per aspect:
   `line_height = 0.075·H` (landscape) / `0.042·H` (9:16), and `size = line_height / 1.46`.
   Reference frame: 47 px. 1080×1920: 55 px. (If we prefer to match TikTok's observed weight
   rather than BBC's, 9:16 caps out around 6.5 % H pitch ≈ clipify; BBC is the defensible floor
   and we should not go below it.)
2. **Lines.** Keep 2 as the cap in both aspects (BBC allows 3 in 9:16; Netflix and our restraint
   say 2 — 3 is a maximum, not a target).
3. **Width.** Enforce *rendered* width: `≤ 0.68·W` landscape, `≤ 0.90·W` 9:16, per line, measured
   after layout. Then derive the char budget backwards: at 55 px CJK portrait, `0.90·1080/55 =
   17.7` → **17 CJK chars/line**; at 47 px landscape `0.68·1920/47 = 27.7` → 27 (below our 24 cap,
   so landscape is unaffected).
4. **Bottom safe area.** `bottom_ratio = 0.052` landscape (unchanged) / **`0.13`** 9:16, with the
   obstacle lift capped at `0.06·H` and a hard rejection if the resulting band intersects the face
   box. 0.13 = BBC's central-75 % and sits just under clipify's 14.6 %.
5. **Top safe area** for graphic elements (not captions): keep the top **10 %** of a 9:16 frame
   clear (`shortcast` uses 8.5 % for its hook band; BBC's central-75 % implies 12.5 %, so 10 % is a
   reasonable middle and is what a status bar + platform header consumes).
6. **Semantics.** One complete sentence per cue (BBC/Netflix); no 1–2 glyph line at the top of a
   two-line cue (Netflix "bottom-heavy pyramid"); break at punctuation, never between
   number+unit or negation+verb (R15).
7. **Machine-checkable.** Every number above becomes a `verify2` assertion (R5) and a line in the
   delivery manifest, so a client can see which envelope was used rather than trusting a claim.

---

## 6. Decisions that should move from code to a model

This is the section that should drive the next implementation step. It is deliberately ruthless:
it lists **every** hard-coded judgement that is really a taste or semantic call, with the exact
symbol, what a model would decide better, the evidence it must see, and the guardrail that keeps
the renderer deterministic.

A design constraint for all of them: **the model returns data, never code; the solver keeps every
invariant; the rules stay as a fallback and are recorded in provenance.** We already have the
machinery for this — `provenance.truth_class` (`MACHINE` in `pipeline.auto_anchors`),
`semotion.llm-decisions/1`, and a deterministic repair loop. Extend `truth_class` to
`MODEL`/`HUMAN` and make QA report the class of every element.

### 6.1 The three that matter most

**M1 — Which words deserve a graphic at all (`pipeline.auto_anchors`).**
Today: a phrase of ≥4 chars and ≥0.45 s is classified by membership in hand-written token lists
(`ENUM`, `CONTRAST`, `SUPERLATIVE`, `CONCLUSION`) with fixed confidences 0.90 / 0.84 / 0.78 / 0.72
/ 0.66, minus `0.015·(len−14)`, then top-N by score with a 0.9 s gap. A model should decide this
from the *whole* transcript: is this a claim, a reversal, a list, a number that matters, or plain
narration? It should also be allowed to say "this clip can carry 3 elements, not 5" — the density
budget is currently a constant (`max_elements_per_10s = 4`, `max_coverage = 0.58`) regardless of
how dense the speech is.
*Evidence it should see:* the full transcript with char-level timings and pauses; the paragraphs
before/after each candidate (context, not just the phrase); the list of candidates the rules
already found *and the ones they rejected* (so it can promote, not only prune); clip duration; the
speaker's words-per-second; the target platform (a 15 s Short vs a 3 min YouTube cut).
*Why a model:* the rule's failure mode is a *false negative* on a beat that has no keyword but is
the point of the story ("我从来没见过这样的球场" has no number, no 但是, no 最). No token list
covers that; an LLM reading the paragraph does.
*Guardrail:* the model proposes `{start,end,kind,reason,confidence}`; `auto_anchors`' density,
minimum-gap and cardinality checks still run; disagreements are logged with both scores; the
deterministic rules remain the fallback and the baseline for comparison.

**M2 — Where the graphic goes (`design2.Designer._score` / `candidates`).**
Today: hard constraints reject person/face/hands/burned-in-text collisions (correct, keep), then
`_score = 0.40·quiet + PREF[kind][treatment] + 0.10 consistency + 0.08 type-fidelity +
0.06 face-distance − 0.18 torso + 0.10 scrim-bonus`. That is a *proxy* for "does this look like a
designed frame", and it has never seen a frame. The 9:16 failure is its signature: a 116 px column
scored well because the ratio gate said 19 % of W was enough.
*Evidence it should see:* three crops per candidate (the frame at window start / middle / end, with
the candidate box and the person mask outline drawn on), the candidate's measured type metrics
(font px, chars/line, lines), the OCR obstacle list, the previous element's box and time, the
palette accent, and the policy envelope (margin, caption band, coverage budget). Not the whole
frame: 3 crops × top-3 candidates is cheap and makes the judgement concrete.
*Why a model:* composition is perceptual. A vision model can tell that a column is too narrow, that
the type would sit on a busy gilt frame, that a lower-third would fall on hands; our scalar cannot.
It can also answer the question no rule encodes — "would a designer put anything here at all?"
*Guardrail:* the model *ranks* candidates that the solver generated and already proved legal; it
cannot invent a box outside the legal set. Deterministic fallback = current `_score`. Record the
chosen index, the runner-up, and the margin; if the margin is < 0.05, prefer the rule (stability).

**M3 — How long a graphic holds and what it does while it holds
(`design2.Designer._element`, `policy.motion`).**
Today: enter 0.42 s `out_expo` + 12 px, exit 0.18 s `in_quad`, stagger 0.075 s, hold from
`hold_s` (default 0.45 s) — identical timing for a two-glyph keyword and a four-row list. The
model *may* set `hold_s` (it does: 0.5 / 0.3 in the reference decisions) and *may* request
`params.enter`, but `enter` is silently dropped (§3) and the duration is not a function of what is
being animated.
*Evidence it should see:* element kind, glyph count, row count, the anchor's duration in seconds
and the next anchor's start (how much room there is), the words per second around it, and the
neighbouring elements' boxes/zones (so two motions do not race).
*Why a model:* hold time is a reading-time judgement ("can I read 9 glyphs in 0.9 s while
listening?") and motion choice is a rhetorical one (a wipe for a concede, a count for a stat). Both
are semantic; neither is a lookup table on `type`.
*Guardrail:* the model returns `{enter, enter_s, hold_s, exit_s, stagger_s}` within hard envelopes
from `policy` (e.g. `enter_s ∈ [0.18, 0.42]`, `hold_s ∈ [0.3, max(0.45, 0.9·read_time)]`,
`exit_s ∈ [0.12, 0.24]`); the renderer clamps and the IR records both requested and applied values.

### 6.2 The full list of hard-coded judgements

| # | Symbol | Current judgement (value) | What a model decides better | Evidence it should see |
|---|---|---|---|---|
| 1 | `pipeline.auto_anchors` classification | `ENUM`/`CONTRAST`/`SUPERLATIVE`/`CONCLUSION` token lists; scores 0.90/0.84/0.78/0.72/0.66 | whether a phrase is a claim / reversal / list / number worth showing, and which ones to skip | M1 evidence |
| 2 | `pipeline._split_phrases` | `max_chars = 13`, pause threshold 0.26 s (or `speech_db − 10`) | phrase boundary by syntax and meaning, not just by char count and silence | transcript + pause map + audio RMS |
| 3 | `pipeline.auto_anchors` repetition merge | common prefix ≥ 2 chars, gap < 2.6 s, ≤ 4 items, prev text ≤ 14 chars → one list | whether repeated structure is a list ("每一年…/每一年…") or coincidence, and how many rows are *useful* | the 3–5 neighbouring phrases + their timings |
| 4 | `pipeline.auto_anchors` selection | sort by score, keep if `abs(Δstart) ≥ 0.9 s`, cap `max_elements` | how many overlays this clip and this speaker can carry; where to leave silence | duration, WPS, candidate density, platform |
| 5 | `design2.TYPE_ALIASES` + `_content_for` | kind → element (keyword/card/list/stat/quote) by alias table | which *form* carries the meaning (a quote card vs a keyword vs a number) | anchor text, params, palette, available area |
| 6 | `design2.Designer._score` | weights 0.40 / 0.10 / 0.08 / 0.06 / −0.18 / +0.10 (policy `layout.weights`) | rank legal candidates by *how the frame looks* | M2 evidence |
| 7 | `design2.Designer.candidates` gates | `min_col_ratio 0.19`, `min_band_ratio 0.115`, `>= 180` px floor, `type_band (0.18, 0.74)`, `band_bottom` bias 0.58, `zones[:8]`, quiet-zone overlap 0.45 | whether a region is *usable* for the type that will actually be set there | candidate box + measured type metrics + frame crops |
| 8 | `design2.Designer._quietness` | divisor 0.085, fallback 0.55, five clip-wide frames | how busy the backdrop is *during this element* and whether type can survive it | 3 crops of the window + edge-density stats |
| 9 | `design2._is_obstacle` | `w < 0.02W`/`h < 0.012H` ignored; subtitle/caption always protected; anything `h ≤ 0.055H` protected; logos in the top/bottom 16 % | which burned-in text is a credit/subtitle (must not be covered) vs venue signage (part of the shot) | OCR text + kind + position + confidence + a crop |
| 10 | `design2.Designer._enforce_coverage` | drop by `priority` map (stat .95 … quote .65) until `≤ 0.58` coverage | which element to sacrifice when the clip is over budget (the *least meaningful*, not the lowest type priority) | the elements' texts, spans, and reasons |
| 11 | `effects2._fitting_font` | shrink by 0.92 until ≤ `max_lines`, floor `0.58·start`; `max_lines` from `policy.type.max_lines` | whether to set smaller type, use a different form, or drop the element | box, text, hierarchy (what else is on screen) |
| 12 | `design2.Designer._element` motion map | per-type style map (`wipe`/`rows`/`count`/`bars`/`columns`), 0.42 s, `out_expo`, 12 px, stagger 0.075, exit 0.18 s | the motion metaphor and its duration (§M3) | M3 evidence |
| 13 | `captions.DEFAULT_STYLE` | 2 lines, 22/24 CJK, 38/42 Latin, cue 30/44 CJK, min display 0.90, merge 0.18, pause break 0.35 | *where a cue should break for meaning* when punctuation and silence disagree (e.g. a long clause with no comma) | the transcript with punctuation + pauses + the visual pacing |
| 14 | `captions._best_split_index` | score `abs(L−R) + 0.05·abs(total−target)`, `+2.0` if over target, `+3.0` if under min, `−1.5` clause, `−1.0` punct | whether a split preserves the sentence's sense (a model reads the words; the score counts characters) | the cue text + its neighbours + the syntax |
| 15 | `render2.render` caption bottom | obstacle search below `0.62·H`, conf ≥ 0.5, `bottom = max(0.035H, H − top + 0.018H)`, no upper bound | where captions belong given the frame, the burned-in text *and* the target platform's UI | OCR boxes + their text, face/hands boxes, platform (9:16 vs 16:9) |
| 16 | `pipeline` director precedence | decisions from `llm_director` may style/drop rule-found anchors but never add; `director_mode = off` falls back to rules | the whole directing step (M1 + M3 combined) | transcript, candidates (accepted *and* rejected), duration, platform |

**Cost and cadence.** All sixteen are per-clip, not per-frame: one call for the directing pass
(M1, rows 1–4, 16), one for layout (M2, rows 6–10) *if* we decide to consult a vision model, one for
motion (M3, row 12), and ideally none for captions (rows 13–15) — caption rules are the one place
where the standard genuinely is a rule (BBC/Netflix), so the right change there is to *measure*,
not to ask. On a 25 s clip that is 2–3 calls, cached by `(transcript_sha, policy_sha, frames_sha)`
so a re-render never re-asks.

**What must stay deterministic.** The solver's hard constraints (never over the face/hands/credits,
never outside the canvas, never over another live element), the density/coverage budget arithmetic,
frame timing, the encoder settings, and the QA thresholds. A model may *rank and choose inside*
the legal set; it must not be able to create an illegal frame.

---

## 7. Defects found while reading (evidence, not opinion)

Recorded here because they are cheap to fix and they undermine otherwise-correct claims. None of
these is a blocker for v2; all of them are fixed by R1–R7.

* **D1 — `_quietness` is time-blind.** `design2.Designer._quietness(box)` averages
  `self.frames[:5]`, which `pipeline.run` samples at 8/25/45/65/85 % of the *clip*
  (`pipeline.py:436`), so every element gets the same score for the same box regardless of its
  window, even though 5 Hz perception frames exist and are used for masks. (R4)
* **D2 — the column gate is canvas-independent.** `>= 180` px and `min_col_ratio 0.19` admit a
  116 px text area at 1080 wide, where a CJK glyph is 52 px: `artifacts/v2/demo_portrait/motion_ir.json`
  contains `lines: ["最难","的地","方"]` in a 116 px box. (R3)
* **D3 — three documented decision fields are silently dropped.** `params.enter`,
  `params.accent`, and `content` are accepted by `llm_director.validate_decisions` and never read by
  `design2._groups`/`_content_for`/`_element`. (R7)
* **D4 — captions are below the broadcast floor and no check exists.** 3.15 % of H measured
  landscape, 1.2 % computed portrait, versus BBC's 7–8 % / 3.9–4.5 %; `verify2` has no caption
  measurement at all (`FONT_FLOOR_PX["caption"]` is never applied to captions). (R1, R5)
* **D5 — caption line width is a character count, not a measurement.** At the portrait size the
  doc recommends, 22 CJK chars would be 112 % of the frame width; `captions.validate` only counts
  characters. (R14)
* **D6 — two sources of truth for caption numbers.** `captions.DEFAULT_STYLE` and
  `policy.FALLBACK["captions"]` carry the same numbers under different key names
  (`target_chars_per_line` vs `target_chars_cjk`), so they can drift. `policy.py`'s own docstring
  argues there should be one.
* **D7 — `layout.candidates` is always empty in the IR** (`design2._element` writes `[]`), so the
  rejected alternatives and the score margin are unrecoverable after the fact. (R6)

---

## 8. Suggested order of work

1. **Week 1 (numbers, no model):** R3, R4, R1, R2, R14, R5. These close D1, D2, D4, D5 and give QA
   a new gate. All are policy/constant plus a check; nothing about them is speculative.
2. **Week 2 (audit surface):** R6, R7, R20, R16, R15, R17. The IR starts recording what it
   rejected, and the decision contract stops lying. After R7 the model has real writable knobs.
3. **Week 3 (motion craft):** R9, R10, R11, R13, R12. Timing becomes a function of what is being
   animated; the IR gains keyframes; drafts get cheap.
4. **Then the model work:** M2 (visual ranking) first — it has the clearest evidence package and
   the tightest guardrail — then M1 (anchor recall), then M3 (motion selection). M2/R6 together
   are what turn the director from "styles text" into "composes a frame".

Do not start with M1. It is the highest-value judgement but the hardest to evaluate without a
labelled set; M2 can be evaluated immediately because R6 puts the alternatives in the IR and the
QA already measures the result.

---

## Appendix A — symbol index

| symbol | file:line | touched by |
|---|---|---|
| `MAX_ELEMENTS_PER_10S`, `MIN_GAP_S`, `MAX_COVERAGE`, `HOLD_S`, `CAPTION_BAND`, `PREF` | `src/semotion/design2.py:40-73` | R8, R17 |
| `DesignConfig` | `src/semotion/design2.py:118` | R3 |
| `Designer.candidates` | `src/semotion/design2.py:391` | R3, R4, M2 |
| `Designer._quietness` | `src/semotion/design2.py:367` | R4 |
| `Designer._content_for` / `_groups` | `src/semotion/design2.py:523` / `:483` | R7, M1 |
| `Designer._enforce_coverage` | `src/semotion/design2.py:658` | §6.2 row 10 |
| `Designer._score` | `src/semotion/design2.py:774` | R6, R17, M2 |
| `Designer._element` | `src/semotion/design2.py:794` | R6, R9, R11, M3 |
| `_is_obstacle` | `src/semotion/design2.py:861` | R19 |
| `_fitting_font` / `fit_element` | `src/semotion/effects2.py:105` / `:192` | R11, §6.2 row 11 |
| `paint_element` | `src/semotion/effects2.py:471` | R9, R10 |
| `CaptionTrack.__init__` / `.draw` | `src/semotion/render2.py:42` / `:63` | R1, R2, R5 |
| `render` caption-bottom block | `src/semotion/render2.py:308` | R2 |
| `_element_transform` | `src/semotion/render2.py:226` | R11 |
| `_scrim_2d` | `src/semotion/render2.py:140` | R10 |
| `auto_anchors` | `src/semotion/pipeline.py:97` | M1, R8, R18 |
| `_split_phrases` / `_cap_length` | `src/semotion/pipeline.py:209` / `:229` | §6.2 row 2 |
| frame sampling | `src/semotion/pipeline.py:436` | R4 |
| `DEFAULT_STYLE` | `src/semotion/captions.py:52` | R14, §6.2 row 13 |
| `_best_split_index` | `src/semotion/captions.py:297` | R15, §6.2 row 14 |
| `_layout_lines` | `src/semotion/captions.py:486` | R15 |
| `wrap_text` | `src/semotion/theme2.py:300` | R15 |
| `reference_height` / `EASES` | `src/semotion/theme2.py:43` / `:86` | R1, R10 |
| `MIN_CONTRAST`, `SAFE_MARGIN_PX`, `FONT_FLOOR_PX` | `src/semotion/verify2.py:81-88` | R5, R16 |
| `FALLBACK` policy table | `src/semotion/policy.py:36` | R1, R2, R9, R14 |
| `build_prompt` / `validate_decisions` | `src/semotion/llm_director.py:605` / `:492` | R7, R8, M1, M3 |

## Appendix B — sources and the exact place I read

| source | link | what was read |
|---|---|---|
| movis | [repo](https://github.com/rezoo/movis) | `movis/motion.py`, `movis/transform.py`, `movis/layer/composition.py`, `docs/overview.rst` |
| clipify | [repo](https://github.com/louisedesadeleer/clipify) | `SKILL.md` (steps 1A–4a), `scripts/build_ass.py`, `scripts/analyze.py` |
| shortcast | [repo](https://github.com/mutonby/shortcast) | `Shortcast/Services/VideoOverlayRenderer.swift`, `Shortcast/Resources/social-content-coach.md` |
| auto-vertical-reframe | [repo](https://github.com/KazKozDev/auto-vertical-reframe) | `src/verthor/auto_reframe.py` (presets, dead zone, `_smooth_damp`, scene detect) |
| whisperX | [repo](https://github.com/m-bain/whisperX) | `whisperx/alignment.py` (CTC alignment, wildcard column, word score) |
| ctc-forced-aligner | [repo](https://github.com/MahmoudAshraf97/ctc-forced-aligner) | README (granularity, `window_size`/`context_size`), `get_alignments`/`get_spans` |
| aeneas | [repo](https://github.com/readbeyond/aeneas) | README (DTW/MFCC, TTS engines, multilevel, word-level caveat) |
| Remotion | [captions](https://www.remotion.dev/docs/captions/create-tiktok-style-captions), [interpolate](https://www.remotion.dev/docs/interpolate), [licence](https://github.com/remotion-dev/remotion/blob/main/LICENSE.md) | pages + licence note already in `docs/REMOTION.md` |
| Revideo | [repo](https://github.com/redotvideo/revideo) | README (generators, `waitFor`, `renderVideo`) |
| Motion Canvas | [repo](https://github.com/motion-canvas/motion-canvas) | README + package layout (skimmed; no change adopted) |
| Material 3 | [tokens](https://m3.material.io/styles/motion/easing-and-duration/tokens-specs) | easing + duration tables |
| Carbon | [motion](https://carbondesignsystem.com/elements/motion/overview/) | easing curves, duration tokens, duration-vs-distance rule, checklist |
| Apple HIG | [motion](https://developer.apple.com/design/human-interface-guidelines/motion) | best practices, feedback, visionOS guidance |
| BBC | [subtitles](https://www.bbc.co.uk/accessibility/forproducts/guides/subtitles/) | 3.1 line length, 3.3 lines, 3.4 breaks, 9.2.1 authoring font size, 9.3 position |
| Netflix | [general](https://partnerhelp.netflixstudios.com/hc/en-us/articles/215758617-Timed-Text-Style-Guide-General-Requirements), [English](https://partnerhelp.netflixstudios.com/hc/en-us/articles/217350977-English-Timed-Text-Style-Guide) | timing min/max, 42 chars, line treatment |
| WCAG 2.2 | [1.4.3](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html) | 4.5:1 / 3:1 large text |
| TikTok | [ad specs](https://ads.tiktok.com/help/article/tiktok-video-ad-specifications) | checked, no typography/safe-area numbers (§1.17) |
