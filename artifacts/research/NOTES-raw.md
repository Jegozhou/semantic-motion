# Raw research notes — evidence log for `docs/BEST-PRACTICES.md`

Date: 2026-10-25. Author: research (task-5).
Purpose: keep the raw, checkable evidence (commands, greps, measured numbers, dead ends)
separate from the distilled document, so a maintainer can re-run any claim.

---

## 1. Sources actually read

Shallow clones (`git clone --depth 1`, most with `--filter=blob:none`) into
`/tmp/sm-research/vendor/`, commit pinned:

| repo | commit | url |
|---|---|---|
| movis | `64a9618` | https://github.com/rezoo/movis |
| clipify | `5f6b75e` | https://github.com/louisedesadeleer/clipify |
| shortcast | `618a507` | https://github.com/mutonby/shortcast |
| auto-vertical-reframe | `1556b3c` | https://github.com/KazKozDev/auto-vertical-reframe |
| whisperX | `5d3678b` | https://github.com/m-bain/whisperX |
| ctc-forced-aligner | `64293cc` | https://github.com/MahmoudAshraf97/ctc-forced-aligner |
| aeneas | `4d200a0` | https://github.com/readbeyond/aeneas |
| remotion | `a64a027` | https://github.com/remotion-dev/remotion |
| revideo | `b5de67a` | https://github.com/redotvideo/revideo |
| motion-canvas | `7b91435` | https://github.com/motion-canvas/motion-canvas |

Web pages fetched to `/tmp/sm-research/raw/*.txt` (text-stripped):
bbc.co.uk subtitle guidelines, partnerhelp.netflixstudios.com general + English
timed-text guides, carbondesignsystem.com motion, m3.material.io easing/tokens
(via text proxy), developer.apple.com HIG motion (via text proxy), WCAG 2.2
Understanding 1.4.3, remotion.dev/docs `interpolate`, `spring`, `sequence`,
`captions/create-tiktok-style-captions`, ads.tiktok.com help article
`tiktok-video-ad-specifications`.

## 2. Dead ends / honest gaps

* `https://github.com/revideo-dev/revideo` → 404. The project moved:
  `https://github.com/redotvideo/revideo` (cloned, README + `packages/2d`).
* `m3.material.io` and `developer.apple.com/design/…` are client-rendered:
  plain `curl` returns a 45-character shell. Fetched through a text renderer
  (`r.jina.ai/<url>`); quotes below come from that rendering of the canonical URL.
* `material-components/material-web` `tokens/_md-sys-motion.scss` is a
  re-export (`@use 'versions/v0_192/md-sys-motion'`); the versioned file is not
  in the shallow tree, so the numbers below come from the m3.material.io token
  table, not from SCSS.
* **TikTok publishes no normative caption-typography or safe-area spec.**
  Checked: `ads.tiktok.com/help/article/tiktok-video-ad-specifications`
  (client-rendered, no typography numbers in the DOM), `.../video-specifications`,
  `support.tiktok.com/.../captions`, creator portal. What exists is ad
  "keep text out of the UI strip" guidance, not a citable number set. The
  document therefore treats "TikTok practice" as *observed convention* and takes
  the hard numbers from BBC (which explicitly specifies 9:16) plus two shipping
  open-source implementations (clipify's ASS preset, shortcast's hook overlay).
* shortcast has **no** burned-in caption engine: it burns a *hook* line
  (`Shortcast/Services/VideoOverlayRenderer.swift`) and writes platform post
  copy (`Shortcast/Resources/social-content-coach.md`). Its caption-layout
  contribution is therefore the hook geometry, not subtitle typography.
* motion-canvas was only skimmed (README + package layout). The one idea taken
  from it (generators advanced by *exact frame time*, not wall clock) is also
  visible in revideo, and our renderer is already frame-indexed, so it yielded
  no change.
* aeneas at word level: its own README says it was *not* designed for word-level
  alignment. Treated as a negative result (see doc).
* Netflix's guides give character limits and line-break rules, not font sizes
  ("tts:fontSize shall be defined as 100%", "relative to video resolution and
  ability to fit 42 characters across screen"). No px numbers to steal.

## 3. Numbers extracted verbatim

### BBC Subtitle Guidelines — https://www.bbc.co.uk/accessibility/forproducts/guides/subtitles/

* Authoring line height: **7 %–8 % of active video height** for 16:9 / 4:3 / 1:1;
  **3.9 %–4.5 % of active video height** for **9:16** (example given: 4.2 %).
  The 4.5 % figure is explicitly derived as `8 % × 9/16`.
* Line count: **2 lines max** for 16:9/4:3/1:1; **3 lines max** for 9:16.
* Line width (online): **68 % of width** for 16:9, **90 % of width** for 4:3,
  1:1 and **9:16**. Teletext/broadcast: 37 monospaced characters.
  Their character equivalence: "37 characters in a 75 % width region of 16:9 ≈
  **25 characters in a 90 % width region of 9:16**".
* Position: subtitles "should not be placed outside the central **90 %**
  vertically" (landscape); for 9:16 this is "**central 75 %** vertically".
  "In vertical (9:16) videos, it is common to position subtitles a little higher up."
* One complete sentence per subtitle; break at punctuation; explicit list of
  prohibited break points (article+noun, preposition+phrase,
  conjunction+clause, pronoun+verb, parts of a complex verb).
* 3 lines at 8 % line height ⇒ region height ≥ 24 %.

### Netflix Timed Text Style Guide (general + English)

* Min duration **5/6 s** per event, max **7 s**.
* **42 characters per line** (English), **max 2 lines**; "text should usually be
  kept to one line"; prefer a **bottom-heavy pyramid**; avoid a 1–2 word top line.
* Font size is **100 %**, relative to resolution and the ability to fit 42 chars;
  no px values.

### Material Design 3 — https://m3.material.io/styles/motion/easing-and-duration/tokens-specs

* Easing tokens (CSS): standard `cubic-bezier(0.2,0,0,1)`, standard decelerate
  `cubic-bezier(0,0,0,1)`, standard accelerate `cubic-bezier(0.3,0,1,1)`,
  **emphasized decelerate `cubic-bezier(0.05,0.7,0.1,1)`**,
  emphasized accelerate `cubic-bezier(0.3,0,0.8,0.15)`.
* Duration tokens: short 50/100/150/200 ms; medium 250/300/350/400;
  long 450/500/550/600; extra-long 700/800/900/1000.
* Pairing: "FAB → sheet 400 ms emphasized", "card → full screen 500 ms long".

### IBM Carbon — https://carbondesignsystem.com/elements/motion/overview/

* Productive standard `cubic-bezier(0.2,0,0.38,0.9)`; expressive standard
  `cubic-bezier(0.4,0.14,0.3,1)`; entrance `(0,0,0.38,0.9)` / `(0,0,0.3,1)`;
  exit `(0.2,0,1,0.9)` / `(0.4,0.14,1,1)`.
* Duration tokens: fast-01 **70 ms**, fast-02 **110 ms**, moderate-01 **150 ms**,
  moderate-02 **240 ms**, slow-01 **400 ms**, slow-02 **700 ms**.
* "Duration should be dynamic based on the size of the animation; the larger the
  change in distance or size, the longer the animation."
* Evaluation checklist asks: does micro-interaction easing use ease-out, and
  does it fall in **90–120 ms**?

### Apple HIG — Motion

* "Add motion purposefully…", "Aim for brevity and precision in feedback
  animations", "In apps, generally avoid adding motion to UI interactions that
  occur frequently", "Consider using fades when you need to relocate an object",
  "avoid displaying motion at the edges of a person's field of view",
  "avoid showing objects that oscillate in a sustained way … around 0.2 Hz".
* Games: 30–60 fps sustained. No duration/easing numbers (Apple expresses these
  through SwiftUI defaults, not a table).

### WCAG 2.2 — Understanding 1.4.3 Contrast (Minimum)

* 4.5:1 for normal text; **3:1 for large text** (≥ 18 pt ≈ 24 px, or ≥ 14 pt
  ≈ 18.66 px bold). 1.4.6 (AAA) enhanced: 7:1 / 4.5:1. 1.4.11 non-text: 3:1.

### Movis (code)

* `movis/motion.py`: `EASING_TO_FUNC` maps `Easing` → function; `Motion.append(keyframe,
  value, easing)` stores a **per-segment** easing callable; `Motion.__getitem__`
  bisects keyframes, normalises `t` in the segment, applies that segment's
  easing, then lerps values (`m + (M-m)*t`). Values are numpy float64
  (`transform_to_numpy`), so positions are sub-pixel.
* `movis/layer/composition.py:65`: `self._cache: Cache = Cache(size_limit=1024*1024*1024)`;
  frame cache keyed by `(CacheType, layer name, key_body)` at `:777`, cleared
  on structural edits; `level`/`preview_level` render the whole composition at
  1/2 or 1/4 resolution (`docs/overview.rst#L206`).
* `movis/transform.py`: `TransformValue` = anchor point, position (float pair),
  uniform/XY scale, rotation, opacity — a single value type animated by `Motion`.

### auto-vertical-reframe (code)

* `src/verthor/auto_reframe.py:80-82` sports preset: `follow_deadzone_px: 28.0`,
  `motion_lead_x: 1.7`, `motion_lead_y: 0.7`.
* `:1801-1810` dead zone applied in *observation* space before filtering:
  `if abs(obs.center_x - state.crop_center_x) <= deadzone: obs.center_x = state.crop_center_x`,
  and zoom is frozen while `abs(obs.zoom - state.zoom) < 0.045`.
* `:1812-1815` confidence-weighted target:
  `obs_weight = clamp(confidence, 0.15, 1.0)`,
  `alpha = clamp(target_alpha * (0.55 + obs_weight*0.65), 0.04, 0.4)`.
* `:1893` `_smooth_damp()` — critically damped spring (Game Programming Gems 4 /
  Unity SmoothDamp): `omega = 2/smooth_time`, rational `exp` approximation,
  returns `(value, velocity)` so the camera has inertia.
* Scene detect via PySceneDetect `AdaptiveDetector`, `--scene-threshold 3.0`,
  `--min-scene-len 15`; "movie" preset locks **one static crop per shot** and
  hard-cuts on scene or subject change.

### clipify (a Claude Code skill — the prompt is the product)

* `SKILL.md` step 1A: "Read the resulting JSON (or `.txt`) and pick 3–5
  candidate clips", with an explicit **signal list** ("punchlines and reactions:
  'what', 'wait', 'no way', laughter", "reversal moments", "awkward pauses",
  "self-roast / quotable one-liners", "audio peaks") and an explicit **candidate
  schema**: `[start, end, why-it's-funny, suggested title]`, "Aim for 10–25 s".
* Step 1.5 "Verify each candidate is in frame": sample one frame per candidate
  and *look at it*, with drop criteria (subject bent down / off-frame / thumb on
  lens / different sound source). "Cheap insurance" — a human-in-the-loop
  verification gate implemented by the agent, not the tool.
* Step 4a: eyeball each face's mouth+chin as `x,y,w,h` on one sample frame, draw
  boxes with `drawbox` to verify, "Iterate at most twice. Don't over-tune".
* Speaker detection: no face model — per-frame motion energy in each mouth ROI
  via `tblend=all_mode=difference,signalstats` → `YAVG`; `analyze.py` merges
  short interjections with **min dwell 1.0 s**.
* `scripts/build_ass.py`: presets `opus` (Arial Black, **size 100** at
  1080×1920, **3 words per chunk**, active-word `&H0000FFFF&` = yellow,
  outline 8, shadow 3, `Alignment=2`, `MarginV=280`), `karaoke` (size 110,
  4 words, green highlight), `minimal` (Helvetica 70, 6 words, no highlight).
  One ASS event per word: text re-emitted with the active word recoloured, so
  the cue switches on every word onset.

### shortcast (code)

* `Shortcast/Services/VideoOverlayRenderer.swift:91-112`: hook font
  `max(24, w * 0.058)`, padding `0.55 * fontSize`, band centred,
  **top margin 8.5 % of height**; fade animation.
* `Shortcast/Resources/social-content-coach.md`: platform copy contracts
  (TikTok hook ≤ 90 chars, 1–2 line caption body, 3–6 hashtags; IG 2–4 short
  paragraphs, 20–30 hashtags; Shorts title 40–60 chars).

### Remotion

* `@remotion/captions` `createTikTokStyleCaptions({captions,
  combineTokensWithinMilliseconds: 1200})` — groups word tokens into "pages"
  (the TikTok look), returns `{text, startMs, durationMs, tokens:[{text,fromMs,toMs}]}`.
  A low combine window ⇒ word-by-word animation; high ⇒ many words per page.
  "spaces must be included in the `text` field before each word" or the page
  merges into one line.
* `interpolate()` exposes `extrapolateLeft` / `extrapolateRight` (`clamp`) so a
  frame outside the input range cannot overshoot.
* `spring()` exposes `damping`, `stiffness`, `mass`, and can overshoot.

### whisperX / ctc-forced-aligner / aeneas

* whisperX `whisperx/alignment.py`: wav2vec2 CTC align model per language
  (`DEFAULT_ALIGN_MODELS_TORCH/HF`), `return_char_alignments` for CJK, a
  **wildcard column = max non-blank score per frame** to absorb unmatched audio,
  per-char `score` = path score, word score = **mean of its char scores**
  (`:392`), output `{"segments": …, "word_segments": …}`. Transcription is
  VAD-chunked before alignment.
* ctc-forced-aligner README: granularity `sentence | word | char`;
  `--window_size 30` s audio chunks with `--context_size 2` s overlap;
  `get_alignments` returns `segments, scores, blank_token`, spans derived from
  the starred token sequence; i.e. **fixed overlapping windows, not VAD**, and a
  per-span CTC score as confidence.
* aeneas: MFCC + **DTW** against a TTS-synthesised reading of the text
  (eSpeak default), multilevel recursive alignment (paragraph → sentence →
  word), "robust against misspelled/mispronounced words, local rearrangements,
  background noise"; word-level is explicitly *not* its design target; no
  confidence output.

## 4. Our own measurements (this checkout, 2026-10-24 tree)

### Caption geometry, burned output of the reference run

`delivery/06_v2_real_footage_16x9/final.mp4` (1920×920), cue at t=3.72 s and
t=6.82 s, measured by thresholding `>232` luma and clustering rows:

* caption ink band **y = 761…789 → 29 px = 3.15 % of H**
* **bottom margin = 14.1 % of H** (not the 5.2 % default — the OCR-obstacle lift
  in `render2.render()` raised it)
* ink width 32 % of W on a long single-line CJK cue
* computed from code: `theme2.reference_height = max(640, min(920, 0.62*1920)) = 920`;
  `CaptionTrack.size = 0.034*920 = 31 px`; `pitch = 1.46*31 = 46 px` (5.0 % H).

At 1080×1920 the same formula gives `ref = 669.6`, `size = 23 px` = **1.19 % of
H / 2.1 % of W** — i.e. the reference landscape caption is already ~2.4× smaller
than BBC's landscape floor (7 % × 920 = 64 px line height), and the portrait
path would be ~3.5× smaller than BBC's 9:16 floor.

### 9:16 layout evidence (portrait demo IR)

`artifacts/v2/demo_portrait/motion_ir.json` (1080×1920, 2026-10-24 23:19):
keyword elements are placed at `box = [916, 755, 116, 236]` with lines
`['最难', '的地', '方']` and `['其实', '是节', '奏']` — a 116 px text area,
2 characters per line, 3 lines. All six elements use `scrim_right`.
Computed: policy `type.ratio.display = 0.078` × `ref 669.6` = **52 px per CJK
glyph**, so a 116 px box fits 2 glyphs; a 4-glyph keyword needs 208 px.
`DesignConfig.min_col_ratio = 0.19` (205 px) admits a column that cannot hold the
type, and `design2.Designer.candidates()` additionally has a canvas-independent
`>= 180` px pixel floor.

### QA of that run (for the record)

`delivery/06_v2_real_footage_16x9/qa_metrics.json`: overall `PASS`,
`legibility.min_contrast = 7.28` (worst element 7.28, best 12.1),
`contrast_vs_source` 1.9–2.0 (i.e. the *source* under the type is bright; the
legibility comes from ink+shadow, not a scrim), person occlusion 0.

### Where our numbers live

`policy.py` `FALLBACK` is the single table of taste: budget (4 per 10 s, gap
0.85 s, coverage 0.58), layout (`min_col_ratio 0.19`, `min_band_ratio 0.115`,
weights), type ratios (display 0.078 … meta 0.021), motion (enter 0.42 s
`out_expo`, exit 0.18 s `in_quad`, stagger 0.075), captions (`bottom_ratio 0.052`,
`size_ratio 0.034`, `line_factor 1.46`), qa (contrast 4.5, margin 0.025).
`captions.DEFAULT_STYLE` duplicates the caption numbers — the two copies can
drift (they already differ in naming: `target_chars_cjk 22` vs `target_chars_per_line 22`).

## 5. Commands worth re-running

```bash
# caption ink measurement (see §4)
ffmpeg -v error -ss 3.72 -i delivery/06_v2_real_footage_16x9/final.mp4 \
  -frames:v 1 -f image2pipe -vcodec png - | python3 -c "…"   # threshold >232, cluster rows

# which frame does _quietness actually look at?
sed -n '367,390p' src/semotion/design2.py    # self.frames[:5]; no t0/t1 argument
sed -n '436,441p' src/semotion/pipeline.py    # frames sampled at 8/25/45/65/85 % of clip

# portrait geometry
python3 -c "import json;d=json.load(open('artifacts/v2/demo_portrait/motion_ir.json'));
print([(e['type'],e['box'],e['fit']['lines']) for e in d['elements']])"
```
