# Semantic Motion v2.1 — research-driven hardening

This addendum records the work done after v2 shipped: a distilled review of open-source
engines and broadcast standards, the model-driven director, an editable design policy, a
real acoustic aligner, and the fixes those surfaced. Everything below is measured on the
artifacts in `delivery/` and `artifacts/`.

## 1. What was researched

`docs/BEST-PRACTICES.md` (721 lines, 17 sources read at file level): movis (keyframe/easing
engine, draft levels, caches), clipify (host-agent moment selection), shortcast,
auto-vertical-reframe (silhouette tracking, dead zone), whisperX / ctc-forced-aligner /
aeneas (forced alignment), Remotion / Revideo / Motion Canvas (composition + determinism),
Material 3 / IBM Carbon / Apple HIG (motion), BBC Subtitle Guidelines / Netflix Timed Text
(captions), WCAG 2.2. Twenty ranked recommendations, each tied to a symbol in this repo;
`docs/IMPLEMENTED-FROM-RESEARCH.md` tracks which landed and why the rest are deferred.

Honest negatives were kept: TikTok publishes no citable caption spec, shortcast has no
burned-in caption engine, and aeneas solves a problem we do not have.

## 2. Judgement moved out of code

**Direction is now a model decision.** `semotion.director` builds a factual brief
(candidate anchors with measured times, features, context, speaker side, timing
confidence), and either a configured OpenAI-compatible model or the host agent chooses what
deserves motion. The brief's output schema contains no timestamps, and
`validate_decisions()` rejects unknown anchors, out-of-order lists, re-claimed anchors,
emphasis strings that are not in the text, **numbers the speaker did not say**, and holds
outside the envelope. Rules remain a labelled fallback (`--director rules`).

**Design numbers are data.** `policies/smart_balanced.json` holds the budget, margins,
column/band ratios, treatment preferences, scoring weights, type scale, motion durations,
caption rules and QA thresholds, with provenance links. The resolved policy hash is
recorded in the IR; `--policy` deep-merges overrides; validation fails closed. The refactor
was verified behaviour-preserving: the same clip produced byte-identical element geometry
and fit before/after.

## 3. Real acoustic alignment

`semotion.forced` implements CTC forced alignment (XLSR-53 character CTC for zh; letter
CTC for en) and the pipeline defaults to `--aligner auto` (CTC when the model is cached,
otherwise the envelope DP).

| truth | method | n | P50 | P90 |
|---|---|---|---|---|
| SYNTH_KNOWN (TTS unit starts) | envelope DP | 147 | 2.75 ms | **7.56 ms** |
| SYNTH_KNOWN (same) | CTC | 147 | 17.04 ms | 72.81 ms |
| REAL_UNVERIFIED (agreement with Whisper word starts) | CTC | 98 | **50 ms** | **132 ms** (83.7 % ≤100 ms) |
| REAL_UNVERIFIED (same) | envelope DP | 98 | 150 ms | 390 ms (34.7 % ≤100 ms) |

The synthetic truth is a splice point (an energy jump), so it structurally favours the
envelope method; the real-speech third opinion favours CTC. No human labels exist, so
neither is claimed as accuracy. Negative control: shuffled characters drop the CTC
path score from −0.084 to −1.195 per frame and confidence 0.809 → 0.289.

## 4. Fixes the research surfaced (with evidence)

| fix | before | after |
|---|---|---|
| Caption type scale (BBC 9.2) | 34 px / 3.15 % H landscape, 23 px portrait | pitch 7.5 % H landscape → **ink 4.89 % H**; portrait 4.8 % H → **ink 3.07 % H** |
| Caption safe area (BBC 9.3) | flat 5.2 % H | 5.2 % landscape / **13 % portrait**, obstacle lift capped at +6 % |
| Caption line budget | 24 chars regardless of size | derived from canvas + font: **15 chars/line portrait**, 24 landscape (20-char cue wraps to 2 lines) |
| Type-aware columns | any column ≥180 px accepted (portrait got 116 px / 2-glyph lines) | must hold ≥3.2 display glyphs; portrait falls back to a band |
| Windows-local quietness | five clip-wide frames | 5 Hz grey frames inside each element's window |
| Fit must fit | oversized fits were height-clamped and spilled out of the scrim | fit fails → smaller tier or another treatment; boxes are confined to the candidate area |
| Band contrast | gradient faded through the text area | holds 86 % to 72 % of the band; portrait legibility 2.33 → **5.84** |
| Decision params | `enter` / `accent` accepted then ignored | applied (`scale_in→scale`, `spring_pop→scale`, `#F2B233` per element) and recorded requested-vs-applied |
| Caption QA | nothing measured captions | `caption_typography` gate (ink height, lines, width, bottom margin, face distance) |
| Non-text contrast | unchecked | WCAG 1.4.11 floor 3.0 reported (advisory) |

## 5. Final measured state

**Landscape (real interview, CC BY, 1920×920, 25 s): 13/13 PASS.**
person/face/hand 0 px · protected text 0 px · legibility min **7.24** · text onset Δ0 on all
five elements · caption ink 4.89 % H, width 48.8 % W, bottom 11.96 % H, 0 caption/face
intersections · margins 48 px · determinism **625/625 frames, mean |Δ| 0.0**.

**Portrait (real 1080×1920 crop): 12 PASS + 1 advisory FAIL.**
legibility 5.84 · captions ink 3.07 % H, width ≤81 % W, bottom 13.5 % H, 2-line cue ·
person 0 px, text 0 px, timing 0 frames. `non_text_graphics` is advisory and fails because
the gold accent rule sits on a bright wall (1.12–2.28); it is reported, not hidden, and does
not affect the verdict.

Deliverables: `delivery/06_v2_real_footage_16x9/` and `delivery/07_v2_portrait_9x16/`.

## 6. Remaining limitations

1. No human-labelled corpus: every timing number is SYNTH_KNOWN or method agreement.
2. The CTC model is CJK-only in practice (English path implemented, unmeasured); 1.3 GB
   download, CPU 1.2 s for 25 s audio.
3. The deterministic rules still split some phrases at acoustic pauses; the model/host
   path is the intended one.
4. Layout treatment choice is deterministic (policy-weighted) and has not been judged by a
   vision model; the candidate table makes that the next step.
5. `non_text_graphics` on bright backgrounds can fail; no automatic accent darkening yet.
