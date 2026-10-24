# Forced alignment: acoustic CTC vs the envelope DP

Status: **implemented, measured, not wired**. `src/semotion/forced.py` is a new
aligner beside `align.py`; nothing shared was modified. This document reports
what it does, what was measured, and which aligner to use when — with every
number labelled by its truth class.

Two truth classes exist in this repo and they are **never mixed**:

| label | what it is | what it can prove |
|---|---|---|
| `SYNTH_KNOWN` | 10 clips in `tests/fixtures/clips/*` built by concatenating per-unit macOS TTS on a timeline measured with ffprobe (`ground_truth.json`) | unit-start placement only; the "truth" is a splice point in synthetic audio |
| `REAL_UNVERIFIED` | `inputs/zhengqinwen_25s.mp4` + the cached Whisper transcript | confidence, monotonicity, agreement, posterior separation — **not accuracy**; there are no human labels |

**There is no human-labelled real-speech set in this repo, so no number below
is real-human accuracy.** The synthetic numbers are honestly better for the
envelope aligner *by construction*: every synthetic unit starts at a spliced
energy jump, which is exactly what a spectral-flux envelope detects.

---

## 1. Method

`forced.py` produces the same `semotion.alignment/1` document as `align.py`,
with `method.name = "ctc-forced"` and the same `chars[] / words[] / segments[] /
quality{}` keys, so a consumer can switch engines without changes.

Pipeline:

1. **Audio** — 16 kHz mono through `align.load_audio` (ffmpeg), so both
   aligners see identical samples; VAD/speech islands come from
   `align.audio_features` (same log-mel + energy hysteresis).
2. **Targets** — the transcript is tokenised with `align.tokenize_text`
   (one Han character = one syllable; numerals expand to their spoken Mandarin
   characters; Latin words in Chinese text have no honest target). Each target
   is looked up in the CTC vocabulary, with **OpenCC t2s** as a lookup fallback
   (several fixtures are Traditional Chinese; see §7).
3. **Posteriors** — a CTC acoustic model runs over the audio and returns frame
   log-posteriors at 20 ms stride.
4. **Monotonic CTC Viterbi** — the known transcript is forced through the
   posteriors with `torchaudio.functional.forced_align` when torchaudio is
   installed (it is here), `merge_tokens` for the spans, and an independent
   NumPy implementation (`forced.ctc_viterbi`) as the fallback. Both are
   covered by a test that asserts they agree within one frame.
5. **Chunking** — speech islands are merged across gaps < 0.4 s, split at local
   energy minima when longer than `max_chunk_s` (default 20 s), and a chunk too
   short for its tokens (`T < 2N+1`) is merged forward instead of being
   stretched. Token allocation is per ASR segment, proportional to overlap.
6. **Assembly** — per-target spans become characters (numeral/Latin spans are
   distributed over their source characters and marked `approx`), words,
   segments, `quality{}` and warnings. Confidence is the CTC posterior of the
   chosen state (mean log-probability, `exp`-ed), not a calibrated probability.

Character span convention: a symbol runs from its first emission frame to the
next symbol's first emission frame (the trailing blanks belong to it), and the
last symbol of a chunk runs to the end of that speech island. Without this
chaining every syllable would be reported as 20 ms.

## 2. Model choice

| candidate | what it is | outcome |
|---|---|---|
| `torchaudio.pipelines.MMS_FA` | 29-symbol **English/IPA phone** CTC model (315.5 M params) | **evaluated, not used**: no Mandarin phones, and mapping English text to its labels needs an espeak-style G2P that is not installed. Download and inference both work (`forced.mms_fa_probe()`), but it cannot align this project's material. |
| `facebook/wav2vec2-lv-60-espeak-cv-ft` | English phoneme CTC | same G2P problem |
| **`jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`** | XLSR-53 fine-tuned on Chinese, **character** CTC, 3503 symbols, 319.0 M params | **shipped zh engine**: one symbol = one Han character = one syllable, exactly the anchor unit of this project. It is *not* a phoneme model. |
| **`facebook/wav2vec2-base-960h`** | English **letter** CTC | shipped en engine; no G2P, no fixture measured (see §5) |

So the requested "preferred MMS_FA" path is not the shipped path, for a
substantive reason: MMS_FA cannot align Mandarin at all. The HuggingFace CTC
fallback is what runs, and it is a real acoustic model — not the envelope.

Cache: `HF_HOME` defaults to `<repo>/.cache/huggingface`
(`SEMOTION_HF_HOME` overrides; existing weights are also found under `HF_HOME`
read-only). Loading is by local snapshot path, so it works offline:
`SEMOTION_FORCED_OFFLINE=1` forbids network access and a missing cache raises
`E_ALIGN_MODEL_MISSING` instead of downloading or guessing.

## 3. Install footprint

| item | size |
|---|---|
| `torch==2.14.1` wheel (macOS arm64, CPU) | 121.4 MiB download |
| `transformers==5.20.2` | 12.2 MiB download (+ safetensors, tokenizers, regex, sympy, networkx, jinja2…) |
| zh model (`jonatasgrosman/…chinese-zh-cn`) | **1.276 GB** (`pytorch_model.bin`, 319 M params) |
| en model (`facebook/wav2vec2-base-960h`) | **0.755 GB** on disk (safetensors + pytorch bin; 0.38 GB is enough) |
| `MMS_FA` weights (evaluated, unused) | 1.262 GB under `.cache/torch` |
| HF cache total | 2.41 GB |
| uv download cache | 1.13 GB |

Everything is inside the repo (`.cache/`), nothing in `$HOME`.

## 4. Measured — `SYNTH_KNOWN` (TTS splice truth, n = 147 units, 10 fixtures)

Metric: nearest predicted character onset vs the unit's assembly `start_s`
(the same rule as `align.score_unit_onsets`), absolute error in ms.

| method | n | P50 | P90 | max | mean |
|---|---|---|---|---|---|
| envelope DP (`align.py`) | 147 | **2.75** | **7.56** | 108.17 | 5.25 |
| CTC forced (`forced.py`) | 147 | 17.04 | 72.81 | 258.63 | 30.29 |

The envelope aligner wins P50 on **10/10** fixtures. Per fixture (P50/P90, ms):

| fixture | energy | CTC | CTC mean conf |
|---|---|---|---|
| fx_a_zh_portrait | 4.47 / 7.04 | 14.54 / 34.06 | 0.77 |
| fx_b_zh_landscape | 3.79 / 6.74 | 10.69 / 74.47 | 0.86 |
| fx_c_zh_cuts | 1.25 / 6.17 | 9.40 / 33.13 | 0.83 |
| fx_d_zh_portrait | 0.80 / 7.50 | 34.75 / 91.53 | 0.74 |
| fx_e_zh_landscape | 2.73 / 7.31 | 7.33 / 80.85 | 0.75 |
| fx_face_pan | 4.44 / 6.02 | 15.50 / 45.11 | 0.72 |
| fx_face_zh_portrait | 3.19 / 7.37 | 9.53 / 65.42 | 0.67 |
| fx_t1_zh_phrase | 3.11 / 7.61 | 23.80 / 83.84 | 0.62 |
| fx_t2_zh_phrase | 3.72 / 7.63 | 19.23 / 69.70 | 0.66 |
| fx_t3_zh_phrase | 3.38 / 7.89 | 24.98 / 65.78 | 0.70 |

**Do not read this as "CTC is worse at syllable onsets".** The synthetic truth
is a splice point, i.e. an abrupt energy jump; the envelope aligner is built to
land on exactly that jump, and a CTC symbol starts at the acoustic realisation
of the character (vowel onset), typically tens of milliseconds after the
splice. On this material the comparison mostly measures the metric's built-in
bias toward envelope peaks. Runtime: energy 0.07–0.12 s per clip, CTC
0.97–2.24 s per clip after a 1.7–2.9 s cold model load (CPU, 4 threads).

## 5. Measured — `REAL_UNVERIFIED` (25.0 s, 10 segments, 123 characters)

### 5.1 CTC confidence

| metric | value |
|---|---|
| per-character posterior, mean | 0.731 |
| median | 0.966 |
| p10 | 0.002 |
| share below 0.35 (`low_conf_rate`) | 23.6 % |
| path score per frame (chunked / single pass) | −0.277 / −0.084 |
| coverage (transcript symbols with a CTC target) | 1.000 |
| monotonicity violations | 0 |
| median syllable | 140 ms (energy: 150 ms) |
| runtime, 25 s clip | 1.21 s (6 chunks) + 2.68 s cold load |

The bimodal distribution is real: confident characters (0.96+) are syllables
the model recognises cleanly, while the low tail is where the transcript and
the acoustics disagree or a syllable is reduced. Confidence is **not**
calibrated against human judgements.

### 5.2 Agreement with the envelope aligner

| metric | value |
|---|---|
| matched characters | 123 / 123 |
| onset \|Δ\| median | 150 ms |
| onset \|Δ\| p90 | 318 ms |
| onset within 50 ms | 22.0 % |
| onset within 100 ms | 36.6 % |
| offset \|Δ\| median | 150 ms |
| confidence vs agreement (Spearman) | 0.003 |

Agreement is **not** accuracy, and a 150 ms median is large enough that the two
methods cannot both be right. Which one is more plausible is testable with a
third, independent-of-the-DP timing signal: Whisper's word times
(`MACHINE_WEAK`, cross-attention based, not used by either aligner):

| against Whisper word times (n = 98 words) | energy DP | CTC |
|---|---|---|
| word-start median \|Δ\| | 150 ms | **50 ms** |
| word-start p90 \|Δ\| | 390 ms | **132 ms** |
| word-start within 100 ms | 34.7 % | **83.7 %** |
| word-end median \|Δ\| | 160 ms | **60 ms** |

The pattern explains itself: signed Δ (CTC − energy) within a segment is
**+50 ms in the first half and −70 ms in the second half**. The envelope
aligner spreads characters across the whole Whisper segment, including its
padded edges; CTC follows the acoustics and finishes the segment earlier. The
Whisper transcript's greedy CTC decode matches the text closely (e.g. 20/20
characters on segment `s1`), so the model is hearing this speaker, not sliding.

Caveat: Whisper word times come from the same ASR that produced the transcript
both aligners consume; this is corroborating evidence, not truth.

### 5.3 Negative control

Same posteriors, same audio, character order shuffled **within each segment**
(same symbol multiset), single 25 s pass:

| | correct order | shuffled |
|---|---|---|
| path score per frame | −0.084 | **−1.195** |
| mean confidence | 0.809 | **0.289** |
| median confidence | 0.988 | **0.000** |

A method that ignored the transcript would score the same on both. It does not,
so the score has real discriminative power.

### 5.4 Degenerate cases

* **Repeated characters** — `远远` (segment `s2`) is aligned to two distinct,
  strictly ordered spans (6.80–6.96 s, 6.96–7.12 s). This is why the module
  does not use the old `merge_tokens(tokens, targets)` API, which collapses
  repeated labels into one span.
* **Out-of-vocabulary symbols** — none (coverage 1.0). OOV characters are
  reported, their tokens marked `approx` and interpolated, and the document is
  **refused** (`E_ALIGN_MODEL_UNSUPPORTED`) when the vocabulary would drop more
  than 15 % of the transcript.
* **Very short characters** — 5 of 123 characters are ≤ 2 frames (40 ms), where
  the model compresses fast syllables; max character length 660 ms.
* **Infeasible chunks** — none (`chunk_feasible: true`); a chunk too short for
  its tokens is merged forward, and a genuinely impossible chunk would be
  reported (and uniformly split) rather than silently produced.
* **No CTC targets** — a transcript that is entirely punctuation/Latin gets a
  clear `E_ALIGN_MODEL_UNSUPPORTED`, never an empty alignment.

## 6. Recommendation

| situation | use |
|---|---|
| **Chinese real human speech** (the product case) | **`forced.py` (CTC)** as the primary aligner. It is the only one of the two that tracks acoustic word boundaries on real audio; the envelope aligner's synthetic advantage does not transfer. |
| Model not cached / no torch / zipped distribution | `align.py` (envelope DP). `forced.align_video(..., allow_fallback=True)` does this automatically and the document's `method.name` tells you which ran. |
| Non-speech audio, music, no transcript | envelope DP — CTC needs speech and a text hypothesis. |
| Very short clips (< ~0.5 s) or a transcript that is mostly symbols | envelope DP; CTC will have too few frames/targets and refuses. |
| Clean TTS with known splice timing (fixtures, regression) | envelope DP — it is near-exact there by construction and ~10–30× faster. |
| **English** | the letter-CTC path is implemented (`facebook/wav2vec2-base-960h`) but **UNMEASURED**: there is no English fixture in this repo and no human-labelled English audio. Do not present it as validated. |
| MMS_FA phonemes | not usable here: English phones only, no Mandarin support, no G2P installed. |

Suggested integration policy: run CTC, and if `method.name != "ctc-forced"`
(fallback fired) or `quality.posterior.low_conf_rate > 0.5` or
`quality.chunk_feasible == false`, treat the file as low-confidence and let the
pipeline degrade gracefully. Do not re-run the envelope aligner and pick the
"better-looking" one — there is no truth to choose with.

Known bias to keep in mind: character CTC onsets land on the character's
acoustic realisation (roughly the vowel), so they are typically **0–60 ms later**
than a consonant attack. For overlays that is well inside tolerance; for
tight audio-visual cuts, prefer `standalone CTC onset − 20 ms` only if measured
on your own material (this document did not measure that correction, so it is
not shipped).

## 7. What did not work (and what it cost)

1. **MMS_FA for Chinese.** `torchaudio.pipelines.MMS_FA` downloads (1.18 GB) and
   runs, but its label set is 29 English/IPA phones (`a i e n o u …`), so it
   cannot align Mandarin. For English it would need an espeak-style G2P, which
   is not installed. It is kept only as `forced.mms_fa_probe()` evidence.
2. **Traditional vs Simplified Chinese.** Four of ten fixtures are Traditional
   (`鏡頭開始移動`, `九成的人會拖延`), while the zh model vocabulary is
   Simplified: 31–39 % of their characters were OOV, which dragged aggregate CTC
   P90 to 208 ms and produced 450 ms errors on single units. Fixed with an
   OpenCC `t2s` lookup key (emitted text stays Traditional); coverage is now
   97–100 % and P90 fell to 72.8 ms.
3. **`torchaudio.functional.merge_tokens(tokens, targets)`** (older API) merges
   repeated labels, which would silently merge `谢,谢` into one character. The
   module derives spans from the alignment instead; with torchaudio 2.11 the API
   changed to `merge_tokens(per-frame tokens, per-frame scores)` and repeats are
   blank-separated, so both paths are handled and tested.
4. **CTC emission spans are 1–3 frames**, so reporting them directly made every
   syllable 20 ms. Spans are now chained to the next symbol's start.
5. **HuggingFace cache path.** Passing `cache_dir=<HF_HOME>` makes the hub look
   for `<HF_HOME>/models--…` instead of `<HF_HOME>/hub/models--…`, so the first
   offline load failed. Loading from the resolved local snapshot path fixed it.
6. **The first 403s from `dl.fbaipublicfiles.com`** were a wrong URL, not a
   blocked network: the real MMS_FA weight URL (`…/ctc_alignment_mling_uroman/
   model.pt`) downloads at ~2–4 MB/s here.
7. **The envelope aligner on real speech.** It is fast and deterministic, but it
   pads characters to Whisper segment bounds (§5.2), which is why it disagrees
   with both CTC and Whisper word times. Its synthetic P90 of 7.56 ms should not
   be quoted as real-speech performance.

Not attempted (would need human labels): a hybrid that takes CTC's ordering and
snaps each onset back to the energy rise; it is the obvious next step, and it
would need a real-speech labelled set to justify.

## 8. Adapter signature

`forced.py` mirrors `align.py`'s entry points so the pipeline can switch with a
flag:

```python
from semotion import forced

# exactly the shape of align.align_video / align.align_transcript
doc = forced.align_video(video_path, transcript, language="zh")
doc = forced.align_transcript(transcript, samples=samples, language="zh")

# optional knobs (all keyword-only)
doc = forced.align_video(
    video_path, transcript, language="zh",
    params={"max_chunk_s": 20.0},   # chunking / VAD / confidence thresholds
    model_id=None,                  # override the HF model
    device="cpu",
    allow_download=False,           # offline: only cached weights
    allow_fallback=True,            # run align.py if the CTC model is missing
)

# capability check before choosing a path
forced.available("zh")              # weights cached? (no download, no load)
forced.model_info("zh")             # cache path, bytes, snapshot, offline env
forced.mms_fa_probe()               # why MMS_FA is not used

# documents
forced.save(doc, path); forced.load(path)
forced.validate(doc)                # align.validate + CTC keys
```

Returned document: `schema == "semotion.alignment/1"`,
`method.name == "ctc-forced"`, `method.model`, `method.device`,
`method.model_info`, `method.runtime_s` (load/align/total), plus
`quality.posterior`, `quality.coverage` (`script_converted`, `oov`,
`approx_tokens`), `quality.n_chunks`, `quality.chunk_feasible`,
`quality.alignment_score_per_frame` and `quality.degenerate_targets`.
It passes `align.validate()` unchanged. When `allow_fallback=True` fires, the
document is the envelope one (`method.name == "envelope-dp/1"`).

Lower-level, model-free API for tests or a different CTC model:

```python
forced.align_posteriors(transcript, log_probs, vocab, blank=0,
                        frame_stride_s=0.02, language="zh", params={...})
forced.build_targets(tokens, vocab, language="zh")   # text -> CTC targets
forced.ctc_viterbi(log_probs, targets, blank=0)      # NumPy CTC DP
```

## 9. Tests and evidence

* `tests/test_forced.py` — 39 tests, ~4 s, **all green** (286 tests in the
  whole suite pass). Covers schema
  compatibility (`align.validate`), monotonicity, Viterbi recovery, repeated
  characters, infeasible audio, phone/character→source mapping, numeral
  expansion vs `align.number_syllables`, Traditional→Simplified lookup,
  OOV refusal, chunking equivalence, negative control, torchaudio/NumPy engine
  agreement, offline refusal. The real-model test runs against `fx_a_zh_portrait`
  when the weights are cached and **skips** when they are not (it never
  downloads).
* `artifacts/p3/forced/evidence.json` — every number above, per fixture, with
  truth classes, environment, model/cache footprint and caveats.
* `artifacts/p3/forced/timeline.png` — Panel A: synthetic unit bars, both
  methods' onsets, per-unit error. Panel B: real clip, segments, both onset
  rows (CTC bar height = confidence), per-character disagreement and the
  negative-control summary.
* `artifacts/p3/forced/fx_a_{energy,ctc}.json`, `real_{energy,ctc}.json` —
  full documents for inspection.

Reproduce (models already cached; offline):

```bash
cd "/Volumes/ZHITAI/Semantic Motion"
export PYTHONPATH=src HF_HOME="$PWD/.cache/huggingface" \
       TORCH_HOME="$PWD/.cache/torch" SEMOTION_FORCED_OFFLINE=1
.venv/bin/python -m pytest tests/test_forced.py -q
.venv/bin/python artifacts/p3/forced/run_bench.py     # ~3 min, CPU
```

Re-download (once, network):

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
uv pip install --python .venv/bin/python torch torchaudio transformers
export HF_HOME="$PWD/.cache/huggingface"
.venv/bin/hf download jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn \
    --exclude flax_model.msgpack
.venv/bin/hf download facebook/wav2vec2-base-960h
```
