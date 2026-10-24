# Model-directed decisions (and why the rules are only a fallback)

The hard part of this product is not rendering. It is deciding **which sentence
deserves a graphic** — and that is an editorial judgement. Semantic Motion therefore
separates two things that used to be mixed:

| layer | who decides | file |
|---|---|---|
| **direction** — what gets motion, how it groups, the wording shown | a language model or the host agent | `director_brief.json` → `decisions.json` |
| **design** — where it goes, type scale, motion, timing | deterministic solver constrained by measurement | `design2.py` + `policies/*.json` |
| **truth** — timestamps | measured once, never chosen | `alignment.json` |

## 1. The brief (facts, no free timestamps)

`--brief-only` stops after perception, alignment and candidate extraction and writes
`out/<name>/director_brief.json`:

```jsonc
{
  "schema": "semotion.director-brief/1",
  "language": "zh",
  "policy": {"budget": {...}, "captions": {...}, "max_element_types": [...]},
  "candidates": [{
    "id": "a01", "text": "但是我还有其他的梦想",
    "start_s": 2.02, "end_s": 3.7, "duration_s": 1.68,
    "features": {"has_number": false, "contrast": true, "chars": 10, ...},
    "context": ["奥运会是我的梦想没错", "但是我还有其他的梦想等着我去实现我的目标"],
    "speaker_side": "centre", "kind_hint": "keyword"
  }],
  "rules": [...], "output_schema": {...}
}
```

The candidate ids are the only legal references. Their times are measured; the brief's
`output_schema` contains no `start_s`/`end_s` at all, so a model cannot invent timing even
if it tries.

## 2. Two ways to decide

**A. Host agent (Claude Code), no API key.** The agent reads the brief, writes
`decisions.json`, and the pipeline validates it. This is the intended Claude path and
what the skill instructions describe.

```bash
python -m semotion.pipeline run --video talk.mp4 --out out/talk --brief-only
# agent reads out/talk/director_brief.json and writes out/talk/decisions.json
python -m semotion.pipeline run --video talk.mp4 --out out/talk \
    --decisions out/talk/decisions.json
```

**B. Any OpenAI-compatible endpoint.** Set the env vars and run with `--director auto`
(uses the model when available) or `--director llm` (fails if the model is unusable).

```bash
export SM_LLM_BASE_URL="https://api.example.com/v1"
export SM_LLM_API_KEY="..."
export SM_LLM_MODEL="..."
python -m semotion.pipeline run --video talk.mp4 --out out/talk --director llm
```

`--director rules` forces the deterministic fallback; `--director off` produces a clean
pass-through render (useful as a control).

## 3. What the model may and may not do

`director.validate_decisions()` rejects, with a reason written to `director_error.json`:

* an anchor id that is not in the brief;
* a list of fewer than two anchors, or anchors out of speech order;
* the same anchor claimed twice;
* an `emphasis` substring that does not occur in the anchor text;
* **a number in an on-screen field that the speaker did not say** (colours and enum
  values are excluded from this scan on purpose);
* a `hold_s` outside the policy envelope.

The model never sets coordinates. The solver may still drop an element if no honest
space exists — a model asking for motion does not override the speaker's safety.

## 4. Policy: the numbers are data now

`policies/smart_balanced.json` holds every default that used to be a constant:
budget (elements per 10 s, coverage, holds), layout (margins, column/band ratios,
treatment preferences, scoring weights), type scale, motion durations/easing, caption
limits, QA thresholds. Each block carries provenance links in `meta.provenance`.

* `--policy path.json` deep-merges over the built-in defaults;
* the resolved policy is recorded in `motion_ir.json → provenance.policy` and in
  `run_summary.json` (`name`, `version`, `sha256`);
* `policy.validate()` fails closed on values that would break readability
  (e.g. coverage > 0.95, contrast < 0.5);
* defaults are covered by tests so a policy edit can be A/B tested by re-rendering the
  same clip and diffing the IR (`elements[].box/window/fit`).

## 5. Where the remaining judgement lives

| judgement | current owner | next step |
|---|---|---|
| which words matter | model / host agent | prompt iteration against a labelled clip set |
| list grouping | model / rules | same |
| treatment choice (scrim vs band vs card) | solver + policy weights | critic loop with vision feedback |
| caption line breaks | aligner + jieba + policy | fine |
| "does it look designed" | human | storyboard review; a vision critic could pre-score |

The critic loop is deliberately not automatic yet: a model that has not seen the frame
cannot judge composition. The supported path today is: render → look at
`storyboard.png` → emit bounded overrides (drop / prefer treatment / retime) → re-run.
