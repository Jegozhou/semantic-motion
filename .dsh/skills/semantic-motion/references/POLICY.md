# Policy reference (`semotion.policy/1`)

`policies/smart_balanced.json` is the whole design envelope. Nothing here decides
*content*; it bounds and weights the deterministic solver. Every block lists its sources
in `meta.provenance` (Material motion, IBM Carbon, Apple HIG, BBC subtitles, Netflix timed
text, WCAG 2.2, movis, clipfix/auto-vertical-reframe, WhisperX).

```bash
python -m semotion.pipeline run ... --policy my_policy.json
```

Deep-merges over the built-in defaults; the resolved policy is recorded in
`motion_ir.json → provenance.policy` (`name`, `version`, `sha256`) and `run_summary.json`.
`policy.validate()` fails closed on values that would break readability.

| key | default | effect |
|---|---|---|
| `budget.max_elements_per_10s` | 4 | density ceiling |
| `budget.min_gap_s` | 0.85 | minimum separation between elements |
| `budget.max_coverage` | 0.58 | maximum share of runtime with an overlay |
| `budget.hold_s` | 0.45 | how long an element lingers after its last word |
| `budget.min_dwell_s` | 1.35 | minimum readable duration |
| `budget.caption_band` | 0.14 | reserved bottom strip for captions |
| `layout.min_margin_px` / `margin_ratio` | 48 / 0.03 | safe margins (48 px wins on 920-high) |
| `layout.min_col_ratio` / `min_band_ratio` | 0.19 / 0.115 | when a side column / band is usable |
| `layout.type_band` | [0.18, 0.74] | vertical band whose silhouette edge bounds a column |
| `layout.treatment_pref` | per type | editorial preference: scrims > bands > card > torso |
| `layout.weights` | quiet .40 / consistency .10 / fidelity .08 / face .06 | scoring weights |
| `layout.scrim.alpha` | 0.86 | scrim strength at its dense edge |
| `type.ratio` | display .078 … | optical type scale vs reference height |
| `type.max_lines` | keyword 2, card 2 | line caps |
| `motion.enter_s` / `exit_s` / `stagger_s` / `preroll_s` | .42 / .18 / .075 / .08 | timing |
| `motion.ink_floor` | 0.80 | opacity on the anchor frame |
| `captions.*` | 2 lines, 22/24 CJK chars, 0.9 s | caption rules |
| `qa.contrast_min` / `safe_margin_frac` / `timing_tolerance_frames` | 4.5 / .025 / 1 | gates |

## Overriding for one project

```json
{"name": "punchy_vertical", "budget": {"max_elements_per_10s": 6, "hold_s": 0.3},
 "type": {"ratio": {"display": 0.085}}, "motion": {"enter_s": 0.3}}
```

Re-render the same clip with two policies and diff `elements[].box/window/fit` plus
`qa_metrics.json` — that is the A/B test.
