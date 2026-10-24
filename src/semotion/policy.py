"""Design policy: every judgement number in one editable, attributable file.

The renderer used to carry its thresholds as module constants. That made taste
invisible and untestable: "how many elements per 10 s" was a literal buried in
``design2.py``. A policy moves those numbers into ``policies/*.json`` with
provenance, so they can be reviewed, diffed, A/B tested and overridden per
project without editing code — and so a model can be asked to choose within an
explicit envelope instead of guessing.

Nothing here decides *content*; it only bounds and weights the deterministic
solver. The creative decision (which words deserve a graphic) belongs to the
director (``semotion.director``) or the host agent.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "semotion.policy/1"
ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = ROOT / "policies"
DEFAULT_NAME = "smart_balanced"

# Embedded fallback: identical to policies/smart_balanced.json. Kept here so the
# package still works from a wheel without the repository layout.
FALLBACK: dict[str, Any] = {
    "schema": SCHEMA,
    "name": DEFAULT_NAME,
    "version": "1.0.0",
    "budget": {"max_elements_per_10s": 4, "min_gap_s": 0.85, "max_coverage": 0.58,
               "hold_s": 0.45, "min_dwell_s": 1.35, "caption_band": 0.14},
    "layout": {
        "margin_ratio": 0.03, "min_margin_px": 48, "min_col_ratio": 0.19,
        "min_band_ratio": 0.115, "allow_torso": True, "text_conf_min": 0.5,
        "type_band": [0.18, 0.74],
        "treatment_pref": {
            "keyword": {"band_top": 0.10, "scrim_right": 0.12, "scrim_left": 0.10,
                        "band_bottom": 0.08, "card": 0.04, "lower_third": 0.0},
            "card": {"scrim_right": 0.12, "scrim_left": 0.10, "card": 0.08,
                     "band_top": 0.05, "band_bottom": 0.04, "lower_third": 0.0},
            "list": {"scrim_right": 0.12, "scrim_left": 0.10, "card": 0.08,
                     "band_top": 0.04, "band_bottom": 0.04, "lower_third": 0.0},
            "stat": {"scrim_right": 0.12, "scrim_left": 0.10, "card": 0.08,
                     "band_top": 0.04, "band_bottom": 0.04, "lower_third": 0.0},
            "barchart": {"card": 0.10, "scrim_right": 0.08, "band_top": 0.05,
                         "lower_third": 0.0},
            "compare": {"card": 0.10, "scrim_right": 0.08, "band_top": 0.05,
                        "lower_third": 0.0},
            "quote": {"band_top": 0.08, "scrim_right": 0.10, "card": 0.05,
                      "lower_third": 0.0},
        },
        "weights": {"quiet": 0.40, "consistency": 0.10, "type_fidelity": 0.08,
                    "face_distance": 0.06, "torso_penalty": 0.18, "scrim_bonus": 0.10},
        "scrim": {"alpha": 0.86, "ramp_soft_px": 46, "vertical_falloff": 0.70},
    },
    "type": {
        "reference": {"min_height": 640, "width_factor": 0.62},
        "ratio": {"display": 0.078, "title": 0.058, "body": 0.030, "meta": 0.021,
                  "stat": 0.150, "stat_unit": 0.052},
        "line_factor": {"display": 1.30, "title": 1.32, "body": 1.52, "meta": 1.45,
                        "stat": 1.0, "stat_unit": 1.0},
        "max_lines": {"keyword": 2, "card": 2, "list_row": 2},
        "min_tier": 0.55,
    },
    "motion": {"enter_s": 0.42, "enter_ease": "out_expo", "exit_s": 0.18,
               "exit_ease": "in_quad", "stagger_s": 0.075, "preroll_s": 0.08,
               "offset_y": 12, "ink_floor": 0.80, "scrim_enter_s": 0.34,
               "scrim_exit_s": 0.22},
    "captions": {"max_lines": 2, "target_chars_cjk": 22, "max_chars_cjk": 24,
                 "max_chars_latin": 42, "min_display_s": 0.9, "merge_gap_s": 0.18,
                 "bottom_ratio": 0.052, "size_ratio": 0.034, "line_factor": 1.46,
                 "respect_segments": True},
    "qa": {"contrast_min": 4.5, "safe_margin_frac": 0.025,
           "timing_tolerance_frames": 1, "person_frame_max": 0.002,
           "person_ratio_max": 0.005, "reliability": 0.95,
           "caption_max_lines": 2, "caption_min_lines": 1,
           "caption_ink_height_landscape": [0.040, 0.060],
           "caption_ink_height_portrait": [0.024, 0.040],
           "caption_bottom_min_landscape": 0.045,
           "caption_bottom_min_portrait": 0.11,
           "caption_width_max": 0.90},
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Policy:
    """Read-only view over a resolved policy document."""

    def __init__(self, doc: dict, *, source: str = "embedded"):
        self.doc = doc
        self.source = source
        self.name = doc.get("name") or DEFAULT_NAME
        self.version = str(doc.get("version") or "0")
        self.sha256 = hashlib.sha256(
            json.dumps(doc, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]

    # ------------------------------------------------------------------ access
    def get(self, path: str, default=None):
        node: Any = self.doc
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, path: str):
        return self.get(path)

    # --------------------------------------------------------------- sections
    @property
    def budget(self) -> dict:
        return self.get("budget", {}) or {}

    @property
    def layout(self) -> dict:
        return self.get("layout", {}) or {}

    @property
    def type(self) -> dict:
        return self.get("type", {}) or {}

    @property
    def motion(self) -> dict:
        return self.get("motion", {}) or {}

    @property
    def captions(self) -> dict:
        return self.get("captions", {}) or {}

    @property
    def qa(self) -> dict:
        return self.get("qa", {}) or {}

    def pref(self, kind: str, treatment: str, default: float = 0.0) -> float:
        table = self.get("layout.treatment_pref", {}) or {}
        return float((table.get(kind) or {}).get(treatment, default))

    def weight(self, name: str, default: float = 0.0) -> float:
        return float((self.get("layout.weights", {}) or {}).get(name, default))

    def as_dict(self) -> dict:
        return {"name": self.name, "version": self.version,
                "sha256": self.sha256, "source": self.source}

    def record(self) -> dict:
        """Provenance block embedded into the Motion IR."""
        return {"policy": self.name, "version": self.version,
                "sha256": self.sha256, "source": self.source,
                "meta": self.get("meta", {}) or {}}


def load(path: str | Path | None = None, *, name: str | None = None,
         overrides: dict | None = None) -> Policy:
    """Load a policy file, falling back to the embedded default.

    ``overrides`` is a partial document deep-merged last (used by tests and by
    project-level tweaks without creating a file).
    """
    if path:
        p = Path(path)
        doc = json.loads(p.read_text(encoding="utf-8"))
        policy = Policy(_deep_merge(FALLBACK, doc), source=str(p))
    else:
        f = POLICY_DIR / f"{name or DEFAULT_NAME}.json"
        if f.exists():
            doc = json.loads(f.read_text(encoding="utf-8"))
            policy = Policy(_deep_merge(FALLBACK, doc), source=str(f))
        else:
            policy = Policy(copy.deepcopy(FALLBACK), source="embedded")
    if overrides:
        policy = Policy(_deep_merge(policy.doc, overrides), source=policy.source)
    validate(policy)
    return policy


def validate(policy: Policy) -> None:
    """Fail closed on a policy that would produce an unreadable or unsafe render."""
    problems: list[str] = []
    b = policy.budget
    if not (0.15 <= float(b.get("hold_s", 0)) <= 3.0):
        problems.append("budget.hold_s must be in [0.15, 3.0]")
    if not (0.3 <= float(b.get("min_gap_s", 0)) <= 5.0):
        problems.append("budget.min_gap_s must be in [0.3, 5.0]")
    if not (1 <= int(b.get("max_elements_per_10s", 0)) <= 10):
        problems.append("budget.max_elements_per_10s must be in [1, 10]")
    if not (0.05 <= float(b.get("max_coverage", 0)) <= 0.95):
        problems.append("budget.max_coverage must be in [0.05, 0.95]")
    if not (0.5 <= float(policy.get("qa.contrast_min", 0)) <= 21):
        problems.append("qa.contrast_min must be in [0.5, 21]")
    if problems:
        raise ValueError("invalid policy: " + "; ".join(problems))


DEFAULT = load()
