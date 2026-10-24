"""Tests for the model-driven director and the design policy."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semotion import director, policy, theme2  # noqa: E402


def _brief():
    anchors = [
        {"id": "a00", "text": "奥运会是我的梦想没错", "start_s": 0.5, "end_s": 1.9},
        {"id": "a01", "text": "但是我还有其他的梦想", "start_s": 2.02, "end_s": 3.7},
        {"id": "a02", "text": "每一年我们都有比赛", "start_s": 11.18, "end_s": 12.7},
        {"id": "a03", "text": "每一年都有更大的困难", "start_s": 13.04, "end_s": 15.7},
        {"id": "a04", "text": "九成的选手都这么说", "start_s": 16.0, "end_s": 17.2},
    ]
    alignment = {
        "segments": [
            {"id": "s0", "start_s": 0.0, "end_s": 2.0, "text": "奥运会是我的梦想没错"},
            {"id": "s1", "start_s": 2.0, "end_s": 3.8, "text": "但是我还有其他的梦想"},
        ],
        "chars": [
            {"segment_id": "s0", "text": "奥", "start_s": 0.5, "end_s": 0.66},
            {"segment_id": "s1", "text": "但", "start_s": 2.02, "end_s": 2.2},
        ],
    }
    return director.build_brief(
        source={"duration_s": 25.0}, canvas={"width": 1920, "height": 920, "fps": 25.0},
        transcript={"language": "zh", "segments": [{"text": "x"}]},
        alignment=alignment, anchors=anchors, perception=None,
        policy=policy.DEFAULT)


def _doc(decisions, **extra):
    return {"schema": director.DECISIONS_SCHEMA, "decision_model": "test",
            "decisions": decisions, **extra}


# ----------------------------------------------------------------------- brief
def test_brief_is_factual_and_has_no_editable_timestamps_schema():
    brief = _brief()
    assert brief["schema"] == director.BRIEF_SCHEMA
    text = json.dumps(brief["output_schema"], ensure_ascii=False)
    assert "start_s" not in text and "end_s" not in text
    assert [c["id"] for c in brief["candidates"]] == ["a00", "a01", "a02", "a03", "a04"]
    assert brief["candidates"][4]["features"]["has_number"] is True
    assert brief["candidates"][1]["features"]["contrast"] is True
    assert brief["candidates"][2]["features"]["chars"] == 9


def test_brief_context_contains_neighbouring_sentences():
    brief = _brief()
    ctx = brief["candidates"][1]["context"]
    assert any("奥运会" in c for c in ctx)


# ------------------------------------------------------------------ validation
def test_valid_decisions_pass_validation():
    brief = _brief()
    doc = _doc([
        {"anchor_id": "a01", "effect": "keyword", "reason": "转折",
         "params": {"emphasis": ["梦想"]}},
        {"anchor_id": "a02", "anchor_ids": ["a02", "a03"], "effect": "list",
         "params": {"kicker": "要点"}},
        {"anchor_id": "a04", "effect": "stat", "params": {"unit": "%", "label": "选手"}},
    ])
    out = director.validate_decisions(doc, brief)
    assert len(out["decisions"]) == 3
    assert out["decisions"][1]["anchor_ids"] == ["a02", "a03"]


def test_unknown_anchor_is_rejected():
    with pytest.raises(director.DecisionError, match="unknown anchor"):
        director.validate_decisions(_doc([{"anchor_id": "zzz", "effect": "keyword"}]), _brief())


def test_list_requires_two_anchors():
    with pytest.raises(director.DecisionError, match="needs two"):
        director.validate_decisions(
            _doc([{"anchor_id": "a01", "anchor_ids": ["a01"], "effect": "list"}]), _brief())


def test_list_must_be_in_speech_order():
    with pytest.raises(director.DecisionError, match="speech order"):
        director.validate_decisions(
            _doc([{"anchor_id": "a03", "anchor_ids": ["a03", "a02"], "effect": "list"}]),
            _brief())


def test_anchor_cannot_be_claimed_twice():
    with pytest.raises(director.DecisionError, match="already claimed"):
        director.validate_decisions(_doc([
            {"anchor_id": "a01", "effect": "keyword"},
            {"anchor_id": "a01", "effect": "card"},
        ]), _brief())


def test_emphasis_must_exist_in_text():
    with pytest.raises(director.DecisionError, match="does not occur"):
        director.validate_decisions(_doc([
            {"anchor_id": "a01", "effect": "keyword", "params": {"emphasis": ["冠军"]}},
        ]), _brief())


def test_invented_number_is_rejected():
    with pytest.raises(director.DecisionError, match="not present"):
        director.validate_decisions(_doc([
            {"anchor_id": "a01", "effect": "stat", "params": {"label": "90% 的选手"}},
        ]), _brief())


def test_number_present_in_text_is_accepted():
    out = director.validate_decisions(_doc([
        {"anchor_id": "a04", "effect": "stat", "params": {"label": "九成", "unit": ""}},
    ]), _brief())
    assert out["decisions"][0]["effect"] == "stat"


def test_hold_outside_envelope_is_rejected():
    with pytest.raises(director.DecisionError, match="hold_s"):
        director.validate_decisions(
            _doc([{"anchor_id": "a01", "effect": "keyword", "hold_s": 9.0}]), _brief())


def test_effect_alias_is_normalised():
    out = director.validate_decisions(
        _doc([{"anchor_id": "a04", "effect": "number"}]), _brief())
    assert out["decisions"][0]["effect"] == "stat"


# -------------------------------------------------------------------- provider
def test_direct_with_injected_transport(tmp_path):
    brief = _brief()
    payload = _doc([
        {"anchor_id": "a01", "effect": "keyword", "reason": "contrast",
         "params": {"emphasis": ["梦想"]}},
    ])
    got = director.direct(brief, transport=lambda messages: json.dumps(payload),
                          out_dir=tmp_path)
    assert got is not None and len(got["decisions"]) == 1
    assert (tmp_path / "decisions.generated.json").exists()


def test_direct_returns_none_and_records_bad_output(tmp_path):
    brief = _brief()
    bad = json.dumps(_doc([{"anchor_id": "nope", "effect": "keyword"}]))
    got = director.direct(brief, transport=lambda messages: bad, out_dir=tmp_path)
    assert got is None
    err = json.loads((tmp_path / "director_error.json").read_text())
    assert "unknown anchor" in err["error"]


def test_direct_without_model_returns_none(monkeypatch):
    for k in ("SM_LLM_BASE_URL", "SM_LLM_API_KEY", "SM_LLM_MODEL",
              "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(k, raising=False)
    assert director.direct(_brief()) is None


# ---------------------------------------------------------------------- policy
def test_default_policy_matches_renderer_type_scale():
    assert theme2.TYPE_RATIO["display"] == pytest.approx(policy.DEFAULT.get("type.ratio.display"))
    assert theme2.MAX_LINES["keyword"] == int(policy.DEFAULT.get("type.max_lines.keyword"))


def test_policy_override_deep_merges():
    p = policy.load(overrides={"budget": {"max_elements_per_10s": 2}})
    assert p.budget["max_elements_per_10s"] == 2
    assert p.budget["min_gap_s"] == policy.DEFAULT.budget["min_gap_s"]


def test_invalid_policy_fails_closed():
    with pytest.raises(ValueError, match="invalid policy"):
        policy.load(overrides={"budget": {"max_coverage": 3.0}})


def test_policy_from_file(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"name": "custom", "budget": {"hold_s": 0.7}}))
    p = policy.load(path)
    assert p.name == "custom" and p.get("budget.hold_s") == 0.7
