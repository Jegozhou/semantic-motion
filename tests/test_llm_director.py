"""Tests for the LLM director and the caption track (``llm-director`` scope).

Everything runs offline: the real decision samples in ``artifacts/llm`` are
validated as-is, and every negative case mutates a small document in memory.

Run:
    PYTHONPATH=src .venv/bin/python -m unittest tests.test_llm_director -v
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from semotion import llm_director as ld  # noqa: E402

LLM_DIR = ROOT / "artifacts/llm"
FX_A = ROOT / "artifacts/g1/final_product/fx_a_zh_portrait"
FX_FACE = ROOT / "artifacts/g2/matrix/fx_face/combo"
FX_B = ROOT / "artifacts/g1/product/fx_b_zh_landscape"

MODEL = "deepseek-v4.1-flash"


# --------------------------------------------------------------------------- helpers
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


_CACHE: dict[str, dict] = {}


def case(name: str) -> dict:
    """Load one real clip: transcript, anchors, vision, canvas, LLM document."""
    if name in _CACHE:
        return _CACHE[name]
    if name == "fx_a":
        base, doc_name = FX_A, "fx_a_zh_portrait"
    elif name == "fx_face":
        base, doc_name = FX_FACE, "fx_face_zh_portrait"
    else:  # pragma: no cover - guard for typos in tests
        raise KeyError(name)
    transcript = load_json(base / "transcript.json")
    anchors = load_json(base / "anchors.json")
    vision = load_json(base / "vision.json")
    canvas = {"width": vision["source"]["width"],
              "height": vision["source"]["height"],
              "fps": vision["source"]["fps"],
              "duration_s": vision["source"]["duration_s"]}
    doc = load_json(LLM_DIR / f"{doc_name}.decisions.json")
    _CACHE[name] = {"transcript": transcript, "anchors": anchors, "vision": vision,
                    "canvas": canvas, "doc": doc, "base": base}
    return _CACHE[name]


def make_doc(decisions: list[dict], **extra) -> dict:
    doc = {
        "schema": ld.SCHEMA_DECISIONS,
        "decision_model": MODEL,
        "prompt_summary": "unit-test document",
        "decisions": decisions,
    }
    doc.update(extra)
    return doc


def make_decision(anchor_id: str, effect: str = "keyword", **overrides) -> dict:
    decision = {
        "anchor_id": anchor_id,
        "anchor_ids": [anchor_id],
        "effect": effect,
        "emphasis": "high",
        "reason": "unit test",
        "hold_s": ld.DEFAULT_HOLD_S,
        "confidence": 0.9,
        "params": {"accent": "#F2B233", "enter": "fade_up", "avoid_face": True},
    }
    decision.update(overrides)
    return decision


def anchors_doc(rows: list[dict], duration: float | None = None) -> dict:
    doc = {"schema": "semotion.anchors/1", "anchors": rows}
    if duration is not None:
        doc["duration_s"] = duration
    return doc


def anchor(anchor_id: str, kind: str, text: str, start: float, end: float,
           confidence: float = 0.9) -> dict:
    return {"id": anchor_id, "kind": kind, "text": text, "start_s": start,
            "end_s": end, "confidence": confidence, "source": "TEST",
            "segment_id": "s0", "word_ids": []}


def transcript_doc(words: list[dict], *, fps: float = 25.0,
                   duration: float | None = None, segments: list[dict] | None = None) -> dict:
    if duration is None:
        duration = max((w["end_s"] for w in words), default=1.0)
    return {"schema": "semotion.transcript/1",
            "source": {"fps": fps, "duration_s": duration},
            "segments": segments or [],
            "words": words}


def word(text: str, start: float, end: float, index: int = 0,
         segment_id: str = "s0") -> dict:
    return {"id": f"w{index:03d}", "segment_id": segment_id, "text": text,
            "start_s": start, "end_s": end, "probability": 0.95}


def errors_of(doc: dict, anchors: dict, transcript: dict | None = None) -> list[str]:
    return ld.validate_decisions(doc, anchors, transcript)


# --------------------------------------------------------------------------- artifacts
class TestArtifactSamples(unittest.TestCase):
    """The checked-in LLM samples must be real, complete and legal."""

    def test_files_and_metadata(self):
        expected = {
            "fx_a_zh_portrait.decisions.json": "fx_a",
            "fx_face_zh_portrait.decisions.json": "fx_face",
        }
        for file_name, prompt_name in expected.items():
            path = LLM_DIR / file_name
            self.assertTrue(path.exists(), f"missing LLM sample: {path}")
            doc = load_json(path)
            self.assertEqual(doc["schema"], ld.SCHEMA_DECISIONS)
            self.assertEqual(doc["decision_model"], MODEL)
            self.assertFalse(doc["decision_model"].startswith("rules-"),
                             "a fallback must never be stored as an LLM sample")
            self.assertTrue(doc["prompt_summary"].strip())
            self.assertIn("created_utc", doc)
            self.assertTrue((LLM_DIR / f"prompt_{prompt_name}.txt").exists(),
                            "the prompt actually used must be archived next to the output")
            self.assertGreaterEqual(len(doc["decisions"]), 3)

    def test_samples_validate_without_errors(self):
        for name in ("fx_a", "fx_face"):
            data = case(name)
            with self.subTest(clip=name):
                self.assertEqual(
                    ld.validate_decisions(data["doc"], data["anchors"],
                                          data["transcript"]),
                    [],
                    f"{name}: checked-in LLM decisions must pass every rule")

    def test_plans_from_samples_are_legal(self):
        for name in ("fx_a", "fx_face"):
            data = case(name)
            with self.subTest(clip=name):
                plan = ld.to_plan(data["doc"], data["anchors"], data["canvas"],
                                  transcript=data["transcript"])
                self.assertEqual(plan["schema"], ld.SCHEMA_PLAN)
                self.assertEqual(plan["style"], "smart_balanced")
                self.assertEqual(plan["canvas"]["width"], data["canvas"]["width"])
                duration = plan["duration_s"]
                self.assertGreater(len(plan["decisions"]), 0)
                anchors = {a["id"]: a for a in data["anchors"]["anchors"]}
                previous_end = -1.0
                for decision in plan["decisions"]:
                    for field in ("id", "anchor_id", "anchor_ids", "effect", "emphasis",
                                  "reason", "confidence", "decision_model", "content",
                                  "window"):
                        self.assertIn(field, decision)
                    self.assertIn(decision["effect"], ld.EFFECT_WHITELIST)
                    self.assertTrue(decision["decision_model"].startswith("llm:"))
                    self.assertTrue(decision["content"]["text"])
                    start = decision["window"]["start_s"]
                    end = decision["window"]["end_s"]
                    self.assertAlmostEqual(start, anchors[decision["anchor_id"]]["start_s"],
                                           places=5,
                                           msg="window start must be the anchor onset")
                    self.assertLess(start, end)
                    self.assertGreaterEqual(start, previous_end - 1e-9,
                                            "windows must not overlap")
                    self.assertLessEqual(end, duration + 1e-6)
                    previous_end = end
                self.assertLessEqual(plan["notes"]["overlay_coverage"],
                                     ld.MAX_COVERAGE + 1e-9)


class TestPrompt(unittest.TestCase):
    def test_prompt_contains_every_contract_requirement(self):
        data = case("fx_a")
        prompt = ld.build_prompt(data["transcript"], data["anchors"],
                                 data["vision"], data["canvas"])
        required = [
            "keyword", "number", "list", "card",          # whitelist + semantics
            "anchor_id", "anchor_ids",
            "时间只能来自锚点",                            # no invented timestamps
            "每 10 秒最多 4", "58%", "0.8",               # rhythm budget
            "avoid_face", "不得跨镜头切换",                # safety
            "数字必须", "content.numeric",                 # exact numbers
            "semotion.llm-decisions/1", "JSON",            # output schema
            "宁可不加",                                    # restraint principle
            "skipped",
        ]
        for needle in required:
            with self.subTest(needle=needle):
                self.assertIn(needle, prompt)
        for row in data["anchors"]["anchors"]:
            self.assertIn(row["id"], prompt)
        self.assertIn("1080", prompt)
        self.assertIn("1920", prompt)
        self.assertIn("plain", prompt)

    def test_prompt_reports_faces_and_cuts(self):
        data = case("fx_face")
        vision = copy.deepcopy(data["vision"])
        vision["shots"] = [
            {"index": 0, "start_s": 0.0, "end_s": 12.0,
             "start_frame": 0, "end_frame": 300},
            {"index": 1, "start_s": 12.0, "end_s": 24.92,
             "start_frame": 300, "end_frame": 623},
        ]
        prompt = ld.build_prompt(data["transcript"], data["anchors"], vision,
                                 data["canvas"])
        self.assertIn("12.00s", prompt)
        self.assertIn("检测到人脸", prompt)


# -------------------------------------------------------------------------- validation
class TestValidationPositive(unittest.TestCase):
    def test_minimal_document_is_valid(self):
        data = case("fx_a")
        doc = make_doc([make_decision("a00")])
        self.assertEqual(errors_of(doc, data["anchors"], data["transcript"]), [])

    def test_empty_decisions_is_legal(self):
        data = case("fx_a")
        doc = make_doc([], skipped=[{"anchor_id": "a00",
                                     "reason": "nothing worth animating here"}])
        self.assertEqual(errors_of(doc, data["anchors"], data["transcript"]), [])

    def test_number_with_consistent_numeric_is_valid(self):
        data = case("fx_a")
        # parse_numeric("九成") == 90 (percent); the declared value must match exactly
        doc = make_doc([make_decision("a03", "number",
                                      content={"text": "九成", "numeric": {"value": 90}})])
        self.assertEqual(errors_of(doc, data["anchors"], data["transcript"]), [])

    def test_collision_ok_exempts_same_onset(self):
        data = case("fx_face")
        doc = make_doc([make_decision("a04"), make_decision("a05")])
        self.assertTrue(any("overlap" in error
                            for error in errors_of(doc, data["anchors"],
                                                   data["transcript"])))
        doc["collision_ok"] = [["a04", "a05"]]
        self.assertEqual(errors_of(doc, data["anchors"], data["transcript"]), [])


class TestValidationNegative(unittest.TestCase):
    def test_unknown_effect(self):
        data = case("fx_a")
        doc = make_doc([make_decision("a00", "explode")])
        errors = errors_of(doc, data["anchors"], data["transcript"])
        self.assertTrue(any("whitelist" in error for error in errors), errors)

    def test_fake_anchor_is_rejected(self):
        data = case("fx_a")
        doc = make_doc([make_decision("a99")])
        errors = errors_of(doc, data["anchors"], data["transcript"])
        self.assertTrue(any("a99" in error and "does not exist" in error
                            for error in errors), errors)

    def test_plain_anchor_is_rejected(self):
        anchors = anchors_doc([anchor("p0", "plain", "大家好", 0.5, 1.0)], duration=5.0)
        transcript = transcript_doc([word("大家好", 0.5, 1.0)])
        doc = make_doc([make_decision("p0")])
        errors = errors_of(doc, anchors, transcript)
        self.assertTrue(any("kind=plain" in error for error in errors), errors)

    def test_explicit_window_that_breaks_the_anchor_onset(self):
        data = case("fx_a")
        doc = make_doc([make_decision("a00", window={"start_s": 0.0, "end_s": 3.0})])
        errors = errors_of(doc, data["anchors"], data["transcript"])
        self.assertTrue(any("window.start_s" in error for error in errors), errors)

    def test_window_beyond_canvas_duration(self):
        data = case("fx_a")
        doc = make_doc([make_decision("a09", window={"start_s": 20.81,
                                                     "end_s": 30.0})])
        errors = errors_of(doc, data["anchors"], data["transcript"])
        self.assertTrue(any("exceeds canvas duration" in error for error in errors), errors)

    def test_overlapping_onsets_are_reported(self):
        data = case("fx_face")
        doc = make_doc([make_decision("a04"), make_decision("a05")])
        errors = errors_of(doc, data["anchors"], data["transcript"])
        self.assertTrue(any("overlap" in error for error in errors), errors)
        self.assertTrue(any("a04" in error or "a05" in error for error in errors), errors)

    def test_rhythm_density_budget(self):
        data = case("fx_a")
        # 5 starts inside 3.32s..13.32s: one too many for "max 4 per 10s"
        decisions = [make_decision("a00"), make_decision("a01"), make_decision("a02"),
                     make_decision("a03"), make_decision("a05")]
        doc = make_doc(decisions)
        errors = errors_of(doc, data["anchors"], data["transcript"])
        self.assertTrue(any("10s" in error and "max" in error for error in errors), errors)

    def test_coverage_budget(self):
        rows = [anchor(f"c{i}", "keyword", f"第{i}段", start, start + 6.0)
                for i, start in enumerate((0.5, 8.5, 16.5))]
        anchors = anchors_doc(rows, duration=24.0)
        doc = make_doc([make_decision("c0"), make_decision("c1"), make_decision("c2")])
        errors = errors_of(doc, anchors, None)
        self.assertTrue(any("coverage" in error and "58%" in error for error in errors),
                        errors)

    def test_number_without_numeric_token(self):
        data = case("fx_a")
        doc = make_doc([make_decision("a00", "number")])
        errors = errors_of(doc, data["anchors"], data["transcript"])
        self.assertTrue(any("no parseable numeric token" in error for error in errors),
                        errors)

    def test_number_value_mismatch(self):
        data = case("fx_a")
        doc = make_doc([make_decision("a03", "number",
                                      content={"text": "九成",
                                               "numeric": {"value": 5}})])
        errors = errors_of(doc, data["anchors"], data["transcript"])
        self.assertTrue(any("must never be rewritten" in error for error in errors),
                        errors)

    def test_hold_over_limit(self):
        data = case("fx_a")
        doc = make_doc([make_decision("a00", hold_s=1.2)])
        errors = errors_of(doc, data["anchors"], data["transcript"])
        self.assertTrue(any("hold_s" in error and "0.8" in error for error in errors),
                        errors)

    def test_avoid_face_is_mandatory(self):
        data = case("fx_a")
        doc = make_doc([make_decision("a00",
                                      params={"accent": "#F2B233", "avoid_face": False})])
        errors = errors_of(doc, data["anchors"], data["transcript"])
        self.assertTrue(any("avoid_face" in error for error in errors), errors)

    def test_unknown_decision_field_is_rejected(self):
        data = case("fx_a")
        doc = make_doc([make_decision("a00", script="print('nope')")])
        errors = errors_of(doc, data["anchors"], data["transcript"])
        self.assertTrue(any("unknown field" in error and "script" in error
                            for error in errors), errors)

    def test_invented_copy_is_rejected(self):
        data = case("fx_a")
        doc = make_doc([make_decision("a00", content={"text": "这段文案根本没人说过"})])
        errors = errors_of(doc, data["anchors"], data["transcript"])
        self.assertTrue(any("no invented copy" in error for error in errors), errors)

    def test_low_confidence_anchor(self):
        anchors = anchors_doc([anchor("l0", "keyword", "也许", 1.0, 2.0,
                                      confidence=0.55)], duration=6.0)
        doc = make_doc([make_decision("l0")])
        errors = errors_of(doc, anchors, None)
        self.assertTrue(any("confidence" in error for error in errors), errors)

    def test_anchor_onset_outside_duration(self):
        anchors = anchors_doc([anchor("z0", "keyword", "结尾", 30.0, 31.0)],
                              duration=24.0)
        doc = make_doc([make_decision("z0")])
        errors = errors_of(doc, anchors, None)
        self.assertTrue(any("outside" in error for error in errors), errors)

    def test_bad_schema_and_missing_metadata(self):
        data = case("fx_a")
        doc = make_doc([make_decision("a00")])
        doc["schema"] = "semotion.llm-decisions/0"
        doc.pop("prompt_summary")
        doc["decision_model"] = ""
        errors = errors_of(doc, data["anchors"], data["transcript"])
        self.assertTrue(any("schema" in error for error in errors), errors)
        self.assertTrue(any("prompt_summary" in error for error in errors), errors)
        self.assertTrue(any("decision_model" in error for error in errors), errors)

    def test_skipped_reference_must_exist(self):
        data = case("fx_a")
        doc = make_doc([make_decision("a00")],
                       skipped=[{"anchor_id": "zzz", "reason": "not a real anchor"}])
        errors = errors_of(doc, data["anchors"], data["transcript"])
        self.assertTrue(any("zzz" in error for error in errors), errors)

    def test_shot_cut_guard_when_vision_is_given(self):
        data = case("fx_face")
        vision = {"shots": [
            {"index": 0, "start_s": 0.0, "end_s": 20.0, "start_frame": 0,
             "end_frame": 500},
            {"index": 1, "start_s": 20.0, "end_s": 24.92, "start_frame": 500,
             "end_frame": 623},
        ]}
        doc = make_doc([make_decision("a13", "card")])
        errors = ld.validate_decisions(doc, data["anchors"], data["transcript"],
                                       vision=vision)
        self.assertTrue(any("straddles the shot cut" in error for error in errors),
                        errors)


# ------------------------------------------------------------------------------ plan
class TestToPlan(unittest.TestCase):
    def test_invalid_document_raises(self):
        data = case("fx_a")
        doc = make_doc([make_decision("a00", "explode")])
        with self.assertRaises(ValueError):
            ld.to_plan(doc, data["anchors"], data["canvas"])

    def test_decision_model_is_namespaced(self):
        data = case("fx_a")
        doc = make_doc([make_decision("a00")])
        plan = ld.to_plan(doc, data["anchors"], data["canvas"])
        self.assertEqual(plan["decisions"][0]["decision_model"], f"llm:{MODEL}")
        self.assertEqual(plan["provenance"]["decision_model"], f"llm:{MODEL}")

    def test_fallback_model_is_not_namespaced(self):
        data = case("fx_a")
        doc = ld.default_decisions_for(data["anchors"])
        plan = ld.to_plan(doc, data["anchors"], data["canvas"])
        self.assertEqual(plan["decisions"][0]["decision_model"], ld.FALLBACK_MODEL)

    def test_list_items_survive_into_the_plan(self):
        data = case("fx_a")
        plan = ld.to_plan(data["doc"], data["anchors"], data["canvas"],
                          transcript=data["transcript"])
        lists = [d for d in plan["decisions"] if d["effect"] == "list"]
        self.assertEqual(len(lists), 1)
        self.assertEqual(len(lists[0]["content"]["items"]), 3)
        self.assertTrue(all(item for item in lists[0]["content"]["items"]))
        self.assertEqual(len(lists[0]["anchor_ids"]), 3)

    def test_colliding_windows_are_dropped_not_overlapped(self):
        data = case("fx_face")
        doc = make_doc([make_decision("a04"), make_decision("a05")],
                       collision_ok=[["a04", "a05"]])
        plan = ld.to_plan(doc, data["anchors"], data["canvas"])
        windows = [(d["window"]["start_s"], d["window"]["end_s"])
                   for d in plan["decisions"]]
        self.assertTrue(all(windows[i][1] <= windows[i + 1][0]
                            for i in range(len(windows) - 1)))
        self.assertEqual(len(plan["decisions"]), 1)
        self.assertTrue(any("window_too_short_after_packing" in item["reason"]
                            for item in plan["notes"]["skipped"]))

    def test_optional_transcript_catches_invented_copy(self):
        data = case("fx_a")
        doc = make_doc([make_decision("a00", content={"text": "凭空捏造"})])
        ld.to_plan(doc, data["anchors"], data["canvas"])  # tolerated: no transcript
        with self.assertRaises(ValueError):
            ld.to_plan(doc, data["anchors"], data["canvas"],
                       transcript=data["transcript"])

    def test_windows_respect_hold_and_next_onset(self):
        data = case("fx_a")
        doc = make_doc([make_decision("a00", hold_s=0.8), make_decision("a02")])
        plan = ld.to_plan(doc, data["anchors"], data["canvas"])
        anchors = {a["id"]: a for a in data["anchors"]["anchors"]}
        for decision in plan["decisions"]:
            anchor = anchors[decision["anchor_id"]]
            self.assertAlmostEqual(decision["window"]["start_s"], anchor["start_s"],
                                   places=5)
            self.assertLessEqual(decision["window"]["end_s"] - anchor["end_s"],
                                 ld.MAX_HOLD_S + 1e-6)


# -------------------------------------------------------------------------- fallback
class TestDefaultDecisions(unittest.TestCase):
    def test_fallback_is_valid_and_deterministic(self):
        for name in ("fx_a", "fx_face"):
            data = case(name)
            with self.subTest(clip=name):
                first = ld.default_decisions_for(data["anchors"])
                second = ld.default_decisions_for(data["anchors"])
                self.assertEqual(json.dumps(first, sort_keys=True),
                                 json.dumps(second, sort_keys=True))
                self.assertEqual(first["decision_model"], ld.FALLBACK_MODEL)
                self.assertEqual(ld.validate_decisions(first, data["anchors"],
                                                       data["transcript"]), [])
                plan = ld.to_plan(first, data["anchors"], data["canvas"])
                self.assertEqual(plan["schema"], ld.SCHEMA_PLAN)
                self.assertLessEqual(plan["notes"]["overlay_coverage"],
                                     ld.MAX_COVERAGE + 1e-9)

    def test_fallback_skips_plain_and_low_confidence(self):
        rows = [anchor("k0", "keyword", "节奏", 1.0, 1.6),
                anchor("p0", "plain", "大家好", 0.2, 0.8),
                anchor("l0", "keyword", "也许", 3.0, 3.6, confidence=0.5)]
        doc = ld.default_decisions_for(anchors_doc(rows, duration=6.0))
        self.assertEqual([d["anchor_id"] for d in doc["decisions"]], ["k0"])
        reasons = {item["anchor_id"]: item["reason"] for item in doc["skipped"]}
        self.assertEqual(reasons["p0"], "plain_speech_no_motion")
        self.assertTrue(reasons["l0"].startswith("low_confidence"))


# --------------------------------------------------------------------------- captions
class TestCaptionCues(unittest.TestCase):
    def _assert_cues_sane(self, cues: list[dict], max_chars: int,
                          max_lines: int) -> None:
        self.assertGreater(len(cues), 0)
        previous_end = -1
        for cue in cues:
            for field in ("startFrame", "endFrame", "lines", "words"):
                self.assertIn(field, cue)
            self.assertIsInstance(cue["startFrame"], int)
            self.assertIsInstance(cue["endFrame"], int)
            self.assertLess(cue["startFrame"], cue["endFrame"])
            self.assertGreaterEqual(cue["startFrame"], previous_end,
                                    "cues must be monotonic and non-overlapping")
            previous_end = cue["endFrame"]
            self.assertGreaterEqual(len(cue["lines"]), 1)
            self.assertLessEqual(len(cue["lines"]), max_lines)
            for line in cue["lines"]:
                # max_chars is a *display width*: CJK 1 unit, Latin/ASCII 0.5
                self.assertLessEqual(ld.display_width(line), max_chars + 1e-9,
                                     f"line too wide: {line!r}")
                self.assertTrue(line, "empty caption line")
                self.assertNotIn(line[-1], "，,。.、；;：:",
                                 f"trailing punctuation in {line!r}")
            self.assertTrue(cue["words"])
            self.assertEqual(cue["words"][0]["startFrame"], cue["startFrame"])
            last_end = cue["startFrame"]
            for token in cue["words"]:
                self.assertLess(token["startFrame"], token["endFrame"])
                self.assertGreaterEqual(token["startFrame"], cue["startFrame"])
                self.assertLessEqual(token["endFrame"], cue["endFrame"])
                self.assertGreaterEqual(token["startFrame"], last_end)
                last_end = token["endFrame"]

    def test_fx_a_captions(self):
        data = case("fx_a")
        cues = ld.caption_cues(data["transcript"], data["canvas"])
        self._assert_cues_sane(cues, 22, 2)
        fps = data["canvas"]["fps"]
        words = [w for w in data["transcript"]["words"] if w["text"].strip()]
        # first frame of every cue is the frame of its first spoken word
        by_first_word = {w["text"] + str(w["start_s"]): round(w["start_s"] * fps)
                         for w in words}
        # caption display text is Simplified and de-punctuated, so match on the
        # normalised word text rather than the raw transcript token
        normalized = {ld._caption_display_text(w["text"]).strip("，,。.、；;：: ")
                      for w in words}
        for cue in cues:
            only = cue["words"][0]["text"].strip("，,。.、；;：: ")
            self.assertIn(only, normalized)
        self.assertEqual(cues[0]["words"][0]["startFrame"],
                         round(words[0]["start_s"] * fps))
        self.assertIn(round(words[-1]["end_s"] * fps),
                      {cues[-1]["endFrame"], cues[-1]["words"][-1]["endFrame"]})

    def test_fx_face_traditional_chinese_captions(self):
        data = case("fx_face")
        cues = ld.caption_cues(data["transcript"], data["canvas"])
        self._assert_cues_sane(cues, 22, 2)

    def test_caption_display_is_simplified_and_unpunctuated(self):
        """Burned captions must read like subtitles: Simplified, no trailing punctuation."""
        for fixture in ("fx_a", "fx_face"):
            data = case(fixture)
            cues = ld.caption_cues(data["transcript"], data["canvas"])
            for cue in cues:
                for line in cue["lines"]:
                    self.assertFalse(line.endswith(("，", ",", "。", ".", "、", "；", ";", "：", ":")),
                                     f"{fixture}: trailing punctuation in {line!r}")
                    self.assertEqual(ld._simplified(line), line,
                                     f"{fixture}: traditional characters in {line!r}")

    def test_caption_copy_is_simplified_with_stable_offsets(self):
        """Captions must be Simplified, and folding must not move character offsets."""
        try:
            from opencc import OpenCC
        except Exception:  # pragma: no cover - optional dependency
            self.skipTest("opencc not installed")
        converter = OpenCC("t2s")
        data = case("fx_face")  # the Traditional fixture
        cues = ld.caption_cues(data["transcript"], data["canvas"])
        raw_words = {w["text"] for w in data["transcript"]["words"] if w["text"].strip()}
        changed = 0
        for cue in cues:
            for line in cue["lines"]:
                self.assertEqual(converter.convert(line), line,
                                 f"traditional characters in caption line {line!r}")
            for token in cue["words"]:
                self.assertEqual(converter.convert(token["text"]), token["text"],
                                 f"traditional characters in word {token['text']!r}")
                if "raw_text" in token:
                    changed += 1
                    self.assertIn(token["raw_text"], raw_words)
                    # 1:1 fold: same length, so any char span stays valid
                    self.assertEqual(len(token["raw_text"]), len(token["text"]),
                                     "t2s folding changed the character count")
                    self.assertNotEqual(token["raw_text"], token["text"])
        self.assertGreater(changed, 0,
                           "the Traditional fixture must actually exercise the fold")

    def test_english_uses_half_width_units(self):
        text = ("Hello world this is a longer sentence about caption wrapping "
                "and readable lines").split()
        words = [word(token, i * 0.30, (i + 1) * 0.30, i)
                 for i, token in enumerate(text)]
        transcript = transcript_doc(words, duration=len(text) * 0.30)
        canvas = {"width": 1080, "height": 1920, "fps": 25.0,
                  "duration_s": len(text) * 0.30}
        cues = ld.caption_cues(transcript, canvas, max_chars=12, max_lines=2)
        self._assert_cues_sane(cues, 12, 2)
        # 12 units ≈ 24 ASCII characters: at least one line uses the extra room
        self.assertTrue(any(len(line) > 12 for cue in cues for line in cue["lines"]),
                        "Latin text should count at half width, not one unit per char")
        joined = " ".join(line for cue in cues for line in cue["lines"])
        for token in text:  # words are never cut
            self.assertIn(token, joined)

    def test_landscape_uses_thirty_char_lines(self):
        transcript = load_json(FX_B / "transcript.json")
        canvas = {"width": 1920, "height": 1080, "fps": 25.0,
                  "duration_s": transcript["source"]["duration_s"]}
        self.assertEqual(ld.default_max_chars(canvas), 30)
        self.assertEqual(ld.default_max_chars({"width": 1080, "height": 1920}), 22)
        cues = ld.caption_cues(transcript, canvas, max_chars=30)
        self._assert_cues_sane(cues, 30, 2)

    def test_landscape_budget_allows_wider_lines(self):
        # a synthetic 50-unit clause: 16:9 rows may exceed the 22-unit portrait budget
        words = [word(str(i % 10), i * 0.20, (i + 1) * 0.20, i) for i in range(50)]
        transcript = transcript_doc(words, duration=10.0)
        canvas = {"width": 1920, "height": 1080, "fps": 25.0, "duration_s": 10.0}
        cues = ld.caption_cues(transcript, canvas, max_chars=30)
        self._assert_cues_sane(cues, 30, 2)
        self.assertTrue(any(len(line) > 22 for cue in cues for line in cue["lines"]),
                        "max_chars=30 must actually permit longer 16:9 lines")

    def test_long_clause_splits_into_several_screens(self):
        words = [word(str(i % 10), i * 0.20, (i + 1) * 0.20, i) for i in range(70)]
        transcript = transcript_doc(words, duration=14.0)
        cues = ld.caption_cues(transcript, {"width": 1080, "height": 1920,
                                            "fps": 25.0, "duration_s": 14.0},
                               max_chars=10, max_lines=2)
        self.assertGreater(len(cues), 1)
        self._assert_cues_sane(cues, 10, 2)
        words_in_cues = sum(len(cue["words"]) for cue in cues)
        self.assertEqual(words_in_cues, 70)

    def test_english_words_are_never_split(self):
        text = ("Hello world this is a longer sentence about caption wrapping "
                "and readable lines").split()
        words = [word(token, i * 0.30, (i + 1) * 0.30, i) for i, token in enumerate(text)]
        transcript = transcript_doc(words, duration=len(text) * 0.30)
        canvas = {"width": 1080, "height": 1920, "fps": 25.0,
                  "duration_s": len(text) * 0.30}
        cues = ld.caption_cues(transcript, canvas, max_chars=12, max_lines=2)
        self._assert_cues_sane(cues, 12, 2)
        rendered = " ".join(cue["lines"][0] for cue in cues)
        for token in text:
            self.assertIn(token, " ".join(line for cue in cues for line in cue["lines"]),
                          f"word was split or dropped: {token!r}")

    def test_oversized_single_token_is_hard_split(self):
        words = [word("abcdefghijklmnopqrstuvwxyz", 0.0, 2.0, 0)]
        transcript = transcript_doc(words, duration=2.5)
        cues = ld.caption_cues(transcript, {"width": 1080, "height": 1920,
                                            "fps": 25.0, "duration_s": 2.5},
                               max_chars=10, max_lines=2)
        self._assert_cues_sane(cues, 10, 2)
        joined = "".join(line for cue in cues for line in cue["lines"])
        self.assertEqual(joined, "abcdefghijklmnopqrstuvwxyz")

    def test_empty_transcript(self):
        self.assertEqual(ld.caption_cues({}, {"width": 1080, "height": 1920}), [])

    def test_invalid_budgets(self):
        transcript = transcript_doc([word("你好", 0.0, 0.5)])
        with self.assertRaises(ValueError):
            ld.caption_cues(transcript, {"width": 1080, "height": 1920}, max_chars=0)
        with self.assertRaises(ValueError):
            ld.caption_cues(transcript, {"width": 1080, "height": 1920}, max_lines=0)


# ------------------------------------------------------------------------ persistence
class TestPersistence(unittest.TestCase):
    def test_roundtrip(self):
        original = load_json(LLM_DIR / "fx_a_zh_portrait.decisions.json")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "decisions.json"
            written = ld.dump_decisions(original, path)
            self.assertEqual(written, path)
            self.assertTrue(path.exists())
            self.assertEqual(ld.load_decisions(path), original)

    def test_dump_rejects_non_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                ld.dump_decisions(["not", "an", "object"], Path(tmp) / "x.json")

    def test_load_rejects_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                ld.load_decisions(path)
            with self.assertRaises(FileNotFoundError):
                ld.load_decisions(Path(tmp) / "missing.json")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
