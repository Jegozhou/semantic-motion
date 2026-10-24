"""Regression tests for Lead-owned modules: fixtures, baseline, director, verify helpers."""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semotion import baseline, media, semantic, verify  # noqa: E402

CLIPS = ROOT / "tests/fixtures/clips"


class FixtureTruthTest(unittest.TestCase):
    """SYNTH_KNOWN truth must be self-consistent and measurable."""

    def test_fx_a_truth_is_monotonic_and_frame_aligned(self):
        gt = json.loads((CLIPS / "fx_a_zh_portrait/ground_truth.json").read_text(encoding="utf-8"))
        units = gt["units"]
        self.assertGreaterEqual(len(units), 12)
        last = 0.0
        for u in units:
            self.assertGreater(u["start_s"], last - 1e-9, "unit starts must advance")
            self.assertGreater(u["end_s"], u["start_s"])
            self.assertEqual(u["truth"], "SYNTH_KNOWN")
            self.assertEqual(u["start_frame"], round(u["start_s"] * gt["fps"]))
            last = u["end_s"]
        self.assertLess(units[-1]["end_s"], gt["duration_s"] + 1e-6)

    def test_ground_truth_has_no_asr_dependency(self):
        gt = json.loads((CLIPS / "fx_a_zh_portrait/ground_truth.json").read_text(encoding="utf-8"))
        self.assertFalse(gt["audio_assembly"]["asr_used_for_truth"])
        self.assertEqual(gt["kind"], "SYNTH_KNOWN")

    def test_real_pd_fixture_is_labelled_machine_weak(self):
        fx = json.loads((CLIPS / "fx_pd_real_16x9_a/fixture.json").read_text(encoding="utf-8"))
        self.assertEqual(fx["kind"], "REAL_PD")
        self.assertEqual(fx["truth_class"], "MACHINE_WEAK")
        self.assertIn("Public domain", fx["source"]["license"])


class MediaTest(unittest.TestCase):
    def test_info_and_hashing(self):
        p = CLIPS / "fx_a_zh_portrait/fx_a_zh_portrait.mp4"
        mi = media.info(p)
        self.assertEqual((mi.width, mi.height), (1080, 1920))
        self.assertAlmostEqual(mi.fps, 25.0)
        self.assertEqual(mi.n_audio_streams, 1)
        self.assertEqual(len(media.sha256_file(p)), 64)
        self.assertEqual(len(media.video_frame_hashes(p, width=64, height=112)), mi.frame_count)

    def test_audio_pcm_hash_is_stable(self):
        p = CLIPS / "fx_a_zh_portrait/fx_a_zh_portrait.mp4"
        a = media.audio_pcm_sha256(p)
        b = media.audio_pcm_sha256(p)
        self.assertEqual(a, b)


class BaselineTest(unittest.TestCase):
    """The G0 baseline must stay reproducible."""

    def test_remux_audio_and_frames_identical(self):
        src = CLIPS / "fx_a_zh_portrait/fx_a_zh_portrait.mp4"
        out = ROOT / "artifacts/regress/remux.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        baseline.remux(src, out)
        rep = baseline.verify_copy(src, out, work=ROOT / "artifacts/regress/verify")
        self.assertEqual(rep["verdict"], "PASS", rep["checks"])
        self.assertTrue(rep["checks"]["audio_pcm_identity"]["pass"])
        self.assertTrue(rep["checks"]["frames"]["identical_hashes"])


def _anchors() -> dict:
    def a(i, kind, text, start, end, conf=0.9):
        return {"id": f"a{i:02d}", "kind": kind, "text": text, "start_s": start,
                "end_s": end, "confidence": conf, "source": "ASR_ALIGNED",
                "segment_id": f"s{i}", "word_ids": [f"w{i}"]}
    return {
        "schema": "semotion.anchors/1",
        "duration_s": 30.0,
        "anchors": [
            a(1, "plain", "大家好", 0.5, 1.2),
            a(2, "keyword", "其实是节奏", 6.3, 7.5),
            a(3, "number", "九成的剪辑", 9.0, 10.2),
            a(4, "list_item", "第一", 12.0, 12.7),
            a(5, "list_item", "第二", 14.0, 14.7),
            a(6, "list_item", "第三", 16.0, 16.7),
            a(7, "keyword", "低置信度", 20.0, 20.6, conf=0.4),
            a(8, "conclusion", "做成了自动化", 24.0, 25.2),
        ],
    }


class DirectorTest(unittest.TestCase):
    def test_plain_speech_gets_no_motion(self):
        plan = semantic.decide(_anchors(), {}, {"width": 1080, "height": 1920, "fps": 25})
        edited = {d["anchor_id"] for d in plan["decisions"]}
        self.assertNotIn("a01", edited, "plain speech must not be animated")
        skipped = {s["anchor_id"]: s["reason"] for s in plan["notes"]["skipped"]}
        self.assertEqual(skipped.get("a01"), "plain_speech_no_motion")

    def test_low_confidence_is_not_animated(self):
        plan = semantic.decide(_anchors(), {}, {"width": 1080, "height": 1920, "fps": 25})
        self.assertNotIn("a07", {d["anchor_id"] for d in plan["decisions"]})

    def test_list_items_merge_into_single_element(self):
        plan = semantic.decide(_anchors(), {}, {"width": 1080, "height": 1920, "fps": 25})
        lists = [d for d in plan["decisions"] if d["effect"] == "list"]
        self.assertEqual(len(lists), 1)
        self.assertEqual(len(lists[0]["content"]["items"]), 3)
        self.assertEqual(lists[0]["anchor_ids"], ["a04", "a05", "a06"])

    def test_effects_are_from_whitelist(self):
        plan = semantic.decide(_anchors(), {}, {"width": 1080, "height": 1920, "fps": 25})
        for d in plan["decisions"]:
            self.assertIn(d["effect"], {"keyword", "number", "list", "card"})
        self.assertEqual(plan["schema"], "semotion.plan/1")

    def test_rhythm_budget_limits_density(self):
        anchors = {"schema": "semotion.anchors/1", "duration_s": 12.0, "anchors": [
            {"id": f"a{i:02d}", "kind": "keyword", "text": f"关键词{i}", "start_s": i * 0.4,
             "end_s": i * 0.4 + 0.8, "confidence": 0.9, "source": "ASR_ALIGNED"} for i in range(1, 20)
        ]}
        plan = semantic.decide(anchors, {}, {"width": 1080, "height": 1920, "fps": 25})
        self.assertLessEqual(len(plan["decisions"]), 4)
        self.assertLessEqual(plan["notes"]["overlay_coverage"], semantic.MAX_OVERLAY_COVERAGE + 1e-6)

    def test_deterministic_output(self):
        p1 = semantic.decide(_anchors(), {}, {"width": 1080, "height": 1920, "fps": 25})
        p2 = semantic.decide(_anchors(), {}, {"width": 1080, "height": 1920, "fps": 25})
        self.assertEqual(json.dumps(p1, sort_keys=True), json.dumps(p2, sort_keys=True))


class VerifyHelperTest(unittest.TestCase):
    def test_truth_matching_prefers_exact_text(self):
        units = [
            {"unit_id": "u1", "text": "其实是节奏", "start_s": 6.3},
            {"unit_id": "u2", "text": "九成的剪辑", "start_s": 9.0},
        ]
        self.assertEqual(verify._match_truth("其实是节奏", units)["unit_id"], "u1")
        self.assertEqual(verify._match_truth("九成的剪辑", units)["unit_id"], "u2")
        self.assertIsNone(verify._match_truth("完全无关的一句话", units))

    def test_inter_area(self):
        self.assertEqual(verify._inter_area([0, 0, 10, 10], [5, 5, 10, 10]), 25.0)
        self.assertEqual(verify._inter_area([0, 0, 10, 10], [20, 20, 5, 5]), 0.0)

    def test_worst_verdict_ordering(self):
        self.assertEqual(verify._worst(["PASS", "FAIL"]), "FAIL")
        self.assertEqual(verify._worst(["PASS", "NOT RUN"]), "NOT RUN")
        self.assertEqual(verify._worst(["PASS", "PASS"]), "PASS")


class CliContractTest(unittest.TestCase):
    def test_cli_help_lists_commands(self):
        p = subprocess.run([sys.executable, "-m", "semotion.cli", "--help"],
                           capture_output=True, text=True, cwd=ROOT,
                           env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
        self.assertEqual(p.returncode, 0, p.stderr)
        for cmd in ("fixtures", "baseline", "asr", "vision", "anchors", "plan", "ir",
                    "render", "verify", "run"):
            self.assertIn(cmd, p.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
