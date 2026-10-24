"""Tests for the speech/vision layer: ASR, vision, anchors, manifest.

Run with::

    PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_asr_vision.py' -v
    # or directly:
    PYTHONPATH=src .venv/bin/python tests/test_asr_vision.py -v

Rules these tests follow
------------------------
* They run the **real** pipeline on the real fixtures (no mocks of the model
  output). First run transcribes fx_a with faster-whisper (~10 s, CPU int8);
  later runs hit the deterministic cache.
* They never download anything: the model is loaded from the local CTranslate2
  snapshot (see ``asr.resolve_model_path``) and every fixture is on disk.
* Timing assertions compare against **SYNTH_KNOWN assembly truth**
  (``ground_truth.json``), never against ASR output. They are dev-set
  engineering checks, not a human-labelled acceptance run.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from semotion import anchors as A  # noqa: E402
from semotion import asr, media, vision  # noqa: E402

CLIPS = ROOT / "tests/fixtures/clips"
ART = ROOT / "artifacts"


def clip(fid: str) -> Path:
    return CLIPS / fid / f"{fid}.mp4"


def ground_truth(fid: str) -> Path:
    return CLIPS / fid / "ground_truth.json"


def synthetic_transcript(text: str, words: list[str], *, start: float = 0.5,
                         step: float = 0.3) -> dict:
    """Build a minimal valid transcript document for rule-only unit tests.

    Word character offsets are derived by walking the text, exactly like
    ``asr.transcribe`` does, so the document exercises the same code paths
    without spending an ASR run.
    """
    words_doc = []
    cursor = 0
    t = start
    for i, w in enumerate(words):
        idx = text.find(w, cursor)
        if idx < 0:
            idx = cursor
        words_doc.append({
            "id": f"w{i}", "segment_id": "s0", "text": w,
            "start_s": round(t, 6), "end_s": round(t + step, 6),
            "probability": 0.9, "char_start": idx, "char_end": idx + len(w),
        })
        cursor = idx + len(w)
        t += step
    return {
        "schema": asr.SCHEMA,
        "source": {"path": "/dev/null", "sha256": "0" * 64, "duration_s": round(t + 0.5, 6),
                   "width": 1080, "height": 1920, "fps": 25.0},
        "language": "zh",
        "model": {"name": "faster-whisper", "size": "medium", "device": "cpu", "version": "test"},
        "segments": [{"id": "s0", "start_s": round(start, 6), "end_s": round(t, 6),
                      "text": text, "avg_logprob": -0.1, "no_speech_prob": 0.0}],
        "words": words_doc,
        "provenance": {"created_utc": "2026-10-24T00:00:00Z", "command": "test",
                       "truth_class": "MACHINE_WEAK"},
    }


# ============================================================================ ASR

class TestAsrOnFixture(unittest.TestCase):
    """Real faster-whisper transcription of the 9:16 SYNTH_KNOWN clip."""

    @classmethod
    def setUpClass(cls):
        cls.video = clip("fx_a_zh_portrait")
        if not cls.video.exists():
            raise unittest.SkipTest(f"fixture missing: {cls.video}")
        cls.out = ART / "asr" / "test_fx_a.transcript.json"
        cls.transcript = asr.transcribe(cls.video, cls.out)
        cls.info = media.info(cls.video)
        cls.truth = A.load_truth(ground_truth("fx_a_zh_portrait"))

    def test_contract_valid(self):
        self.assertEqual(self.transcript["schema"], "semotion.transcript/1")
        self.assertEqual(asr.validate(self.transcript), [])
        self.assertEqual(self.transcript["provenance"]["truth_class"], "MACHINE_WEAK")

    def test_matches_source_media(self):
        src = self.transcript["source"]
        self.assertEqual(src["sha256"], media.sha256_file(self.video))
        self.assertAlmostEqual(src["duration_s"], self.info.duration, places=3)
        self.assertEqual((src["width"], src["height"]), (self.info.width, self.info.height))
        self.assertAlmostEqual(src["fps"], self.info.fps, places=3)

    def test_segment_and_word_counts(self):
        # 14 spoken units are grouped by whisper into ~7 segments; the contract
        # only requires that the transcript is real (not empty) and word-level.
        self.assertGreaterEqual(len(self.transcript["segments"]), 5)
        self.assertLessEqual(len(self.transcript["segments"]), 12)
        self.assertGreater(len(self.transcript["words"]), 30)
        for seg in self.transcript["segments"]:
            self.assertGreaterEqual(float(seg["end_s"]), float(seg["start_s"]))
            self.assertTrue(seg["text"])

    def test_word_probabilities_and_char_spans(self):
        segs = {s["id"]: s for s in self.transcript["segments"]}
        for word in self.transcript["words"]:
            self.assertIsNotNone(word["probability"])
            self.assertGreaterEqual(word["probability"], 0.0)
            self.assertLessEqual(word["probability"], 1.0)
            text = segs[word["segment_id"]]["text"]
            sliced = text[word["char_start"]:word["char_end"]]
            self.assertEqual(sliced, word["text"],
                             f"{word['id']} char span does not round-trip: {sliced!r}")

    def test_word_times_are_ordered_and_inside_media(self):
        duration = float(self.transcript["source"]["duration_s"])
        previous = -1.0
        for word in self.transcript["words"]:
            start, end = float(word["start_s"]), float(word["end_s"])
            self.assertLessEqual(start, end)
            self.assertGreaterEqual(start, 0.0)
            self.assertLessEqual(end, duration + 0.5)
            self.assertGreaterEqual(start, previous - 1e-6)
            previous = start

    def test_cache_key_contains_sha_and_model_and_is_reproducible(self):
        sha = media.sha256_file(self.video)
        cache = asr.cache_path(sha, "medium", "cpu")
        self.assertIn(sha, cache.name)
        self.assertIn("medium", cache.name)
        self.assertIn("cpu", cache.name)
        self.assertTrue(cache.exists(), "transcribe() must persist the cache file")
        again = asr.transcribe(self.video, self.out)
        self.assertEqual(json.dumps(again, sort_keys=True),
                         json.dumps(self.transcript, sort_keys=True))

    def test_helpers(self):
        phrase = asr.find_phrase(self.transcript, "其实是节奏")
        self.assertIsNotNone(phrase)
        self.assertAlmostEqual(phrase["start_s"], 6.52, delta=0.30)
        self.assertEqual(len(phrase["word_ids"]), 4)

        window = asr.words_in_range(self.transcript, 11.30, 11.90)
        self.assertEqual([w["text"] for w in window], ["第一,"])

        self.assertEqual(asr.word_by_id(self.transcript, "w0")["text"], "大家好,")
        self.assertIsNone(asr.word_by_id(self.transcript, "nope"))

        segs = asr.segments_in_range(self.transcript, 6.4, 6.6)
        self.assertTrue(segs)

    def test_validate_detects_bad_documents(self):
        broken = json.loads(json.dumps(self.transcript))
        broken["schema"] = "wrong"
        broken["provenance"]["truth_class"] = "HUMAN"   # ASR may never self-certify
        broken["words"][0]["start_s"] = 999.0
        problems = asr.validate(broken)
        self.assertGreaterEqual(len(problems), 3)

    def test_missing_input_is_a_media_error(self):
        with self.assertRaises(media.MediaError) as ctx:
            asr.transcribe(ROOT / "tests/fixtures/clips/does_not_exist.mp4", ART / "asr/x.json")
        self.assertEqual(ctx.exception.code, "E_INPUT_MISSING")

    def test_no_network_model_resolution(self):
        resolved = asr.resolve_model_path("medium")
        self.assertTrue(Path(resolved).is_dir(), f"model should resolve to a local dir, got {resolved}")
        self.assertTrue((Path(resolved) / "model.bin").exists())


# ========================================================================= VISION

class TestVisionOnFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx_a = vision.analyze(clip("fx_a_zh_portrait"),
                                  ART / "vision" / "test_fx_a.vision.json")
        cls.fx_c = vision.analyze(clip("fx_c_zh_cuts"),
                                  ART / "vision" / "test_fx_c.vision.json")
        cls.pd = None
        if clip("fx_pd_real_16x9_a").exists():
            cls.pd = vision.analyze(clip("fx_pd_real_16x9_a"),
                                    ART / "vision" / "test_fx_pd_a.vision.json")

    def test_contract_valid_and_covers_whole_clip(self):
        doc = self.fx_a
        self.assertEqual(doc["schema"], "semotion.vision/1")
        self.assertEqual(vision.validate(doc), [])
        duration = float(doc["source"]["duration_s"])
        self.assertAlmostEqual(doc["shots"][0]["start_s"], 0.0, places=6)
        self.assertAlmostEqual(doc["shots"][-1]["end_s"], duration, places=3)
        self.assertEqual(doc["shots"][-1]["end_frame"], doc["source"]["n_frames"])
        for shot in doc["shots"]:
            self.assertLessEqual(shot["start_s"], shot["end_s"])

    def test_face_sampling_rate(self):
        doc = self.fx_a
        fps = float(doc["source"]["fps"])
        step = doc["provenance"]["sample_step_frames"]
        self.assertEqual(step, int(round(-(-fps // 5.0))))     # ceil(fps / 5)
        self.assertGreaterEqual(float(doc["provenance"]["scanned_fps"]), 5.0 - 1e-6)

    def test_no_face_is_reported_as_no_face(self):
        """Generated artwork has no real face: the layer must not invent one."""
        doc = self.fx_a
        self.assertEqual(doc["faces"], [])
        self.assertEqual(doc["face_tracks"], [])
        self.assertIsNone(vision.face_box_at(doc, 5.0))
        self.assertIsNone(vision.face_union_box(doc))
        self.assertEqual(doc["safe_zones"]["card"], [0.06, 0.30, 0.88, 0.42])

    def test_shot_detection_matches_ground_truth(self):
        truth = json.loads(ground_truth("fx_c_zh_cuts").read_text(encoding="utf-8"))
        shots = self.fx_c["shots"]
        self.assertEqual(len(shots), len(truth["shots"]))
        for got, want in zip(shots, truth["shots"]):
            self.assertAlmostEqual(got["start_s"], want["start_s"], delta=0.25)
            self.assertAlmostEqual(got["end_s"], want["end_s"], delta=0.25)
        cuts = self.fx_c["provenance"]["shot_detection"]["cuts"]
        self.assertEqual([round(c["t_s"], 2) for c in cuts], [8.0, 16.0])

    def test_static_fixture_has_no_false_cut(self):
        self.assertEqual(len(self.fx_a["shots"]), 1)

    def test_real_face_track(self):
        if self.pd is None:
            self.skipTest("real public-domain fixture not present")
        doc = self.pd
        self.assertEqual(vision.validate(doc), [])
        tracks = vision.face_tracks(doc)
        self.assertGreaterEqual(len(tracks), 1)
        track = tracks[0]
        x, y, w, h = track["bbox_union"]
        frame_area = doc["source"]["width"] * doc["source"]["height"]
        self.assertLess(w * h, 0.6 * frame_area, "a face box must not be the whole frame")
        self.assertGreater(w, 40)
        self.assertLessEqual(x + w, doc["source"]["width"])
        self.assertLessEqual(y + h, doc["source"]["height"])
        self.assertGreaterEqual(track["coverage"], 0.8)
        self.assertGreaterEqual(len(track["frames"]), 5)
        # nearest-neighbour interpolation returns the sampled box itself
        box = vision.box_at_frame(track, track["frames"][0]["frame"])
        self.assertEqual(box, track["frames"][0]["box"])
        mid = vision.box_at_frame(track, track["frames"][0]["frame"] + 1)
        self.assertEqual(len(mid), 4)

    def test_card_zone_avoids_face(self):
        if self.pd is None:
            self.skipTest("real public-domain fixture not present")
        doc = self.pd
        card = doc["safe_zones"]["card"]
        track = doc["face_tracks"][0]
        x, y, w, h = track["bbox_union"]
        face = [x / doc["source"]["width"], y / doc["source"]["height"],
                w / doc["source"]["width"], h / doc["source"]["height"]]
        overlap = vision._rect_overlap(card, face)
        self.assertLess(overlap, 0.05, f"card zone {card} overlaps the face {face}")

    def test_track_association_is_deterministic(self):
        per_frame = {
            0: [{"box": [100, 100, 80, 80], "score": 0.9, "t_s": 0.0}],
            5: [{"box": [104, 102, 80, 80], "score": 0.9, "t_s": 0.2}],
            10: [{"box": [300, 100, 80, 80], "score": 0.9, "t_s": 0.4}],
        }
        tracks = vision._associate_tracks(per_frame)
        self.assertEqual(len(tracks), 2)                      # near box kept, far box split
        self.assertEqual([len(t["frames"]) for t in tracks], [2, 1])
        self.assertEqual(vision._associate_tracks(per_frame), tracks)

    def test_invalid_document_is_rejected(self):
        broken = json.loads(json.dumps(self.fx_a))
        broken["shots"] = []
        broken["safe_zones"]["card"] = [0.9, 0.9, 0.5, 0.5]
        problems = vision.validate(broken)
        self.assertGreaterEqual(len(problems), 2)


# ========================================================================= ANCHORS

class TestAnchorsOnFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.transcript = asr.transcribe(clip("fx_a_zh_portrait"),
                                        ART / "asr" / "test_fx_a.transcript.json")
        cls.truth = A.load_truth(ground_truth("fx_a_zh_portrait"))
        cls.raw = A.classify(cls.transcript, use_cache=False)
        cls.refined = A.refine_onsets(clip("fx_a_zh_portrait"), cls.raw)

    def test_contract_valid_and_required_fields(self):
        doc = self.refined
        self.assertEqual(doc["schema"], "semotion.anchors/1")
        self.assertEqual(A.validate(doc), [])
        self.assertAlmostEqual(doc["duration_s"],
                               float(self.transcript["source"]["duration_s"]), places=3)
        self.assertEqual(doc["source_sha256"], self.transcript["source"]["sha256"])
        for anchor in doc["anchors"]:
            for key in ("id", "kind", "text", "start_s", "end_s", "confidence",
                        "source", "segment_id", "word_ids"):
                self.assertIn(key, anchor)
            self.assertEqual(anchor["source"], "ASR_ALIGNED")
            self.assertGreaterEqual(anchor["confidence"], 0.0)
            self.assertLessEqual(anchor["confidence"], 1.0)

    def test_classifier_covers_keyword_list_and_number(self):
        kinds = {a["kind"] for a in self.refined["anchors"]}
        self.assertIn("keyword", kinds)
        self.assertIn("list_item", kinds)
        self.assertIn("number", kinds)
        texts = [a["text"] for a in self.refined["anchors"]]
        self.assertIn("第一", texts)
        self.assertIn("九成", texts)
        self.assertTrue(any("其实" in t for t in texts))
        self.assertTrue(any("三个" in t for t in texts))

    def test_times_come_from_real_words(self):
        words = {w["id"]: w for w in self.transcript["words"]}
        for anchor in self.refined["anchors"]:
            spelled = [words[i] for i in anchor["word_ids"]]
            self.assertTrue(spelled, f"{anchor['id']} has no words")
            self.assertAlmostEqual(anchor["end_s"],
                                   max(float(w["end_s"]) for w in spelled), places=5)
            # start may be snapped by refine, but never further than the window
            self.assertLessEqual(
                abs(float(anchor["start_s"]) - min(float(w["start_s"]) for w in spelled)),
                0.26 + 1e-6)
            self.assertLess(float(anchor["start_s"]), float(anchor["end_s"]))

    def test_keyword_phrase_merge(self):
        merged = [a for a in self.refined["anchors"] if a.get("merged_words")]
        self.assertTrue(merged, "expected at least one merged keyword phrase")
        texts = " ".join(a["text"] for a in merged)
        self.assertIn("其实是节奏", texts)
        for anchor in self.refined["anchors"]:
            if anchor["kind"] in ("number", "list_item"):
                self.assertNotIn("merged_words", anchor,
                                 f"{anchor['kind']} must keep its exact span")

    def test_refine_block_contract(self):
        for anchor in self.refined["anchors"]:
            refine = anchor.get("refine")
            self.assertIsNotNone(refine, f"{anchor['id']} missing refine")
            self.assertEqual(refine["method"], "energy_onset")
            self.assertEqual(refine["window_ms"], 250)
            self.assertIsInstance(refine["delta_ms"], int)
            self.assertIsInstance(refine["accepted"], bool)
            if anchor["start_s"] != round(float(refine["asr_start_s"]), 6):
                self.assertTrue(refine["accepted"])
        accepted = self.refined["provenance"]["refine"]["n_accepted"]
        total = self.refined["provenance"]["refine"]["n_anchors"]
        self.assertEqual(accepted, total)

    def test_anchor_level_timing_vs_synth_truth(self):
        before = A.timing_errors(self.raw, self.truth)
        after = A.timing_errors(self.refined, self.truth)
        self.assertGreaterEqual(after["samples"], 4, "need enough matched events")
        self.assertLessEqual(after["p90_ms"], 100)
        self.assertLessEqual(after["max_ms"], 100)
        self.assertLessEqual(after["p50_ms"], before["p50_ms"] + 5)

    def test_word_onset_timing_vs_synth_truth(self):
        calibration = A.unit_onset_anchors(self.transcript, self.truth)
        before = A.timing_errors(calibration, self.truth)
        after = A.timing_errors(A.refine_onsets(clip("fx_a_zh_portrait"), calibration),
                                self.truth)
        self.assertGreaterEqual(before["samples"], 10)
        self.assertGreaterEqual(after["samples"], 10)
        # ASR word starts carry a systematic positive bias on this fixture
        self.assertGreaterEqual(before["p50_ms"], 50)
        self.assertLessEqual(after["p50_ms"], 20)
        self.assertLessEqual(after["p90_ms"], 100)
        self.assertLessEqual(after["max_ms"], 150)

    def test_cache_separation(self):
        cache = A.anchors_cache_path(self.transcript)
        self.assertTrue(cache.exists())
        cached = json.loads(cache.read_text(encoding="utf-8"))
        self.assertTrue(all(a.get("refine") is None for a in cached["anchors"]),
                        "classify() cache must not contain refined anchors")
        second = A.classify(self.transcript)     # cache hit
        self.assertEqual([a["id"] for a in second["anchors"]],
                         [a["id"] for a in self.raw["anchors"]])
        self.assertEqual(second["provenance"]["refine"], None)

    def test_empty_transcript_is_handled(self):
        empty = synthetic_transcript("", [])
        empty["segments"] = []
        doc = A.classify(empty, use_cache=False, write_cache=False)
        self.assertEqual(doc["anchors"], [])
        self.assertEqual(A.validate(doc), [])

    def test_chinese_rules_traditional_and_percent(self):
        text = "先說結論，三個指標，百分之七十的問題。"
        words = ["先", "說", "結", "論,", "三", "個", "指", "標,", "百", "分", "之",
                 "七", "十", "的", "問", "題。"]
        transcript = synthetic_transcript(text, words)
        doc = A.classify(transcript, use_cache=False, write_cache=False)
        kinds = {a["kind"] for a in doc["anchors"]}
        self.assertIn("conclusion", kinds)       # 先說結論
        self.assertIn("number", kinds)           # 三個 / 百分之七十
        numbers = [a["text"] for a in doc["anchors"] if a["kind"] == "number"]
        self.assertTrue(any("百分之七十" in t for t in numbers))
        self.assertTrue(any(a["text"].startswith("三個") for a in doc["anchors"]))

    def test_filler_numbers_are_not_anchors(self):
        text = "今天聊一个真实的痛点，改一次就要重做一遍。"
        words = ["今天", "聊", "一个", "真实", "的", "痛点,", "改", "一次", "就要",
                 "重做", "一遍。"]
        doc = A.classify(synthetic_transcript(text, words), use_cache=False,
                         write_cache=False)
        number_texts = [a["text"] for a in doc["anchors"] if a["kind"] == "number"]
        self.assertNotIn("一个", number_texts)
        self.assertNotIn("一次", number_texts)
        self.assertNotIn("一遍", number_texts)

    def test_load_truth(self):
        self.assertEqual(len(self.truth), 14)
        for unit in self.truth:
            for key in ("unit_id", "text", "start_s", "end_s", "role", "truth"):
                self.assertIn(key, unit)
            self.assertEqual(unit["truth"], "SYNTH_KNOWN")
            self.assertLess(unit["start_s"], unit["end_s"])


# ======================================================================== MANIFEST

class TestManifest(unittest.TestCase):
    REQUIRED = ("id", "kind", "truth_class", "split", "path", "sha256", "duration_s",
                "width", "height", "aspect", "fps", "language", "source", "license",
                "summary", "person_position", "camera_motion", "label_types",
                "usable_for", "not_usable_for")

    @classmethod
    def setUpClass(cls):
        path = ROOT / "tests/fixtures/manifest.json"
        cls.doc = json.loads(path.read_text(encoding="utf-8"))
        cls.entries = {e["id"]: e for e in cls.doc["fixtures"]}

    def test_every_clip_is_registered(self):
        on_disk = {d.name for d in CLIPS.iterdir() if d.is_dir() and (d / f"{d.name}.mp4").exists()}
        self.assertEqual(on_disk - set(self.entries), set(), "unregistered fixture clips")

    def test_required_fields(self):
        for fid, entry in self.entries.items():
            for key in self.REQUIRED:
                self.assertIn(key, entry, f"{fid} missing {key}")
            self.assertIn(entry["truth_class"], ("SYNTH_KNOWN", "HUMAN", "MACHINE_WEAK"))
            self.assertIn(entry["split"], ("train", "dev", "holdout", "calibration"))
            self.assertGreater(entry["duration_s"], 0)
            self.assertGreater(entry["fps"], 0)

    def test_licenses_are_explicit(self):
        self.assertEqual(self.entries["fx_a_zh_portrait"]["license"],
                         "synthetic (macOS say + generated artwork), no third-party rights")
        for fid in ("fx_pd_real_16x9_a", "fx_pd_real_16x9_b"):
            entry = self.entries[fid]
            self.assertEqual(entry["kind"], "REAL_PD")
            self.assertEqual(entry["truth_class"], "MACHINE_WEAK")
            self.assertIn("Public domain", entry["license"])
            self.assertFalse(entry["timing_truth"])
            self.assertTrue(any("timing" in s for s in entry["not_usable_for"]))

    def test_sha256_matches_files(self):
        for fid, entry in self.entries.items():
            path = ROOT / entry["path"]
            self.assertTrue(path.exists(), f"{fid}: missing {path}")
            self.assertEqual(entry["sha256"], media.sha256_file(path))

    def test_split_policy_is_honest(self):
        self.assertIn("holdout", self.doc["split_policy"])
        self.assertIn("calibration", self.doc["split_policy"])
        self.assertIn("non-blind", self.doc["split_policy"]["note"])
        counts = self.doc["counts"]
        self.assertEqual(counts["fixtures"], len(self.entries))
        self.assertGreaterEqual(counts["synth_known_units"], 50)
        self.assertEqual(counts["calibration_units"],
                         sum(e["units"]["count"] for e in self.entries.values()
                             if e["split"] == "calibration"))

    def test_calibration_split_is_registered(self):
        calibration = {fid: e for fid, e in self.entries.items() if e["split"] == "calibration"}
        self.assertGreaterEqual(len(calibration), 3)
        for fid, entry in calibration.items():
            self.assertEqual(entry["kind"], "SYNTH_KNOWN")
            self.assertEqual(entry["truth_class"], "SYNTH_KNOWN")
            self.assertTrue(entry["timing_truth"])
            self.assertGreaterEqual(entry["units"]["count"], 18)
            self.assertLessEqual(entry["units"]["count"], 20)
            self.assertTrue(40.0 <= entry["duration_s"] <= 60.0 + 1e-6,
                            f"{fid}: duration {entry['duration_s']} outside 40-60 s")
        self.assertGreaterEqual(self.doc["timing_calibration"]["events_measured"],
                                self.doc["timing_calibration"]["min_events_required"])

    def test_calibration_unit_text_lengths(self):
        """Every calibration unit must be 5-9 characters (short, single-phrase)."""
        for fid, entry in self.entries.items():
            if entry["split"] != "calibration":
                continue
            gt = json.loads((ROOT / entry["ground_truth"]).read_text(encoding="utf-8"))
            for unit in gt["units"]:
                self.assertTrue(5 <= len(unit["text"]) <= 9,
                                f"{fid} {unit['unit_id']} text length {len(unit['text'])}")

    def test_calibration_prefix_events_from_cached_transcripts(self):
        """Each calibration clip must yield >= 15 unit-initial anchor events.

        Skipped when the ASR transcripts have not been generated yet (they are
        produced by tests/eval_timing.py); the assertion never invents data.
        """
        checked = 0
        for fid, entry in self.entries.items():
            if entry["split"] != "calibration":
                continue
            transcript_path = ART / "anchors" / f"{fid}.transcript.json"
            gt_path = ROOT / entry["ground_truth"]
            if not transcript_path.exists() or not gt_path.exists():
                continue
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
            doc = A.classify(transcript)
            units = A.load_truth(gt_path)
            pairs, _unmatched = A.prefix_matches(doc, units)
            events = len({p["unit_id"] for p in pairs})
            self.assertGreaterEqual(events, 15,
                                    f"{fid}: only {events} unit-initial anchor events")
            checked += 1
        if checked == 0:
            self.skipTest("no cached calibration transcripts; run tests/eval_timing.py first")

    def test_calibration_report_meets_threshold(self):
        path = ART / "anchors" / "calibration_report.json"
        if not path.exists():
            self.skipTest("calibration report not generated yet")
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["truth_class"], "SYNTH_KNOWN")
        self.assertEqual(report["split"], "calibration")
        totals = report["totals"]
        self.assertGreaterEqual(totals["events"], totals["min_events_required"])
        self.assertEqual(totals["verdict"], "PASS")
        self.assertLessEqual(totals["after"]["p90_ms"], 100)
        for fid, fixture in report["fixtures"].items():
            self.assertGreaterEqual(fixture["events"], 15, f"{fid}: too few events")


if __name__ == "__main__":
    unittest.main(verbosity=2)
