"""Tests for the Motion IR, layout, rasteriser and renderer (motion-render scope).

No network, no model downloads: everything uses the synthetic fixture under
``tests/fixtures/clips`` and the bundled OFL font. Ground-truth unit times are
``SYNTH_KNOWN`` assembly times, used here only to build explicit test IRs.

Run:
    PYTHONPATH=src .venv/bin/python -m unittest tests.test_ir_render -v
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from semotion import design, effects, ir as irmod, media, raster, render as rmod  # noqa: E402
from semotion.effects import number as number_effect  # noqa: E402

CLIP_DIR = ROOT / "tests/fixtures/clips/fx_a_zh_portrait"
SOURCE = CLIP_DIR / "fx_a_zh_portrait.mp4"
GT = json.loads((CLIP_DIR / "ground_truth.json").read_text(encoding="utf-8"))
WORK = ROOT / "artifacts/tmp/tests_ir_render"

CANVAS = {"width": GT["width"], "height": int(GT["height"]), "fps": float(GT["fps"])}
ZONES = {"title": [0.08, 0.06, 0.84, 0.12],
         "caption": [0.08, 0.78, 0.84, 0.11],
         "card": [0.06, 0.30, 0.88, 0.42]}
KIND = {"plain": "plain", "keyword": "keyword", "number": "number",
        "list_item": "list_item"}


# --------------------------------------------------------------------------- fixtures
def anchors_from_truth() -> dict:
    out = []
    for i, u in enumerate(GT["units"], start=1):
        out.append({"id": f"a{i:02d}", "kind": KIND.get(u["role"], "plain"),
                    "text": u["text"], "start_s": u["start_s"], "end_s": u["end_s"],
                    "confidence": 0.9, "source": "SYNTH_KNOWN",
                    "segment_id": u["unit_id"], "word_ids": [u["unit_id"]]})
    return {"schema": "semotion.anchors/1", "anchors": out,
            "duration_s": GT["duration_s"]}


def transcript_from_truth(units: list[dict], *, source: Path, duration: float,
                          sha: str) -> dict:
    return {
        "schema": "semotion.transcript/1",
        "source": {"path": str(source), "sha256": sha, "duration_s": duration,
                   "width": CANVAS["width"], "height": CANVAS["height"],
                   "fps": CANVAS["fps"]},
        "language": "zh",
        "segments": [{"id": u["unit_id"], "start_s": u["start_s"], "end_s": u["end_s"],
                      "text": u["text"]} for u in units],
        "words": [{"id": u["unit_id"], "segment_id": u["unit_id"], "text": u["text"],
                   "start_s": u["start_s"], "end_s": u["end_s"], "probability": 0.99}
                  for u in units],
    }


def vision_doc() -> dict:
    return {"schema": "semotion.vision/1",
            "source": {"width": CANVAS["width"], "height": CANVAS["height"],
                       "fps": CANVAS["fps"], "n_frames": GT["n_frames"]},
            "faces": [], "face_tracks": [],
            "shots": [{"index": 0, "start_s": 0.0, "end_s": GT["duration_s"],
                       "start_frame": 0, "end_frame": GT["n_frames"]}],
            "safe_zones": copy.deepcopy(ZONES)}


def _decision(did: str, effect: str, anchor_ids: list[str], anchors: dict,
              emphasis: str = "high", content: dict | None = None) -> dict:
    by_id = {a["id"]: a for a in anchors["anchors"]}
    start = min(by_id[i]["start_s"] for i in anchor_ids)
    end = max(by_id[i]["end_s"] for i in anchor_ids)
    hold = {"keyword": 0.30, "number": 0.35, "list": 0.45, "card": 0.50}[effect]
    text = by_id[anchor_ids[0]]["text"]
    payload = dict(content or {"text": text})
    if effect == "list":
        payload.setdefault("text", text)
        payload.setdefault("items", [by_id[i]["text"] for i in anchor_ids])
    return {"id": did, "anchor_id": anchor_ids[0], "anchor_ids": list(anchor_ids),
            "effect": effect, "emphasis": emphasis, "reason": "unit_test",
            "confidence": 0.9, "decision_model": "test-manual",
            "content": payload,
            "window": {"start_s": round(start, 6), "end_s": round(end + hold, 6)}}


def plan_all_four(anchors: dict) -> dict:
    """keyword + number + list + card, matching the fixture's unit times."""
    return {
        "schema": "semotion.plan/1", "style": "smart_balanced", "canvas": dict(CANVAS),
        "decisions": [
            _decision("d01", "card", ["a02"], anchors, "medium"),
            _decision("d02", "keyword", ["a04"], anchors),
            _decision("d03", "number", ["a05"], anchors),
            _decision("d04", "list", [f"a{i:02d}" for i in range(7, 13)], anchors, "high"),
            _decision("d05", "keyword", ["a14"], anchors),
        ],
    }


def build_full_ir() -> dict:
    anchors = anchors_from_truth()
    tr = transcript_from_truth(GT["units"], source=SOURCE,
                               duration=float(GT["duration_s"]), sha=GT["sha256"])
    return irmod.build(tr, vision_doc(), anchors, plan_all_four(anchors), SOURCE, CANVAS)


def _box(el: dict) -> list[int]:
    return [int(v) for v in el["layout"]["box"]]


def _inter(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy


# --------------------------------------------------------------------------- tests
class TestIrBuildAndValidate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        WORK.mkdir(parents=True, exist_ok=True)
        cls.ir = build_full_ir()

    def test_build_produces_valid_ir_with_four_effect_types(self) -> None:
        errs = irmod.validate(self.ir)
        self.assertEqual(errs, [], f"validate errors: {errs}")
        types = {e["type"] for e in self.ir["elements"]}
        self.assertEqual(types, {"keyword", "number", "list", "card"})
        self.assertEqual(self.ir["schema"], "semotion.motion-ir/1")
        self.assertEqual(self.ir["ir_version"], "1.0.0")

    def test_frames_match_seconds_and_source(self) -> None:
        fps = float(self.ir["source"]["fps"])
        n = int(self.ir["source"]["n_frames"])
        for el in self.ir["elements"]:
            w = el["window"]
            self.assertEqual(w["start_frame"], round(w["start_s"] * fps))
            self.assertEqual(w["end_frame"], round(w["end_s"] * fps))
            self.assertGreaterEqual(w["start_frame"], 0)
            self.assertGreater(w["end_frame"], w["start_frame"])
            self.assertLessEqual(w["end_frame"], n)

    def test_boxes_inside_canvas_and_off_faces(self) -> None:
        cw, ch = CANVAS["width"], CANVAS["height"]
        face = self.ir["semantic"]["face_safe"]["bbox_union"]
        for el in self.ir["elements"]:
            x, y, w, h = _box(el)
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + w, cw)
            self.assertLessEqual(y + h, ch)
            if face:
                area = face[2] * face[3]
                self.assertLessEqual(_inter(_box(el), face) / area, 0.01,
                                     f"{el['id']} overlaps the face-safe box")

    def test_hash_ignores_itself_and_roundtrips(self) -> None:
        h1 = irmod.compute_ir_hash(self.ir)
        self.assertEqual(h1, self.ir["provenance"]["ir_sha256"])
        tampered = copy.deepcopy(self.ir)
        tampered["provenance"]["ir_sha256"] = "0" * 64
        self.assertEqual(irmod.compute_ir_hash(tampered), h1)
        path = WORK / "roundtrip.json"
        irmod.dump(self.ir, path)
        loaded = irmod.load(path)
        self.assertEqual(loaded, self.ir)
        self.assertEqual(irmod.validate(loaded), [])

    def test_number_semantics_are_preserved(self) -> None:
        el = next(e for e in self.ir["elements"] if e["type"] == "number")
        numeric = el["content"]["numeric"]
        self.assertEqual(numeric["value"], 90.0)     # 九成 -> 90 %
        self.assertEqual(numeric["final"], "90%")
        self.assertEqual(el["content"]["text"], "九成的剪辑")
        self.assertEqual(el["content"]["numeric_display"], "90%的剪辑")
        cases = {"都卡在三个问题": "3个", "百分之三十": "30%", "3.5倍": "3.5倍",
                 "第一": "1", "九成": "90%"}
        for text, expected in cases.items():
            info = number_effect.parse_numeric(text)
            self.assertIsNotNone(info, text)
            self.assertEqual(info["final"], expected, text)
        # terminal value is exact (no rounding drift at the end of the count-up)
        terminal = number_effect._numstr(numeric, numeric["value"])
        self.assertEqual(terminal, numeric["final"])

    def test_list_rows_merge_ordinals(self) -> None:
        el = next(e for e in self.ir["elements"] if e["type"] == "list")
        rows = el["content"]["rows"]
        self.assertEqual([r["marker"] for r in rows], ["1", "2", "3"])
        self.assertEqual([r["text"] for r in rows],
                         ["不知道什么时候加字", "加了又挡住脸", "改一次就要重做一遍"])
        # onsets are anchor-relative and monotonic
        ats = [r["at_s"] for r in rows]
        self.assertEqual(ats, sorted(ats))

    def test_list_is_wide_and_single_line(self) -> None:
        el = next(e for e in self.ir["elements"] if e["type"] == "list")
        box = self.ir["elements"][0]["layout"]["box"]
        x, y, w, h = [int(v) for v in el["layout"]["box"]]
        self.assertGreaterEqual(w, int(0.6 * CANVAS["width"]), "list panel must be prominent")
        spec = raster.element_spec(el, CANVAS)
        self.assertTrue(all(len(r["lines"]) == 1 for r in spec["rows"]),
                        [r["lines"] for r in spec["rows"]])
        self.assertGreaterEqual(spec["font_px"], 56)
        self.assertLessEqual(spec["font_px"], 68)

    def test_frame_count_source_is_decoded_stream(self) -> None:
        src = self.ir["source"]
        info = media.info(SOURCE)
        self.assertEqual(src["n_frames"], int(info.stream_frames))
        self.assertEqual(src["frame_source"], "nb_frames")
        if int(info.stream_frames) != round(src["duration_s"] * src["fps"]):
            self.assertTrue(irmod.validate(self.ir) == [])


class TestIrInvariants(unittest.TestCase):
    """Each negative case must be reported with the element id, not swallowed."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.base = build_full_ir()

    def _mutate(self, fn) -> list[str]:
        doc = copy.deepcopy(self.base)
        fn(doc)
        irmod.seal(doc)
        return irmod.validate(doc)

    def _el(self, doc: dict, etype: str) -> dict:
        return next(e for e in doc["elements"] if e["type"] == etype)

    def test_box_outside_canvas(self) -> None:
        def mutate(doc):
            self._el(doc, "keyword")["layout"]["box"] = [1000, 900, 400, 200]
        errs = self._mutate(mutate)
        self.assertTrue(any("not fully inside canvas" in e for e in errs), errs)

    def test_window_beyond_hold_budget(self) -> None:
        def mutate(doc):
            el = self._el(doc, "keyword")
            fps = float(doc["source"]["fps"])
            el["window"]["end_s"] += 0.8
            el["window"]["end_frame"] = round(el["window"]["end_s"] * fps)
            el["window"]["hold_s"] = round(el["window"]["end_s"]
                                          - el["window"]["anchor_end_s"], 6)
        errs = self._mutate(mutate)
        self.assertTrue(any("0.8s hold" in e for e in errs), errs)

    def test_onset_before_anchor_tolerance(self) -> None:
        def mutate(doc):
            el = self._el(doc, "keyword")
            fps = float(doc["source"]["fps"])
            el["window"]["start_s"] -= 0.8
            el["window"]["start_frame"] = round(el["window"]["start_s"] * fps)
        errs = self._mutate(mutate)
        self.assertTrue(any("before anchor start" in e for e in errs), errs)

    def test_hold_s_declared_and_bounded(self) -> None:
        fps = float(self.base["source"]["fps"])
        for el in self.base["elements"]:
            win = el["window"]
            self.assertIn("hold_s", win, el["id"])
            self.assertGreaterEqual(win["hold_s"], 0.0)
            self.assertLessEqual(win["hold_s"], 0.8 + 0.5 / fps + 1e-6, el["id"])
            # onset never precedes the anchor start by more than 250 ms
            self.assertGreaterEqual(win["start_s"],
                                    win["anchor_start_s"] - 0.25 - 0.5 / fps - 1e-6)

    def test_missing_hold_s_rejected(self) -> None:
        def mutate(doc):
            el = self._el(doc, "keyword")
            el["window"].pop("hold_s", None)
        errs = self._mutate(mutate)
        self.assertTrue(any("hold_s is not declared" in e for e in errs), errs)

    def test_undeclared_simultaneous_elements(self) -> None:
        def mutate(doc):
            src = self._el(doc, "keyword")
            twin = copy.deepcopy(src)
            twin["id"] = "el_dup"
            twin["content"]["text"] = src["content"]["text"]
            doc["elements"].append(twin)
            for track in doc["tracks"]:
                if src["id"] in track["elements"]:
                    track["elements"].append("el_dup")
            doc["collision_ok"] = []
        errs = self._mutate(mutate)
        self.assertTrue(any("not declared in collision_ok" in e for e in errs), errs)

    def test_overlap_declared_passes(self) -> None:
        def mutate(doc):
            docs = doc
            src = self._el(docs, "keyword")
            twin = copy.deepcopy(src)
            twin["id"] = "el_dup"
            docs["elements"].append(twin)
            for track in docs["tracks"]:
                if src["id"] in track["elements"]:
                    track["elements"].append("el_dup")
            docs["collision_ok"] = [[src["id"], "el_dup"]]
        errs = self._mutate(mutate)
        self.assertEqual(errs, [], errs)

    def test_face_overlap_without_allowance(self) -> None:
        def mutate(doc):
            el = self._el(doc, "card")
            doc["semantic"]["face_safe"]["bbox_union"] = _box(el)
        errs = self._mutate(mutate)
        self.assertTrue(any("face-safe box" in e for e in errs), errs)

    def test_allow_face_overlap_requires_reason(self) -> None:
        def mutate(doc):
            el = self._el(doc, "card")
            doc["semantic"]["face_safe"]["bbox_union"] = _box(el)
            el["layout"]["allow_face_overlap"] = True
        errs = self._mutate(mutate)
        self.assertTrue(any("recorded reason" in e for e in errs), errs)

    def test_fabricated_text(self) -> None:
        def mutate(doc):
            self._el(doc, "keyword")["content"]["text"] = "这段文案不存在"
        errs = self._mutate(mutate)
        self.assertTrue(any("not found in anchors or transcript" in e for e in errs), errs)

    def test_unknown_anchor(self) -> None:
        def mutate(doc):
            self._el(doc, "keyword")["anchor_refs"] = ["a99"]
        errs = self._mutate(mutate)
        self.assertTrue(any("a99" in e and "not found" in e for e in errs), errs)

    def test_track_membership(self) -> None:
        def mutate(doc):
            eid = doc["elements"][0]["id"]
            for track in doc["tracks"]:
                track["elements"] = [e for e in track["elements"] if e != eid]
        errs = self._mutate(mutate)
        self.assertTrue(any("belongs to no track" in e for e in errs), errs)

        def mutate2(doc):
            eid = doc["elements"][0]["id"]
            doc["tracks"].append({"id": "t_dup", "z": 25, "elements": [eid]})
        errs2 = self._mutate(mutate2)
        self.assertTrue(any("belongs to 2 tracks" in e for e in errs2), errs2)

    def test_bad_hash(self) -> None:
        doc = copy.deepcopy(self.base)
        doc["elements"][0]["style"]["font_size_px"] += 2
        errs = irmod.validate(doc)
        self.assertTrue(any("ir_sha256 mismatch" in e for e in errs), errs)

    def test_empty_content_rejected(self) -> None:
        def mutate(doc):
            self._el(doc, "keyword")["content"]["text"] = "  "
        errs = self._mutate(mutate)
        self.assertTrue(any("content.text is empty" in e for e in errs), errs)


class TestDesign(unittest.TestCase):
    def _element(self, etype: str, text: str, canvas: dict) -> dict:
        font = {"keyword": 96, "number": 84, "list": 72, "card": 78}[etype]
        el = {"id": f"el_{etype}", "type": etype,
              "style": {"font_size_px": font, "padding_px": 36, "radius_px": 28,
                        "bg": "rgba(12,16,28,0.72)", "color": "#F5F7FA",
                        "accent": "#F2B233"},
              "content": {"text": text},
              "window": {"start_frame": 0, "end_frame": 25}}
        if etype == "list":
            el["content"] = {"text": "第一", "items": ["第一", "第二步"],
                             "rows": [{"marker": "1", "text": "第一", "at_s": 0.0},
                                      {"marker": "2", "text": "第二步", "at_s": 0.5}]}
        return el

    def test_safe_area_boxes_with_real_face(self) -> None:
        canvas = {"width": 1280, "height": 720, "fps": 25.0}
        vision = {"face_tracks": [{"id": "f0", "bbox_union": [491, 87, 280, 392],
                                   "coverage": 0.97}],
                  "safe_zones": {"title": [0.05, 0.05, 0.55, 0.14],
                                 "caption": [0.05, 0.80, 0.90, 0.12],
                                 "card": [0.55, 0.26, 0.40, 0.46]}}
        info = design.safe_area_boxes(canvas, vision)
        self.assertEqual(info["face_union"], [491, 87, 280, 392])
        self.assertEqual(info["face_box"], [491 - 32, 87 - 32, 280 + 64, 392 + 64])
        self.assertTrue(info["has_space"])
        face = info["face_box"]
        face_area = face[2] * face[3]
        for etype, text in (("keyword", "其实是节奏"), ("number", "九成的剪辑"),
                            ("list", "第一"), ("card", "今天聊一个真实的痛点")):
            el = self._element(etype, text, canvas)
            lay = design.layout(el, canvas, vision, [])
            box = lay["box"]
            self.assertGreaterEqual(box[0], 0)
            self.assertGreaterEqual(box[1], 0)
            self.assertLessEqual(box[0] + box[2], canvas["width"])
            self.assertLessEqual(box[1] + box[3], canvas["height"])
            self.assertLessEqual(_inter(box, face) / face_area, 0.01,
                                 f"{etype}: box {box} overlaps face safe box {face}")
            self.assertIn(lay["zone"], ("title", "card", "caption"))
            self.assertFalse(lay["degraded"], lay.get("reason"))

    def test_portrait_and_landscape_both_place(self) -> None:
        for canvas in ({"width": 1080, "height": 1920, "fps": 25.0},
                       {"width": 1920, "height": 1080, "fps": 25.0}):
            vision = {"faces": [], "face_tracks": [],
                      "safe_zones": copy.deepcopy(ZONES)}
            for etype, text in (("keyword", "其实是节奏"), ("number", "九成的剪辑"),
                                ("list", "第一"), ("card", "今天聊一个真实的痛点")):
                lay = design.layout(self._element(etype, text, canvas), canvas, vision, [])
                box = lay["box"]
                self.assertGreaterEqual(box[0], 0)
                self.assertGreaterEqual(box[1], 0)
                self.assertLessEqual(box[0] + box[2], canvas["width"])
                self.assertLessEqual(box[1] + box[3], canvas["height"])

    def test_face_covered_canvas_degrades_instead_of_crashing(self) -> None:
        canvas = {"width": 1080, "height": 1920, "fps": 25.0}
        vision = {"face_tracks": [{"id": "f0", "bbox_union": [0, 0, 1080, 1920]}],
                  "safe_zones": copy.deepcopy(ZONES)}
        lay = design.layout(self._element("keyword", "其实是节奏", canvas), canvas, vision, [])
        self.assertTrue(lay["degraded"])
        self.assertTrue(lay["reason"])
        box = lay["box"]
        self.assertGreaterEqual(box[0], 0)
        self.assertGreaterEqual(box[1], 0)
        self.assertLessEqual(box[0] + box[2], 1080)
        self.assertLessEqual(box[1] + box[3], 1920)

    def test_occupied_boxes_are_respected_only_when_simultaneous(self) -> None:
        canvas = {"width": 1080, "height": 1920, "fps": 25.0}
        vision = {"faces": [], "face_tracks": [], "safe_zones": copy.deepcopy(ZONES)}
        el = self._element("keyword", "其实是节奏", canvas)
        el["window"] = {"start_frame": 0, "end_frame": 25}
        first = design.layout(el, canvas, vision, [])
        occupied = [{"id": "other", "box": first["box"],
                     "window": {"start_frame": 0, "end_frame": 25}}]
        second = design.layout(el, canvas, vision, occupied)
        inter = _inter(first["box"], second["box"])
        small = min(first["box"][2] * first["box"][3], second["box"][2] * second["box"][3])
        self.assertLessEqual(inter / small, 0.05)
        # non-overlapping windows must not be treated as occupied
        occupied_later = [{"id": "other", "box": first["box"],
                           "window": {"start_frame": 100, "end_frame": 125}}]
        third = design.layout(el, canvas, vision, occupied_later)
        self.assertEqual(third["box"], first["box"])

    def test_collisions_helper(self) -> None:
        boxes = [{"id": "a", "box": [0, 0, 100, 100]},
                 {"id": "b", "box": [50, 50, 100, 100]},
                 {"id": "c", "box": [500, 500, 10, 10]}]
        cols = design.collisions(boxes)
        self.assertEqual(len(cols), 1)
        ida, idb, ratio = cols[0]
        self.assertEqual((ida, idb), ("a", "b"))
        self.assertAlmostEqual(ratio, 2500 / 10000, places=6)


class TestRaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ir = build_full_ir()
        cls.canvas = dict(CANVAS)

    def test_each_effect_renders_its_window_only_and_stays_inside_box(self) -> None:
        for el in self.ir["elements"]:
            out = WORK / f"raster_{el['id']}"
            paths = raster.render_element_png(el, self.canvas, out)
            w = el["window"]
            expected = w["end_frame"] - w["start_frame"]
            self.assertEqual(len(paths), expected, el["id"])
            self.assertEqual(paths[0].name, f"f{w['start_frame']:05d}.png")
            self.assertEqual(paths[-1].name, f"f{w['end_frame'] - 1:05d}.png")
            mid = paths[len(paths) // 2]
            img = Image.open(mid)
            self.assertEqual(img.mode, "RGBA", el["id"])
            self.assertEqual(img.size, (CANVAS["width"], CANVAS["height"]))
            alpha = np.asarray(img.getchannel("A"))
            ys, xs = np.nonzero(alpha)
            self.assertGreater(xs.size, 0, f"{el['id']} painted nothing")
            bx, by, bw, bh = _box(el)
            self.assertGreaterEqual(int(xs.min()), bx, el["id"])
            self.assertGreaterEqual(int(ys.min()), by, el["id"])
            self.assertLessEqual(int(xs.max()) + 1, bx + bw, el["id"])
            self.assertLessEqual(int(ys.max()) + 1, by + bh, el["id"])

    def test_first_frame_is_already_visible(self) -> None:
        """Rendered onset must land on window.start_frame (no systematic 1-frame lag)."""
        for el in self.ir["elements"]:
            out = WORK / f"onset_{el['id']}"
            paths = raster.render_element_png(el, self.canvas, out)
            first = np.asarray(Image.open(paths[0]).getchannel("A"))
            mid = np.asarray(Image.open(paths[len(paths) // 2]).getchannel("A"))
            self.assertGreater(int(first.max()), 20, f"{el['id']}: first frame invisible")
            self.assertGreater(int((first > 8).sum()), 100, f"{el['id']}: first frame too faint")
            # the entrance must still be progressing (first frame is not the final pose)
            self.assertLess(float(first.mean()), float(mid.mean()) + 1e-6, el["id"])

    def test_raster_is_deterministic(self) -> None:
        el = next(e for e in self.ir["elements"] if e["type"] == "number")
        a = raster.render_element_png(el, self.canvas, WORK / "det_a")
        b = raster.render_element_png(el, self.canvas, WORK / "det_b")
        self.assertEqual(len(a), len(b))
        for pa, pb in zip(a, b):
            self.assertEqual(pa.read_bytes(), pb.read_bytes(), pa.name)


def _trim_clip(dst: Path, start_s: float, dur_s: float) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    media.run([media.FFMPEG, "-v", "error", "-y", "-ss", f"{start_s:.3f}",
               "-t", f"{dur_s:.3f}", "-i", str(SOURCE),
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
               "-ar", "48000", "-movflags", "+faststart", str(dst)])
    return dst


class TestRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        WORK.mkdir(parents=True, exist_ok=True)
        cls.ir = build_full_ir()

    def test_render_short_clip_10s(self) -> None:
        clip = _trim_clip(WORK / "clip_10s.mp4", 6.0, 10.0)
        info = media.info(clip)
        sha = media.sha256_file(clip)
        units = []
        for u in GT["units"]:
            if u["start_s"] >= 6.0 and u["end_s"] <= 16.0:
                units.append({**u, "start_s": u["start_s"] - 6.0,
                              "end_s": u["end_s"] - 6.0})
        self.assertTrue(units, "no ground-truth units inside the 10 s window")
        anchors = {"schema": "semotion.anchors/1", "duration_s": info.duration,
                   "anchors": [{"id": f"a{i:02d}",
                                "kind": KIND.get(u["role"], "plain"), "text": u["text"],
                                "start_s": u["start_s"], "end_s": u["end_s"],
                                "confidence": 0.9, "source": "SYNTH_KNOWN",
                                "segment_id": u["unit_id"], "word_ids": [u["unit_id"]]}
                               for i, u in enumerate(units, start=1)]}
        by_text = {a["text"]: a for a in anchors["anchors"]}
        canvas = {"width": info.width, "height": info.height, "fps": info.fps}
        tr = transcript_from_truth(units, source=clip, duration=info.duration, sha=sha)
        decisions = []
        for text, effect in ((units[1]["text"] if len(units) > 1 else units[0]["text"], "keyword"),):
            a = by_text[text]
            decisions.append(_decision("d01", effect, [a["id"]], anchors))
        docs = irmod.build(tr, vision_doc(), anchors,
                           {"schema": "semotion.plan/1", "style": "smart_balanced",
                            "canvas": canvas, "decisions": decisions}, clip, canvas)
        self.assertEqual(irmod.validate(docs), [])
        out = WORK / "clip_10s_motion.mp4"
        man = rmod.render(docs, clip, out)
        self.assertEqual(man["checks"]["verdict"], "PASS")
        probed = media.info(out)
        self.assertAlmostEqual(probed.duration, info.duration, delta=1.0 / info.fps + 0.01)
        self.assertEqual((probed.width, probed.height), (info.width, info.height))
        self.assertEqual(probed.video_codec, "h264")
        self.assertEqual(probed.audio_codec, "aac")
        self.assertEqual(man["output"]["n_frames"], int(round(info.duration * info.fps)))
        self.assertGreaterEqual(man["overlay"]["rendered_frames"], 1)

    def test_render_full_fixture(self) -> None:
        out = WORK / "fx_a_full.mp4"
        man = rmod.render(self.ir, SOURCE, out)
        self.assertEqual(man["checks"]["verdict"], "PASS", man["checks"]["problems"])
        info = media.info(out)
        self.assertEqual(man["output"]["n_frames"], int(self.ir["source"]["n_frames"]))
        self.assertAlmostEqual(info.duration, float(self.ir["source"]["duration_s"]),
                               delta=1.0 / float(self.ir["source"]["fps"]) + 0.01)
        self.assertEqual((info.width, info.height), (CANVAS["width"], CANVAS["height"]))
        self.assertEqual(info.video_codec, "h264")
        self.assertEqual(info.audio_codec, "aac")
        self.assertGreaterEqual(man["audio"]["correlation"], 0.99)
        man_path = out.parent / "render_manifest.json"
        self.assertTrue(man_path.exists())
        loaded = json.loads(man_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["schema"], "semotion.render/1")
        self.assertEqual(loaded["ir_sha256"], self.ir["provenance"]["ir_sha256"])
        self.assertTrue(loaded["commands"])
        self.assertTrue(loaded["commands"][0].startswith("ffmpeg "))
        self.assertTrue(loaded["tool_versions"]["ffmpeg"])

    def test_determinism_two_renders_identical_frames(self) -> None:
        a = WORK / "det_run_a.mp4"
        b = WORK / "det_run_b.mp4"
        rmod.render(self.ir, SOURCE, a)
        rmod.render(self.ir, SOURCE, b)
        ha = media.video_frame_hashes(a)
        hb = media.video_frame_hashes(b)
        self.assertEqual(len(ha), len(hb))
        self.assertEqual(ha, hb, "frame hashes differ between two renders of the same IR")

    def test_preview_manifest_reports_actual_resolution(self) -> None:
        out = WORK / "fx_a_preview.mp4"
        man = rmod.render(self.ir, SOURCE, out, preview=True)
        self.assertEqual(man["checks"]["verdict"], "PASS", man["checks"]["problems"])
        self.assertEqual(man["output"]["width"], CANVAS["width"] // 2)
        self.assertEqual(man["output"]["height"], CANVAS["height"] // 2)
        self.assertTrue(man["preview"])

    def test_render_refuses_wrong_source_and_invalid_ir(self) -> None:
        other = _trim_clip(WORK / "other_source.mp4", 0.0, 2.0)
        with self.assertRaises(media.MediaError) as ctx:
            rmod.render(self.ir, other, WORK / "should_fail.mp4")
        self.assertEqual(ctx.exception.code, "E_IR_INVALID")

        bad = copy.deepcopy(self.ir)
        bad["elements"][0]["layout"]["box"] = [-5, -5, 10, 10]
        with self.assertRaises(media.MediaError) as ctx2:
            rmod.render(bad, SOURCE, WORK / "should_fail2.mp4")
        self.assertEqual(ctx2.exception.code, "E_IR_INVALID")


def tearDownModule() -> None:
    # frames and test renders are regenerable; keep the tree small
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
