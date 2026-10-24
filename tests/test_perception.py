"""Tests for the perception v2 layer (``semotion.perception/1``).

Fast unit tests cover the geometry/mask/zone/text maths without Apple Vision.
A handful of integration tests run the real Vision path on a 0.8 s synthetic
clip (skipped when Apple Vision / ffmpeg are unavailable). Everything writes
into ``tmp_path``; the shared artifacts cache is redirected with
``SEMOTION_CACHE_ROOT``.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from semotion import media, perception


# --------------------------------------------------------------------------- helpers

def _person_grid(gw: int, gh: int, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    grid = np.zeros((gh, gw), dtype=bool)
    grid[int(y0 * gh):int(y1 * gh), int(x0 * gw):int(x1 * gw)] = True
    return grid


def _doc_skeleton(width: int = 1920, height: int = 920, n: int = 5,
                  zones: list | None = None, text: list | None = None) -> dict:
    """A minimal document that passes validate(); used for negative tests."""
    fps = 25.0
    frames = []
    for i in range(n):
        frames.append({
            "frame": i * 5, "t_s": round(i * 5 / fps, 6), "person_bbox": [100, 100, 200, 200],
            "mask": "masks/%04d.png" % i, "mask_scale": 4.0, "mask_coverage": 0.1,
            "pose": {"neck": [200, 200], "left_shoulder": None, "right_shoulder": None,
                     "left_wrist": None, "right_wrist": None, "left_hip": None,
                     "right_hip": None, "left_elbow": None, "right_elbow": None,
                     "nose": [210, 160]},
            "hands": [], "face": {"bbox": [180, 120, 80, 90], "conf": 0.9},
        })
    if zones is None:
        zones = [
            {"bbox": [1480, 300, 420, 220], "kind": "side", "score": 0.85,
             "person_overlap": 0.0, "text_overlap": 0.0,
             "edge_density": 0.02, "contrast_luma": 0.1},
            {"bbox": [30, 300, 420, 220], "kind": "side", "score": 0.80,
             "person_overlap": 0.0, "text_overlap": 0.0,
             "edge_density": 0.03, "contrast_luma": 0.1},
            {"bbox": [1480, 40, 420, 220], "kind": "background", "score": 0.75,
             "person_overlap": 0.0, "text_overlap": 0.0,
             "edge_density": 0.05, "contrast_luma": 0.1},
        ]
    if text is None:
        text = []
    shots = [{"index": 0, "start_s": 0.0, "end_s": 1.0, "start_frame": 0, "end_frame": 25}]
    return {
        "schema": perception.SCHEMA,
        "ir_compatible": perception.LEGACY_SCHEMA,
        "source": {"path": "/tmp/x.mp4",
                   "sha256": "a" * 64, "width": width, "height": height,
                   "fps": fps, "n_frames": 25, "duration_s": 1.0},
        "backend": {"name": "test"},
        "capabilities": {"person_mask": True, "body_pose": True, "hands": True,
                         "existing_text": True, "scene_cuts": True},
        "sample_hz": 5.0,
        "frames": frames,
        "regions": {
            "frame_size": [width, height],
            "face_union": [180, 120, 80, 90],
            "person_union": [100, 100, 400, 400],
            "hands_union": None,
            "face_safe": [148, 88, 144, 154],
            "person_safe": [68, 68, 464, 464],
            "safe_margin_px": 32,
            "existing_text": text,
            "quiet_zones": zones,
            "person_coverage_curve": [[0.0, 0.1], [0.2, 0.1]],
            "shots": [{"start_s": 0.0, "end_s": 1.0}],
        },
        "faces": [{"frame": 0, "t_s": 0.0, "box": [180, 120, 80, 90], "score": 0.9}],
        "face_tracks": [{"id": "f0", "frames": [{"frame": 0, "t_s": 0.0,
                                                "box": [180, 120, 80, 90], "score": 0.9}],
                         "bbox_union": [180, 120, 80, 90], "coverage": 0.2,
                         "n_frames": 1, "first_frame": 0, "last_observed_frame": 0}],
        "shots": shots,
        "safe_zones": {"title": [0.06, 0.06, 0.88, 0.14],
                       "caption": [0.1, 0.8, 0.8, 0.12],
                       "card": [0.05, 0.26, 0.4, 0.46]},
        "provenance": {"created_utc": "2026-01-01T00:00:00Z", "command": "test",
                       "scanned_fps": 5.0, "seed": 0},
    }


# --------------------------------------------------------------------------- masks

def test_mask_to_grid_scale_and_binary():
    mask = np.zeros((384, 512), dtype=np.uint8)
    mask[:, :256] = 255
    grid = perception.mask_to_grid(mask, 1920, 920, 4)
    assert grid.shape == (230, 480)                      # ceil(920/4) x ceil(1920/4)
    assert grid.dtype == np.uint8
    assert set(np.unique(grid).tolist()) <= {0, 255}
    assert abs(float((grid > 127).mean()) - 0.5) < 0.01
    assert (grid[:, :230] == 255).all() and (grid[:, 250:] == 0).all()

    finer = perception.mask_to_grid(mask, 1920, 920, 2)
    assert finer.shape == (460, 960)


def test_mask_to_grid_handles_empty_and_odd_sizes():
    grid = perception.mask_to_grid(np.zeros((384, 512), np.uint8), 1919, 917, 4)
    assert grid.shape == (230, 480) and not grid.any()
    assert perception.mask_to_grid(None, 100, 50, 4).shape == (13, 25)


def test_grid_bbox_maps_back_to_pixels_and_clamps():
    grid = np.zeros((230, 480), dtype=np.uint8)
    grid[40:60, 100:150] = 255                           # rows 40..59, cols 100..149
    box = perception.grid_bbox(grid, 4, 1920, 920)
    assert box == [400, 160, 200, 80]                    # x,y,w,h in pixels
    clamped = perception.grid_bbox(grid, 4, 1000, 200)
    assert clamped == [400, 160, 200, 40]
    assert perception.grid_bbox(np.zeros((10, 10), np.uint8), 4) is None


def test_clamp_and_grow_box_are_int_pixels():
    assert perception.clamp_box([-10.4, -3.2, 100.6, 50.4], 1920, 920) == [0, 0, 90, 47]
    assert perception.clamp_box([1910, 900, 50, 50], 1920, 920) == [1910, 900, 10, 20]
    assert perception.clamp_box([2000, 2000, 50, 50], 1920, 920) is None
    grown = perception.grow_box([100, 100, 200, 100], 32, 1920, 920)
    assert grown == [68, 68, 264, 164]
    assert perception.grow_box(None, 32, 1920, 920) is None


# --------------------------------------------------------------------------- zones

def _zones(person, text, edge=None, luma=None, face_center=(600.0, 400.0),
           person_safe=None, w=1920, h=920, scale=4):
    gh, gw = person.shape
    edge = np.full((gh, gw), 0.03) if edge is None else edge
    luma = np.full((gh, gw), 0.5) if luma is None else luma
    return perception.quiet_zones(person_union=person, text_grid=text,
                                  edge_grid=edge, luma_grid=luma,
                                  frame_w=w, frame_h=h, mask_scale=scale,
                                  face_center=list(face_center),
                                  person_safe=person_safe)


def test_quiet_zones_reject_person_and_text_overlap():
    gw, gh = 480, 230
    person = _person_grid(gw, gh, 0.05, 0.45, 0.20, 1.0)
    text = _person_grid(gw, gh, 0.35, 0.65, 0.90, 1.0)
    zones = _zones(person, text)
    assert 3 <= len(zones) <= perception.ZONE_MAX
    scores = [z["score"] for z in zones]
    assert scores == sorted(scores, reverse=True)
    for zone in zones:
        assert zone["person_overlap"] == 0.0
        assert zone["text_overlap"] == 0.0
        assert zone["edge_density"] >= 0.0
        x, y, w, h = zone["bbox"]
        gx0, gx1 = int(x / 4), int(np.ceil((x + w) / 4))
        gy0, gy1 = int(y / 4), int(np.ceil((y + h) / 4))
        assert not person[gy0:gy1, gx0:gx1].any(), f"zone touches person: {zone['bbox']}"
        assert not text[gy0:gy1, gx0:gx1].any(), f"zone touches text: {zone['bbox']}"
        assert w >= perception.ZONE_MIN_W_FRAC * 1920
        assert h >= perception.ZONE_MIN_H_FRAC * 920
        assert 0.0 <= zone["score"] <= 1.0
        assert zone["kind"] in ("background", "lower_third", "side")


def test_quiet_zones_empty_when_person_fills_frame():
    person = np.ones((230, 480), dtype=bool)
    zones = _zones(person, np.zeros((230, 480), dtype=bool))
    assert zones == []


def test_quiet_zones_prefer_clear_space_and_are_deterministic():
    gw, gh = 480, 230
    person = _person_grid(gw, gh, 0.30, 0.70, 0.20, 1.0)
    text = np.zeros((gh, gw), dtype=bool)
    safe = [560, 180, 800, 740]
    first = _zones(person, text, person_safe=safe)
    second = _zones(person, text, person_safe=safe)
    assert first == second
    assert first[0]["score_parts"]["clear"] == 1.0
    assert first[0]["bbox"][0] + first[0]["bbox"][2] <= safe[0] or first[0]["bbox"][0] >= safe[0] + safe[2]
    # every reported zone still has a zero-overlap mask (re-verified above)
    for zone in first:
        assert zone["person_overlap"] == 0.0


def test_quiet_zones_score_responds_to_edge_density():
    gw, gh = 480, 230
    person = np.zeros((gh, gw), dtype=bool)
    text = np.zeros((gh, gw), dtype=bool)
    edge = np.full((gh, gw), 0.02)
    edge[:, : gw // 2] = 0.15          # left half busy
    zones = _zones(person, text, edge=edge)
    assert zones[0]["bbox"][0] >= 1920 // 2
    assert zones[0]["score"] > zones[-1]["score"]


# --------------------------------------------------------------------------- text

def _samples(texts_by_sample, boxes_by_sample, width=1920, height=920):
    out = []
    for texts, boxes in zip(texts_by_sample, boxes_by_sample):
        out.append([{"text": t, "conf": c, "box": list(b)}
                    for t, c, b in zip(texts, boxes)])
    return out


def _det(text, conf, box):
    return {"text": text, "conf": conf, "box": list(box)}


def test_group_text_subtitle_and_logo():
    n = 40
    changing = ["今天比赛打得不错", "我对自己表现满意", "接下来还要继续努力",
                "感谢球迷的支持", "巴黎的场地很好", "希望在决赛再见面"]
    samples = []
    for i in range(n):
        samples.append([
            _det("中新视频", 0.9, [90, 55, 267, 87]),                    # top-left logo
            _det(changing[i % len(changing)], 0.35, [880, 850, 220, 60]),  # changing line
        ])
    entries = perception.group_existing_text(
        samples, width=1920, height=920, duration_s=8.0, fps=25.0, sample_step=5)
    by_text = {e["text"]: e for e in entries}
    assert by_text["中新视频"]["kind"] == "logo"
    assert by_text["中新视频"]["t0"] == 0.0 and by_text["中新视频"]["t1"] == 8.0
    subtitle = next(e for e in entries if e["kind"] == "subtitle")
    assert subtitle["text"] in changing
    assert subtitle["bbox"] == [880, 850, 220, 60]
    assert subtitle["t0"] == 0.0 and subtitle["t1"] == 8.0


def test_group_text_constant_banner_and_credits_are_other():
    n = 30
    samples = []
    for i in range(n):
        dets = [_det("走过的巴黎纪念", 0.8, [880, 846, 210, 58])]  # constant, centred
        if i >= 20:
            dets.append(_det("版权所有未经许可请勿转载", 0.8, [123, 812, 411, 51]))
        samples.append(dets)
    entries = perception.group_existing_text(
        samples, width=1920, height=920, duration_s=6.0, fps=25.0, sample_step=5)
    kinds = {e["text"]: e["kind"] for e in entries}
    assert kinds["走过的巴黎纪念"] == "other"
    assert kinds["版权所有未经许可请勿转载"] == "other"


def test_group_text_drops_unstable_and_merges_a_short_dropout():
    n = 30
    samples = []
    for i in range(n):
        dets = []
        if i != 11:                                   # 1-sample OCR dropout at 11
            dets.append(_det("台标", 0.9, [100, 60, 160, 60]))
        if i == 3:                                    # one-off false positive
            dets.append(_det("x", 0.2, [800, 400, 50, 30]))
        samples.append(dets)
    entries = perception.group_existing_text(
        samples, width=1920, height=920, duration_s=6.0, fps=25.0, sample_step=5)
    assert [e["text"] for e in entries] == ["台标"]
    logo = entries[0]
    assert logo["kind"] == "logo"
    assert logo["t0"] == 0.0 and logo["t1"] == 6.0     # dropout is bridged
    # only 2 consecutive samples is not enough to be reported
    short = [[_det("y", 0.9, [10, 10, 20, 20])] for _ in range(2)]
    assert perception.group_existing_text(
        short, width=1920, height=920, duration_s=0.4, fps=25.0, sample_step=5) == []


def test_group_text_empty_and_single_sample():
    assert perception.group_existing_text([], width=100, height=100, duration_s=0.0,
                                           fps=25.0, sample_step=5) == []
    one = [[_det("A", 0.9, [10, 10, 20, 20])]]
    assert perception.group_existing_text(one, width=100, height=100, duration_s=0.2,
                                          fps=25.0, sample_step=5) == []


def test_text_mask_padding_covers_gap():
    entries = [{"t0": 0.0, "t1": 1.0, "bbox": [400, 800, 200, 60],
                "text": "x", "conf": 1.0, "kind": "subtitle"}]
    grid = perception._text_mask(entries, width=1920, height=920, gw=480, gh=230,
                                 pad_frac=0.006)
    assert grid[:, :90].sum() == 0          # far left untouched
    assert grid[195:230, 95:155].any()      # padded box is rasterized


# --------------------------------------------------------------------------- options/cache

def test_options_key_stable_and_sensitive():
    a = perception.options_key(perception.Options())
    b = perception.options_key(perception.Options())
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    c = perception.options_key(perception.Options(mask_scale=8))
    assert a != c
    key1 = perception.cache_key("a" * 64, perception.Options())
    key2 = perception.cache_key("a" * 64, perception.Options(mask_scale=8))
    key3 = perception.cache_key("b" * 64, perception.Options())
    assert len(key1) == 32 and key1 != key2 and key1 != key3


def test_cache_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SEMOTION_CACHE_ROOT", str(tmp_path / "cache"))
    assert perception.cache_root() == tmp_path / "cache"
    assert str(perception.cache_dir("a" * 64)).startswith(str(tmp_path / "cache"))


# --------------------------------------------------------------------------- validate

def test_validate_accepts_minimal_document():
    doc = _doc_skeleton()
    assert perception.validate(doc) == []


@pytest.mark.parametrize("mutate, needle", [
    (lambda d: d.update({"schema": "wrong"}), "schema"),
    (lambda d: d["regions"].update({"frame_size": [1, 2]}), "frame_size"),
    (lambda d: d["regions"]["quiet_zones"][0].update({"person_overlap": 0.5}), "person_overlap"),
    (lambda d: d["regions"]["quiet_zones"][0].update({"text_overlap": 0.5}), "text_overlap"),
    (lambda d: d["regions"]["quiet_zones"][0].update({"kind": "nope"}), "kind"),
    (lambda d: d["regions"]["quiet_zones"][0].update({"bbox": [-10, 0, 100, 100]}), "outside frame"),
    (lambda d: d["regions"]["quiet_zones"].append(
        {"bbox": [10, 10, 400, 200], "kind": "side", "score": 0.99,
         "person_overlap": 0, "text_overlap": 0, "edge_density": 0, "contrast_luma": 0}),
     "descending"),
    (lambda d: d["frames"][0].update({"mask_coverage": 2.0}), "mask_coverage"),
    (lambda d: d["frames"][0].update({"person_bbox": [0, 0, 5000, 10]}), "outside frame"),
    (lambda d: d["regions"]["existing_text"].append(
        {"t0": 2.0, "t1": 1.0, "bbox": [0, 0, 10, 10], "text": "x", "conf": 1.0,
         "kind": "subtitle"}), "bad time range"),
])
def test_validate_rejects_mutated_documents(mutate, needle):
    doc = _doc_skeleton()
    mutate(doc)
    problems = perception.validate(doc)
    assert any(needle in p for p in problems), problems


def test_validate_allows_zero_zones_without_person_mask():
    doc = _doc_skeleton(zones=[])
    doc["capabilities"]["person_mask"] = False
    assert perception.validate(doc) == []


# --------------------------------------------------------------------------- integration

def _make_clip(path: Path) -> Path:
    from semotion import media as m

    m.run([m.FFMPEG, "-v", "error", "-y", "-f", "lavfi",
           "-i", "testsrc2=duration=0.8:size=640x360:rate=25",
           "-pix_fmt", "yuv420p", str(path)])
    return path


@pytest.fixture(scope="module")
def vision_ready():
    if not perception.vision_available():
        pytest.skip("Apple Vision (pyobjc) not available")
    return True


@pytest.fixture()
def workspace(tmp_path, monkeypatch, vision_ready):
    monkeypatch.setenv("SEMOTION_CACHE_ROOT", str(tmp_path / "cache"))
    clip = _make_clip(tmp_path / "tiny.mp4")
    return tmp_path, clip


def test_integration_analyze_document_schema_and_artifacts(workspace):
    tmp_path, clip = workspace
    out = tmp_path / "perc" / "perception.json"
    meta: dict = {}
    doc = perception.analyze(clip, out, mask_dir=tmp_path / "perc" / "masks",
                             debug_path=tmp_path / "perc" / "debug.png",
                             sample_hz=5.0, meta=meta)
    assert meta["cache"] == "miss"
    assert perception.validate(doc) == []
    assert doc["schema"] == perception.SCHEMA
    assert doc["ir_compatible"] == perception.LEGACY_SCHEMA
    assert doc["source"]["width"] == 640 and doc["source"]["height"] == 360
    assert 3 <= len(doc["frames"]) <= 8
    assert doc["capabilities"]["person_mask"] is True
    # legacy fields are structurally valid for v1 consumers
    legacy_problems = [p for p in perception.vision.validate(doc)
                       if not p.startswith("schema")]
    assert legacy_problems == []
    # mask PNGs: one per sample, 8-bit L, 1/4 linear scale
    from PIL import Image

    masks = sorted((tmp_path / "perc" / "masks").glob("*.png"))
    assert len(masks) == len(doc["frames"])
    image = Image.open(masks[0])
    assert image.mode == "L"
    assert image.size == (160, 90)
    for frame in doc["frames"]:
        assert (tmp_path / "perc" / frame["mask"]).exists(), frame["mask"]
    debug = tmp_path / "perc" / "debug.png"
    assert debug.exists() and debug.stat().st_size > 20_000
    # JSON is written with sorted keys and no wall-clock floats
    text = out.read_text(encoding="utf-8")
    assert text == json.dumps(json.loads(text), ensure_ascii=False, indent=2,
                              sort_keys=True) + "\n"


def test_integration_cache_hit_is_identical(workspace):
    tmp_path, clip = workspace
    out1 = tmp_path / "run1" / "perception.json"
    out2 = tmp_path / "run2" / "perception.json"
    meta1: dict = {}
    meta2: dict = {}
    perception.analyze(clip, out1, mask_dir=tmp_path / "run1" / "masks",
                       debug_path=tmp_path / "run1" / "d.png",
                       sample_hz=5.0, meta=meta1)
    perception.analyze(clip, out2, mask_dir=tmp_path / "run2" / "masks",
                       debug_path=tmp_path / "run2" / "d.png",
                       sample_hz=5.0, meta=meta2)
    assert meta1["cache"] == "miss"
    assert meta2["cache"] == "hit"
    assert meta2["runtime_s"] < max(2.0, meta1["runtime_s"])

    def norm(path: Path) -> dict:
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc.pop("provenance", None)
        doc["source"]["path"] = "<p>"
        for frame in doc["frames"]:
            frame["mask"] = Path(frame["mask"]).name
        return doc

    assert norm(out1) == norm(out2)
    assert (tmp_path / "run2" / "masks" / "0000.png").exists()
    assert (tmp_path / "run2" / "d.png").exists()


def test_integration_force_is_deterministic(workspace):
    tmp_path, clip = workspace
    a = tmp_path / "a" / "perception.json"
    b = tmp_path / "b" / "perception.json"
    perception.analyze(clip, a, mask_dir=tmp_path / "a" / "masks",
                       debug_path=tmp_path / "a" / "d.png", force=True)
    first = a.read_text(encoding="utf-8")
    perception.analyze(clip, b, mask_dir=tmp_path / "b" / "masks",
                       debug_path=tmp_path / "b" / "d.png", force=True)
    second = b.read_text(encoding="utf-8")

    def strip(text: str) -> dict:
        doc = json.loads(text)
        doc.pop("provenance", None)
        doc["source"]["path"] = "<p>"
        for frame in doc["frames"]:
            frame["mask"] = Path(frame["mask"]).name
        return doc

    assert strip(first) == strip(second)
    import hashlib

    def hashes(folder: Path) -> dict:
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((folder / "masks").glob("*.png"))}

    assert hashes(tmp_path / "a") == hashes(tmp_path / "b")


def test_cli_validate_only(workspace, capsys):
    tmp_path, clip = workspace
    out = tmp_path / "cli" / "perception.json"
    assert perception.main(["--video", str(clip), "--out", str(out),
                            "--mask-dir", str(tmp_path / "cli" / "masks"),
                            "--fps", "5"]) == 0
    assert "VALIDATE   : OK" in capsys.readouterr().out
    assert perception.main(["--validate-only", str(out)]) == 0
