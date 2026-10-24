"""Tests for verify2 (rendered-frame QA) and repair (deterministic auto-repair).

Every check is exercised twice: a positive control that must PASS and a negative
control that must FAIL *with the expected number*. The negative controls are
synthetic on purpose — they prove the metric is not vacuous without needing the
perception module or the full renderer.

Run:
    PYTHONPATH=src .venv/bin/python -m pytest tests/test_verify2.py -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from semotion import ir as ir_mod  # noqa: E402
from semotion.repair import plan_repairs  # noqa: E402
from semotion.verify2 import (  # noqa: E402
    FAIL, NOT_RUN, PASS, PERSON_FRAME_MAX, PERSON_PERSON_MAX, compare_renders,
    diff_footprint, element_font_size, element_min_font, element_text,
    font_floor_px, glyph_and_contrast, is_subsequence, layout_checks, load_masks,
    measure_caption_ink, caption_thresholds, caption_verdict, caption_policy_view,
    discover_caption_cues, non_text_graphic_contrast, normalise_display_text,
    occlusion_metrics, timing_series_row, verify, Verify2Error,
)

FFMPEG = "ffmpeg"
FPS = 10.0
W, H = 160, 88
MASK_SCALE = 4
N = 8
BAD_BOX = [24, 14, 44, 44]
GOOD_BOX = [104, 24, 40, 40]
PERSON_RECT = [16, 12, 60, 64]     # pixels: x, y, w, h
FACE_BOX = [30, 14, 24, 24]
WINDOW = {"start_s": 0.2, "end_s": 0.6, "start_frame": 2, "end_frame": 6,
          "hold_s": 0.0, "anchor_start_s": 0.2, "anchor_end_s": 0.6}


# --------------------------------------------------------------------- fixtures
def _base_frames() -> list[np.ndarray]:
    """Deterministic synthetic scene: gradient + dark person rectangle."""
    yy, xx = np.mgrid[0:H, 0:W]
    frames = []
    for f in range(N):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[..., 0] = (40 + xx * 120 // W).astype(np.uint8)
        img[..., 1] = (60 + yy * 100 // H + f).astype(np.uint8)
        img[..., 2] = (80 + (xx + yy) * 60 // (W + H)).astype(np.uint8)
        x, y, w, h = PERSON_RECT
        img[y:y + h, x:x + w] = (24, 26, 30)
        frames.append(img)
    return frames


def _with_box(frames: list[np.ndarray], box: list[int], color=(8, 10, 16),
              frames_range=range(2, 6)) -> list[np.ndarray]:
    out = [f.copy() for f in frames]
    x, y, w, h = box
    for i in frames_range:
        out[i][y:y + h, x:x + w] = color
    return out


def _encode(frames: list[np.ndarray], path: Path) -> Path:
    h, w = frames[0].shape[:2]
    argv = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(int(FPS)), "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", str(path)]
    proc = subprocess.run(argv, input=b"".join(f.tobytes() for f in frames),
                          capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")[-1500:]
    return path


def _union_boxes(boxes):
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return [x0, y0, x1 - x0, y1 - y0]


def _mask_doc(tmp: Path, *, boxes=None, text_regions=None, zones=None,
              samples=None, hands=None) -> Path:
    """Write a minimal contract-shaped perception document + mask PNGs."""
    masks = tmp / "masks"
    masks.mkdir(parents=True, exist_ok=True)
    mw, mh = W // MASK_SCALE, H // MASK_SCALE
    rows = []
    samples = samples if samples is not None else list(range(0, N, 2))
    for n, frame in enumerate(samples):
        m = np.zeros((mh, mw), dtype=np.uint8)
        for box in (boxes if boxes is not None else [PERSON_RECT]):
            x, y, w, h = (int(round(v / MASK_SCALE)) for v in box)
            m[y:y + h, x:x + w] = 255
        rel = f"masks/{n:04d}.png"
        Image.fromarray(m).save(tmp / rel)
        union = _union_boxes(boxes if boxes is not None else [PERSON_RECT])
        rows.append({"frame": int(frame), "t_s": round(frame / FPS, 6),
                     "person_bbox": union, "mask": rel,
                     "mask_scale": float(MASK_SCALE),
                     "mask_coverage": float((m > 127).mean()),
                     "pose": None,
                     "hands": [{"side": "right", "bbox": list(h), "conf": 0.9}
                               for h in (hands or [])],
                     "face": {"bbox": FACE_BOX, "conf": 0.9}})
    doc = {
        "schema": "semotion.perception/1",
        "source": {"path": str(tmp / "base.mp4"), "sha256": "a" * 64,
                   "width": W, "height": H, "fps": FPS, "n_frames": N,
                   "duration_s": N / FPS},
        "backend": {"name": "test-synthetic"},
        "capabilities": {"person_mask": True, "body_pose": False, "hands": False,
                         "existing_text": bool(text_regions), "scene_cuts": False},
        "sample_hz": round(FPS / 2, 3),
        "frames": rows,
        "regions": {
            "frame_size": [W, H],
            "face_union": FACE_BOX,
            "person_union": PERSON_RECT,
            "hands_union": None,
            "face_safe": [FACE_BOX[0] - 8, FACE_BOX[1] - 8,
                          FACE_BOX[2] + 16, FACE_BOX[3] + 16],
            "person_safe": [PERSON_RECT[0] - 8, PERSON_RECT[1] - 8,
                            PERSON_RECT[2] + 16, PERSON_RECT[3] + 16],
            "safe_margin_px": 8,
            "existing_text": text_regions or [],
            "quiet_zones": zones or [],
            "person_coverage_curve": [],
            "shots": [{"start_s": 0.0, "end_s": N / FPS}],
        },
    }
    path = tmp / "perception.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _ir(box: list[int], *, element_id: str = "el1", font: int = 36,
        window: dict | None = None, source_path: str = "base.mp4",
        seal: bool = False) -> dict:
    win = dict(WINDOW) if window is None else dict(window)
    doc = {
        "schema": "semotion.motion-ir/1",
        "ir_version": "1.0.0",
        "source": {"path": source_path, "sha256": "a" * 64, "width": W,
                   "height": H, "fps": FPS, "duration_s": N / FPS,
                   "n_frames": N, "frame_source": "nb_frames",
                   "timebase": {"num": 1, "den": int(FPS)}},
        "canvas": {"width": W, "height": H, "fps": FPS},
        "style": "smart_balanced",
        "semantic": {
            "transcript_schema": "semotion.transcript/1",
            "transcript_text": "QA DEMO",
            "anchors": [{"id": "a1", "kind": "keyword", "text": "QA DEMO",
                         "start_s": 0.2, "end_s": 0.6, "confidence": 1.0,
                         "source": "TEST", "segment_id": "s0", "word_ids": []}],
        },
        "tracks": [{"id": "t_content", "z": 20, "elements": [element_id]}],
        "elements": [{
            "id": element_id, "type": "keyword",
            "anchor_ref": "a1", "anchor_refs": ["a1"],
            "window": win,
            "style": {"font_size_px": font, "color": "#F5F7FA",
                      "bg": "rgba(8,10,16,0.82)"},
            "motion": {"enter": {"type": "fade", "dur_s": 0.1},
                       "exit": {"type": "fade", "dur_s": 0.1}},
            "content": {"text": "QA DEMO"},
            "layout": {"box": list(box), "avoid_face": True},
        }],
        "collision_ok": [],
        "provenance": {"generator": "tests/test_verify2.py"},
    }
    if seal:
        ir_mod.seal(doc)
    return doc


# ------------------------------------------------------------------ pure checks
def test_footprint_threshold_separates_overlay_from_noise():
    base = np.zeros((20, 20, 3), np.uint8)
    final = base.copy()
    final[5:10, 5:10] = 200
    fp = diff_footprint(final, base, 12)
    assert int(fp.sum()) == 25
    assert not fp[0, 0]
    # a soft encoder-like perturbation below the threshold is not an overlay
    noisy = base.copy()
    noisy[:] = 3
    assert int(diff_footprint(noisy, base, 12).sum()) == 0


def test_occlusion_negative_control_face_overlay_fails_with_numbers():
    mw, mh = W // MASK_SCALE, H // MASK_SCALE
    person = np.zeros((mh, mw), np.uint8)
    px, py, pw, ph = (int(round(v / MASK_SCALE)) for v in PERSON_RECT)
    person[py:py + ph, px:px + pw] = 255
    fp = np.zeros((H, W), dtype=bool)
    x, y, w, h = BAD_BOX
    fp[y:y + h, x:x + w] = True
    occ = occlusion_metrics(fp, BAD_BOX, person, [FACE_BOX], W * H)
    assert occ["face_overlap_px"] > 0, "face overlay must be detected"
    assert occ["person_overlap_px"] > 0
    assert occ["person_frame_fraction"] > PERSON_FRAME_MAX
    assert occ["person_person_fraction"] > PERSON_PERSON_MAX
    assert occ["person_overlap_px"] > 100, \
        "the synthetic box must cover a large part of the person rectangle"
    assert occ["person_person_fraction"] > 0.5

    # positive control: same footprint in a person-free zone
    fp2 = np.zeros((H, W), dtype=bool)
    x, y, w, h = GOOD_BOX
    fp2[y:y + h, x:x + w] = True
    occ2 = occlusion_metrics(fp2, GOOD_BOX, person, [FACE_BOX], W * H)
    assert occ2["face_overlap_px"] == 0
    assert occ2["person_overlap_px"] == 0


def test_wcag_contrast_pass_and_fail():
    white = np.full((40, 40, 3), 245, np.uint8)
    dark = np.full((40, 40, 3), 16, np.uint8)
    rendered = dark.copy()
    rendered[15:25, 15:25] = white[0, 0]
    row = glyph_and_contrast(rendered, dark, [0, 0, 40, 40], (245, 247, 250))
    assert row is not None and row["contrast"] >= 4.5, row
    # same geometry, mid-grey glyphs on mid-grey background -> ~1.2:1, must FAIL
    bg = np.full((40, 40, 3), 140, np.uint8)
    rendered2 = bg.copy()
    rendered2[15:25, 15:25] = (160, 160, 160)
    row2 = glyph_and_contrast(rendered2, np.full((40, 40, 3), 110, np.uint8),
                              [0, 0, 40, 40], (160, 160, 160))
    assert row2 is not None and row2["contrast"] < 4.5, row2


def test_timing_onsets_signed_errors_and_tolerance():
    counts = np.zeros(N, dtype=int)
    counts[2:6] = 500
    row = timing_series_row(counts, WINDOW, FPS, anchor_start_s=0.2,
                            n_frames=N, box_area=44 * 44)
    assert row["verdict"] == PASS
    assert row["appearance_frame"] == 2
    assert row["disappearance_frame"] == 6
    assert row["signed_error_anchor_frames"] == 0

    late = np.zeros(N, dtype=int)
    late[5:6] = 500          # appears 3 frames late, well beyond 1-frame tolerance
    bad = timing_series_row(late, WINDOW, FPS, anchor_start_s=0.2,
                            n_frames=N, box_area=44 * 44)
    assert bad["verdict"] == FAIL
    assert bad["appearance_frame"] == 5
    assert bad["signed_error_start_frames"] == 3
    assert bad["signed_error_anchor_frames"] == 3


def test_layout_negative_controls_and_font_floor():
    assert font_floor_px("keyword", 160, 88) == 32

    ir = _ir([10, 10, 20, 20], font=20)
    res = layout_checks(ir, W, H, safe_margin_px=8)
    assert res["verdict"] == FAIL
    problems = res["elements"][0]["problems"]
    assert any("font" in p for p in problems)

    off = _ir([150, 10, 40, 20])
    res = layout_checks(off, W, H, safe_margin_px=8)
    assert res["verdict"] == FAIL
    assert res["elements"][0]["on_canvas"] is False

    # two simultaneously visible elements with overlapping boxes -> collision
    ir2 = _ir([10, 10, 40, 40])
    el2 = json.loads(json.dumps(ir2["elements"][0]))
    el2["id"] = "el2"
    el2["layout"]["box"] = [20, 20, 40, 40]
    ir2["elements"].append(el2)
    ir2["tracks"][0]["elements"].append("el2")
    res2 = layout_checks(ir2, W, H, safe_margin_px=8)
    assert res2["collision_threshold"] == 0.01
    assert len(res2["collisions"]) == 1
    assert res2["verdict"] == FAIL


# ------------------------------------------------------------- end-to-end (MP4)
def test_end_to_end_face_overlay_fails_then_passes(tmp_path):
    base_frames = _base_frames()
    base = _encode(base_frames, tmp_path / "base.mp4")
    bad = _encode(_with_box(base_frames, BAD_BOX), tmp_path / "bad.mp4")
    good = _encode(_with_box(base_frames, GOOD_BOX), tmp_path / "good.mp4")
    masks = _mask_doc(tmp_path, samples=[0, 2, 4, 6])
    ir_bad = _ir(BAD_BOX)
    ir_good = _ir(GOOD_BOX)

    m_bad = verify(ir=ir_bad, source=base, rendered=bad, masks=masks,
                   safe_margin_px=8, make_sheet=False)
    assert m_bad["verdict"] == FAIL
    occ = m_bad["checks"]["person_occlusion"]
    assert occ["face_overlap_px_total"] > 0
    assert occ["max_person_frame_fraction"] > PERSON_FRAME_MAX
    assert occ["max_person_fraction"] > PERSON_PERSON_MAX
    assert m_bad["checks"]["timing"]["verdict"] == PASS
    assert m_bad["checks"]["existing_text_collision"]["verdict"] == NOT_RUN

    m_good = verify(ir=ir_good, source=base, rendered=good, masks=masks,
                    safe_margin_px=8, make_sheet=False)
    assert m_good["verdict"] == PASS, json.dumps(m_good["checks"], indent=1)[:1500]
    assert m_good["checks"]["person_occlusion"]["face_overlap_px_total"] == 0
    assert m_good["checks"]["person_occlusion"]["max_person_frame_fraction"] == 0.0

    # overlay-only pass: footprint comes from the layer, not the diff
    layer = np.zeros((H, W, 3), np.uint8)
    gx, gy, gw, gh = GOOD_BOX
    layer[gy:gy + gh, gx:gx + gw] = 200
    overlay = _encode([layer if 2 <= i < 6 else np.zeros((H, W, 3), np.uint8)
                       for i in range(N)], tmp_path / "overlay.mp4")
    m_ovl = verify(ir=ir_good, source=base, rendered=good, masks=masks,
                   overlay=overlay, safe_margin_px=8, make_sheet=False)
    assert m_ovl["checks"]["footprint"]["overlay_only"] is True
    assert m_ovl["checks"]["footprint"]["by_element"][0]["union_px"] > 0


def test_existing_text_collision_is_measured(tmp_path):
    base_frames = _base_frames()
    base = _encode(base_frames, tmp_path / "base.mp4")
    final = _encode(_with_box(base_frames, GOOD_BOX), tmp_path / "final.mp4")
    text = {"t0": 0.0, "t1": 1.0, "bbox": [100, 20, 50, 30],
            "text": "LIVE", "conf": 0.9, "kind": "logo"}
    masks = _mask_doc(tmp_path, text_regions=[text])
    metrics = verify(ir=_ir(GOOD_BOX), source=base, rendered=final, masks=masks,
                     safe_margin_px=8, make_sheet=False)
    check = metrics["checks"]["existing_text_collision"]
    assert check["verdict"] == FAIL
    assert check["max_overlap_px"] > 0


def test_determinism_tolerance(tmp_path):
    frames = _base_frames()
    a = _encode(frames, tmp_path / "a.mp4")
    b = _encode(frames, tmp_path / "b.mp4")
    det = compare_renders(a, b, W, H)
    assert det["verdict"] == PASS
    assert det["frames_compared"] == N
    assert det["max_pixel_diff"] == 0.0

    perturbed = [f.copy() for f in frames]
    perturbed[4][40:50, 40:50] = 255
    c = _encode(perturbed, tmp_path / "c.mp4")
    det2 = compare_renders(a, c, W, H)
    assert det2["verdict"] == FAIL
    assert det2["max_pixel_diff"] > 32


def test_missing_mask_capability_is_not_run(tmp_path):
    base_frames = _base_frames()
    base = _encode(base_frames, tmp_path / "base.mp4")
    final = _encode(_with_box(base_frames, GOOD_BOX), tmp_path / "final.mp4")
    masks = _mask_doc(tmp_path)
    doc = json.loads(masks.read_text())
    doc["capabilities"]["person_mask"] = False
    masks.write_text(json.dumps(doc))
    metrics = verify(ir=_ir(GOOD_BOX), source=base, rendered=final, masks=masks,
                     safe_margin_px=8, make_sheet=False)
    assert metrics["checks"]["person_occlusion"]["verdict"] == NOT_RUN


# ----------------------------------------------------------------------- repair
def _repair_setup(tmp_path: Path):
    """Canvas 320x180, person on the left, one quiet zone on the right."""
    global W, H
    W, H = 320, 180
    masks = _mask_doc(tmp_path, boxes=[[0, 0, 160, 180]],
                      zones=[{"bbox": [200, 20, 100, 120], "kind": "background",
                              "score": 0.9, "person_overlap": 0.0,
                              "text_overlap": 0.0}])
    ir = _ir([40, 40, 80, 80], font=40, seal=True,
             window={"start_s": 0.2, "end_s": 0.6, "start_frame": 2,
                     "end_frame": 6, "hold_s": 0.0,
                     "anchor_start_s": 0.2, "anchor_end_s": 0.6})
    return ir, masks


@pytest.fixture(autouse=True)
def _restore_canvas():
    global W, H
    saved = (W, H)
    yield
    W, H = saved


def test_repair_relocates_and_is_idempotent(tmp_path):
    ir, masks_path = _repair_setup(tmp_path)
    mask_doc = load_masks(masks_path)
    metrics = {"verdict": FAIL, "failing_elements": {"el1": {}},
               "elements": [{"id": "el1", "verdict": FAIL,
                             "occlusion": {"verdict": FAIL}}]}
    repaired, doc = plan_repairs(ir, metrics, mask_doc, safe_margin_px=8)
    assert doc["noop"] is False
    assert len(doc["repairs"]) == 1
    entry = doc["repairs"][0]
    assert entry["element_id"] == "el1"
    assert "relocate" in entry["action"]
    assert entry["after"]["box"] != entry["before"]["box"]
    assert entry["evidence"]["placement"]["person_overlap_px_before"] > 0
    assert doc["dropped"] == []
    # mask-check the new box: zero person overlap
    new_box = entry["after"]["box"]
    masks = np.array(Image.open(tmp_path / "masks/0000.png")) > 127
    mx, my = W / masks.shape[1], H / masks.shape[0]
    x, y, w, h = new_box
    sub = masks[int(y / my):int(np.ceil((y + h) / my)),
                int(x / mx):int(np.ceil((x + w) / mx))]
    assert not sub.any(), "repaired box still touches the person mask"
    assert ir_mod.validate(repaired) == [], ir_mod.validate(repaired)

    # second run with the SAME stale metrics: the IR already satisfies the target
    repaired2, doc2 = plan_repairs(repaired, metrics, mask_doc, safe_margin_px=8)
    assert doc2["noop"] is True
    assert doc2["repairs"] == [] and doc2["dropped"] == []
    assert json.dumps(repaired2, sort_keys=True) == json.dumps(repaired, sort_keys=True)

    # determinism: repeating from the same input gives byte-identical documents
    ir_b, masks_b = _repair_setup(tmp_path / "again")
    repaired_b, doc_b = plan_repairs(ir_b, metrics, load_masks(masks_b),
                                     safe_margin_px=8)
    assert json.dumps(repaired_b, sort_keys=True) == json.dumps(repaired, sort_keys=True)
    assert json.dumps(doc_b["repairs"], sort_keys=True) == json.dumps(doc["repairs"],
                                                                     sort_keys=True)
    assert doc_b["summary"] == doc["summary"]


def test_repair_retimes_to_anchor_and_second_run_is_noop(tmp_path):
    masks = _mask_doc(tmp_path, boxes=[[0, 0, 20, 20]])   # far from the box
    mask_doc = load_masks(masks)
    window = {"start_s": 0.0, "end_s": 0.4, "start_frame": 0, "end_frame": 4,
              "hold_s": None, "anchor_start_s": 0.3, "anchor_end_s": 0.7}
    ir = _ir([110, 30, 40, 40], font=36, window=window, seal=True)
    metrics = {"verdict": FAIL, "failing_elements": {"el1": {}},
               "elements": [{"id": "el1", "verdict": FAIL,
                             "timing": {"verdict": FAIL, "window_start_frame": 0,
                                        "appearance_frame": 3,
                                        "anchor_onset_frame": 3,
                                        "signed_error_anchor_frames": 3}}]}
    repaired, doc = plan_repairs(ir, metrics, mask_doc, safe_margin_px=8)
    entry = doc["repairs"][0]
    assert "retime" in entry["action"]
    assert entry["after"]["window"]["start_frame"] == 3
    assert entry["after"]["window"]["end_frame"] == 7
    assert entry["after"]["window"]["start_s"] == pytest.approx(0.3)
    assert entry["before"]["window"]["start_frame"] == 0
    repaired2, doc2 = plan_repairs(repaired, metrics, mask_doc, safe_margin_px=8)
    assert doc2["noop"] is True
    assert doc2["unfixable"] == []       # stale metric, but the IR already matches
    assert json.dumps(repaired2, sort_keys=True) == json.dumps(repaired, sort_keys=True)


def test_repair_adds_scrim_for_contrast(tmp_path):
    masks = _mask_doc(tmp_path, boxes=[[0, 0, 20, 20]])
    ir = _ir([110, 30, 40, 40], font=36, seal=True)
    metrics = {"verdict": FAIL, "failing_elements": {"el1": {}},
               "elements": [{"id": "el1", "verdict": FAIL,
                             "legibility": {"verdict": FAIL, "min_contrast": 2.1}}]}
    repaired, doc = plan_repairs(ir, metrics, load_masks(masks), safe_margin_px=8)
    entry = doc["repairs"][0]
    assert entry["action"] == "scrim"
    assert entry["after"]["bg"] != entry["before"]["bg"]
    assert "0.88" in entry["after"]["bg"]
    repaired2, doc2 = plan_repairs(repaired, metrics, load_masks(masks),
                                   safe_margin_px=8)
    assert doc2["noop"] is True and doc2["unfixable"] == []
    assert json.dumps(repaired2, sort_keys=True) == json.dumps(repaired, sort_keys=True)


def test_repair_drops_when_no_safe_placement_exists(tmp_path):
    # person mask covers the whole canvas and there are no zones
    masks = _mask_doc(tmp_path, boxes=[[0, 0, W, H]], zones=[])
    ir = _ir([40, 40, 80, 80], font=40, seal=True)
    repaired, doc = plan_repairs(ir, None, load_masks(masks), safe_margin_px=8)
    assert doc["noop"] is False
    assert doc["summary"]["dropped"] == 1
    assert repaired["elements"] == []
    assert repaired["tracks"] == []
    assert doc["dropped"][0]["element_id"] == "el1"
    repaired2, doc2 = plan_repairs(repaired, None, load_masks(masks),
                                   safe_margin_px=8)
    assert doc2["noop"] is True
    assert json.dumps(repaired2, sort_keys=True) == json.dumps(repaired, sort_keys=True)


def test_load_masks_rejects_broken_document(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "semotion.perception/1"}))
    with pytest.raises(Verify2Error):
        load_masks(bad)


def test_element_text_and_font_mapping():
    assert element_text({"content": {"text": "正文"}}) == "正文"
    assert element_text({"content": {"title": "卡片标题"}}) == "卡片标题"
    assert element_text({"content": {"rows": [{"text": "一"}, {"text": "二"}]}}) == "一\n二"
    # v2: fit.font_size_px is main; display for keyword; title for list; body never
    assert element_font_size({"fit": {"font_size_px": 65, "display_px": 71,
                                      "body_px": 26}}) == 65
    assert element_font_size({"fit": {"display_px": 0, "title_px": 53,
                                      "body_px": 26}}) == 53
    assert element_font_size({"style": {"font_size_px": 40},
                              "fit": {"font_size_px": 65}}) == 40
    assert element_min_font({"fit": {"min_font_px": 31}}) == 31


def test_ocr_normalisation_folds_traditional_and_punctuation():
    assert normalise_display_text("停滯 在，一個位置！") == normalise_display_text("停滞在一个位置")
    assert is_subsequence("abc", "aXbYc") is True
    assert is_subsequence("abcd", "abc") is False


def test_layout_min_font_authoritative_vs_warning():
    # v2-shaped element: fit.min_font_px is the module's own floor -> hard fail
    ir = _ir([10, 10, 60, 40], font=0)
    el = ir["elements"][0]
    el.pop("style", None)
    el["fit"] = {"font_size_px": 30, "title_px": 30, "body_px": 26,
                 "min_font_px": 40}
    res = layout_checks(ir, 320, 180, safe_margin_px=8)
    assert res["verdict"] == FAIL
    assert any("min_font_px" in p for p in res["elements"][0]["problems"])

    # without min_font_px the legacy v1 floor is a warning for v2 typography
    ir2 = _ir([10, 10, 60, 40], font=0)
    el2 = ir2["elements"][0]
    el2.pop("style", None)
    el2["fit"] = {"font_size_px": 20, "title_px": 20, "body_px": 12}
    res2 = layout_checks(ir2, 1920, 920, safe_margin_px=8)
    assert res2["verdict"] == PASS
    assert res2["elements"][0]["font_floor_warning"] is not None


def _ir_with(el, *, box, treatment=None, backdrop=None, font=36, seal=False):
    ir = _ir(box, font=font, seal=seal)
    ir["elements"] = [el]
    ir["tracks"][0]["elements"] = [el["id"]]
    if treatment:
        el.setdefault("layout", {})["treatment"] = treatment
    if backdrop:
        el["backdrop"] = backdrop
    return ir


def test_lower_third_allows_body_but_not_hands(tmp_path):
    base_frames = _base_frames()
    base = _encode(base_frames, tmp_path / "base.mp4")
    box = [16, 38, 50, 36]     # over clothing, below the face box
    final = _encode(_with_box(base_frames, box), tmp_path / "final.mp4")
    # person mask covers the element box; face is above it
    masks_body = _mask_doc(tmp_path / "body", boxes=[PERSON_RECT], hands=[])
    # same, but a hand rectangle now sits inside the element box
    masks_hand = _mask_doc(tmp_path / "hand", boxes=[PERSON_RECT],
                           hands=[[20, 50, 10, 10]])

    def make_ir(seal=False):
        el = {
            "id": "el1", "type": "keyword", "anchor_ref": "a1",
            "anchor_refs": ["a1"], "window": dict(WINDOW),
            "style": {"font_size_px": 36, "color": "#F5F7FA"},
            "content": {"text": "QA DEMO"},
            "layout": {"box": list(box), "treatment": "lower_third",
                       "avoid_face": True},
        }
        ir = _ir(box, font=36)
        ir["elements"] = [el]
        ir["tracks"][0]["elements"] = ["el1"]
        return ir

    passed = verify(ir=make_ir(), source=base, rendered=final, masks=masks_body,
                    safe_margin_px=8, make_sheet=False)
    occ = passed["checks"]["person_occlusion"]
    assert occ["verdict"] == PASS
    assert occ["max_person_fraction"] > PERSON_PERSON_MAX, \
        "body overlap must still be measured"
    assert occ["by_element"][0]["policy"] == "lower_third"

    failed = verify(ir=make_ir(), source=base, rendered=final, masks=masks_hand,
                    safe_margin_px=8, make_sheet=False)
    occ2 = failed["checks"]["person_occlusion"]
    assert occ2["verdict"] == FAIL
    assert occ2["by_element"][0]["hand_overlap_px"] > 0


def test_scrim_person_overlap_fails(tmp_path):
    base_frames = _base_frames()
    base = _encode(base_frames, tmp_path / "base.mp4")
    box = [110, 30, 20, 20]
    final = _encode(_with_box(base_frames, box), tmp_path / "final.mp4")
    masks = _mask_doc(tmp_path, boxes=[[110, 20, 40, 60]])
    el = {
        "id": "el1", "type": "keyword", "anchor_ref": "a1",
        "anchor_refs": ["a1"], "window": dict(WINDOW),
        "style": {"font_size_px": 36, "color": "#F5F7FA"},
        "content": {"text": "QA DEMO"},
        "layout": {"box": list(box), "treatment": "scrim_right"},
        "backdrop": {"kind": "scrim_right", "box": [100, 0, 60, 88],
                     "alpha": 0.86},
    }
    ir = _ir_with(el, box=box, treatment="scrim_right",
                  backdrop=el["backdrop"])
    metrics = verify(ir=ir, source=base, rendered=final, masks=masks,
                     safe_margin_px=8, make_sheet=False)
    scrim = metrics["checks"]["scrim"]
    assert scrim["verdict"] == FAIL
    assert scrim["by_element"][0]["person_overlap_px"] > 0


# --------------------------------------------------- caption typography (R5)
def _caption_canvas(w=1920, h=920) -> np.ndarray:
    return np.zeros((h, w, 3), np.uint8)


def _draw_line(img, y0, y1, x0, x1, colour=(245, 247, 250)):
    img[y0:y1, x0:x1] = colour
    return img


def test_caption_measure_and_verdict_landscape():
    img = _caption_canvas()
    _draw_line(img, 800, 844, 700, 1200)          # 44 px ink = 4.78 % H
    m = measure_caption_ink(img, [])
    assert m is not None
    assert m["lines"] == 1
    assert abs(m["ink_height_ratio"] - 44 / 920) < 1e-6
    assert abs(m["ink_width_ratio"] - 500 / 1920) < 1e-6
    th = caption_thresholds("landscape", {})
    verdict, problems = caption_verdict(m, "landscape", th)
    assert verdict == PASS, problems

    two = _caption_canvas()
    _draw_line(two, 700, 742, 650, 1250)
    _draw_line(two, 790, 832, 650, 1250)
    m2 = measure_caption_ink(two, [])
    assert m2["lines"] == 2
    assert caption_verdict(m2, "landscape", th)[0] == PASS


def test_caption_negative_controls_landscape():
    th = caption_thresholds("landscape", {})
    # ink too small: 15 px = 1.6 % H (below 2.8 %)
    small = _caption_canvas()
    _draw_line(small, 860, 875, 700, 1200)
    m = measure_caption_ink(small, [])
    assert m["ink_height_ratio"] < 0.028
    assert caption_verdict(m, "landscape", th)[0] == FAIL

    # too wide: 1840 px = 95.8 % W
    wide = _caption_canvas()
    _draw_line(wide, 800, 844, 40, 1880)
    m = measure_caption_ink(wide, [])
    assert m["ink_width_ratio"] > 0.90
    assert caption_verdict(m, "landscape", th)[0] == FAIL

    # bottom margin 0
    low = _caption_canvas()
    _draw_line(low, 900, 920, 700, 1200)
    m = measure_caption_ink(low, [])
    assert m["bottom_margin_ratio"] < 0.045
    assert caption_verdict(m, "landscape", th)[0] == FAIL

    # caption ink over the face
    face = _caption_canvas()
    _draw_line(face, 800, 844, 700, 1200)
    m = measure_caption_ink(face, [[700, 800, 300, 80]])
    assert m["face_intersection_px"] > 0
    assert caption_verdict(m, "landscape", th)[0] == FAIL


def test_caption_portrait_envelope():
    th = caption_thresholds("portrait", {})
    img = _caption_canvas(1080, 1920)
    _draw_line(img, 1600, 1665, 200, 900)         # 66 px = 3.44 % H
    m = measure_caption_ink(img, [])
    assert abs(m["bottom_margin_ratio"] - (1920 - 1665) / 1920) < 1e-6
    assert caption_verdict(m, "portrait", th)[0] == PASS

    low = _caption_canvas(1080, 1920)
    _draw_line(low, 1800, 1865, 200, 900)         # bottom margin 2.9 %
    m_low = measure_caption_ink(low, [])
    assert m_low["bottom_margin_ratio"] < 0.11
    assert caption_verdict(m_low, "portrait", th)[0] == FAIL

    tiny = _caption_canvas(1080, 1920)
    _draw_line(tiny, 1600, 1650, 200, 900)        # 50 px = 2.6 %
    m_tiny = measure_caption_ink(tiny, [])
    assert m_tiny["ink_height_ratio"] < 0.030
    assert caption_verdict(m_tiny, "portrait", th)[0] == FAIL


def test_caption_cue_discovery_and_policy_wiring():
    import numpy as np
    counts = np.array([0, 50, 60, 0, 0, 30, 40, 0])
    cues = discover_caption_cues(counts, min_pixels=20, max_gap=2)
    assert cues == [{"start_frame": 1, "end_frame": 3},
                    {"start_frame": 5, "end_frame": 7}]
    view = caption_policy_view({})
    assert view["policy"]["name"] == "smart_balanced"
    assert view["captions"]["max_lines"] == 2
    assert view["captions"]["bottom_ratio"]["portrait"] == 0.13
    # policy.qa override is honoured and reported
    th = caption_thresholds("portrait", {"qa": {"caption_max_lines": 1,
                                                "caption_bottom_min_portrait": 0.15}})
    assert th["values"]["max_lines"] == 1
    assert th["values"]["bottom_min"] == 0.15
    assert "caption_bottom_min_portrait" in th["source"]


# --------------------------------------------------- non-text graphics (R16)
def test_non_text_graphics_contrast_report():
    ink = np.zeros((100, 100, 3), np.uint8)
    final = np.full((100, 100, 3), 35, np.uint8)
    ink[10:30, 10:30] = (245, 247, 250)       # glyph block (excluded)
    final[10:30, 10:30] = (245, 247, 250)
    ink[45:49, 20:80] = (235, 200, 152)       # gold accent rule
    final[45:49, 20:80] = (235, 200, 152)     # rendered over dark background
    row = non_text_graphic_contrast(ink, final, [0, 0, 100, 100], (245, 247, 250))
    assert row is not None
    assert row["graphic_pixels"] >= 100
    assert row["contrast"] >= 3.0, row
    assert row["verdict"] == PASS

    # a near-background gold rule is below the 3.0 floor
    low_ink = np.zeros((100, 100, 3), np.uint8)
    low_final = np.full((100, 100, 3), 210, np.uint8)
    low_ink[45:49, 20:80] = (235, 200, 152)
    low_final[45:49, 20:80] = (215, 200, 170)
    low = non_text_graphic_contrast(low_ink, low_final, [0, 0, 100, 100],
                                    (245, 247, 250))
    assert low is not None
    assert low["contrast"] < 3.0
    assert low["verdict"] == FAIL
