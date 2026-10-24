import json
from pathlib import Path

import pytest

from conftest import REAL_PROJECTS
from semantic_motion.compiler import parse_value


def test_parse_value():
    assert parse_value("2,000+") == {"prefix": "", "value": 2000.0, "suffix": "+", "decimals": 0, "separator": ","}
    assert parse_value("$1.5B")["value"] == 1.5 and parse_value("$1.5B")["prefix"] == "$"
    assert parse_value("两千多") == {"value": 2000.0, "suffix": "+"}
    assert parse_value(3.5) == {"value": 3.5}
    with pytest.raises(Exception):
        parse_value("no number here")


@pytest.mark.skipif(not REAL_PROJECTS, reason="set SM_TEST_PROJECTS to analysed .smproj dirs with plan.json")
@pytest.mark.parametrize("path", REAL_PROJECTS)
def test_real_project_compiles_deterministically_and_never_covers_face(path):
    from semantic_motion.compiler import compile_project
    from semantic_motion.project import Project
    from semantic_motion.qa import pre_qa

    proj = Project(path)
    tl1, rep1 = compile_project(proj, write=False)
    tl2, _ = compile_project(proj, write=False)
    assert json.dumps(tl1, sort_keys=True) == json.dumps(tl2, sort_keys=True)
    assert len([e for e in tl1["elements"] if e["type"] != "captions"]) > 0
    pre = pre_qa(proj, tl1, rep1)
    assert "checks" in pre and "ok" in pre
    bad = [c for c in pre["checks"] if c["level"] == "error"
           or c["code"] in {"FACE_OCCLUDED", "HEAD_OCCLUDED", "COLLISION", "OUTSIDE_SAFE"}]
    assert pre["ok"] and not bad, bad


@pytest.mark.skipif(not REAL_PROJECTS, reason="set SM_TEST_PROJECTS to analysed .smproj dirs with plan.json")
@pytest.mark.parametrize("path", REAL_PROJECTS)
def test_rendered_footprint_matches_qa_box(path):
    """Regression for a QA bug where the box was scaled twice: what is drawn must sit on the QA box
    (plus the soft shadow), otherwise face/safe-area checks measure the wrong rectangle."""
    import numpy as np

    from semantic_motion.compiler import compile_project
    from semantic_motion.project import Project
    from semantic_motion.qa import element_box_at, full_reveal_time
    from semantic_motion.render.engine import Compositor

    proj = Project(path)
    tl, _ = compile_project(proj, write=False)
    u = min(tl["W"], tl["H"]) / 100.0
    for e in tl["elements"]:
        if e["type"] == "captions" or "box" not in e:
            continue
        t = min(full_reveal_time(e) + 0.3, e["t_end"] - 0.15)
        a = Compositor(tl, only={e["id"]}).compose(np.zeros((tl["H"], tl["W"], 4), np.uint8), t, alpha_only=True)[:, :, 3]
        ys, xs = np.where(a > 8)
        assert len(xs), f"{e['id']} draws nothing at {t:.2f}s"
        x, y, w, h = element_box_at(e)
        pad = 3.5 * u
        assert xs.min() >= x - pad and ys.min() >= y - pad, e["id"]
        assert xs.max() <= x + w + pad and ys.max() <= y + h + pad, e["id"]
        assert (xs.max() - xs.min()) >= 0.6 * w and (ys.max() - ys.min()) >= 0.5 * h, e["id"]


def test_camera_never_crops_burned_in_text():
    """Punch-ins keep burned-in subtitles and the face in frame by moving the zoom centre (or shrinking the zoom),
    and are refused when subtitles span the full width."""
    from types import SimpleNamespace

    import numpy as np

    from semantic_motion.compiler import Compiler

    gh, gw = 45, 80
    over = np.zeros((gh, gw), np.float32)
    over[38:42, 22:58] = 1.0                       # subtitle band at y 0.84-0.93
    vd = SimpleNamespace(overlay_text=over, shot_text=np.zeros((1, gh, gw), np.float32),
                         sample_shot=np.zeros(10, np.int32), faces=[[[0.42, 0.12, 0.16, 0.3]]] * 10,
                         samples_between=lambda a, b: np.arange(10))
    me = SimpleNamespace(vd=vd)
    z, f = Compiler._camera_keep_text(me, 0.0, 2.0, 1.1, (0.5, 0.33))
    k = 1 - 1 / z
    assert f[1] * k + 1 / z >= 42 / gh - 1e-6        # subtitles' bottom edge still visible
    assert f[1] * k <= 0.12 + 1e-6                   # face top still visible
    # no overlay text -> unchanged
    vd.overlay_text = np.zeros((gh, gw), np.float32)
    assert Compiler._camera_keep_text(me, 0.0, 2.0, 1.1, (0.5, 0.33)) == (1.1, (0.5, 0.33))
    # full-width subtitles -> impossible
    full = np.zeros((gh, gw), np.float32)
    full[38:42, :] = 1.0
    vd.overlay_text = full
    assert Compiler._camera_keep_text(me, 0.0, 2.0, 1.06, (0.5, 0.33)) is None
