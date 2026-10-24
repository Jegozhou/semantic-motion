import numpy as np

from semantic_motion.layout import Fields, solve_box


def _fields(gw=32, gh=18, face=(12, 2, 8, 7), seed=0):
    rng = np.random.default_rng(seed)
    hard = np.zeros((gh, gw), bool)
    x, y, w, h = face
    hard[y:y + h, x:x + w] = True
    head = hard.astype(np.float32)
    person = np.zeros((gh, gw), np.float32)
    person[y:, x - 2:x + w + 2] = 1.0
    cost = rng.random((gh, gw)).astype(np.float32) + 8.0 * person
    z = np.zeros((gh, gw), np.float32)
    return Fields(cost=cost, hard=hard, head=head, person=person, text=z, lum=z, clutter=z,
                  face_box=[x / gw, y / gh, w / gw, h / gh], person_box=None, open_side="left")


def test_solver_never_places_on_hard_cells():
    W, H, gw, gh = 1920, 1080, 32, 18
    rng = np.random.default_rng(1)
    for trial in range(40):
        fx, fy = int(rng.integers(0, gw - 8)), int(rng.integers(0, gh - 7))
        f = _fields(face=(fx, fy, 8, 7), seed=trial)
        bw, bh = float(rng.uniform(200, 900)), float(rng.uniform(100, 500))
        pref = (float(rng.random()), float(rng.random()))
        best = solve_box(f, [{"w": bw, "h": bh, "layout_w": bw}], W, H, pref, [], "auto", (gw, gh))
        if best is None:
            continue
        x, y, w, h = best["box"]
        cw, ch = W / gw, H / gh
        x0, y0 = int(np.floor(x / cw)), int(np.floor(y / ch))
        x1, y1 = int(np.ceil((x + w) / cw)), int(np.ceil((y + h) / ch))
        assert not f.hard[y0:y1, x0:x1].any(), (trial, best)


def test_solver_respects_avoid_boxes_and_preference():
    W, H, gw, gh = 1920, 1080, 32, 18
    f = _fields(face=(20, 2, 8, 7))
    first = solve_box(f, [{"w": 600, "h": 200, "layout_w": 600}], W, H, (0.2, 0.3), [], "left", (gw, gh))
    assert first is not None and first["box"][0] < W / 2
    second = solve_box(f, [{"w": 600, "h": 200, "layout_w": 600}], W, H, (0.2, 0.3), [tuple(first["box"])], "left", (gw, gh))
    ax, ay, aw, ah = first["box"]
    bx, by, bw, bh = second["box"]
    assert bx >= ax + aw or bx + bw <= ax or by >= ay + ah or by + bh <= ay


def test_solver_returns_none_when_nothing_fits():
    f = _fields()
    f.hard[:] = True
    assert solve_box(f, [{"w": 300, "h": 100, "layout_w": 300}], 1920, 1080, (0.5, 0.5), [], "auto", (32, 18)) is None
