"""Scratch: IR + render + verify on the real-face 16:9 fixture (1280x720, 30 fps)."""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semotion import ir as irmod, media, render as rmod, verify  # noqa: E402

CLIP_DIR = ROOT / "tests/fixtures/clips/fx_pd_real_16x9_a"
SRC = CLIP_DIR / "fx_pd_real_16x9_a.mp4"
VISION = json.loads((ROOT / "artifacts/g1/probe/vision_pd.json").read_text(encoding="utf-8"))
OUT_DIR = ROOT / "artifacts/g1/real_face"

ITEMS = [
    ("a01", "keyword", "口播视频最难的地方", 2.0, 4.2),
    ("a02", "number", "九成的剪辑", 5.4, 7.0),
    ("a03", "list_item", "第一", 8.4, 8.9),
    ("a04", "list_item", "不知道什么时候加字", 9.0, 10.7),
    ("a05", "list_item", "第二", 10.8, 11.2),
    ("a06", "list_item", "加了又挡住脸", 11.4, 13.0),
    ("a07", "card", "其实是节奏", 15.0, 17.2),
]


def main() -> int:
    info = media.info(SRC)
    sha = media.sha256_file(SRC)
    canvas = {"width": info.width, "height": info.height, "fps": info.fps}
    anchors = {"schema": "semotion.anchors/1", "duration_s": info.duration,
               "anchors": [{"id": i, "kind": k, "text": t, "start_s": s, "end_s": e,
                            "confidence": 0.9, "source": "SYNTH_KNOWN",
                            "segment_id": i, "word_ids": [i]}
                           for (i, k, t, s, e) in ITEMS]}
    by = {a["id"]: a for a in anchors["anchors"]}

    def dec(did, effect, ids, emphasis="high", content=None):
        st = min(by[i]["start_s"] for i in ids)
        en = max(by[i]["end_s"] for i in ids)
        hold = {"keyword": 0.30, "number": 0.35, "list": 0.45, "card": 0.50}[effect]
        return {"id": did, "anchor_id": ids[0], "anchor_ids": ids, "effect": effect,
                "emphasis": emphasis, "reason": "scratch", "confidence": 0.9,
                "decision_model": "scratch-manual",
                "content": dict(content or {"text": by[ids[0]]["text"]}),
                "window": {"start_s": round(st, 6), "end_s": round(en + hold, 6)}}

    list_ids = ["a03", "a04", "a05", "a06"]
    plan = {"schema": "semotion.plan/1", "style": "smart_balanced", "canvas": canvas,
            "decisions": [dec("d01", "keyword", ["a01"]),
                          dec("d02", "number", ["a02"]),
                          dec("d03", "list", list_ids, "high",
                              {"text": "第一",
                               "items": [by[i]["text"] for i in list_ids]}),
                          dec("d04", "card", ["a07"], "medium")]}
    transcript = {
        "schema": "semotion.transcript/1",
        "source": {"path": str(SRC), "sha256": sha, "duration_s": info.duration,
                   "width": info.width, "height": info.height, "fps": info.fps},
        "language": "zh",
        "segments": [{"id": i, "start_s": s, "end_s": e, "text": t}
                     for (i, _k, t, s, e) in ITEMS],
        "words": [{"id": i, "segment_id": i, "text": t, "start_s": s, "end_s": e,
                   "probability": 0.99} for (i, _k, t, s, e) in ITEMS],
    }
    doc = irmod.build(transcript, VISION, anchors, plan, SRC, canvas)
    errs = irmod.validate(doc)
    print("validate:", errs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    irmod.dump(doc, OUT_DIR / "motion_ir.json")

    face = doc["semantic"]["face_safe"]["bbox_union"]
    area = face[2] * face[3]
    print("face union:", face)
    for el in doc["elements"]:
        box = el["layout"]["box"]
        inter = (max(0, min(box[0] + box[2], face[0] + face[2]) - max(box[0], face[0]))
                 * max(0, min(box[1] + box[3], face[1] + face[3]) - max(box[1], face[1])))
        print(f"  {el['id']} {el['type']:8s} box={box} zone={el['layout']['zone']} "
              f"font={el['style']['font_size_px']} face_overlap={inter / area:.4%} "
              f"degraded={el['layout']['degraded']}")

    out = OUT_DIR / "fx_pd_motion.mp4"
    man = rmod.render(doc, SRC, out)
    print("render:", man["checks"]["verdict"], man["output"]["n_frames"],
          f"{man['output']['width']}x{man['output']['height']}")
    m = verify.metrics(doc, man, source=SRC, rendered=out, truth_json=None,
                       vision=VISION, gate="G2-realface", work=OUT_DIR)
    print("verdicts:", m["verdicts"])
    print("planned:", json.dumps(m["position"]["planned"], ensure_ascii=False))
    print("observed:", json.dumps(m["position"]["observed"], ensure_ascii=False))
    print("collision:", m["position"]["collision"]["verdict"],
          m["position"]["collision"]["pairs"], "/", m["position"]["collision"]["compared_pairs"])
    print("render checks:", m["render"]["verdict"], m["render"]["audio"]["rms_delta_db"])
    return 0 if all(v in ("PASS", "NOT RUN") for v in m["verdicts"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
