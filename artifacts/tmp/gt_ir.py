"""Local scratch builder: fixture ground-truth IR with all four effects."""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semotion import ir as irmod  # noqa: E402

CLIP = ROOT / "tests/fixtures/clips/fx_a_zh_portrait"
SRC = CLIP / "fx_a_zh_portrait.mp4"
GT = json.loads((CLIP / "ground_truth.json").read_text(encoding="utf-8"))
CANVAS = {"width": GT["width"], "height": GT["height"], "fps": float(GT["fps"])}
KIND = {"plain": "plain", "keyword": "keyword", "number": "number", "list_item": "list_item"}


def anchors() -> dict:
    return {"schema": "semotion.anchors/1", "duration_s": GT["duration_s"],
            "anchors": [{"id": f"a{i:02d}", "kind": KIND.get(u["role"], "plain"),
                         "text": u["text"], "start_s": u["start_s"], "end_s": u["end_s"],
                         "confidence": 0.9, "source": "SYNTH_KNOWN",
                         "segment_id": u["unit_id"], "word_ids": [u["unit_id"]]}
                        for i, u in enumerate(GT["units"], start=1)]}


def transcript(a: dict) -> dict:
    return {"schema": "semotion.transcript/1",
            "source": {"path": str(SRC), "sha256": GT["sha256"],
                       "duration_s": GT["duration_s"], "width": GT["width"],
                       "height": GT["height"], "fps": GT["fps"]},
            "language": "zh",
            "segments": [{"id": u["unit_id"], "start_s": u["start_s"], "end_s": u["end_s"],
                          "text": u["text"]} for u in GT["units"]],
            "words": [{"id": u["unit_id"], "segment_id": u["unit_id"], "text": u["text"],
                       "start_s": u["start_s"], "end_s": u["end_s"], "probability": 0.99}
                      for u in GT["units"]]}


def vision() -> dict:
    return {"schema": "semotion.vision/1",
            "source": {"width": GT["width"], "height": GT["height"], "fps": GT["fps"],
                       "n_frames": GT["n_frames"]},
            "faces": [], "face_tracks": [],
            "shots": [{"index": 0, "start_s": 0.0, "end_s": GT["duration_s"],
                       "start_frame": 0, "end_frame": GT["n_frames"]}],
            "safe_zones": {"title": [0.08, 0.06, 0.84, 0.12],
                           "caption": [0.08, 0.78, 0.84, 0.11],
                           "card": [0.06, 0.30, 0.88, 0.42]}}


def plan(a: dict) -> dict:
    by = {x["id"]: x for x in a["anchors"]}

    def dec(did, effect, ids, emphasis="high", content=None):
        st = min(by[i]["start_s"] for i in ids)
        en = max(by[i]["end_s"] for i in ids)
        hold = {"keyword": 0.30, "number": 0.35, "list": 0.45, "card": 0.50}[effect]
        payload = dict(content or {"text": by[ids[0]]["text"]})
        return {"id": did, "anchor_id": ids[0], "anchor_ids": ids, "effect": effect,
                "emphasis": emphasis, "reason": "scratch", "confidence": 0.9,
                "decision_model": "scratch-manual", "content": payload,
                "window": {"start_s": round(st, 6), "end_s": round(en + hold, 6)}}

    items = [by[f"a{i:02d}"]["text"] for i in range(7, 13)]
    return {"schema": "semotion.plan/1", "style": "smart_balanced", "canvas": dict(CANVAS),
            "decisions": [dec("d01", "card", ["a02"], "medium"),
                          dec("d02", "keyword", ["a04"]),
                          dec("d03", "number", ["a05"]),
                          dec("d04", "list", [f"a{i:02d}" for i in range(7, 13)], "high",
                              {"text": by["a07"]["text"], "items": items}),
                          dec("d05", "keyword", ["a14"])]}


def build() -> dict:
    a = anchors()
    return irmod.build(transcript(a), vision(), a, plan(a), SRC, CANVAS)


if __name__ == "__main__":
    doc = build()
    print("validate:", irmod.validate(doc))
    for el in doc["elements"]:
        print(el["id"], el["type"], el["layout"]["box"], "font", el["style"]["font_size_px"],
              "hold", el["window"]["hold_s"])
    out = ROOT / "artifacts/g1/full_test/motion_ir_gt.json"
    irmod.dump(doc, out)
    print("wrote", out)
