"""Word-onset timing benchmark against synthetic ground truth (LABELLED SYNTHETIC).

Text-to-speech (Microsoft Edge neural voices via `edge-tts`) gives exact word-boundary times. We render
the speech over a plain background video, run the normal `sm analyze` pipeline (Whisper + onset
refinement) and compare every recognised word onset with the TTS boundary:

    python tests/bench_timing.py --out /tmp/bench    # needs network for edge-tts

Reported: median / p90 absolute onset error (ms) for raw Whisper word times and for the refined
times the engine actually uses, and the share of words within 1 and 2 frames at 25 fps.
This measures the timing layer only; TTS speech is cleaner than real recordings, so real-world
errors are larger. Results are written to <out>/timing_benchmark.json.
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent / "skills" / "semantic-motion" / "engine"
sys.path.insert(0, str(ENGINE))

SCRIPTS = {
    "zh": ("zh-CN-YunxiNeural",
           "大家好，今天我们聊一个很实际的问题。第一，团队用了三个月，把成本降低了百分之四十。"
           "第二，新版本上线以后，日活用户从两万增长到了八万五千。但是，真正重要的不是数字，而是用户的信任。"
           "所以接下来，我会分三步讲清楚：先定目标，再做实验，最后复盘。"),
    "en": ("en-US-GuyNeural",
           "Hi everyone, today I want to talk about something practical. First, in three months the team cut costs "
           "by forty percent. Second, after the new release, daily users grew from twenty thousand to eighty five "
           "thousand. But the number that really matters is trust. So here is the plan in three steps: set a goal, "
           "run the experiment, and review what you learned."),
}


async def tts(text: str, voice: str, mp3: Path) -> list[dict]:
    import edge_tts

    com = edge_tts.Communicate(text, voice, boundary="WordBoundary")
    marks = []
    with open(mp3, "wb") as f:
        async for ch in com.stream():
            if ch["type"] == "audio":
                f.write(ch["data"])
            elif ch["type"] == "WordBoundary":
                marks.append({"text": ch["text"], "start": ch["offset"] / 1e7, "end": (ch["offset"] + ch["duration"]) / 1e7})
    return marks


def make_video(mp3: Path, mp4: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=0x30343a:s=1280x720:r=25",
                    "-i", str(mp3), "-shortest", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "160k", str(mp4)], check=True)


def char_onsets(words: list[dict]) -> list[tuple[str, float]]:
    """Explode words into normalized characters carrying the onset of the word they start (others None)."""
    from semantic_motion.text import norm

    out = []
    for w in words:
        n = norm(w["text"])
        for k, c in enumerate(n):
            out.append((c, w["start"] if k == 0 else None))
    return out


def compare(gt: list[dict], hyp: list[dict]) -> list[float]:
    """Onset errors (s) for every ground-truth word whose first character aligns with a hypothesis word start."""
    a, b = char_onsets(gt), char_onsets(hyp)
    sm = difflib.SequenceMatcher(a=[c for c, _ in a], b=[c for c, _ in b], autojunk=False)
    errs = []
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            ta, tb = a[blk.a + k][1], b[blk.b + k][1]
            if ta is not None and tb is not None:
                errs.append(tb - ta)
    return errs


def stats(errs: list[float], fps: float = 25.0) -> dict:
    import numpy as np

    e = np.abs(np.array(errs)) * 1000
    if not len(e):
        return {}
    fr = 1000 / fps
    return {"n": int(len(e)), "median_ms": round(float(np.median(e)), 1), "p90_ms": round(float(np.percentile(e, 90)), 1),
            "max_ms": round(float(e.max()), 1), "mean_signed_ms": round(float(np.mean(errs) * 1000), 1),
            "within_1_frame": round(float((e <= fr).mean()), 3), "within_2_frames": round(float((e <= 2 * fr).mean()), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/sm_bench")
    ap.add_argument("--langs", default="zh,en")
    ap.add_argument("--asr-model", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sm = str(HERE.parent / "skills" / "semantic-motion" / "scripts" / "sm")
    results = {"label": "SYNTHETIC (edge-tts ground truth)", "langs": {}}
    for lang in a.langs.split(","):
        voice, text = SCRIPTS[lang]
        mp3, mp4 = out / f"tts_{lang}.mp3", out / f"tts_{lang}.mp4"
        gt_path = out / f"tts_{lang}.truth.json"
        if not gt_path.exists():
            marks = asyncio.run(tts(text, voice, mp3))
            gt_path.write_text(json.dumps(marks, ensure_ascii=False, indent=1))
            make_video(mp3, mp4)
        gt = json.loads(gt_path.read_text())
        cmd = [sm, "analyze", str(mp4), "--language", lang] + (["--asr-model", a.asr_model] if a.asr_model else [])
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proj = out / f"tts_{lang}.smproj"
        raw = json.loads((proj / "analysis" / "asr_raw.json").read_text())["words"]
        ref = json.loads((proj / "analysis" / "transcript.json").read_text())["words"]
        r = {"voice": voice, "gt_words": len(gt), "raw_whisper": stats(compare(gt, raw)), "refined": stats(compare(gt, ref))}
        # anchor-level check: phrases a director would anchor on
        from semantic_motion.anchors import Anchors
        tr = json.loads((proj / "analysis" / "transcript.json").read_text())
        anc = Anchors(tr)
        probes = {"zh": ["百分之四十", "两万", "八万五千", "信任", "先定目标"], "en": ["forty percent", "twenty thousand", "trust", "set a goal"]}[lang]
        a_err = []
        for ph in probes:
            try:
                res = anc.resolve({"phrase": ph})
            except Exception as ex:  # noqa: BLE001
                r.setdefault("anchor_failures", []).append(f"{ph}: {ex}")
                continue
            # ground-truth onset of the phrase: first GT word whose text starts the phrase
            from semantic_motion.text import norm
            gtc = char_onsets(gt)
            full = "".join(c for c, _ in gtc)
            k = full.find(norm(ph))
            t_gt = next((t for c, t in gtc[k::-1] if t is not None), None) if k >= 0 else None
            if t_gt is not None:
                a_err.append({"phrase": ph, "anchor_t": round(res["t"], 3), "truth_t": round(t_gt, 3),
                              "err_ms": round((res["t"] - t_gt) * 1000, 1)})
        r["anchors"] = a_err
        results["langs"][lang] = r
        print(json.dumps({lang: {k: r[k] for k in ("raw_whisper", "refined")}}, ensure_ascii=False))
    (out / "timing_benchmark.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"written {out / 'timing_benchmark.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
