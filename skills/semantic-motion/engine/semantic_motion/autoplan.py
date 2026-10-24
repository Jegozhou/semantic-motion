"""Heuristic baseline planner (no LLM). Used when no plan exists, for smoke tests, and as a
starting point Claude can refine. It is deliberately conservative: numbers -> stat,
enumerations -> list, strongly stressed key terms -> keyword, one camera punch on the most
emphatic line. Claude's semantic plan (written from brief.md) is the primary path."""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

from .brief import LIST_MARKERS_EN, LIST_MARKERS_ZH
from .text import find_numbers, is_cjk, norm

STOP_EN = set("the a an and or but so to of in on for with at by from is are was were be been it this that these those "
              "i you he she we they me my your our their his her its as if then than there here what which who whom "
              "will would can could should just also very really about into out up down over more most some any all "
              "not no do does did have has had".split())


def _zh_terms(text: str, k: int = 3) -> list[str]:
    try:
        import jieba.analyse  # type: ignore

        return [w for w in jieba.analyse.extract_tags(text, topK=k * 2, allowPOS=("n", "nr", "ns", "nt", "nz", "vn", "eng"))
                if 2 <= len(w) <= 8][:k]
    except Exception:
        return []


def _en_terms(text: str, k: int = 3) -> list[str]:
    ws = re.findall(r"[A-Za-z][A-Za-z'-]+", text)
    cands = [w for w in ws if w.lower() not in STOP_EN and len(w) > 3]
    cands.sort(key=lambda w: (-(w[0].isupper()), -len(w)))
    out = []
    for w in cands:
        if w.lower() not in [o.lower() for o in out]:
            out.append(w)
    return out[:k]


def _term_conf(term: str, ws: list[dict]) -> float:
    """Lowest ASR word confidence under a term — never put a possibly misheard word on screen."""
    chars, probs = [], []
    for w in ws:
        for c in norm(w["text"]):
            chars.append(c)
            probs.append(float(w.get("p", 1.0)))
    full = "".join(chars)
    k = full.find(norm(term))
    if k < 0:
        return 0.0
    return min(probs[k:k + len(norm(term))] or [0.0])


def auto_plan(proj, speaker: Optional[str] = None, style: str = "balanced") -> dict:
    tr = proj.transcript()
    info = proj.media()
    cjk = tr.get("cjk")
    sents = tr["sentences"]
    words = tr["words"]
    dur = info.duration
    els: list[dict] = []
    busy: list[tuple[float, float]] = []

    def free(t0: float, t1: float, gap: float = 1.2) -> bool:
        return all(t1 + gap <= a or t0 >= b + gap for a, b in busy)

    target = max(2, int(round(dur / 60 * (5 if style == "balanced" else 3 if style == "minimal" else 8))))
    # 1) speaker name (only when supplied: never guess names from ASR)
    if speaker:
        name, _, role = speaker.partition("|")
        s0 = sents[0]
        els.append({"id": "speaker", "type": "lower_third", "at": f"{s0['id']}:" + norm(s0["text"])[:2],
                    "until": {"hold": 3.5}, "content": {"name": name.strip(), **({"role": role.strip()} if role.strip() else {})},
                    "intent": "identify the speaker"})
    # 2) numbers -> stat
    for s in sents:
        if len([e for e in els if e["type"] != "lower_third"]) >= target:
            break
        nums = [n for n in find_numbers(s["text"]) if (n["value"] >= 10 or n["suffix"] in ("%", "倍", "x", "X"))]
        if not nums:
            continue
        n = nums[0]
        if not free(s["start"], s["end"]):
            continue
        # the same number repeated later ("20 years ... 20 years since") is not news: show it once
        if any(e["type"] == "stat" and abs(e["content"]["value"] - n["value"]) < 1e-9 for e in els):
            continue
        rest = s["text"][n["span"][1]:]
        rest = re.split(r"[，。,.!?！？；;]", rest)[0]
        rest = rest.lstrip("多余个张位名")
        label = rest[:12] if cjk else " ".join(rest.split()[:5])
        unit = ""
        m = re.match(r"^(多)?([个张位名人次年天元件家倍%])", s["text"][n["span"][1]:])
        if m:
            unit = m.group(2)
        val = n["value"]
        content = {"value": val, "suffix": ("+" if s["text"][n["span"][1]:].startswith("多") else "") + (n["suffix"] or ""),
                   "unit": unit if unit not in ("%",) else "", "label": label.strip() or s["text"][:12]}
        els.append({"id": f"stat{s['id']}", "type": "stat", "at": f"{s['id']}:{n['raw']}", "until": "sentence",
                    "content": content, "importance": 3, "intent": "auto: spoken number"})
        busy.append((s["start"], s["end"] + 1.0))
    # 3) enumerations -> list
    markers = LIST_MARKERS_ZH if cjk else LIST_MARKERS_EN
    run = []
    for s in sents + [None]:
        hit = s is not None and any(norm(s["text"]).startswith(norm(m)) or (" " + m + " ") in (" " + s["text"].lower() + " ")
                                    for m in markers[:6])
        if hit:
            run.append(s)
            continue
        if len(run) >= 2 and free(run[0]["start"], run[-1]["end"]):
            items = []
            for r in run[:4]:
                txt = re.split(r"[，。,.!?！？；;]", r["text"])[0]
                items.append({"text": txt[:14] if cjk else " ".join(txt.split()[:6]), "at": f"{r['id']}:{norm(r['text'])[:3]}"})
            els.append({"id": f"list{run[0]['id']}", "type": "list", "at": items[0]["at"], "until": f"{run[-1]['id']}:"
                        + norm(run[-1]["text"])[-2:], "content": {"items": items}, "intent": "auto: enumeration"})
            busy.append((run[0]["start"], run[-1]["end"] + 1.0))
        run = []
    # 4) stressed key terms -> keyword, spread over the video
    scored = []
    for s in sents:
        ws = words[s["w0"]: s["w1"] + 1]
        e = max([w.get("emph", 0) for w in ws] or [0])
        if s["end"] - s["start"] < 1.0:
            continue
        scored.append((e, s))
    scored.sort(key=lambda x: -x[0])
    for e, s in scored:
        if len(els) >= target + (1 if speaker else 0):
            break
        if not free(s["start"], s["end"], gap=3.0):
            continue
        terms = _zh_terms(s["text"]) if cjk else _en_terms(s["text"])
        terms = [t for t in terms if norm(t) in norm(s["text"]) and _term_conf(t, ws) >= 0.8]
        if not terms:
            continue
        kw = terms[0]
        els.append({"id": f"kw{s['id']}", "type": "keyword", "at": f"{s['id']}:{kw}", "until": "sentence",
                    "content": {"text": kw, "highlight": kw}, "motion": {"reveal": "word"}, "intent": "auto: stressed term"})
        busy.append((s["start"], s["end"] + 1.0))
    # 5) one camera punch on the most emphatic sentence (inside a single shot)
    shots = proj.shots()
    for e, s in scored[:5]:
        sh = next((x for x in shots if x["start"] <= s["start"] < x["end"]), None)
        if sh and sh["end"] - s["start"] > 1.6 and sh.get("speaker"):
            els.append({"id": f"cam{s['id']}", "type": "camera", "move": "punch_in", "at": f"{s['id']}:" + norm(s["text"])[:2],
                        "until": "sentence", "intent": "auto: emphasis"})
            break
    els.sort(key=lambda e: e.get("at", ""))
    return {"version": 1, "style": style, "accent": "auto", "platform": proj.settings.get("platform", "auto"),
            "generator": "auto-plan (heuristic baseline)", "corrections": {}, "elements": els}
