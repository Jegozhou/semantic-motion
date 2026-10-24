"""Semantic anchors: resolve plan references ("3:两千多张", {"s":3,"phrase":"..."}, {"w":"w0012"}, {"t":12.3})
to exact spoken times using the refined word/character timeline."""
from __future__ import annotations

import re
from typing import Any, Optional

from .asr import char_timeline
from .text import find_phrase, norm, numeric_variants


class AnchorError(ValueError):
    pass


_SHORT = re.compile(r"^\s*(?:s)?(\d+)\s*[:：]\s*(.+?)\s*$")


def parse_anchor(a: Any) -> dict:
    """Normalize every accepted anchor syntax to a dict."""
    if a is None:
        raise AnchorError("missing anchor")
    if isinstance(a, (int, float)):
        return {"t": float(a)}
    if isinstance(a, dict):
        d = dict(a)
        if "sentence" in d and "s" not in d:
            d["s"] = d.pop("sentence")
        if "text" in d and "phrase" not in d:
            d["phrase"] = d.pop("text")
        return d
    s = str(a).strip()
    d: dict = {}
    m = re.search(r"@(start|end|mid)\s*$", s)
    if m:
        d["edge"] = m.group(1)
        s = s[: m.start()].strip()
    m = re.search(r"([+-]\d*\.?\d+)s?\s*$", s)
    if m and not re.fullmatch(r"[+-]?\d*\.?\d+s?", s):
        d["offset"] = float(m.group(1))
        s = s[: m.start()].strip()
    if re.fullmatch(r"w\d+", s):
        d["w"] = s
    elif re.fullmatch(r"\d*\.?\d+s", s) or re.fullmatch(r"t=\d*\.?\d+", s):
        d["t"] = float(s.replace("t=", "").rstrip("s"))
    elif re.fullmatch(r"s\d+", s):
        d["s"] = int(s[1:])
    else:
        m = _SHORT.match(s)
        if m:
            d["s"] = int(m.group(1))
            d["phrase"] = m.group(2)
        else:
            d["phrase"] = s
    return d


class Anchors:
    def __init__(self, transcript: dict, corrections: Optional[dict] = None):
        self.tr = transcript
        self.words = transcript["words"]
        self.sents = {s["id"]: s for s in transcript["sentences"]}
        self.chars = char_timeline(transcript)
        self.full = "".join(c["c"] for c in self.chars)
        self.corrections = corrections or {}
        self.s_range: dict[int, tuple[int, int]] = {}
        for i, c in enumerate(self.chars):
            s = c["s"]
            a, b = self.s_range.get(s, (i, i))
            self.s_range[s] = (min(a, i), i + 1)

    def sentence(self, sid: int) -> dict:
        if sid not in self.sents:
            raise AnchorError(f"sentence {sid} does not exist (1..{len(self.sents)})")
        return self.sents[sid]

    def _search(self, needle: str, sid: Optional[int], n: int = 1, after: float = -1.0,
                raw: Optional[str] = None) -> tuple[int, int, float, bool]:
        cands = [needle]
        for wrong, right in self.corrections.items():
            nr, nw = norm(right), norm(wrong)
            if nr and nr in needle:
                cands.append(needle.replace(nr, nw))
        # spoken numbers -> digit forms the ASR actually wrote ("百分之四十" -> "40", "twenty thousand" -> "20000")
        for v in numeric_variants(raw if raw is not None else needle):
            if v not in cands:
                cands.append(v)
        ranges = []
        if sid is not None:
            if sid not in self.s_range:
                raise AnchorError(f"sentence {sid} does not exist")
            a, b = self.s_range[sid]
            ranges.append((a, b, False))
            # tolerate an off-by-one sentence reference (re-segmentation, planner slip)
            for nb in (sid + 1, sid - 1):
                if nb in self.s_range:
                    ra, rb = self.s_range[nb]
                    ranges.append((min(a, ra), max(b, rb), True))
        ranges.append((0, len(self.full), sid is not None))
        for a, b, spilled in ranges:
            hay = self.full[a:b]
            # exact matches of every candidate first, fuzzy only afterwards
            for fuzzy, cand in [(False, c) for c in cands] + [(True, c) for c in cands]:
                start = 0
                hits = []
                while True:
                    r = find_phrase(hay, cand, start, fuzzy=fuzzy)
                    if r is None:
                        break
                    i0, i1, sc = r
                    if sc < 0.99 or (hits and i0 <= hits[-1][0]):
                        hits.append((i0, i1, sc))
                        break
                    hits.append((i0, i1, sc))
                    start = i0 + 1
                hits = [h for h in hits if self.chars[a + h[0]]["t0"] >= after - 1e-6] or hits
                if len(hits) >= n:
                    i0, i1, sc = hits[n - 1]
                    return a + i0, a + i1, sc, spilled
        raise AnchorError(f"phrase '{needle}' not found" + (f" in sentence {sid}" if sid is not None else ""))

    def resolve(self, a: Any, after: float = -1.0) -> dict:
        d = parse_anchor(a)
        edge = d.get("edge", "start")
        off = float(d.get("offset", 0.0))
        out: dict = {"spec": d}
        if "t" in d:
            t = float(d["t"])
            out.update(t=t + off, t0=t, t1=t, s=self._sid_at(t), score=1.0, kind="time")
            return out
        if "w" in d:
            wid = d["w"]
            idx = int(str(wid).lstrip("w")) if not isinstance(wid, int) else wid
            if not (0 <= idx < len(self.words)):
                raise AnchorError(f"word {wid} does not exist")
            w = self.words[idx]
            t0, t1 = w["start"], w["end"]
            out.update(t0=t0, t1=t1, s=w.get("s"), w0=idx, w1=idx, score=1.0, kind="word", text=w["text"].strip())
        elif "phrase" in d:
            needle = norm(str(d["phrase"]))
            if not needle:
                raise AnchorError(f"empty phrase in anchor {a!r}")
            sid = int(d["s"]) if d.get("s") is not None else None
            i0, i1, sc, spilled = self._search(needle, sid, int(d.get("n", 1)), after, raw=str(d["phrase"]))
            c0, c1 = self.chars[i0], self.chars[i1 - 1]
            w0, w1 = c0["w"], c1["w"]
            # phrase that starts at a word boundary uses the word's refined onset exactly
            t0 = c0["t0"]
            t1 = c1["t1"]
            out.update(t0=round(t0, 3), t1=round(t1, 3), s=c0["s"], w0=w0, w1=w1, score=round(sc, 3), kind="phrase",
                       text="".join(self.words[k]["text"] for k in range(w0, w1 + 1)).strip(), matched=self.full[i0:i1],
                       char_range=[i0, i1], spilled=spilled,
                       char_times=[round(self.chars[k]["t0"], 3) for k in range(i0, i1)])
            if sid is not None and c0["s"] != sid:
                out["warning"] = f"phrase found in sentence {c0['s']} instead of {sid}"
            if sc < 0.99:
                out["warning"] = (out.get("warning", "") + f" fuzzy match {sc:.2f} -> '{self.full[i0:i1]}'").strip()
        elif "s" in d:
            s = self.sentence(int(d["s"]))
            t0, t1 = s["start"], s["end"]
            out.update(t0=t0, t1=t1, s=s["id"], w0=s["w0"], w1=s["w1"], score=1.0, kind="sentence", text=s["text"])
        else:
            raise AnchorError(f"cannot interpret anchor {a!r}")
        base = out["t0"] if edge == "start" else (out["t1"] if edge == "end" else (out["t0"] + out["t1"]) / 2)
        out["t"] = round(base + off, 3)
        return out

    def _sid_at(self, t: float) -> Optional[int]:
        best = None
        for s in self.tr["sentences"]:
            if s["start"] - 0.05 <= t <= s["end"] + max(0.05, s.get("pause_after", 0)):
                return s["id"]
            if s["start"] <= t:
                best = s["id"]
        return best

    def unit_times(self, units: list[str], anchor: dict, window_s: int = 1) -> list[Optional[float]]:
        """Map display text units (already split by the text engine) to spoken onsets near the anchor.

        The display text may start before the anchored phrase ("With you in spirit" anchored on "in spirit"),
        so every occurrence of the first unit in the window is tried as a start; units are then matched in order.
        Latin units must start on a word boundary (so "in" never matches inside "marking"). The alignment with
        the most matched units, closest to the anchor, wins."""
        if anchor.get("kind") not in ("phrase", "sentence", "word"):
            return [None] * len(units)
        sid = anchor.get("s")
        a, b = 0, len(self.full)
        if sid in self.s_range:
            a = self.s_range.get(sid - window_s, self.s_range[sid])[0]
            b = self.s_range.get(sid + window_s, self.s_range[sid])[1]
        apos = anchor.get("char_range", [a, a])[0]
        normed = [norm(u) for u in units]

        def word_start(k: int) -> bool:
            return k == 0 or self.chars[k]["w"] != self.chars[k - 1]["w"]

        def find(n: str, lo: int) -> int:
            latin = n[:1].isascii() and n[:1].isalnum()
            k = self.full.find(n, lo, b)
            while k >= 0 and latin and not word_start(k):
                k = self.full.find(n, k + 1, b)
            return k

        def align(start: int) -> tuple[list, int, int]:
            out: list[Optional[float]] = []
            pos, first, hits = start, None, 0
            for n in normed:
                if not n:
                    out.append(None)
                    continue
                k = find(n, pos)
                # a unit spoken much later than the previous one is not the same utterance
                if k < 0 or (first is not None and self.chars[k]["t0"] - self.chars[max(pos - 1, 0)]["t0"] > 2.5):
                    out.append(None)
                    continue
                first = k if first is None else first
                out.append(round(self.chars[k]["t0"], 3))
                pos = k + len(n)
                hits += 1
            return out, hits, (first if first is not None else start)

        lead = next((n for n in normed if n), "")
        starts = []
        k = find(lead, a) if lead else -1
        while k >= 0 and len(starts) < 40:
            starts.append(k)
            k = find(lead, k + 1)
        best = None
        for st in starts or [max(a, apos)]:
            out, hits, first = align(st)
            score = hits - 0.002 * abs(first - apos)
            if best is None or score > best[0]:
                best = (score, out)
        return best[1] if best else [None] * len(units)
