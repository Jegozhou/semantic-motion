"""Plan compiler: plan.json (semantic intent) -> timeline.json (exact frames, boxes, styles).

Deterministic: the same plan + analysis always produces the same timeline. Every automatic
decision (timing snap, conflict trim, layout move, backing change, drop) is written to the
compile report so the planner (Claude) can see and override it.
"""
from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Any, Optional

import numpy as np

from .anchors import AnchorError, Anchors, parse_anchor
from .elements import load_all
from .elements.base import RenderCtx, get_element_class
from .layout import box_stats, build_fields, preference_point, solve_box
from .render.fonts import FontBook
from .style import resolve_style, safe_rect
from .text import cjk_ratio, find_numbers, is_cjk, norm
from .util import SMError, sha1_obj

PRIMARY = {"keyword", "stat", "list", "compare", "chart", "flow", "quote", "title", "custom"}
SECONDARY = {"lower_third", "callout"}
OVERLAY = {"captions"}
CARD_TYPES = {"list", "compare", "chart", "flow", "quote"}
REQUIRED = {"keyword": ["text"], "stat": ["value"], "list": ["items"], "compare": ["left", "right"], "chart": [],
            "flow": ["steps"], "quote": ["text"], "title": ["text"], "lower_third": ["name"], "callout": ["text"],
            "custom": ["layers"]}
SIZE_MUL = {"xs": 0.72, "s": 0.86, "m": 1.0, "l": 1.16, "xl": 1.32}
CAMERA_MOVES = {"punch_in", "push_in", "ease_in", "reset"}


def _strings(obj) -> list[str]:
    out = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("highlight", "unit_times", "at", "target", "kind", "style", "mode", "direction", "marker", "d", "shape",
                     "fill", "stroke", "keys", "align", "valign", "reveal", "winner"):
                continue
            out += _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _strings(v)
    return out


INTERNAL_SIZE = {"keyword", "stat", "captions", "custom"}  # sized by content.size_mul / content.size themselves


def _size_variants(t: str, variants: list, c: dict) -> list:
    """placement.size for elements without internal type scaling: scale every candidate box by size_mul
    (the renderer applies the same factor through the element's canvas scale)."""
    k = float(c.get("size_mul", 1.0)) if t not in INTERNAL_SIZE else 1.0
    if abs(k - 1.0) < 1e-3:
        return variants
    return [{**v, "w": v["w"] * k, "h": v["h"] * k} for v in variants]


def _is_anchor(v) -> bool:
    """A custom-layer time is a spoken anchor when it names a sentence phrase ('6:phrase'), not 'land+0.2'/'@12.3'."""
    return isinstance(v, str) and bool(re.search(r"[:：]", v)) and not v.strip().startswith("@")


def reading_time(content: dict, style: dict) -> float:
    txt = " ".join(_strings(content))
    cjk = sum(1 for c in txt if is_cjk(c))
    latin_words = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'.,%-]*", txt))
    r = style["reading"]
    return r["base"] + cjk / r["cjk_cps"] + latin_words / r["latin_wps"]


def parse_value(v) -> dict:
    """'2,000+' -> {value: 2000, suffix: '+'}; '两千多' -> 2000 + '+'; 3.5 -> 3.5."""
    if isinstance(v, (int, float)):
        return {"value": float(v)}
    s = str(v).strip()
    m = re.match(r"^([^\d\-+.]*)([-+]?\d[\d,]*\.?\d*)\s*(.*)$", s)
    if m:
        d = {"prefix": m.group(1), "value": float(m.group(2).replace(",", "")), "suffix": m.group(3),
             "decimals": len(m.group(2).split(".")[1]) if "." in m.group(2) else 0}
        if "," in m.group(2):
            d["separator"] = ","
        return d
    nums = find_numbers(s)
    if nums:
        n = nums[0]
        rest = s[n["span"][1]:]
        return {"value": float(n["value"]), "suffix": (n["suffix"] or "") + ("+" if rest.startswith("多") else "")}
    raise SMError(f"cannot parse numeric value {v!r}")


class Compiler:
    def __init__(self, proj, plan: Optional[dict] = None):
        load_all()
        self.proj = proj
        self.plan = deepcopy(plan if plan is not None else proj.plan())
        self.info = proj.media()
        self.tr = proj.transcript()
        self.vd = proj.vision()
        self.af = proj.audio()
        self.shots = proj.shots()
        self.W, self.H = self.info.width, self.info.height
        self.orientation = self.info.orientation if self.info.orientation != "square" else "landscape"
        self.issues: list[dict] = []
        p = self.plan
        platform = p.get("platform") or proj.settings.get("platform") or "auto"
        self.platform = platform
        self.safe = tuple(p["safe_area"]) if p.get("safe_area") else safe_rect(platform, self.orientation)
        rgb = np.mean(self.vd.shot_rgb, axis=0) if len(self.vd.shot_rgb) else None
        self.style = resolve_style(p.get("style", "balanced"), p.get("accent", "auto"), rgb, p.get("style_overrides"))
        self.fb = FontBook(p.get("font"))
        self.ctx = RenderCtx(self.W, self.H, self.fb, self.style, self.orientation)
        self.anchors = Anchors(self.tr, p.get("corrections"))
        self.pauses = [(a, b) for a, b in self.af.pauses(0.2)]
        self.cuts = self._cuts()
        self.duration = self.info.duration
        self.chin = 0.06 if self.orientation == "portrait" else None

    # ------------------------------------------------------------------ utilities
    def issue(self, level: str, code: str, msg: str, el: Optional[str] = None, **kw) -> None:
        self.issues.append({"level": level, "code": code, "element": el, "msg": msg, **kw})

    def _cuts(self) -> list[dict]:
        out = []
        for a, b in zip(self.shots, self.shots[1:]):
            same = bool(a.get("speaker") and b.get("speaker"))
            pa, pb = a.get("person_box"), b.get("person_box")
            if same and pa and pb:
                ix = max(0, min(pa[2], pb[2]) - max(pa[0], pb[0])) * max(0, min(pa[3], pb[3]) - max(pa[1], pb[1]))
                ua = (pa[2] - pa[0]) * (pa[3] - pa[1]) + (pb[2] - pb[0]) * (pb[3] - pb[1]) - ix
                same = ix / max(ua, 1e-6) > 0.3
            out.append({"t": b["start"], "kind": "jump" if same else "scene"})
        return out

    def _snap_to_pause(self, t: float, lo: float, hi: float) -> float:
        best = None
        for a, b in self.pauses:
            if lo <= a <= hi:
                d = abs(a - t)
                if best is None or d < best[0]:
                    best = (d, a, b)
        if best is None:
            return t
        _, a, b = best
        return min(a + 0.15, b)

    def _sentence_end(self, sid: Optional[int]) -> Optional[float]:
        if sid is None:
            return None
        s = self.anchors.sents.get(sid)
        return s["end"] if s else None

    # ------------------------------------------------------------------ normalization
    def _normalize(self, e: dict, i: int) -> dict:
        e = deepcopy(e)
        e.setdefault("id", f"e{i + 1}")
        t = e.get("type")
        if t == "camera":
            return e
        if t not in REQUIRED and t != "captions":
            raise SMError(f"element {e['id']}: unknown type '{t}'")
        c = e.setdefault("content", {})
        # accept flat content (fields at top level) for convenience
        for k in list(e.keys()):
            if k in ("text", "sub", "value", "label", "items", "left", "right", "data", "steps", "name", "role", "title",
                     "kicker", "by", "layers", "unit", "suffix", "prefix", "highlight", "kind", "target", "size"):
                if k not in c and k not in ("size",):
                    c[k] = e.pop(k)
        for req in REQUIRED.get(t, []):
            if req not in c:
                raise SMError(f"element {e['id']} ({t}): content.{req} is required")
        if t == "stat":
            pv = parse_value(c["value"])
            for k, v in pv.items():
                if k == "value" or k not in c:
                    c[k] = v
        if t == "chart" and not c.get("data") and c.get("value") is None:
            raise SMError(f"element {e['id']} (chart): content.data or content.value is required")
        pl = e.get("placement") or {}
        if isinstance(pl, str):
            pl = {"region": pl}
        e["placement"] = pl
        e.setdefault("importance", 2)
        e.setdefault("motion", {})
        return e

    # ------------------------------------------------------------------ timing
    def _resolve(self, e: dict, key: str, a, edge_default: str = "start", after: float = -1.0) -> Optional[dict]:
        try:
            d = parse_anchor(a)
            if edge_default != "start" and "edge" not in d:
                d["edge"] = edge_default
            r = self.anchors.resolve(d, after)
            if r.get("warning"):
                self.issue("warn", "ANCHOR_FUZZY", r["warning"], e.get("id"), field=key)
            return r
        except AnchorError as ex:
            self.issue("error", "ANCHOR_NOT_FOUND", f"{key}: {ex}", e.get("id"), field=key, anchor=a)
            return None

    def _time_element(self, e: dict) -> Optional[dict]:
        m = self.style["motion"]
        t = e["type"]
        c = e["content"]
        # item anchors (lists, flows, charts, compare sides)
        item_lists = {"list": c.get("items"), "flow": c.get("steps"), "chart": c.get("data")}
        items = item_lists.get(t) or []
        if t == "compare":
            items = [c.get("left") or {}, c.get("right") or {}]
        item_res = []
        for k, it in enumerate(items):
            if isinstance(it, dict) and it.get("at") is not None:
                r = self._resolve(e, f"items[{k}].at", it["at"])
                if r:
                    lf = m["land_frac"] * m["enter"]
                    it["t_land"] = round(r["t"], 3)
                    it["_anchor"] = {"t": r["t"], "s": r.get("s"), "text": r.get("text")}
                    item_res.append(r)
        if t == "custom":
            # layer timings may be spoken anchors too ("6:English Wikipedia", "6:first started+1.2"):
            # resolve them to absolute '@t' so every beat of a procedural animation lands on its word
            lead_in = m["land_frac"] * float(e["motion"].get("enter", m["enter"]))
            for li, L in enumerate(c.get("layers", [])):
                if not isinstance(L, dict):
                    continue
                if "at" in L and "from" not in L:
                    L["from"] = L.pop("at")
                if _is_anchor(L.get("from")):
                    r = self._resolve(e, f"layers[{li}].from", L["from"])
                    if r:
                        L["from"] = f"@{max(0.0, r['t'] - lead_in):.3f}"
                        item_res.append(r)
                for prop, ks in (L.get("keys") or {}).items():
                    for k in ks:
                        if isinstance(k, list) and k and _is_anchor(k[0]):
                            r = self._resolve(e, f"layers[{li}].keys.{prop}", k[0])
                            if r:
                                k[0] = f"@{r['t']:.3f}"
                                item_res.append(r)
        at = e.get("at")
        if at is None and item_res:
            A = item_res[0]
        elif at is None:
            self.issue("error", "NO_ANCHOR", "element has no 'at' anchor", e["id"])
            return None
        else:
            A = self._resolve(e, "at", at)
            if A is None:
                return None
        enter = float(e["motion"].get("enter", m["enter"]))
        exit_ = float(e["motion"].get("exit", m["exit"]))
        lead = float(e["motion"].get("lead", 0.0))  # >0 lands earlier than the word (anticipation)
        t_land = A["t"] - lead
        t_in = t_land - m["land_frac"] * enter
        if t_in < 0:
            t_in, t_land = 0.0, m["land_frac"] * enter
        # ---- span end
        until = e.get("until", "auto")
        span_end = None
        explicit = False
        if isinstance(until, dict) and "hold" in until:
            span_end = t_land + float(until["hold"])
            explicit = True
        elif until in ("auto", "sentence", None):
            sids = [A.get("s")] + [r.get("s") for r in item_res]
            ends = [self._sentence_end(s) for s in sids if s is not None]
            span_end = max(ends) if ends else t_land + 2.5
            if t == "keyword" and until == "auto" and A.get("kind") == "phrase":
                # keywords hold through the rest of their clause but not much longer
                span_end = min(span_end, max(A["t1"] + 1.6, t_land + 1.8)) if span_end - t_land > 4.0 else span_end
        elif until == "phrase":
            span_end = A.get("t1", t_land + 1.5)
        elif until == "next":
            span_end = t_land + self.style["max_hold"].get(t, 5.0)
            e["_until_next"] = True
        else:
            U = self._resolve(e, "until", until, edge_default="end", after=A["t0"])
            if U is None:
                span_end = (self._sentence_end(A.get("s")) or t_land + 2.5)
            else:
                span_end = U["t"]
                explicit = True
        if item_res:
            span_end = max(span_end, max(r["t"] for r in item_res) + 1.2)
        read = reading_time(c, self.style)
        min_vis = max(1.0, read * (0.85 if t in CARD_TYPES else 1.0))
        t_out = max(span_end + m["tail"], t_land + min_vis)
        mh = self.style["max_hold"].get(t, 8.0)
        if not explicit and t_out - t_land > mh:
            t_out = t_land + mh
            self.issue("info", "MAX_HOLD", f"held {mh:.1f}s max for {t}", e["id"])
        elif explicit and t_out - t_land > mh * 1.6:
            self.issue("warn", "LONG_HOLD", f"{t} stays {t_out - t_land:.1f}s (type max {mh:.1f}s)", e["id"])
        # exit inside a nearby pause so the graphic leaves while the speaker breathes
        if not e.get("no_snap"):
            t_snap = self._snap_to_pause(t_out, t_out - 0.25, t_out + 0.5)
            if t_snap >= t_land + min_vis * 0.9:
                t_out = t_snap
        t_out = min(t_out, self.duration - exit_ - 0.02)
        t_end = t_out + exit_
        exit_mode = "anim"
        # ---- shot cuts
        for cut in self.cuts:
            ct = cut["t"]
            if t_in < ct < t_in + 0.8 * enter and ct - t_in <= 0.25:
                shift = ct - t_in
                t_in, t_land = ct, t_land + shift
                self.issue("info", "ENTER_ON_CUT", f"entrance aligned to cut at {ct:.2f}s (+{shift * 1000:.0f} ms)", e["id"])
            if cut["kind"] == "scene" and t_land + 0.3 < ct < t_end and not e.get("across_cuts"):
                if ct - t_land >= max(0.9, 0.6 * min_vis):
                    t_out, t_end = ct - exit_, ct
                else:
                    t_end, exit_mode = ct, "cut"
                self.issue("info", "END_AT_CUT", f"ends with the scene cut at {ct:.2f}s", e["id"])
                break
        e.update(t_in=round(t_in, 3), t_land=round(t_land, 3), t_out=round(t_out, 3), t_end=round(t_end, 3),
                 exit_mode=exit_mode, enter_dur=enter, exit_dur=exit_, min_vis=round(min_vis, 2), read_time=round(read, 2))
        if item_res and t in CARD_TYPES and min(r["t"] for r in item_res) - t_land > 1.2:
            self.issue("warn", "EMPTY_CARD", f"card is shown {min(r['t'] for r in item_res) - t_land:.1f}s before its first "
                                             "item is spoken; anchor 'at' closer to the first item", e["id"])
        e["_anchor"] = {k: A.get(k) for k in ("t", "t0", "t1", "s", "w0", "w1", "text", "score", "kind")}
        e["_A"] = A
        # stat: count across the spoken number
        if t == "stat" and c.get("count", True) is not False:
            c.setdefault("count_t0", round(t_land, 3))
            c.setdefault("count_t1", round(min(t_land + 1.1, max(t_land + 0.6, A.get("t1", t_land) + 0.1)), 3))
        return e

    # ------------------------------------------------------------------ conflicts
    def _resolve_conflicts(self, els: list[dict]) -> list[dict]:
        gap = self.style["motion"]["min_gap"]
        layers = {"primary": PRIMARY, "secondary": SECONDARY}
        kept: list[dict] = []
        dropped = set()
        for name, types in layers.items():
            group = sorted([e for e in els if e["type"] in types and e.get("layer", name) == name], key=lambda e: e["t_in"])
            maxc = int(self.style["density"]["max_primary"]) if name == "primary" else 1
            active: list[dict] = []
            for e in group:
                active = [p for p in active if p["t_end"] + gap > e["t_in"] and p["id"] not in dropped]
                if len(active) < maxc:
                    active.append(e)
                    continue
                p = max(active, key=lambda q: q["t_end"])
                need_out = e["t_in"] - gap - p["exit_dur"]
                if need_out - p["t_land"] >= p["min_vis"] * 0.8 or p.get("_until_next"):
                    old = p["t_out"]
                    p["t_out"] = round(max(p["t_land"] + 0.6, need_out), 3)
                    p["t_end"] = round(p["t_out"] + p["exit_dur"], 3)
                    p["exit_mode"] = "anim"
                    self.issue("info", "TRIM_FOR_NEXT", f"exit moved {old:.2f}->{p['t_out']:.2f}s so {e['id']} can enter", p["id"])
                    active = [q for q in active if q is not p] + [e]
                    continue
                delay = p["t_land"] + p["min_vis"] * 0.8 + p["exit_dur"] + gap - e["t_in"]
                if delay <= 0.45 and not e.get("strict"):
                    p["t_out"] = round(p["t_land"] + p["min_vis"] * 0.8, 3)
                    p["t_end"] = round(p["t_out"] + p["exit_dur"], 3)
                    for k in ("t_in", "t_land"):
                        e[k] = round(e[k] + delay, 3)
                    if e["t_out"] < e["t_land"] + e["min_vis"]:
                        e["t_out"] = round(e["t_land"] + e["min_vis"], 3)
                        e["t_end"] = round(e["t_out"] + e["exit_dur"], 3)
                    self.issue("warn", "DELAYED", f"entrance delayed {delay * 1000:.0f} ms to avoid overlapping {p['id']}", e["id"])
                    active = [q for q in active if q is not p] + [e]
                    continue
                loser = e if e.get("importance", 2) <= p.get("importance", 2) else p
                winner = p if loser is e else e
                if loser is p:
                    p["t_out"] = round(e["t_in"] - gap - p["exit_dur"], 3)
                    p["t_end"] = round(e["t_in"] - gap, 3)
                    if p["t_out"] - p["t_land"] < 0.5:
                        dropped.add(p["id"])
                        self.issue("warn", "DROPPED", f"dropped: overlaps more important {e['id']}", p["id"])
                    else:
                        self.issue("warn", "CUT_SHORT", f"cut short for more important {e['id']}", p["id"])
                    active = [q for q in active if q is not p] + [e]
                else:
                    dropped.add(e["id"])
                    self.issue("warn", "DROPPED", f"dropped: overlaps {winner['id']} (raise importance or move anchor)", e["id"])
            kept += [e for e in group if e["id"] not in dropped]
        kept += [e for e in els if e["type"] not in PRIMARY | SECONDARY or e.get("layer") not in (None, "primary", "secondary")]
        seen = set()
        out = []
        for e in sorted(kept, key=lambda e: e["t_in"]):
            if e["id"] in seen or e["id"] in dropped:
                continue
            seen.add(e["id"])
            out.append(e)
        return out

    # ------------------------------------------------------------------ camera
    def _camera(self, cams: list[dict]) -> list[dict]:
        track = []
        cs = self.style["camera"]
        for e in cams:
            move = e.get("move", "punch_in")
            if move not in CAMERA_MOVES:
                self.issue("error", "BAD_CAMERA", f"unknown camera move '{move}'", e.get("id"))
                continue
            A = self._resolve(e, "at", e.get("at"))
            if A is None:
                continue
            until = e.get("until", "sentence")
            if isinstance(until, dict) and "hold" in until:
                t1 = A["t"] + float(until["hold"])
            elif until in ("sentence", "auto", None):
                t1 = (self._sentence_end(A.get("s")) or A["t"] + 2.0) + 0.25
            else:
                U = self._resolve(e, "until", until, edge_default="end", after=A["t0"])
                t1 = U["t"] if U else A["t"] + 2.0
            t0 = A["t"] - (0.0 if move == "punch_in" else 0.1)
            # never straddle a scene cut: end the move at the next cut
            for cut in self.cuts:
                if t0 + 0.2 < cut["t"] < t1:
                    t1 = cut["t"]
                    break
            t1 = self._snap_to_pause(t1, t1 - 0.2, t1 + 0.4)
            if t1 - t0 < 0.8:
                self.issue("warn", "CAMERA_TOO_SHORT", f"camera move would last only {t1 - t0:.2f}s before the next cut; "
                                                       "skipped (anchor it earlier inside the shot)", e.get("id"))
                continue
            zoom = float(e.get("zoom", cs["punch"] if move == "punch_in" else cs["push"]))
            focus = e.get("focus", "face")
            if isinstance(focus, (list, tuple)):
                f = (float(focus[0]), float(focus[1]))
            else:
                idx = self.vd.samples_between(t0, t1)
                fbs = [fb for i in idx for fb in self.vd.faces[int(i)]]
                if fbs:
                    fa = np.array(fbs)
                    f = (float(np.median(fa[:, 0] + fa[:, 2] / 2)), float(np.median(fa[:, 1] + fa[:, 3] * 0.45)))
                else:
                    f = (0.5, 0.4)
            if move != "reset" and not isinstance(focus, (list, tuple)):
                adj = self._camera_keep_text(t0, t1, zoom, f)
                if adj is None:
                    self.issue("warn", "CAMERA_CROPS_TEXT", "camera move would crop burned-in text (subtitles/name tag) "
                                                            "at any face-safe framing; skipped", e.get("id"))
                    continue
                if (round(adj[0], 3), tuple(round(v, 3) for v in adj[1])) != (round(zoom, 3), tuple(round(v, 3) for v in f)):
                    self.issue("info", "CAMERA_ADJUSTED", f"zoom {zoom:.3f}->{adj[0]:.3f}, focus "
                                                          f"({f[0]:.2f},{f[1]:.2f})->({adj[1][0]:.2f},{adj[1][1]:.2f}) "
                                                          "to keep burned-in text inside the frame", e.get("id"))
                zoom, f = adj
            in_dur = {"punch_in": float(e.get("dur", 0.0)), "push_in": max(0.6, t1 - t0), "ease_in": 0.5, "reset": 0.0}[move]
            track.append({"id": e.get("id"), "move": move, "t0": round(t0, 3), "t1": round(t1, 3), "zoom": zoom,
                          "focus": [round(f[0], 4), round(f[1], 4)], "in_dur": in_dur,
                          "out_dur": float(e.get("out_dur", 0.0 if move == "punch_in" else cs["release_dur"]))})
        track.sort(key=lambda c: c["t0"])
        for a, b in zip(track, track[1:]):
            if b["t0"] < a["t1"]:
                a["t1"] = b["t0"]
                a["out_dur"] = 0.0
        return track

    def _camera_keep_text(self, t0: float, t1: float, zoom: float, f: tuple) -> Optional[tuple]:
        """A punch/push must not crop burned-in subtitles or name tags, nor the face. Returns (zoom, focus) - the
        requested framing when it already keeps them, otherwise the closest focus (and, if needed, a smaller zoom)
        that does; None when no zoom >= 1.02 can keep both."""
        vd = self.vd
        idx = vd.samples_between(t0, t1)
        shots = sorted({int(vd.sample_shot[int(i)]) for i in idx}) if len(idx) else []
        # burned-in overlays only (subtitles, name tags, logos); per-shot detections extend them along their rows
        # (subtitle lines vary in width) but background text elsewhere in the scene may be cropped like any object
        over = np.asarray(vd.overlay_text, dtype=np.float32) > 0.2
        band = over.any(axis=1)
        txt = over.copy()
        for si in shots:
            txt |= (np.asarray(vd.shot_text[si], dtype=np.float32) > 0.2) & band[:, None]
        ys, xs = np.where(txt)
        if len(xs) == 0:
            return zoom, f
        gh, gw = txt.shape
        boxes = [(xs.min() / gw, ys.min() / gh, (xs.max() + 1) / gw, (ys.max() + 1) / gh)]
        fbs = [fb for i in idx for fb in vd.faces[int(i)]]
        if fbs:
            fa = np.array(fbs)
            boxes.append((float(fa[:, 0].min()), float(fa[:, 1].min()), float((fa[:, 0] + fa[:, 2]).max()),
                          float((fa[:, 1] + fa[:, 3]).max())))
        z = zoom
        while z >= 1.02 - 1e-9:
            k = 1.0 - 1.0 / z           # visible window = [c*k, c*k + 1/z] for focus coordinate c
            out = []
            for ax in (0, 1):
                lo, hi = 0.0, 1.0
                for b in boxes:
                    b0, b1 = b[ax], b[ax + 2]
                    lo = max(lo, (b1 - 1.0 / z) / k)
                    hi = min(hi, b0 / k)
                if lo > hi + 1e-9:
                    break
                out.append(min(max(f[ax], lo), hi))
            if len(out) == 2:
                return round(float(z), 4), (float(out[0]), float(out[1]))
            z -= 0.01
        return None

    # ------------------------------------------------------------------ layout
    def _spec_for_render(self, e: dict) -> dict:
        keep = {k: v for k, v in e.items() if not k.startswith("_")}
        return keep

    def _layout(self, els: list[dict], camera: list[dict]) -> list[dict]:
        placed: list[dict] = []
        prev_by_side: dict = {}
        for e in els:
            t = e["type"]
            cls = get_element_class(t)
            pl = e["placement"]
            c = e["content"]
            size = pl.get("size", "m")
            if size in SIZE_MUL and "size_mul" not in c:
                c["size_mul"] = SIZE_MUL[size]
            if t == "captions":
                placed.append(e)
                continue
            f = build_fields(self.vd, e["t_in"], e["t_end"], self.safe, camera, chin_margin=self.chin)
            avoid = [q["box"] for q in placed if q.get("box") and q["t_in"] < e["t_end"] and e["t_in"] < q["t_end"]]
            region = pl.get("region", "auto")
            # side consistency only carries over between graphics that are close in time and only from
            # auto-placed ones: a user's "move X left" must not drag unrelated later graphics with it
            last = prev_by_side.get("last_el")
            prev_side = prev_by_side.get("last") if (last is not None and e["t_in"] - last["t_end"] < 4.0 and
                                                       last["placement"].get("region", "auto") == "auto") else None
            backing = e.get("backing", "auto")
            if backing == "auto":
                backing_guess = "glass" if t in CARD_TYPES or t in ("lower_third",) else "none"
            else:
                backing_guess = backing
            # explicit box overrides the solver
            if pl.get("box"):
                bx = pl["box"]
                spec = {**self._spec_for_render(e), "backing": backing_guess}
                el = cls({**spec, "layout_w": bx[2] * self.W}, self.ctx)
                w, h = el.size()
                e.update(box=[bx[0] * self.W, bx[1] * self.H, w, h], layout_w=bx[2] * self.W, scale=1.0, backing=backing_guess)
                st = box_stats(f, e["box"], self.W, self.H)
                if st.get("head_max", 0) > 0.35:
                    self.issue("warn", "BOX_ON_FACE", "explicit box overlaps the face", e["id"], stats=st)
                e["_layout"] = {"explicit": True, **st}
                placed.append(e)
                continue
            px, py, side = preference_point(region, t, f, self.orientation, prev_side)
            if pl.get("x") is not None:
                px = float(pl["x"])
            if pl.get("y") is not None:
                py = float(pl["y"])
            if t == "callout" and c.get("target") is not None:
                tg = c["target"]
                if tg in ("face", "hand"):
                    tg = self._target_point(tg, e)
                    c["target"] = tg
                if isinstance(tg, (list, tuple)):
                    px = tg[0] + (0.16 if tg[0] < 0.5 else -0.16)
                    py = tg[1] - 0.08
            res = None
            for attempt in range(2):
                spec = {**self._spec_for_render(e), "backing": backing_guess}
                spec["align"] = pl.get("align") or ("center" if (self.orientation == "portrait" and side == "center") else "left")
                variants = cls.variants(spec, self.ctx)
                variants = _size_variants(e["type"], variants, c)
                if t == "custom":
                    variants = [{"layout_w": self.W, "w": cls(spec, self.ctx).size()[0], "h": cls(spec, self.ctx).size()[1]}]
                cont = prev_by_side.get(side)
                cont_box = cont["box"] if cont and e["t_in"] - cont["t_end"] < 4.0 else None
                res = solve_box(f, variants, self.W, self.H, (px, py), avoid, region, (self.vd.gw, self.vd.gh),
                                continuity=cont_box)
                if res is None:
                    break
                st = box_stats(f, res["box"], self.W, self.H)
                if backing == "auto" and backing_guess == "none":
                    busy = st.get("lum_p90", 0) > 0.62 or st.get("lum_mean", 0) > 0.5 or st.get("clutter", 0) > 1.15
                    if busy:
                        backing_guess = "glass"
                        self.issue("info", "AUTO_BACKING", f"glass backing added (background lum {st.get('lum_mean')}, "
                                                           f"clutter {st.get('clutter')})", e["id"])
                        continue
                break
            if res is None:
                # relax: no preference, smaller
                spec = {**self._spec_for_render(e), "backing": backing_guess}
                variants = cls.variants(spec, self.ctx)
                variants = _size_variants(e["type"], variants, c)
                res = solve_box(f, variants, self.W, self.H, (0.5, 0.5), avoid, "any", (self.vd.gw, self.vd.gh),
                                scale_steps=(0.7, 0.6))
                if res is not None:
                    self.issue("warn", "SHRUNK", "no room at normal size; placed smaller", e["id"])
            if res is None:
                clip = self._clip_to_shots(e, cls, backing_guess, avoid, (px, py), camera)
                if clip is not None:
                    res, f, spec, variants = clip
                    backing_guess = spec.get("backing", backing_guess)
            if res is None:
                self.issue("error", "NO_SPACE", "no face-safe space for this element during its lifetime "
                                                "(shorten text, use a smaller size, or choose another moment)", e["id"])
                continue
            st = box_stats(f, res["box"], self.W, self.H)
            bx, by, bw, bh = res["box"]
            align = spec.get("align", "left")
            if self.orientation != "portrait" and pl.get("align") is None and t in ("keyword", "stat", "title") and \
                    f.face_box and (bx + bw / 2) / self.W < f.face_box[0]:
                align = "left"
            vpatch = variants[res["variant"]].get("content_patch") if res.get("variant") is not None and res["variant"] < len(variants) else None
            if vpatch:
                c.update(vpatch)
            k_size = float(c.get("size_mul", 1.0)) if t not in INTERNAL_SIZE else 1.0
            e.update(box=[round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)], layout_w=res["layout_w"],
                     scale=round(res["scale"] * k_size, 4),
                     backing=backing_guess, align=align, side=side)
            e["_layout"] = {"cost": res["cost"], "variant": res["variant"], "pref": [round(px, 3), round(py, 3)], **st}
            if t == "callout" and isinstance(c.get("target"), (list, tuple)):
                e["target_px"] = [c["target"][0] * self.W, c["target"][1] * self.H]
            prev_by_side[side] = e
            prev_by_side["last"] = side
            prev_by_side["last_el"] = e
            placed.append(e)
        return placed

    def _clip_to_shots(self, e: dict, cls, backing: str, avoid: list, pref: tuple, camera: list):
        """When no face-safe box exists for the whole lifetime (typically because it spans a jump cut into a
        tighter close-up), keep the graphic only for the contiguous shots that do have room. The entrance may
        move to the next cut if that costs <= 0.8 s; the exit becomes a hard cut on the shot boundary."""
        bounds = [0.0] + [c["t"] for c in self.cuts] + [self.duration]
        segs = [(a, b) for a, b in zip(bounds, bounds[1:]) if b > e["t_in"] + 0.05 and a < e["t_end"] - 0.05]
        if len(segs) < 2:
            return None
        m = self.style["motion"]
        lead_in = m["land_frac"] * e.get("enter_dur", m["enter"])
        c = e["content"]
        items = c.get("items") or c.get("steps") or c.get("data") or []
        last_item = max([it["t_land"] for it in items if isinstance(it, dict) and it.get("t_land") is not None] or [0.0])
        need = max(1.0, 0.55 * e.get("min_vis", 1.0))
        cands = []
        for i in range(len(segs)):
            for j in range(i, len(segs)):
                a, b = max(e["t_in"], segs[i][0]), min(e["t_end"], segs[j][1])
                if a == e["t_in"] and b == e["t_end"]:
                    continue
                land = max(e["t_land"], a + lead_in)
                if land - e["t_land"] > 0.8 or b - land < need or (last_item and b < last_item + 0.8):
                    continue
                cands.append((2.0 * (land - e["t_land"]) - (b - land), a, b, land))
        for _, a, b, land in sorted(cands):
            # stay clear of the analysis samples that sit exactly on the shot boundaries
            f = build_fields(self.vd, a + 0.03, b - 0.08, self.safe, camera, chin_margin=self.chin)
            spec = {**self._spec_for_render(e), "backing": backing, "t_in": a, "t_land": land}
            spec["align"] = "center" if self.orientation == "portrait" else "left"
            variants = cls.variants(spec, self.ctx)
            variants = _size_variants(e["type"], variants, c)
            if e["type"] == "custom":
                el = cls(spec, self.ctx)
                variants = [{"layout_w": self.W, "w": el.size()[0], "h": el.size()[1]}]
            res = solve_box(f, variants, self.W, self.H, pref, avoid, "any", (self.vd.gw, self.vd.gh),
                            scale_steps=(1.0, 0.86, 0.72, 0.6))
            if res is None:
                continue
            if e.get("backing", "auto") == "auto" and backing == "none":
                st = box_stats(f, res["box"], self.W, self.H)
                if st.get("lum_p90", 0) > 0.62 or st.get("lum_mean", 0) > 0.5 or st.get("clutter", 0) > 1.15:
                    gspec = {**spec, "backing": "glass"}
                    gvar = cls.variants(gspec, self.ctx) if e["type"] != "custom" else variants
                    gres = solve_box(f, gvar, self.W, self.H, pref, avoid, "any", (self.vd.gw, self.vd.gh),
                                     scale_steps=(1.0, 0.86, 0.72, 0.6))
                    if gres is not None:
                        res, spec, variants = gres, gspec, gvar
                        self.issue("info", "AUTO_BACKING", f"glass backing added (background lum {st.get('lum_mean')}, "
                                                           f"clutter {st.get('clutter')})", e["id"])
            old = (e["t_in"], e["t_end"])
            shift = land - e["t_land"]
            e["t_in"], e["t_land"] = round(a if a > e["t_in"] else e["t_in"], 3), round(land, 3)
            if a > old[0]:
                e["t_in"] = round(a, 3)
            if b < old[1] - 1e-3:
                e["t_end"] = round(b, 3)
                e["t_out"] = round(min(e["t_out"], b), 3)
                e["exit_mode"] = "cut"
            if e["type"] == "stat" and shift > 0:
                c["count_t0"] = round(c.get("count_t0", land) + shift, 3)
                c["count_t1"] = round(c.get("count_t1", land + 1.0) + shift, 3)
            self.issue("warn", "CLIPPED_TO_SHOT", f"no face-safe room for the whole span {old[0]:.2f}-{old[1]:.2f}s "
                                                  f"(framing changes at a cut); shown {e['t_in']:.2f}-{e['t_end']:.2f}s"
                       + (f", entrance delayed {shift * 1000:.0f} ms to the cut" if shift > 0.01 else ""), e["id"])
            return res, f, spec, variants
        return None

    def _target_point(self, what: str, e: dict) -> list:
        idx = self.vd.samples_between(e.get("t_in", 0), e.get("t_end", 1))
        if what == "hand":
            pts = [h for i in idx for h in self.vd.hands[int(i)]]
            if pts:
                a = np.array(pts)
                return [float(np.median(a[:, 0])), float(np.median(a[:, 1]))]
        fbs = [fb for i in idx for fb in self.vd.faces[int(i)]]
        if fbs:
            a = np.array(fbs)
            return [float(np.median(a[:, 0] + a[:, 2] / 2)), float(np.median(a[:, 1] + a[:, 3] * 1.15))]
        return [0.5, 0.5]

    # ------------------------------------------------------------------ text-to-speech sync
    def _unit_sync(self, e: dict) -> None:
        t = e["type"]
        A = e.get("_A")
        if not A or t not in ("keyword", "quote", "title"):
            return
        reveal = (e.get("motion") or {}).get("reveal", "word" if t == "keyword" else "whole")
        if reveal != "spoken":
            return
        cls = get_element_class(t)
        el = cls({**self._spec_for_render(e)}, self.ctx)
        lay = getattr(el, "lay", None)
        if lay is None:
            return
        units = [u.text for u in lay.units]
        times = self.anchors.unit_times(units, A)
        e["content"]["unit_times"] = times
        got = [x for x in times if x is not None]
        if got and min(got) < e["t_land"] - 0.02:
            # the display text starts before the anchored phrase: enter with its first spoken word (<= 0.8 s)
            shift = min(0.8, e["t_land"] - min(got))
            e["t_in"], e["t_land"] = round(e["t_in"] - shift, 3), round(e["t_land"] - shift, 3)
            if e.get("_anchor"):
                e["_anchor"] = {**e["_anchor"], "t": round(min(got), 3), "spoken_start": True}
            self.issue("info", "SPOKEN_START", f"starts {shift * 1000:.0f} ms earlier, with the first displayed word", e["id"])
        miss = sum(1 for x in times if x is None)
        if miss and miss == len(times):
            e["motion"]["reveal"] = "word"
            self.issue("info", "SPOKEN_FALLBACK", "display text not found in speech; using staggered reveal", e["id"])

    # ------------------------------------------------------------------ captions
    def _captions(self) -> Optional[dict]:
        cap = self.plan.get("captions") or {}
        if not cap.get("enabled"):
            return None
        if float(self.vd.overlay_text[int(self.vd.gh * 0.75):].max()) > 0.5 and not cap.get("force"):
            self.issue("warn", "CAPTIONS_SKIPPED", "the video already has burned-in subtitles; captions not added "
                                                   "(set captions.force=true to add anyway)")
            return None
        corr = self.plan.get("corrections") or {}
        max_u = int(cap.get("max_units", 14 if self.tr.get("cjk") else 7))
        chunks = []
        for s in self.tr["sentences"]:
            ws = self.tr["words"][s["w0"]: s["w1"] + 1]
            cur: list = []

            def push():
                if not cur:
                    return
                txt = ("".join(w["text"] for w in cur)).strip()
                for a, b in corr.items():
                    txt = txt.replace(a, b)
                txt = re.sub(r"[，。,.!！?？;；:：]+$", "", txt)
                chunks.append({"text": txt, "t0": cur[0]["start"], "t1": cur[-1]["end"], "words": [w["start"] for w in cur]})
                cur.clear()

            n = 0
            for w in ws:
                cur.append(w)
                n += sum(1 for ch in w["text"] if is_cjk(ch)) if self.tr.get("cjk") else 1
                if n >= max_u or w["text"].strip()[-1:] in "，,、；;":
                    push()
                    n = 0
            push()
        for a, b in zip(chunks, chunks[1:]):
            if b["t0"] - a["t1"] < 0.3:
                a["t1"] = b["t0"] - 0.01
        e = {"id": "captions", "type": "captions", "layer": "overlay", "t_in": 0.0, "t_land": 0.0,
             "t_out": self.duration, "t_end": self.duration, "content": {"chunks": chunks, "highlight": cap.get("highlight", "spoken")},
             "backing": cap.get("backing", "none"), "placement": {}, "motion": {}}
        cls = get_element_class("captions")
        w = (self.safe[2] - self.safe[0]) * self.W * 0.92
        el = cls({**e, "layout_w": w}, self.ctx)
        # per-unit spoken times: map layout units onto chunk word times by character position
        for ch, lay in zip(e["content"]["chunks"], el.lays):
            times = []
            wi = 0
            acc = 0
            wlens = []
            for t0 in ch["words"]:
                wlens.append(t0)
            ch["unit_times"] = self._units_to_times(lay, ch)
        bw, bh = el.size()
        y_bottom = self.safe[3] * self.H - 1.5 * self.ctx.u
        if self.orientation == "portrait":
            y_bottom = min(y_bottom, 0.80 * self.H)
        e.update(box=[(self.W - bw) / 2, y_bottom - bh, bw, bh], layout_w=w, scale=1.0)
        return e

    def _units_to_times(self, lay, ch) -> list:
        # distribute chunk word onsets across units proportionally by normalized length
        n_units = len(lay.units)
        if not ch["words"]:
            return [None] * n_units
        total = sum(len(norm(u.text)) or 0 for u in lay.units) or 1
        out, acc = [], 0
        for u in lay.units:
            frac = acc / total
            k = min(len(ch["words"]) - 1, int(frac * len(ch["words"])))
            out.append(ch["words"][k])
            acc += len(norm(u.text))
        return out

    # ------------------------------------------------------------------ main
    def run(self) -> tuple[dict, dict]:
        els_raw = self.plan.get("elements", [])
        if not isinstance(els_raw, list):
            raise SMError("plan.elements must be a list")
        ids = [e.get("id") for e in els_raw if e.get("id")]
        if len(ids) != len(set(ids)):
            raise SMError("duplicate element ids in plan")
        cams, els = [], []
        for i, e in enumerate(els_raw):
            if e.get("enabled", True) is False:
                continue
            ne = self._normalize(e, i)
            (cams if ne.get("type") == "camera" else els).append(ne)
        camera = self._camera(cams)
        timed = []
        for e in els:
            te = self._time_element(e)
            if te is not None:
                timed.append(te)
        timed = self._resolve_conflicts(timed)
        for e in timed:
            self._unit_sync(e)
        placed = self._layout(timed, camera)
        cap = self._captions()
        if cap:
            placed.append(cap)
        out_els = []
        for e in placed:
            d = self._spec_for_render(e)
            d["anchor"] = e.get("_anchor")
            d["layout"] = e.get("_layout")
            d["layer"] = e.get("layer") or ("primary" if e["type"] in PRIMARY else "secondary" if e["type"] in SECONDARY else "overlay")
            out_els.append(d)
        tl = {"version": 1, "W": self.W, "H": self.H, "fps": self.info.fps, "fps_str": self.info.fps_str,
              "duration": self.duration, "orientation": self.orientation, "platform": self.platform,
              "safe": list(self.safe), "style": self.style, "font": self.fb.describe(), "camera": camera,
              "elements": out_els, "cuts": self.cuts}
        tl["hash"] = sha1_obj({k: v for k, v in tl.items() if k != "hash"})
        n_err = sum(1 for i in self.issues if i["level"] == "error")
        report = {"ok": n_err == 0, "errors": n_err, "warnings": sum(1 for i in self.issues if i["level"] == "warn"),
                  "issues": self.issues, "n_elements": len(out_els), "n_planned": len(els_raw), "camera_moves": len(camera)}
        return tl, report


def compile_project(proj, plan: Optional[dict] = None, write: bool = True) -> tuple[dict, dict]:
    from .util import write_json

    comp = Compiler(proj, plan)
    tl, rep = comp.run()
    if write:
        write_json(proj.p("build", "timeline.json"), tl)
        write_json(proj.p("build", "compile_report.json"), rep)
    return tl, rep
