"""brief.md: a compact, structured description of the video written for the planner (Claude).

It contains everything needed to decide *what* to show and *when/where* — numbered sentences
with exact times, speech rate, pauses and vocal emphasis; numbers, enumerations, contrasts and
low-confidence ASR words; shots with framing, open side and free regions; burned-in text.
"""
from __future__ import annotations

import re

import numpy as np

from .text import cjk_ratio, find_numbers
from .util import fmt_time

LIST_MARKERS_ZH = ["第一", "第二", "第三", "首先", "其次", "然后", "最后", "一是", "二是", "三是", "另外", "还有", "以及"]
LIST_MARKERS_EN = ["first", "second", "third", "finally", "next", "then", "also", "another", "lastly"]
CONTRAST_ZH = ["但是", "可是", "然而", "而是", "不是", "相比", "比", "反而", "却", "vs", "对比", "以前", "现在"]
CONTRAST_EN = ["but", "however", "instead", "versus", "vs", "compared", "whereas", "rather than", "used to", "now"]
CAUSE_ZH = ["因为", "所以", "导致", "因此", "于是", "结果", "如果", "就会"]
CAUSE_EN = ["because", "so ", "therefore", "leads to", "if ", "then", "which means"]


def _framing(face_box) -> str:
    if not face_box:
        return "no face"
    h = face_box[3]
    return "extreme close-up" if h > 0.5 else "close-up" if h > 0.32 else "medium" if h > 0.18 else "wide"


def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def build_brief(proj) -> str:
    info = proj.media()
    tr = proj.transcript()
    shots = proj.shots()
    af = proj.audio()
    vd = proj.vision()
    st = proj.settings
    words = tr["words"]
    cjk = tr.get("cjk")
    dur = info.duration
    lines = []
    lines.append(f"# Video brief — {proj.source.name}")
    lines.append("")
    lines.append(f"- duration **{dur:.2f}s**, {info.width}x{info.height} ({info.orientation}), {info.fps:.3f} fps"
                 f"{', HDR→SDR' if info.is_hdr else ''}{', VFR normalized to CFR' if info.vfr_suspected else ''}")
    lang = tr.get("language")
    total_units = sum(sum(1 for c in s["text"] if '\u4e00' <= c <= '\u9fff') if cjk else len(s["text"].split()) for s in tr["sentences"])
    speech_time = float(af.vad.sum() * af.hop)
    lines.append(f"- language **{lang}**, ASR `{tr.get('asr_model')}`, {len(tr['sentences'])} sentences, "
                 f"speech {speech_time:.1f}s, avg rate {total_units / max(speech_time, 1e-6):.1f} {'chars' if cjk else 'words'}/s")
    lines.append(f"- platform setting: `{st.get('platform', 'auto')}` · style: `{st.get('style', 'balanced')}`")
    # density guidance
    dens_lo, dens_hi = max(1, round(dur / 60 * 3)), max(2, round(dur / 60 * 7))
    lines.append(f"- suggested density for *balanced*: **{dens_lo}–{dens_hi} elements**, ≥40% of the time graphics-free, "
                 "max one primary graphic on screen")
    burned = vd.overlay_text
    if burned.max() > 0.4:
        ys, xs = np.where(burned > 0.4)
        lines.append(f"- ⚠ burned-in text detected (subtitles/logos) at x {_pct(xs.min() / vd.gw)}–{_pct((xs.max() + 1) / vd.gw)}, "
                     f"y {_pct(ys.min() / vd.gh)}–{_pct((ys.max() + 1) / vd.gh)} — keep graphics away; do not add captions")
    lines.append("")
    # ---------------------------------------------------------------- shots
    lines.append("## Shots (framing → where graphics can go)")
    lines.append("")
    lines.append("| shot | time | framing | face box (x,y,w,h) | open side | free regions (x0,y0,x1,y1) | notes |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in shots:
        fr = ", ".join("(" + ",".join(f"{v:.2f}" for v in r) + ")" for r in (s.get("free_regions") or [])[:2])
        fb = s.get("face_box")
        fbs = ",".join(f"{v:.2f}" for v in fb) if fb else "—"
        notes = []
        if not s.get("speaker"):
            notes.append("no speaker (B-roll?)")
        if s.get("hands_active"):
            notes.append("hands visible")
        if s.get("camera") == "moving":
            notes.append("camera moving")
        if s.get("lum_mean", 0) > 0.6:
            notes.append("bright")
        lines.append(f"| {s['id']} | {s['start']:.2f}–{s['end']:.2f} | {_framing(fb)} | {fbs} | {s.get('open_side')} | {fr or '—'} | "
                     f"{'; '.join(notes)} |")
    lines.append("")
    if len(shots) > 1:
        lines.append(f"Cuts at: {', '.join(f'{s['start']:.2f}' for s in shots[1:])} s. Graphics may continue across jump cuts "
                     "(the layout avoids the face in every shot they span); camera moves end at the next cut.")
        lines.append("")
    # ---------------------------------------------------------------- transcript
    lines.append("## Transcript (anchor with `\"<sentence>:<exact phrase>\"`)")
    lines.append("")
    lowconf = []
    for s in tr["sentences"]:
        ws = words[s["w0"]: s["w1"] + 1]
        emph = sorted([w for w in ws if w.get("emph", 0) > 1.2], key=lambda w: -w.get("emph", 0))[:3]
        tags = []
        nums = find_numbers(s["text"])
        if nums:
            tags.append("numbers: " + ", ".join(f"{n['raw']}(={n['value']:g}{n['suffix']})" for n in nums[:4]))
        low = s["text"].lower()
        mk = [m for m in (LIST_MARKERS_ZH if cjk else LIST_MARKERS_EN) if m in low]
        if mk:
            tags.append("enumeration: " + "/".join(mk[:3]))
        ct = [m for m in (CONTRAST_ZH if cjk else CONTRAST_EN) if m in low]
        if ct:
            tags.append("contrast: " + "/".join(ct[:3]))
        ca = [m.strip() for m in (CAUSE_ZH if cjk else CAUSE_EN) if m in low]
        if ca:
            tags.append("cause/effect: " + "/".join(ca[:3]))
        if s["text"].strip().endswith(("?", "？")) or any(q in s["text"] for q in ("吗", "呢？", "为什么", "怎么")):
            tags.append("question")
        if emph:
            tags.append("stressed: " + ", ".join(f"“{w['text'].strip()}”@{w['start']:.2f}" for w in emph))
        for w in ws:
            if w.get("p", 1) < 0.5 and w["text"].strip():
                lowconf.append((s["id"], w))
        pause = f", pause {s['pause_after']:.2f}s" if s.get("pause_after", 0) >= 0.3 else ""
        lines.append(f"**S{s['id']}** `{s['start']:.2f}–{s['end']:.2f}` ({s['rate']:.1f} {'c' if cjk else 'w'}/s{pause})  ")
        lines.append(f"{s['text']}  ")
        if tags:
            lines.append(f"<sub>{' · '.join(tags)}</sub>")
        lines.append("")
    if lowconf:
        lines.append("## Possibly mis-recognized words (low ASR confidence)")
        lines.append("")
        lines.append("Check names/terms; put fixes in `plan.corrections` (anchors still match the spoken/ASR text). "
                     "Never show a word on screen you are not sure about.")
        lines.append("")
        for sid, w in lowconf[:30]:
            lines.append(f"- S{sid} “{w['text'].strip()}” @ {w['start']:.2f}s (p={w['p']:.2f})")
        lines.append("")
    # ---------------------------------------------------------------- pauses / rhythm
    pauses = [(a, b) for a, b in af.pauses(0.45) if a > 0.3 and b < dur - 0.3]
    if pauses:
        lines.append("## Pauses ≥ 0.45 s (natural beats for entrances/exits and camera resets)")
        lines.append("")
        lines.append(", ".join(f"{a:.2f}–{b:.2f}" for a, b in pauses[:40]))
        lines.append("")
    lines.append("## Next step")
    lines.append("")
    lines.append("Write `plan.json` (see the skill's reference/plan-schema.md and reference/design-guide.md), then run "
                 "`sm build <project>`; inspect `qa/contact_sheet.jpg` and `qa/report.md`.")
    return "\n".join(lines) + "\n"


def write_brief(proj) -> str:
    txt = build_brief(proj)
    proj.p("brief.md").write_text(txt, encoding="utf-8")
    return str(proj.p("brief.md"))
