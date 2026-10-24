"""Semantic Motion — caption grouping and export (JSON / SRT / ASS).

Input is a ``semotion.alignment/1`` document (character/word times from
:mod:`semotion.align`).  The builder groups words into caption cues, wraps them
into at most two short lines, quantises everything to video frames, and exports
JSON + SRT + ASS.

Rules implemented (see ``docs/ALIGNMENT.md`` §5)
------------------------------------------------
CJK
    * ≤ 2 lines, target ~22 characters per line, hard maximum 24;
    * no line ends in punctuation (``，。、！？；：…`` …), no punctuation-only
      line (both are checked by :func:`validate`);
    * cues break at sentence punctuation, long pauses (≥ 0.35 s) or a hard
      character cap; lines break at phrase punctuation when possible;
    * display ≥ 0.9 s per cue; gaps shorter than 0.18 s are merged (or split
      back-to-back at a phrase boundary when the merge would be too long);
    * ``respect_segments`` keeps every cue inside one ASR segment (default on
      when the alignment carries segment ids) and ``respect_words`` never lets a
      cue or line start inside an ASR word (``word.group_id``).
Latin
    * ≤ 42 characters per line, ≤ 2 lines, break at word boundaries.
All
    * times are quantised to frames: ``start_s = start_frame / fps`` exactly,
      so a renderer can show the cue on the intended frame.  SRT/ASS are derived
      from the same frames (SRT ms rounding is ≤ 0.5 ms).
    * CJK "characters" counts are Unicode code points; a Latin word counts one
      per character.  Digits are one character each.

The document is ``semotion.captions/1``.  It is deliberately renderer-agnostic:
it carries both frame numbers and seconds.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "SCHEMA",
    "DEFAULT_STYLE",
    "build_captions",
    "build_from_file",
    "export",
    "write_json",
    "write_srt",
    "write_ass",
    "validate",
    "seconds_to_frame",
    "frame_to_srt",
    "frame_to_ass",
]

SCHEMA = "semotion.captions/1"

DEFAULT_STYLE: dict[str, Any] = {
    # per-language readability budgets
    "cjk": {
        "max_lines": 2,
        "target_chars_per_line": 22,
        "max_chars_per_line": 24,
        "min_chars_per_line": 6,
        "target_cue_chars": 30,
        "max_cue_chars": 44,
    },
    "latin": {
        "max_lines": 2,
        "target_chars_per_line": 38,
        "max_chars_per_line": 42,
        "min_chars_per_line": 8,
        "target_cue_chars": 58,
        "max_cue_chars": 84,
    },
    # timing
    "min_display_s": 0.90,
    "merge_gap_s": 0.18,
    "pause_break_s": 0.35,
    # punctuation classes
    "sentence_punct": "。！？!?…",
    "clause_punct": "，,、；;：:",
    # display
    "strip_trailing_punct": True,
    # ASS defaults
    # match what the v2 renderer actually burns in (assets/fonts/)
    "font_cjk": "Noto Sans SC",
    "font_latin": "Inter",
}


# --------------------------------------------------------------------------- utils

_TRAILING_PUNCT = re.compile(r"[\s，。、！？；：,.!?;:…—·~～\"“”‘’)\]}】》」』]+$")


def _is_cjk_doc(language: Any) -> bool:
    return str(language or "").lower().startswith(("zh", "ja", "ko"))


def strip_trailing_punct(text: str) -> str:
    """Remove trailing punctuation/whitespace so a caption line never ends in it."""
    return _TRAILING_PUNCT.sub("", text or "")


def _count_chars(text: str) -> int:
    return len(text)


def seconds_to_frame(seconds: float, fps: float) -> int:
    if fps <= 0:
        raise ValueError("fps must be positive")
    return int(round(float(seconds) * float(fps)))


def frame_to_srt(frame: int, fps: float) -> str:
    ms = int(round(frame * 1000.0 / fps))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    seconds, ms = divmod(ms, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def frame_to_ass(frame: int, fps: float) -> str:
    cs = int(round(frame * 100.0 / fps))
    hours, cs = divmod(cs, 360_000)
    minutes, cs = divmod(cs, 6_000)
    seconds, cs = divmod(cs, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{cs:02d}"


# ----------------------------------------------------------------------- atomising


@dataclass
class _Atom:
    """One unbreakable caption unit (a Han character, a number, a word)."""

    text: str
    start_s: float
    end_s: float
    punct: str
    word_id: str
    segment_id: str
    kind: str
    #: ASR word group (``<segment>:<word id>``); cues/lines never split one.
    group_id: str | None = None


def _atoms(alignment: dict[str, Any]) -> list[_Atom]:
    atoms: list[_Atom] = []
    for word in alignment.get("words") or []:
        text = str(word.get("text") or "").strip()
        if not text:
            continue
        atoms.append(_Atom(
            text=text,
            start_s=float(word["start_s"]),
            end_s=float(word["end_s"]),
            punct=str(word.get("trailing_punct") or ""),
            word_id=str(word.get("id")),
            segment_id=str(word.get("segment_id")),
            kind=str(word.get("kind") or "han"),
            group_id=str(word["group_id"]) if word.get("group_id") else None,
        ))
    return atoms


def _joined(atoms: Sequence[_Atom], cjk: bool) -> str:
    if cjk:
        return "".join(a.text for a in atoms)
    return " ".join(a.text for a in atoms)


def _joined_raw(atoms: Sequence[_Atom], cjk: bool) -> str:
    """Joined text including each atom's trailing punctuation."""
    if cjk:
        return "".join(a.text + a.punct for a in atoms)
    return " ".join(a.text + a.punct for a in atoms)


@dataclass
class _Cue:
    atoms: list[_Atom]
    start_s: float = 0.0
    end_s: float = 0.0

    def __post_init__(self) -> None:
        if self.atoms:
            self.start_s = float(self.atoms[0].start_s)
            self.end_s = float(self.atoms[-1].end_s)


def _has_sentence_end(atom: _Atom, style: dict[str, Any]) -> bool:
    return any(ch in atom.punct for ch in style["sentence_punct"])


def _cue_ends_sentence(cue: "_Cue", style: dict[str, Any]) -> bool:
    return bool(cue.atoms) and _has_sentence_end(cue.atoms[-1], style)


def _has_clause(atom: _Atom, style: dict[str, Any]) -> bool:
    return any(ch in atom.punct for ch in style["clause_punct"])


# ----------------------------------------------------------------- cue segmentation


def _segments_to_cues(atoms: list[_Atom], cjk: bool, style: dict[str, Any], *,
                      respect_segments: bool = False,
                      respect_words: bool = True) -> list[_Cue]:
    """Group atoms into cues.

    With ``respect_segments`` a cue never mixes two ASR segments and the pause
    heuristic is dropped inside a segment (a segment is the natural caption
    unit); sentence-final punctuation and the hard character cap still break.
    With ``respect_words`` a cue never starts inside an ASR word.
    """
    cfg = style["cjk" if cjk else "latin"]
    target = int(cfg["target_cue_chars"])
    hard = int(cfg["max_cue_chars"])
    pause_break = float(style["pause_break_s"])
    cues: list[_Cue] = []
    current: list[_Atom] = []
    for i, atom in enumerate(atoms):
        current.append(atom)
        nxt = atoms[i + 1] if i + 1 < len(atoms) else None
        text_len = _count_chars(_joined(current, cjk))
        if nxt is None:
            cues.append(_Cue(current))
            current = []
            break
        gap = float(nxt.start_s) - float(atom.end_s)
        same_word = bool(respect_words and atom.group_id
                         and atom.group_id == nxt.group_id)
        break_here = False
        if respect_segments and atom.segment_id != nxt.segment_id:
            break_here = True
        elif _has_sentence_end(atom, style):
            break_here = True
        elif text_len >= hard:
            break_here = True
        elif not respect_segments:
            if gap >= pause_break:
                break_here = True
            elif text_len >= target and (gap >= 0.12 or _has_clause(atom, style)):
                break_here = True
        if break_here and same_word:
            # never split a Whisper word across two cues; the hard cap is
            # re-applied by _split_long_cues at a legal boundary
            break_here = False
        if break_here:
            cues.append(_Cue(current))
            current = []
    if current:  # defensive
        cues.append(_Cue(current))
    return cues


def _cue_chars(cue: _Cue, cjk: bool, style: dict[str, Any]) -> int:
    return _count_chars(strip_trailing_punct(_joined_raw(cue.atoms, cjk)))


def _merge(a: _Cue, b: _Cue) -> _Cue:
    return _Cue(list(a.atoms) + list(b.atoms))


def _merge_short_gaps(cues: list[_Cue], cjk: bool, style: dict[str, Any], *,
                      respect_segments: bool = False) -> list[_Cue]:
    """Merge cues separated by less than ``merge_gap_s`` when the result still fits.

    A cue is never merged across an ASR segment when ``respect_segments`` is on;
    the earlier cue is simply held until the next one starts.
    """
    cfg = style["cjk" if cjk else "latin"]
    max_chars = int(cfg["max_cue_chars"])
    gap_limit = float(style["merge_gap_s"])
    out: list[_Cue] = []
    for cue in cues:
        if out:
            gap = cue.start_s - out[-1].end_s
            merged = _merge(out[-1], cue)
            same_segment = (not respect_segments) or (
                out[-1].atoms[-1].segment_id == cue.atoms[0].segment_id)
            if gap < gap_limit:
                if same_segment and not _cue_ends_sentence(out[-1], style) \
                        and _cue_chars(merged, cjk, style) <= max_chars:
                    out[-1] = merged
                    continue
                # keep the boundary but avoid a blank flash
                out[-1].end_s = max(out[-1].end_s, cue.start_s)
        out.append(cue)
    return out


def _best_split_index(atoms: list[_Atom], cjk: bool, style: dict[str, Any],
                      *, target: int | None = None,
                      respect_words: bool = True) -> int:
    """Choose the atom index where a cue should be split for readability.

    Prefers balanced lines inside the character budget, then phrase punctuation,
    then closeness to target.  With ``respect_words`` a split inside an ASR word
    is never chosen unless no legal split exists.
    """
    cfg = style["cjk" if cjk else "latin"]
    max_chars = int(cfg["max_chars_per_line"])
    min_chars = int(cfg["min_chars_per_line"])
    target = int(target or cfg["target_cue_chars"])
    n = len(atoms)
    if n < 2:
        return 0

    def evaluate(j: int, guard: bool) -> float | None:
        if guard and respect_words and atoms[j - 1].group_id \
                and atoms[j - 1].group_id == atoms[j].group_id:
            return None
        left = _count_chars(strip_trailing_punct(_joined(atoms[:j], cjk)))
        right = _count_chars(strip_trailing_punct(_joined(atoms[j:], cjk)))
        if left > max_chars or right > max_chars:
            return None
        score = abs(left - right) + abs((left + right) - target) * 0.05
        if max(left, right) > int(cfg["target_chars_per_line"]) + 2:
            score += 2.0
        if min(left, right) < min_chars:
            score += 3.0
        if _has_clause(atoms[j - 1], style):
            score -= 1.5
        elif atoms[j - 1].punct:
            score -= 1.0
        return score

    for guard in (True, False):
        best_idx = 0
        best_score = math.inf
        for j in range(1, n):
            score = evaluate(j, guard)
            if score is not None and score < best_score:
                best_score = score
                best_idx = j
        if best_idx:
            return best_idx
    return 0


def _split_long_cues(cues: list[_Cue], cjk: bool, style: dict[str, Any], *,
                     respect_words: bool = True) -> list[_Cue]:
    cfg = style["cjk" if cjk else "latin"]
    max_chars = int(cfg["max_cue_chars"])
    out: list[_Cue] = []
    stack = list(reversed(cues))
    guard = 0
    while stack:
        guard += 1
        if guard > 10_000:  # pragma: no cover - pathological
            out.extend(reversed(stack))
            break
        cue = stack.pop()
        if _cue_chars(cue, cjk, style) <= max_chars or len(cue.atoms) < 2:
            out.append(cue)
            continue
        j = _best_split_index(cue.atoms, cjk, style, target=max_chars,
                              respect_words=respect_words)
        if j <= 0:
            j = max(1, len(cue.atoms) // 2)
        left = _Cue(cue.atoms[:j])
        right = _Cue(cue.atoms[j:])
        stack.append(right)
        stack.append(left)
    out.sort(key=lambda c: c.start_s)
    return out


def _absorb_small_gaps(cues: list[_Cue], cjk: bool, style: dict[str, Any], *,
                       respect_segments: bool = False) -> list[_Cue]:
    """Never leave a sub-``merge_gap_s`` blank gap between two cues.

    Short gaps appear after min-display extension even when the initial grouping
    did not produce them.  When the merged text fits, merge; otherwise simply
    hold the earlier cue until the next one starts (a longer display is harmless,
    a flicker is not).
    """
    cfg = style["cjk" if cjk else "latin"]
    max_chars = int(cfg["max_cue_chars"])
    gap_limit = float(style["merge_gap_s"])
    out: list[_Cue] = []
    for cue in cues:
        if out:
            gap = cue.start_s - out[-1].end_s
            if 1e-9 < gap < gap_limit:
                merged = _merge(out[-1], cue)
                same_segment = (not respect_segments) or (
                    out[-1].atoms[-1].segment_id == cue.atoms[0].segment_id)
                if same_segment and not _cue_ends_sentence(out[-1], style) and \
                        _cue_chars(merged, cjk, style) <= max_chars:
                    out[-1] = merged
                    continue
                out[-1].end_s = cue.start_s
        out.append(cue)
    return out


def _enforce_min_display(cues: list[_Cue], cjk: bool, style: dict[str, Any],
                         fps: float, duration_s: float | None, *,
                         min_display_floor: float | None = None,
                         respect_segments: bool = False) -> list[_Cue]:
    """Guarantee the minimum display time by extending or merging neighbours."""
    min_display = float(min_display_floor if min_display_floor is not None
                        else style["min_display_s"])
    cfg = style["cjk" if cjk else "latin"]
    max_chars = int(cfg["max_cue_chars"])
    cues = [c for c in cues if c.atoms]
    guard = 0
    i = 0
    while i < len(cues):
        guard += 1
        if guard > 10_000:  # pragma: no cover - pathological
            break
        cue = cues[i]
        need = min_display - (cue.end_s - cue.start_s)
        if need <= 1e-9:
            i += 1
            continue
        next_start = cues[i + 1].start_s if i + 1 < len(cues) else (
            duration_s if duration_s is not None else cue.start_s + need)
        # 1) extend the end into the following gap
        if next_start - cue.end_s > 1e-6:
            cue.end_s = min(next_start, cue.end_s + need)
            need = min_display - (cue.end_s - cue.start_s)
        if need <= 1e-9:
            i += 1
            continue
        # 2) pull the start back into the preceding gap — but never across a
        #    pause that was long enough to cause the cue break in the first
        #    place (that would reveal the caption during silence)
        prev_end = cues[i - 1].end_s if i > 0 else 0.0
        gap_before = cue.start_s - prev_end
        if 1e-6 < gap_before <= float(style["pause_break_s"]) + 1e-9:
            cue.start_s = max(prev_end, cue.start_s - need)
            need = min_display - (cue.end_s - cue.start_s)
        if need <= 1e-9:
            i += 1
            continue
        # 3) merge with a neighbour when the text still fits
        if i + 1 < len(cues):
            cue_id = cue.atoms[-1].segment_id
            next_id = cues[i + 1].atoms[0].segment_id
            same_segment = (not respect_segments) or cue_id == next_id
            if same_segment:
                merged = _merge(cue, cues[i + 1])
                if _cue_chars(merged, cjk, style) <= max_chars:
                    cues[i] = merged
                    cues.pop(i + 1)
                    continue
            # shared boundary: hold the current cue up to the next cue's start
            cue.end_s = min(cues[i + 1].start_s, cue.start_s + min_display)
            need = min_display - (cue.end_s - cue.start_s)
            if need <= 1e-9:
                i += 1
                continue
        if i > 0:
            cue_id = cue.atoms[0].segment_id
            prev_id = cues[i - 1].atoms[-1].segment_id
            same_segment = (not respect_segments) or cue_id == prev_id
            if same_segment:
                merged = _merge(cues[i - 1], cue)
                if _cue_chars(merged, cjk, style) <= max_chars:
                    cues[i - 1] = merged
                    cues.pop(i)
                    i = max(0, i - 1)
                    continue
        # 4) last resort: keep the cue short rather than overlapping; validate()
        #    reports it and the frame-level pass in build_captions enforces the
        #    minimum where the timeline allows it.
        i += 1
    return cues


# ------------------------------------------------------------------------ line layout


def _strip_line(text: str) -> str:
    return strip_trailing_punct(text)


def _layout_lines(cue: _Cue, cjk: bool, style: dict[str, Any], *,
                  respect_words: bool = True) -> list[str]:
    cfg = style["cjk" if cjk else "latin"]
    max_chars = int(cfg["max_chars_per_line"])
    min_chars = int(cfg["min_chars_per_line"])
    atoms = cue.atoms
    raw = _joined_raw(atoms, cjk)
    total = _count_chars(strip_trailing_punct(raw))
    if total <= max_chars or len(atoms) < 2:
        line = _strip_line(raw)
        return [line] if line else []

    def choose(guard: bool) -> tuple[int, float]:
        best_j, best_score = 0, math.inf
        for j in range(1, len(atoms)):
            if guard and respect_words and atoms[j - 1].group_id \
                    and atoms[j - 1].group_id == atoms[j].group_id:
                continue
            left = _strip_line(_joined_raw(atoms[:j], cjk))
            right = _strip_line(_joined_raw(atoms[j:], cjk))
            if not left or not right:
                continue
            llen, rlen = _count_chars(left), _count_chars(right)
            if llen > max_chars or rlen > max_chars:
                continue
            score = abs(llen - rlen)
            if min(llen, rlen) < min_chars and total >= 2 * min_chars:
                score += 4.0
            if _has_clause(atoms[j - 1], style):
                score -= 1.5
            elif atoms[j - 1].punct:
                score -= 1.0
            if score < best_score:
                best_score = score
                best_j = j
        return best_j, best_score

    best_j, _ = choose(True)
    if best_j <= 0 and respect_words:
        best_j, _ = choose(False)
    if best_j <= 0:
        # No split respects the budget: fall back to a character-level split of
        # the joined text (only reachable for a single over-long atom).
        text = _strip_line(raw)
        cut = min(max_chars, max(1, len(text) // 2))
        return [text[:cut], text[cut:]] if len(text) > max_chars else [text]
    left = _strip_line(_joined_raw(atoms[:best_j], cjk))
    right = _strip_line(_joined_raw(atoms[best_j:], cjk))
    return [left, right]


# -------------------------------------------------------------------------- building


def _cue_boundary_stats(cues: list[_Cue]) -> tuple[int, int]:
    """(cues starting inside an ASR word, cue boundaries at a segment change)."""
    mid_word = 0
    segment_breaks = 0
    for a, b in zip(cues, cues[1:]):
        if not a.atoms or not b.atoms:
            continue
        left, right = a.atoms[-1], b.atoms[0]
        if left.group_id and left.group_id == right.group_id:
            mid_word += 1
        if left.segment_id != right.segment_id:
            segment_breaks += 1
    return mid_word, segment_breaks


def build_captions(alignment: dict[str, Any], *, fps: float | None = None,
                   style: dict[str, Any] | None = None,
                   respect_segments: bool | None = None,
                   respect_words: bool = True) -> dict[str, Any]:
    """Group an alignment document into frame-accurate caption cues.

    ``respect_segments`` (default auto: on when the alignment carries ASR segment
    ids) keeps every cue inside one ASR segment: cues are never merged across a
    segment and pause heuristics inside a segment are ignored.  ``respect_words``
    keeps cues and lines from starting inside an ASR word (``word.group_id``).
    """
    if alignment.get("schema") != "semotion.alignment/1":
        raise ValueError(f"expected semotion.alignment/1, got {alignment.get('schema')!r}")
    src = alignment.get("source") or {}
    language = src.get("language") or "zh"
    cjk = _is_cjk_doc(language)
    fps = float(fps if fps is not None else (src.get("fps") or 25.0))
    cfg = {**DEFAULT_STYLE}
    if style:
        for key, value in style.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key] = {**cfg[key], **value}
            else:
                cfg[key] = value
    atom_list = _atoms(alignment)
    duration_s = float(src.get("duration_s") or 0.0) or None

    if respect_segments is None:
        seg_ids = {str(w.get("segment_id") or "")
                   for w in alignment.get("words") or []}
        respect_segments = bool(seg_ids - {""})

    # the frame grid is coarser than the 0.9 s rule: enforce the rule *in
    # frames* so quantisation can never shave a cue below it
    min_frames = max(1, int(math.ceil(float(cfg["min_display_s"]) * fps - 1e-9)))
    min_display_floor = max(float(cfg["min_display_s"]), min_frames / fps)

    cues = _segments_to_cues(atom_list, cjk, cfg,
                             respect_segments=bool(respect_segments),
                             respect_words=respect_words)
    cues = _merge_short_gaps(cues, cjk, cfg, respect_segments=bool(respect_segments))
    cues = _split_long_cues(cues, cjk, cfg, respect_words=respect_words)
    cues = _enforce_min_display(cues, cjk, cfg, fps, duration_s,
                                min_display_floor=min_display_floor,
                                respect_segments=bool(respect_segments))
    # splitting/merging can re-create short leftovers; two passes are enough
    cues = _split_long_cues(cues, cjk, cfg, respect_words=respect_words)
    cues = _enforce_min_display(cues, cjk, cfg, fps, duration_s,
                                min_display_floor=min_display_floor,
                                respect_segments=bool(respect_segments))
    cues = _absorb_small_gaps(cues, cjk, cfg, respect_segments=bool(respect_segments))
    mid_word_starts, segment_breaks = _cue_boundary_stats(cues)
    duration_frames = int(math.ceil(duration_s * fps - 1e-9)) if duration_s else None
    out_cues: list[dict[str, Any]] = []
    prev_end_frame = -1
    for idx, cue in enumerate(cues):
        if not cue.atoms:
            continue
        lines = _layout_lines(cue, cjk, cfg, respect_words=respect_words)
        lines = [ln for ln in lines if ln]
        if not lines:
            continue
        start_frame = seconds_to_frame(cue.start_s, fps)
        end_frame = max(start_frame + 1, seconds_to_frame(cue.end_s, fps))
        if duration_frames is not None:
            end_frame = min(end_frame, duration_frames)
        if start_frame <= prev_end_frame:
            start_frame = prev_end_frame  # keep the timeline monotone, no overlap
        if end_frame <= start_frame:
            end_frame = start_frame + 1
        if end_frame - start_frame < min_frames:
            # borrow frames from the preceding gap first, then from the end
            shortfall = min_frames - (end_frame - start_frame)
            start_frame = max(prev_end_frame, 0, start_frame - shortfall)
            if duration_frames is not None:
                end_frame = min(max(end_frame, start_frame + min_frames), duration_frames)
            else:
                end_frame = max(end_frame, start_frame + min_frames)
        # never overlap the next cue once times are quantised
        if idx + 1 < len(cues):
            next_start = seconds_to_frame(cues[idx + 1].start_s, fps)
            if end_frame > next_start:
                end_frame = max(start_frame + 1, next_start)
        prev_end_frame = end_frame
        char_count = sum(_count_chars(ln) for ln in lines)
        out_cues.append({
            "id": f"cue{len(out_cues)}",
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "start_s": round(start_frame / fps, 6),
            "end_s": round(end_frame / fps, 6),
            "duration_s": round((end_frame - start_frame) / fps, 6),
            "lines": lines,
            "text": _strip_line(_joined_raw(cue.atoms, cjk)),
            "display_text": " ".join(lines) if not cjk else "".join(lines),
            "char_count": char_count,
            "max_line_chars": max(_count_chars(ln) for ln in lines),
            "word_ids": [a.word_id for a in cue.atoms],
            "segment_ids": sorted({a.segment_id for a in cue.atoms}),
        })

    violations: list[str] = []
    if respect_words and mid_word_starts:
        violations.append(f"{mid_word_starts} cue(s) start inside an ASR word")
    for cue in out_cues:
        cfg_used = cfg["cjk" if cjk else "latin"]
        if len(cue["lines"]) > int(cfg_used["max_lines"]):
            violations.append(f"{cue['id']}: too many lines")
        for line in cue["lines"]:
            if _count_chars(line) > int(cfg_used["max_chars_per_line"]):
                violations.append(f"{cue['id']}: line over {cfg_used['max_chars_per_line']} chars")
            if strip_trailing_punct(line) != line:
                violations.append(f"{cue['id']}: line ends in punctuation")
        if cue["duration_s"] + 1e-6 < float(cfg["min_display_s"]):
            violations.append(f"{cue['id']}: display {cue['duration_s']:.3f}s < "
                              f"{cfg['min_display_s']}s")

    return {
        "schema": SCHEMA,
        "source": {
            "alignment_schema": alignment.get("schema"),
            "path": src.get("path"),
            "duration_s": src.get("duration_s"),
            "fps": fps,
            "width": src.get("width"),
            "height": src.get("height"),
            "language": language,
        },
        "style": cfg,
        "options": {
            "respect_segments": bool(respect_segments),
            "respect_words": bool(respect_words),
        },
        "cues": out_cues,
        "quality": {
            "n_cues": len(out_cues),
            "mid_word_cue_starts": int(mid_word_starts),
            "segment_boundary_breaks": int(segment_breaks),
            "n_lines": sum(len(c["lines"]) for c in out_cues),
            "max_line_chars": max((c["max_line_chars"] for c in out_cues), default=0),
            "min_duration_s": round(min((c["duration_s"] for c in out_cues), default=0.0), 6),
            "max_duration_s": round(max((c["duration_s"] for c in out_cues), default=0.0), 6),
            "mean_duration_s": round(
                sum(c["duration_s"] for c in out_cues) / len(out_cues), 6) if out_cues else 0.0,
            "violations": violations,
        },
    }


def build_from_file(path: str | Path, *, fps: float | None = None,
                    style: dict[str, Any] | None = None,
                    respect_segments: bool | None = None,
                    respect_words: bool = True) -> dict[str, Any]:
    alignment = json.loads(Path(path).read_text(encoding="utf-8"))
    return build_captions(alignment, fps=fps, style=style,
                          respect_segments=respect_segments,
                          respect_words=respect_words)


# --------------------------------------------------------------------------- export


def write_json(doc: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _cue_lines(cue: dict[str, Any]) -> list[str]:
    return [ln for ln in cue.get("lines") or [] if ln]


def write_srt(doc: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = float(doc["source"]["fps"])
    blocks: list[str] = []
    for i, cue in enumerate(doc.get("cues") or [], start=1):
        start = frame_to_srt(int(cue["start_frame"]), fps)
        end = frame_to_srt(int(cue["end_frame"]), fps)
        lines = "\n".join(_cue_lines(cue))
        blocks.append(f"{i}\n{start} --> {end}\n{lines}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def _ass_font_size(width: int, height: int) -> int:
    if width <= 0 or height <= 0:
        return 60
    if width / max(height, 1) >= 1.4:  # landscape
        size = round(width * 0.032)
    else:  # portrait / square
        size = round(width * 0.058)
    return int(min(max(size, 24), 96))


def _ass_escape(text: str) -> str:
    return (text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")
            .replace("\r", "").replace("\n", "\\N"))


def write_ass(doc: dict[str, Any], path: str | Path, *,
              play_res: tuple[int, int] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    src = doc.get("source") or {}
    width = int((play_res[0] if play_res else src.get("width")) or 1080)
    height = int((play_res[1] if play_res else src.get("height")) or 1920)
    fps = float(src.get("fps") or 25.0)
    style = doc.get("style") or DEFAULT_STYLE
    cjk = _is_cjk_doc(src.get("language"))
    font = str(style.get("font_cjk") if cjk else style.get("font_latin"))
    size = _ass_font_size(width, height)
    margin_v = int(round(height * (0.12 if cjk else 0.08)))
    margin_h = int(round(width * 0.06))
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        f"-1,0,0,0,100,100,0,0,1,3,0,2,{margin_h},{margin_h},{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines: list[str] = []
    for cue in doc.get("cues") or []:
        start = frame_to_ass(int(cue["start_frame"]), fps)
        end = frame_to_ass(int(cue["end_frame"]), fps)
        text = "\\N".join(_ass_escape(ln) for ln in _cue_lines(cue))
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return path


def export(doc: dict[str, Any], out_dir: str | Path, stem: str) -> dict[str, str]:
    """Write JSON + SRT + ASS for ``doc`` and return the absolute paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "json": str(write_json(doc, out_dir / f"{stem}.captions.json")),
        "srt": str(write_srt(doc, out_dir / f"{stem}.srt")),
        "ass": str(write_ass(doc, out_dir / f"{stem}.ass")),
    }


# ------------------------------------------------------------------------ validation


def validate(doc: dict[str, Any]) -> list[str]:
    """Return caption-spec problems (empty list == valid)."""
    problems: list[str] = []
    if doc.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}")
    src = doc.get("source") or {}
    fps = float(src.get("fps") or 0.0)
    if fps <= 0:
        problems.append("source.fps missing/invalid")
    cjk = _is_cjk_doc(src.get("language"))
    style = doc.get("style") or DEFAULT_STYLE
    cfg = style["cjk" if cjk else "latin"]
    min_display = float(style.get("min_display_s", 0.9))
    prev_end = -1
    gap_limit = float(style.get("merge_gap_s", 0.18))
    for i, cue in enumerate(doc.get("cues") or []):
        cid = cue.get("id") or f"cues[{i}]"
        start_frame = int(cue.get("start_frame", -1))
        end_frame = int(cue.get("end_frame", -1))
        if start_frame < 0 or end_frame <= start_frame:
            problems.append(f"{cid}: bad frame interval {start_frame}-{end_frame}")
        if start_frame < prev_end:
            problems.append(f"{cid}: overlaps previous cue")
        if prev_end >= 0 and fps > 0:
            gap = (start_frame - prev_end) / fps
            # 0 = back-to-back; otherwise must clear the flicker threshold
            if 1e-9 < gap < gap_limit - 1.0 / fps - 1e-9:
                problems.append(f"{cid}: flicker gap {gap:.3f}s < {gap_limit}s")
        prev_end = max(prev_end, end_frame)
        if fps > 0 and abs(float(cue.get("start_s", -1)) - start_frame / fps) > 1e-4:
            problems.append(f"{cid}: start_s is not frame-accurate")
        if fps > 0 and abs(float(cue.get("end_s", -1)) - end_frame / fps) > 1e-4:
            problems.append(f"{cid}: end_s is not frame-accurate")
        duration = (end_frame - start_frame) / fps if fps > 0 else 0.0
        if duration + 1e-6 < min_display:
            problems.append(f"{cid}: display {duration:.3f}s < {min_display}s")
        lines = cue.get("lines") or []
        if not lines:
            problems.append(f"{cid}: no lines")
        if len(lines) > int(cfg["max_lines"]):
            problems.append(f"{cid}: {len(lines)} lines > {cfg['max_lines']}")
        for line in lines:
            if _count_chars(line) > int(cfg["max_chars_per_line"]):
                problems.append(f"{cid}: line has {_count_chars(line)} chars "
                                f"> {cfg['max_chars_per_line']}")
            if strip_trailing_punct(line) != line:
                problems.append(f"{cid}: line ends in punctuation: {line!r}")
            if not re.search(r"[^\s，。、！？；：,.!?;:…—·~～\"“”‘’()\[\]【】《》]", line):
                problems.append(f"{cid}: punctuation-only line")
    options = doc.get("options") or {}
    quality = doc.get("quality") or {}
    if quality.get("violations"):
        problems.extend(str(v) for v in quality["violations"])
    if options.get("respect_words") and int(quality.get("mid_word_cue_starts", 0)) > 0:
        problems.append(f"{quality['mid_word_cue_starts']} cue(s) start inside an ASR word")
    if options.get("respect_segments"):
        for i, cue in enumerate(doc.get("cues") or []):
            if len(set(cue.get("segment_ids") or [])) > 1:
                problems.append(f"{cue.get('id') or i}: cue mixes ASR segments")
    return problems
