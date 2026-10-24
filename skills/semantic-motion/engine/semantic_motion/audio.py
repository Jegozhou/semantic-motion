"""Audio features at 10 ms resolution: loudness envelope, adaptive VAD, pauses, onsets, pitch.

These are used to (1) refine ASR word boundaries, (2) find natural pauses for element exits,
(3) score vocal emphasis (loudness + pitch excursion), and (4) measure speech rate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOP = 0.01  # seconds


@dataclass
class AudioFeatures:
    sr: int
    hop: float
    db: np.ndarray          # loudness per hop (dBFS)
    vad: np.ndarray         # bool speech mask per hop
    onset: np.ndarray       # onset strength per hop (>=0)
    f0: np.ndarray          # pitch Hz per hop (0 = unvoiced)
    noise_floor: float
    speech_level: float
    duration: float

    def idx(self, t: float) -> int:
        return int(np.clip(round(t / self.hop), 0, len(self.db) - 1))

    def is_speech(self, t: float) -> bool:
        return bool(self.vad[self.idx(t)])

    def pauses(self, min_len: float = 0.18) -> list[tuple[float, float]]:
        out = []
        v = self.vad
        n = len(v)
        i = 0
        while i < n:
            if not v[i]:
                j = i
                while j < n and not v[j]:
                    j += 1
                if (j - i) * self.hop >= min_len:
                    out.append((i * self.hop, j * self.hop))
                i = j
            else:
                i += 1
        return out

    def mean_db(self, t0: float, t1: float) -> float:
        a, b = self.idx(t0), max(self.idx(t1), self.idx(t0) + 1)
        return float(np.mean(self.db[a:b]))

    def to_npz_dict(self) -> dict:
        return dict(sr=self.sr, hop=self.hop, db=self.db.astype(np.float32), vad=self.vad, onset=self.onset.astype(np.float32),
                    f0=self.f0.astype(np.float32), noise_floor=self.noise_floor, speech_level=self.speech_level, duration=self.duration)

    @classmethod
    def from_npz(cls, z) -> "AudioFeatures":
        return cls(sr=int(z["sr"]), hop=float(z["hop"]), db=z["db"], vad=z["vad"].astype(bool), onset=z["onset"], f0=z["f0"],
                   noise_floor=float(z["noise_floor"]), speech_level=float(z["speech_level"]), duration=float(z["duration"]))


def _frame_rms_db(x: np.ndarray, sr: int, hop_s: float, win_s: float) -> np.ndarray:
    hop = int(round(sr * hop_s))
    win = int(round(sr * win_s))
    n = max(1, int(np.ceil(len(x) / hop)))
    pad = np.pad(x, (win // 2, win + hop))
    # cumulative sum of squares for O(n) windowed energy
    c = np.concatenate([[0.0], np.cumsum(pad.astype(np.float64) ** 2)])
    starts = np.arange(n) * hop
    e = (c[starts + win] - c[starts]) / win
    return 10 * np.log10(e + 1e-10)


def _smooth(v: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return v
    ker = np.ones(k) / k
    return np.convolve(v, ker, mode="same")


def _yin_f0(x: np.ndarray, sr: int, hop_s: float, n_frames: int, vad: np.ndarray, fmin=70.0, fmax=400.0) -> np.ndarray:
    """Lightweight autocorrelation pitch tracker on voiced frames (enough for emphasis cues)."""
    f0 = np.zeros(n_frames, np.float32)
    win = int(sr * 0.04)
    hop = int(round(sr * hop_s))
    lag_min, lag_max = int(sr / fmax), int(sr / fmin)
    step = 2  # evaluate every 2nd hop, then fill
    for i in range(0, n_frames, step):
        if not vad[i]:
            continue
        s = i * hop
        fr = x[s:s + win]
        if len(fr) < win:
            break
        fr = fr - fr.mean()
        if np.max(np.abs(fr)) < 1e-4:
            continue
        ac = np.correlate(fr, fr, mode="full")[win - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        seg = ac[lag_min:lag_max]
        if len(seg) == 0:
            continue
        k = int(np.argmax(seg))
        if seg[k] > 0.45:
            f0[i] = sr / (lag_min + k)
            if i + 1 < n_frames:
                f0[i + 1] = f0[i]
    return f0


def analyze(samples: np.ndarray, sr: int = 16000, with_pitch: bool = True) -> AudioFeatures:
    dur = len(samples) / sr
    db = _frame_rms_db(samples, sr, HOP, 0.03)
    db_s = _smooth(db, 3)
    finite = db_s[np.isfinite(db_s)]
    noise = float(np.percentile(finite, 8)) if len(finite) else -80.0
    speech = float(np.percentile(finite, 92)) if len(finite) else -20.0
    span = max(speech - noise, 1.0)
    on_thr = noise + max(6.0, 0.30 * span)
    off_thr = on_thr - min(4.0, 0.08 * span)
    vad = np.zeros(len(db_s), bool)
    state = False
    for i, v in enumerate(db_s):
        if not state and v > on_thr:
            state = True
        elif state and v < off_thr:
            state = False
        vad[i] = state
    # fill short gaps (< 120 ms) and drop short blips (< 60 ms)
    vad = _close_gaps(vad, int(0.12 / HOP), True)
    vad = _close_gaps(vad, int(0.06 / HOP), False)
    onset = np.maximum(0.0, np.diff(_smooth(db, 2), prepend=db[0]))
    f0 = _yin_f0(samples, sr, HOP, len(db), vad) if with_pitch else np.zeros(len(db), np.float32)
    return AudioFeatures(sr=sr, hop=HOP, db=db.astype(np.float32), vad=vad, onset=onset.astype(np.float32), f0=f0,
                         noise_floor=noise, speech_level=speech, duration=dur)


def _close_gaps(mask: np.ndarray, max_len: int, fill_value: bool) -> np.ndarray:
    """Set runs of (not fill_value) shorter than max_len, bounded on both sides, to fill_value."""
    m = mask.copy()
    n = len(m)
    i = 0
    while i < n:
        if m[i] != fill_value:
            j = i
            while j < n and m[j] != fill_value:
                j += 1
            if 0 < i and j < n and (j - i) < max_len:
                m[i:j] = fill_value
            i = j
        else:
            i += 1
    return m


def emphasis_scores(af: AudioFeatures, spans: list[tuple[float, float]]) -> list[float]:
    """Per-span vocal emphasis z-score combining loudness and pitch excursion vs. local context."""
    if not spans:
        return []
    loud = np.array([af.mean_db(a, b) for a, b in spans])
    pitch = []
    for a, b in spans:
        i0, i1 = af.idx(a), max(af.idx(b), af.idx(a) + 1)
        seg = af.f0[i0:i1]
        seg = seg[seg > 0]
        pitch.append(float(np.median(seg)) if len(seg) else np.nan)
    pitch = np.array(pitch)
    out = []
    n = len(spans)
    for i in range(n):
        lo, hi = max(0, i - 12), min(n, i + 13)
        ctx_l = loud[lo:hi]
        zl = (loud[i] - np.median(ctx_l)) / (np.std(ctx_l) + 1.5)
        ctx_p = pitch[lo:hi]
        ctx_p = ctx_p[np.isfinite(ctx_p)]
        zp = 0.0
        if np.isfinite(pitch[i]) and len(ctx_p) > 3:
            zp = (np.log(pitch[i]) - np.median(np.log(ctx_p))) / (np.std(np.log(ctx_p)) + 0.05)
        out.append(float(0.65 * zl + 0.35 * zp))
    return out
