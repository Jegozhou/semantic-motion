import numpy as np

from semantic_motion.audio import AudioFeatures
from semantic_motion.asr import build_transcript, refine_words


def _features(speech_spans, dur=6.0, hop=0.01, nf=-50.0, sp=-12.0):
    n = int(dur / hop) + 1
    db = np.full(n, nf)
    for a, b in speech_spans:
        db[int(a / hop):int(b / hop)] = sp
    vad = db > (nf + 15)
    return AudioFeatures(sr=16000, hop=hop, db=db, vad=vad, onset=np.zeros(n), f0=np.zeros(n),
                         noise_floor=nf, speech_level=sp, duration=dur)


def test_soft_start_fixes_early_onset_after_pause():
    af = _features([(0.0, 1.0), (2.0, 2.8)])
    words = [{"text": "Hello", "start": 0.0, "end": 1.0}, {"text": "Second", "start": 1.45, "end": 2.6}]
    out = refine_words(words, af)
    # Whisper stamped 'Second' 550 ms early inside the pause; it is moved to the real onset (2.0 - 20 ms)
    assert abs(out[1]["start"] - 1.98) < 0.015
    assert out[1]["asr_start"] == 1.45


def test_onsets_inside_speech_are_untouched():
    af = _features([(0.0, 3.0)])
    words = [{"text": "a", "start": 0.5, "end": 1.0}, {"text": "b", "start": 1.0, "end": 1.6}]
    out = refine_words(words, af)
    assert [w["start"] for w in out] == [0.5, 1.0]


def test_silence_end_pulled_back_and_monotonic():
    af = _features([(0.0, 1.0), (2.0, 3.0)])
    words = [{"text": "one", "start": 0.2, "end": 1.6}, {"text": "two", "start": 2.0, "end": 2.9}]
    out = refine_words(words, af)
    assert out[0]["end"] < 1.6 and out[0]["end"] <= 1.31   # pulled back toward the voiced region (<= max_shift)
    for a, b in zip(out, out[1:]):
        assert a["start"] <= b["start"] and a["end"] <= b["start"] + 1e-9


def test_no_beat_ends_on_dangling_function_word():
    text = ("so the thing is that we built this tool and the result was that people could finally edit "
            "the whole encyclopedia together in real time from anywhere in the world without asking anyone "
            "for permission which was a radical idea")
    toks = text.split()
    raw, t = [], 0.0
    for i, tok in enumerate(toks):
        raw.append({"text": " " + tok, "start": t, "end": t + 0.28, "probability": 0.9})
        # a long hesitation right after a dangling "the" must not end the beat there
        t += 0.28 + (0.5 if (tok == "the" and i == 10) else 0.02)
    tr = build_transcript(raw, "en", None, refine=False)
    assert len(tr["sentences"]) >= 2
    dangling = {"the", "a", "an", "and", "of", "to", "that", "in", "for", "with"}
    for s in tr["sentences"][:-1]:
        assert s["text"].split()[-1].lower().strip(",.") not in dangling, s["text"]
    assert all(len(s["text"].split()) <= 30 for s in tr["sentences"])
