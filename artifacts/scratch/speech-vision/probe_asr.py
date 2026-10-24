import glob, json, time, wave
from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel
from semotion import media

video = Path('tests/fixtures/clips/fx_a_zh_portrait/fx_a_zh_portrait.mp4')
tmp = Path('artifacts/scratch/speech-vision/a.wav')
t0 = time.time()
media.extract_audio(video, tmp, rate=16000, channels=1)
print("extract s:", round(time.time()-t0,2))
with wave.open(str(tmp),'rb') as w:
    sr = w.getframerate(); n = w.getnframes(); raw = w.readframes(n)
audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)/32768.0
print("sr", sr, "samples", len(audio), "dur", len(audio)/sr)
mp = sorted(glob.glob('/Volumes/ZHITAI/models/faster-whisper/models--Systran--faster-whisper-medium/snapshots/*'))[0]
m = WhisperModel(mp, device='cpu', compute_type='int8')
t0 = time.time()
segs, info = m.transcribe(audio, language='zh', beam_size=5, word_timestamps=True,
                          condition_on_previous_text=False, vad_filter=False,
                          temperature=0.0)
segs = list(segs)
print("transcribe s:", round(time.time()-t0,2), "lang", info.language, info.language_probability)
out = []
for s in segs:
    for w in (s.words or []):
        out.append((w.word, round(w.start,3), round(w.end,3), round(w.probability,3)))
    print(f"[seg {s.id}] {s.start:.3f}-{s.end:.3f} lp={s.avg_logprob:.3f} nsp={s.no_speech_prob:.3f} {s.text!r} words={len(s.words or [])}")
print("total words:", len(out))
for w in out: print(w)
json.dump({"texts":[o[0] for o in out]}, open('artifacts/scratch/speech-vision/words_raw.json','w'), ensure_ascii=False, indent=1)
