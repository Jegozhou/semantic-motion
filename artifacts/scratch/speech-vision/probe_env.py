import json, wave
from pathlib import Path
import numpy as np

video = Path('tests/fixtures/clips/fx_a_zh_portrait/fx_a_zh_portrait.mp4')
wav = Path('artifacts/scratch/speech-vision/a.wav')
from semotion import media
media.extract_audio(video, wav, rate=16000, channels=1)
with wave.open(str(wav),'rb') as w:
    sr=w.getframerate(); a=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)/32768.0
hop=160  # 10ms
n=len(a)//hop
env=np.sqrt((a[:n*hop].reshape(n,hop)**2).mean(axis=1))
peak=env.max(); noise=np.percentile(env,10)
print("dur",len(a)/sr,"peak",round(float(peak),4),"noise_p10",round(float(noise),5))
gt = json.loads(Path('tests/fixtures/clips/fx_a_zh_portrait/ground_truth.json').read_text())
# find first frame where env exceeds 3*noise after being low, searching around truth start
for u in gt['units']:
    t0=u['start_s']; e0=max(0,int((t0-0.35)/0.01)); e1=min(n,int((t0+0.35)/0.01))
    seg=env[e0:e1]
    thr=max(noise*3.0, peak*0.02)
    above=np.where(seg>thr)[0]
    first = e0+above[0] if len(above) else None
    # rise-based: max positive diff
    d=np.diff(seg, prepend=seg[0])
    i=int(np.argmax(d))
    print(f"{u['unit_id']} {u['text'][:8]:<10} truth={t0:.3f} thr_on={first*0.01 if first is not None else None} rise_at={(e0+i)*0.01:.2f} rise={d[i]:.3f}")
