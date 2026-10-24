import glob, os, time
from pathlib import Path
import numpy as np

snaps = sorted(glob.glob('/Volumes/ZHITAI/models/faster-whisper/models--Systran--faster-whisper-medium/snapshots/*'))
print("snapshots:", snaps)
mp = Path(snaps[0])
print("has model.bin:", (mp/'model.bin').exists(), (mp/'model.bin').stat().st_size/1e6, "MB")

from faster_whisper import WhisperModel
t0 = time.time()
m = WhisperModel(str(mp), device='cpu', compute_type='int8')
print("load s:", round(time.time()-t0, 2))
t0 = time.time()
m2 = WhisperModel(str(mp), device='cpu', compute_type='int8')
print("load2 s:", round(time.time()-t0, 2))
