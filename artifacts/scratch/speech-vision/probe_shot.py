import numpy as np, cv2
from pathlib import Path
from semotion import media, vision as V

def metrics(prev, cur):
    tp = cv2.resize(prev,(64,36),interpolation=cv2.INTER_AREA); tc = cv2.resize(cur,(64,36),interpolation=cv2.INTER_AREA)
    g1 = cv2.cvtColor(tp, cv2.COLOR_RGB2GRAY).astype(np.int16); g2 = cv2.cvtColor(tc, cv2.COLOR_RGB2GRAY).astype(np.int16)
    md = float(np.abs(g2-g1).mean()/255.0)
    rgb = float(np.abs(tc.astype(np.int16)-tp.astype(np.int16)).mean()/255.0)
    h1 = cv2.cvtColor(tp, cv2.COLOR_RGB2HSV); h2 = cv2.cvtColor(tc, cv2.COLOR_RGB2HSV)
    bins=[30,10,10]; ranges=[0,180,0,256,0,256]
    a=cv2.calcHist([h1],[0,1,2],None,bins,ranges); b=cv2.calcHist([h2],[0,1,2],None,bins,ranges)
    a/=max(a.sum(),1); b/=max(b.sum(),1)
    bh=float(cv2.compareHist(a,b,cv2.HISTCMP_BHATTACHARYYA))
    return md, rgb, bh

for name in ['fx_a_zh_portrait','fx_c_zh_cuts','fx_b_zh_landscape','fx_pd_real_16x9_a','fx_pd_real_16x9_b']:
    p = Path(f'tests/fixtures/clips/{name}/{name}.mp4')
    mi = media.info(p)
    step = max(1, int(np.ceil(mi.fps/5.0)))
    dw,dh = V._det_size(mi.width, mi.height, 640)
    frames = V._decode_sampled(p, step=step, det_w=dw, det_h=dh)
    rows=[]
    for i in range(1, frames.shape[0]):
        md,rgb,bh = metrics(frames[i-1], frames[i])
        rows.append((i*step, md, rgb, bh))
    rows_sorted = sorted(rows, key=lambda r: -(0.5*r[2]+0.5*r[3]))[:5]
    print(f'== {name} fps={mi.fps} step={step} n={len(rows)}')
    print('   top5 by 0.5*rgb+0.5*bh:', [(r[0], round(r[1],4), round(r[2],4), round(r[3],4)) for r in rows_sorted])
    med = np.median([r[2] for r in rows]); p90=np.percentile([r[2] for r in rows],90)
    print(f'   rgb_diff median={med:.4f} p90={p90:.4f}; bh median={np.median([r[3] for r in rows]):.4f} p90={np.percentile([r[3] for r in rows],90):.4f}')
