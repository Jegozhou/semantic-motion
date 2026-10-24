"""Aggregate SYNTH_KNOWN timing accuracy for the speech/vision layer (dev set)."""
import json
from pathlib import Path
import numpy as np
from semotion import asr, anchors as A

ROOT = Path('.')
OUT = Path('artifacts/anchors')
FIXTURES = ['fx_a_zh_portrait', 'fx_b_zh_landscape', 'fx_c_zh_cuts']

def stats(errs):
    e = np.asarray(errs, dtype=float)
    if not len(e): return {}
    return {"n": int(len(e)), "p50_ms": int(round(np.percentile(e,50))),
            "p90_ms": int(round(np.percentile(e,90))), "max_ms": int(round(e.max())),
            "mean_ms": int(round(e.mean()))}

report = {"per_fixture": {}, "totals": {}}
word_before, word_after, anchor_before, anchor_after = [], [], [], []
for fid in FIXTURES:
    clip = ROOT / f'tests/fixtures/clips/{fid}/{fid}.mp4'
    gt = ROOT / f'tests/fixtures/clips/{fid}/ground_truth.json'
    if not clip.exists():
        print('skip', fid); continue
    tr = asr.transcribe(clip, OUT / f'{fid}.transcript.json')
    truth = A.load_truth(gt)
    doc = A.classify(tr)
    ref = A.refine_onsets(clip, doc, out_json=OUT / f'{fid}.anchors.json')
    cal = A.unit_onset_anchors(tr, truth)
    cal_ref = A.refine_onsets(clip, cal, out_json=OUT / f'{fid}.calibration.json')

    ab = A.timing_errors(doc, truth); aa = A.timing_errors(ref, truth)
    wb = A.timing_errors(cal, truth); wa = A.timing_errors(cal_ref, truth)
    word_before += wb['errors_ms']; word_after += wa['errors_ms']
    anchor_before += ab['errors_ms']; anchor_after += aa['errors_ms']
    report['per_fixture'][fid] = {
        "segments": len(tr['segments']), "words": len(tr['words']),
        "anchors": len(ref['anchors']), "kinds": ref['provenance']['kinds'],
        "accepted_refine": ref['provenance']['refine']['n_accepted'],
        "word_onset": {"before": stats(wb['errors_ms']), "after": stats(wa['errors_ms']),
                       "unmatched_units": len(wb['unmatched_units'])},
        "anchor_onset": {"before": stats(ab['errors_ms']), "after": stats(aa['errors_ms']),
                         "samples": ab['samples']},
    }
    print(fid, json.dumps(report['per_fixture'][fid], ensure_ascii=False))

report['totals'] = {
    "word_onset": {"before": stats(word_before), "after": stats(word_after),
                   "all_errors_before": word_before, "all_errors_after": word_after},
    "anchor_onset": {"before": stats(anchor_before), "after": stats(anchor_after),
                     "all_errors_before": anchor_before, "all_errors_after": anchor_after},
    "truth_class": "SYNTH_KNOWN (per-unit TTS assembly, measured by ffprobe; ASR not used)",
    "split": "dev",
    "note": "Exploratory: < 50 independent events, spec §5 requires >= 50 for a P90 acceptance claim.",
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'timing_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print('TOTALS', json.dumps(report['totals'], ensure_ascii=False))
