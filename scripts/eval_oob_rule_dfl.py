"""External check of the out-of-bounds dead-ball rule on Sportec Open DFL.

1. Precision of BallStatus=0 inside rule runs on the six training matches
   (this is where MARGIN_M / MIN_FRAMES were fixed).
2. Development effect on J03WPY: run-58 saved probabilities decoded with the
   recorded costs (0, 2, 1), then the rule applied at label level.

    python scripts/eval_oob_rule_dfl.py --dfl data/external/sportec \
        --cache data/local-v08/run-58-pretrained-attention/campaign-audit/probabilities.npz
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm  # noqa: E402
from possession.external import read_sportec  # noqa: E402
from possession.temporal import viterbi_labels  # noqa: E402
from possession.dead_ball_rules import oob_dead_mask, runs_of  # noqa: E402

TRAIN = ['J03WMX', 'J03WN1', 'J03WOH', 'J03WOY', 'J03WQQ', 'J03WR9']


def score(truths, preds):
    t, p = [], []
    for a, b in zip(truths, preds):
        t += [a, np.array(['X'])]; p += [b, np.array(['X'])]
    r = pm.evaluate(pm.label_codes(np.concatenate(t)), pm.label_codes(np.concatenate(p)))
    return {k: r[k] for k in ('spell_f1_mean', 'macro_f1', 'n_gt_spells', 'n_pred_spells')}


def halves(folder, match):
    meta = next(Path(folder).glob('*matchinformation*' + match + '.xml'))
    trk = next(Path(folder).glob('*positions*' + match + '.xml'))
    return read_sportec(meta, trk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dfl', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', default=str(ROOT / 'data/local-v08/oob_rule_dfl.json'))
    a = ap.parse_args()
    flagged = dead = runs = bad = 0
    for m in TRAIN:
        for h in halves(a.dfl, m):
            mask = oob_dead_mask(h.ball, h.pitch)
            flagged += int(mask.sum()); dead += int((h.labels[mask] == 'D').sum())
            for s, e in runs_of(mask):
                runs += 1; bad += int((h.labels[s:e] == 'D').mean() < .9)
    dev = halves(a.dfl, 'J03WPY')
    with np.load(a.cache) as z:
        probs = [z['p0'], z['p1']]; truth = [z['truth0'], z['truth1']]
    assert all(np.array_equal(h.labels, t) for h, t in zip(dev, truth)), 'alignment'
    base = [viterbi_labels(p, 0., 2., 1.) for p in probs]
    new = []
    for b, h in zip(base, dev):
        b = b.copy(); b[oob_dead_mask(h.ball, h.pitch) & np.isin(b, ['H', 'A'])] = 'D'; new.append(b)
    out = dict(train_precision=dead / max(1, flagged), train_flagged_frames=flagged, train_runs=runs,
               train_runs_below_90pct_dead=bad, dev_before=score(truth, base), dev_after=score(truth, new))
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n'); print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
