"""Diagnose a frozen model on external development matches only."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from evaluation import possession_metrics as pm
from possession.temporal import symmetric_probability, viterbi_labels
from possession.data import sha256
from scripts.run_local import load_halves, selected_model, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--omit-features',action='store_true',help='Save probabilities/truth without duplicating the large feature bank')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    checkpoint_hash=sha256(args.run/'selected.pt')
    model, state = selected_model(args.run, 'cuda')
    halves, names = load_halves(state['config']['validation_matches'], ROOT/state['config'].get('features_dir', 'data/external/features-v07'))
    rows, truth, prediction, arrays = [], [], [], {}
    for i, half in enumerate(halves):
        p = symmetric_probability(model, half['x'], names, state['mean'], state['scale'], 'cuda')
        labels = np.repeat(viterbi_labels(p, *state['decoder_costs']), state['config']['sample_step'])[:len(half['truth'])]
        g, y = pm.label_codes(half['truth']), pm.label_codes(labels)
        runs = pm.valid_runs(g)
        gs, ps = pm.possession_spells(g, runs), pm.possession_spells(y, runs)
        overlaps = pm.spell_overlaps(gs, ps)
        flags = pm.matched_gt_flags(overlaps, len(gs), .5)
        for spell, matched in zip(gs, flags):
            a, b, label = spell
            rows.append(dict(half=i, start=int(a), stop=int(b), label=int(label),
                             seconds=(b-a)/25., matched=bool(matched)))
        arrays.update({f'p{i}': p, f'truth{i}': half['truth']})
        if not args.omit_features:
            arrays[f'x{i}']=half['x']
        truth.extend([g, np.array([pm.X])]); prediction.extend([y, np.array([pm.X])])
    buckets = []
    for low, high in ((0, 1), (1, 2), (2, 5), (5, 10), (10, 30), (30, 10000)):
        group = [r for r in rows if low <= r['seconds'] < high]
        buckets.append(dict(min_seconds=low, max_seconds=high, total=len(group), matched=sum(r['matched'] for r in group)))
    scores = pm.evaluate(np.concatenate(truth), np.concatenate(prediction))
    if sha256(args.run/'selected.pt')!=checkpoint_hash:
        raise ValueError('Checkpoint changed during its audit')
    report = dict(scope='External development only; never independent evidence.', run=str(args.run),
                  selected_checkpoint_sha256=checkpoint_hash,protocol_fingerprint=state['fingerprint'],
                  audit_source_sha256=sha256(Path(__file__)),scorer_sha256=sha256(ROOT/'evaluation/possession_metrics.py'),
                  feature_arrays_included=not args.omit_features,
                  scores=scores, duration_buckets=buckets, spells=rows)
    save_json(args.out/'audit.json', report)
    np.savez_compressed(args.out/'probabilities.npz', **arrays)
    print(json.dumps(dict(scores={k: scores[k] for k in ('spell_f1_mean', 'macro_f1', 'n_gt_spells', 'n_pred_spells')},
                          duration_buckets=buckets), indent=2), flush=True)


if __name__ == '__main__':
    main()
