"""Matched external-only geometry/memory learners without teacher base margins.

Reuses verified coordinate-only caches. The parent is blended only during
external development evaluation, so the learners cannot copy its training logits.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np

from possession.data import sha256
from possession.spatial_memory import symmetric_features
from scripts.research.spatial_memory.run_external import TRAIN, DEV, blend, dump, isolate, score_parts


PLAN = dict(train_matches=TRAIN, development_matches=[DEV], independent_test_matches=[],
            views=['native', 'corrupt'], fits=['geometry_control', 'spatial_memory_independent'],
            correction_strengths=[0., .05, .1, .25, .5, 1.],
            native_spell_f1_max_decline=.01,
            learner_inputs='Coordinates/masks only; no parent logits, possession labels or identities as features',
            selection='Maximum mean native/corrupt development spell F1 subject to native guard; strength zero retains parent',
            parameters=dict(n_estimators=600, max_depth=5, learning_rate=.05,
                            min_child_weight=20, reg_lambda=10., subsample=.8,
                            colsample_bytree=1., max_bin=128, random_state=20321115),
            head_orientation_used=False, gaze_measured=False,
            provided_match_data_used=False, public_labels_used=False,
            limitation='Existing development match only; no independent test or BAR-ATH performance estimate.')


def independent_probability(model, memory, names, selected_columns):
    swapped = symmetric_features(memory, names)
    first = model.predict_proba(memory[:, selected_columns])
    second = model.predict_proba(swapped[:, selected_columns])[:, [1, 0, 2]]
    return ((first + second) / 2.).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    out = args.study / 'independent'
    out.mkdir(exist_ok=True)
    if (out / 'selection.json').exists():
        raise ValueError('Bounded independent study already completed')
    dump(out / 'plan.json', PLAN)
    denied = isolate(args.study)
    import torch
    import xgboost as xgb
    from possession.temporal import training_weights, viterbi_labels
    parent = ROOT / 'models/coreorigin-video-v1'
    state = torch.load(parent / 'selected.pt', map_location='cpu', weights_only=False)
    cache = args.study / 'cache'
    for path in cache.glob('*.npz'):
        receipt = json.loads(path.with_suffix('.json').read_text())
        if sha256(path) != receipt['sha256']:
            raise ValueError('External memory cache changed')
    xs, ys, ws = [], [], []
    names = None
    for match in TRAIN:
        for i in range(2):
            for view in PLAN['views']:
                with np.load(cache / f'{match}_{i}_{view}.npz', allow_pickle=False) as z:
                    truth, memory = z['truth'], z['memory']
                    if names is not None and names != z['names'].tolist():
                        raise ValueError('Feature schema differs')
                    names = z['names'].tolist()
                    keep = np.arange(len(truth)) % 5 == 0
                    for edge in np.flatnonzero(truth[1:] != truth[:-1]) + 1:
                        keep[max(0, edge - 12):edge + 13] = True
                    keep &= np.isin(truth, ['H', 'A', 'D'])
                    y = np.array([{'H':0, 'A':1, 'D':2}.get(value, -1) for value in truth], np.int32)
                    xs.append(memory[keep]); ys.append(y[keep]); ws.append(training_weights(truth, .5)[keep])
    x, y, weights = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws).astype(np.float32)
    del xs, ys, ws
    x = np.concatenate([x, symmetric_features(x, names)])
    y = np.concatenate([y, np.where(y == 0, 1, np.where(y == 1, 0, y))])
    weights = np.tile(weights, 2); weights /= weights.mean()
    columns = dict(geometry_control=[i for i, name in enumerate(names) if '_current_' in name or name.startswith('ball_seen')],
                   spatial_memory_independent=list(range(len(names))))
    dump(out / 'training_receipt.json', dict(rows=len(y), memory_names=names, columns=columns,
          plan=PLAN, xgboost=xgb.__version__, no_teacher_logits_in_training=True,
          provided_match_read=False, data_guard_verified=True))
    dev = {(i, view):dict(np.load(cache / f'{DEV}_{i}_{view}.npz', allow_pickle=False))
           for i in range(2) for view in PLAN['views']}
    baseline = {}
    for view in PLAN['views']:
        h = [dev[i, view] for i in range(2)]
        baseline[view] = score_parts([a['truth'] for a in h],
                                    [viterbi_labels(a['probability'], *state['decoder_costs']) for a in h])
    records = []
    for kind in PLAN['fits']:
        model = xgb.XGBClassifier(**PLAN['parameters'], tree_method='hist', device=args.device,
                                   n_jobs=4, objective='multi:softprob', num_class=3, eval_metric='mlogloss')
        print(f'Fitting independent {kind}: {len(y)} external rows / {len(columns[kind])} features', flush=True)
        model.fit(x[:, columns[kind]], y, sample_weight=weights, verbose=False)
        model.set_params(device='cpu')
        model.save_model(out / f'{kind}.ubj')
        probability = {key:independent_probability(model, h['memory'], names, columns[kind]) for key, h in dev.items()}
        for strength in PLAN['correction_strengths']:
            scores = {}
            for view in PLAN['views']:
                h = [dev[i, view] for i in range(2)]
                pred = [viterbi_labels(blend(a['probability'], probability[i, view], strength), *state['decoder_costs'])
                        for i, a in enumerate(h)]
                scores[view] = score_parts([a['truth'] for a in h], pred)
            entry = dict(kind=kind, strength=strength, scores=scores,
                         eligible=scores['native']['spell_f1_mean'] >= baseline['native']['spell_f1_mean'] - PLAN['native_spell_f1_max_decline'],
                         mean_spell_f1=float(np.mean([r['spell_f1_mean'] for r in scores.values()])))
            records.append(entry)
            print(json.dumps(entry), flush=True)
        dump(out / 'development_results.json', dict(parent=baseline, candidates=records))
    best = max([r for r in records if r['eligible']], key=lambda r:(r['mean_spell_f1'], -r['strength'], r['kind'] == 'spatial_memory_independent'))
    result = dict(complete=True, correction_family='independent', selected=best,
                  selected_columns=columns[best['kind']], parent_scores=baseline,
                  selected_model_sha256=sha256(out / (best['kind'] + '.ubj')),
                  parent_sha256=sha256(parent / 'selected.pt'), memory_names=names,
                  decoder_costs=state['decoder_costs'], plan=PLAN,
                  preparation_sha256=sha256(args.study / 'preparation.json'),
                  source_sha256={name:sha256(ROOT / name) for name in
                    ['possession/spatial_memory.py', 'scripts/research/spatial_memory/run_external.py',
                     'scripts/research/spatial_memory/fit_independent.py']},
                  data_guard_verified=True, unexpected_competition_access=denied,
                  competition_data_read=False, public_labels_read=False, full_match_competition_score=None)
    dump(out / 'selection.json', result)
    print('Independent external selection frozen: ' + json.dumps(best), flush=True)


if __name__ == '__main__':
    main()
