"""External paired full-kinematics ablation, with and without spatial memory.

The earlier small feature-only learners omitted the existing kinematic bank.
This comparison retains it in both arms. Sampling is fixed to bound RAM use.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from possession.data import sha256
from possession.external import swap_features
from possession.spatial_memory import symmetric_features
from scripts.research.spatial_memory.run_external import TRAIN, DEV, blend, dump, isolate, score_parts


PLAN = dict(train_matches=TRAIN, development_matches=[DEV], independent_test_matches=[],
            views=['native', 'corrupt'], fits=['kinematics_control', 'kinematics_memory'],
            max_sampled_rows_per_half_view=3000, boundary_sample_fraction=.5,
            seed=20321116, correction_strengths=[0., .05, .1, .25, .5, 1.],
            native_spell_f1_max_decline=.01,
            parameters=dict(n_estimators=600, max_depth=5, learning_rate=.05,
                            min_child_weight=10, reg_lambda=10., subsample=.8,
                            colsample_bytree=.8, max_bin=64, random_state=20321116),
            learner_inputs='Existing 960 coordinate-only kinematic features, optionally 128 spatial-memory features; no teacher probabilities.',
            selection='Maximum mean native/corrupt development spell F1 with native guard; retain parent at strength zero',
            provided_match_data_used=False, public_labels_used=False,
            head_orientation_used=False, gaze_measured=False,
            limitation='Exploratory development on the existing external match, not independent test or BAR-ATH performance.')


def full_probability(model, base, base_names, memory, memory_names, use_memory):
    matrix = np.column_stack([base, memory]) if use_memory else base
    first = model.predict_proba(matrix)
    swapped_base = swap_features(base, base_names)
    swapped = np.column_stack([swapped_base, symmetric_features(memory, memory_names)]) if use_memory else swapped_base
    second = model.predict_proba(swapped)[:, [1, 0, 2]]
    return ((first + second) / 2.).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    out = args.study / 'full_context'
    out.mkdir(exist_ok=True)
    if (out / 'selection.json').exists():
        raise ValueError('This external study has already completed')
    dump(out / 'plan.json', PLAN)
    denied = isolate(args.study)
    import torch
    import xgboost as xgb
    from possession.temporal import training_weights, viterbi_labels
    parent = ROOT / 'models/coreorigin-video-v1'
    state = torch.load(parent / 'selected.pt', map_location='cpu', weights_only=False)
    cache = args.study / 'cache'
    rng = np.random.default_rng(PLAN['seed'])
    base_names, memory_names = None, None
    matrices, labels, weights = [], [], []
    cache_hashes = {}
    for match in TRAIN + [DEV]:
        for view in PLAN['views']:
            base_path = ROOT / ('data/external/features-v25' + ('-corrupt' if view == 'corrupt' else '')) / (match + '.npz')
            digest = sha256(base_path)
            with np.load(base_path, allow_pickle=False) as z:
                if z['names'].tolist() != state['names']:
                    raise ValueError('Existing kinematic bank differs from parent')
                base_names = z['names'].tolist()
                for i in range(2):
                    path = cache / f'{match}_{i}_{view}.npz'
                    receipt = json.loads(path.with_suffix('.json').read_text())
                    if receipt['fingerprint']['base_cache_sha256'] != digest or sha256(path) != receipt['sha256']:
                        raise ValueError('Feature/memory cache provenance differs')
                    cache_hashes[path.name] = receipt['sha256']
                    if match == DEV:
                        continue
                    with np.load(path, allow_pickle=False) as m:
                        truth, memory = m['truth'], m['memory']
                        memory_names = m['names'].tolist()
                        np.testing.assert_array_equal(truth, z[f'truth{i}'])
                        valid = np.flatnonzero(np.isin(truth, ['H', 'A', 'D']))
                        near = np.zeros(len(truth), bool)
                        for edge in np.flatnonzero(truth[1:] != truth[:-1]) + 1:
                            near[max(0, edge - 12):edge + 13] = True
                        edge_rows = np.flatnonzero(near & np.isin(truth, ['H', 'A', 'D']))
                        n = min(len(valid), PLAN['max_sampled_rows_per_half_view'])
                        sampled = rng.choice(valid, n, replace=False)
                        count = min(len(edge_rows), n // 2)
                        if count:
                            sampled[:count] = rng.choice(edge_rows, count, replace=False)
                        sampled = np.unique(sampled)
                        x = np.column_stack([z[f'x{i}'][sampled], memory[sampled]]).astype(np.float32)
                        matrices.append(x)
                        labels.append(np.array([{'H':0, 'A':1, 'D':2}[value] for value in truth[sampled]], np.int32))
                        weights.append(training_weights(truth, .5)[sampled].astype(np.float32))
    x, y, weight = np.concatenate(matrices), np.concatenate(labels), np.concatenate(weights)
    del matrices, labels, weights
    boundary = len(base_names)
    swapped = np.column_stack([swap_features(x[:, :boundary], base_names), symmetric_features(x[:, boundary:], memory_names)])
    x = np.concatenate([x, swapped]); del swapped
    y = np.concatenate([y, np.where(y == 0, 1, np.where(y == 1, 0, y))])
    weight = np.tile(weight, 2); weight /= weight.mean()
    dump(out / 'training_receipt.json', dict(rows=len(y), features=x.shape[1], plan=PLAN,
          external_cache_sha256=cache_hashes, data_guard_verified=True,
          teacher_logits_used_as_features=False, competition_data_read=False))
    dev = {(i, view):dict(np.load(cache / f'{DEV}_{i}_{view}.npz', allow_pickle=False))
           for i in range(2) for view in PLAN['views']}
    baselines = {}
    for view in PLAN['views']:
        h = [dev[i, view] for i in range(2)]
        baselines[view] = score_parts([a['truth'] for a in h], [viterbi_labels(a['probability'], *state['decoder_costs']) for a in h])
    records = []
    for kind in PLAN['fits']:
        use_memory = kind == 'kinematics_memory'
        model = xgb.XGBClassifier(**PLAN['parameters'], tree_method='hist', device=args.device,
                                  n_jobs=4, objective='multi:softprob', num_class=3, eval_metric='mlogloss')
        print(f'Fitting {kind}: {len(y)} external rows', flush=True)
        model.fit(x if use_memory else x[:, :boundary], y, sample_weight=weight, verbose=False)
        model.set_params(device='cpu'); model.save_model(out / f'{kind}.ubj')
        corrected = {}
        for view in PLAN['views']:
            base_path = ROOT / ('data/external/features-v25' + ('-corrupt' if view == 'corrupt' else '')) / (DEV + '.npz')
            with np.load(base_path, allow_pickle=False) as z:
                for i in range(2):
                    corrected[i, view] = full_probability(model, z[f'x{i}'], base_names,
                        dev[i, view]['memory'], memory_names, use_memory)
        for strength in PLAN['correction_strengths']:
            scores = {}
            for view in PLAN['views']:
                h = [dev[i, view] for i in range(2)]
                prediction = [viterbi_labels(blend(a['probability'], corrected[i, view], strength), *state['decoder_costs'])
                              for i, a in enumerate(h)]
                scores[view] = score_parts([a['truth'] for a in h], prediction)
            entry = dict(kind=kind, strength=strength, scores=scores,
                         eligible=scores['native']['spell_f1_mean'] >= baselines['native']['spell_f1_mean'] - PLAN['native_spell_f1_max_decline'],
                         mean_spell_f1=float(np.mean([r['spell_f1_mean'] for r in scores.values()])))
            records.append(entry); print(json.dumps(entry), flush=True)
        dump(out / 'development_results.json', dict(parent=baselines, candidates=records))
    best = max([r for r in records if r['eligible']], key=lambda r:(r['mean_spell_f1'], -r['strength'], r['kind'] == 'kinematics_memory'))
    result = dict(complete=True, correction_family='full_context', selected=best,
                  use_memory=best['kind'] == 'kinematics_memory', parent_scores=baselines,
                  selected_model_sha256=sha256(out / (best['kind'] + '.ubj')),
                  parent_sha256=sha256(parent / 'selected.pt'), base_names=base_names, memory_names=memory_names,
                  decoder_costs=state['decoder_costs'], plan=PLAN,
                  source_sha256={name:sha256(ROOT / name) for name in
                    ['possession/spatial_memory.py', 'scripts/research/spatial_memory/run_external.py',
                     'scripts/research/spatial_memory/fit_full_context.py']},
                  data_guard_verified=True, unexpected_competition_access=denied,
                  competition_data_read=False, public_labels_read=False, full_match_competition_score=None)
    dump(out / 'selection.json', result)
    print('Full-context external selection frozen: ' + json.dumps(best), flush=True)


if __name__ == '__main__':
    main()
