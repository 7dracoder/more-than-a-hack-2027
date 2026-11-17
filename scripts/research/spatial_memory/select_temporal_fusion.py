"""Select a missing-ball temporal ensemble using external development only.

No fitting or calibration on BAR-ATH. Both temporal models are externally
trained. A coarse, predeclared grid tests whole-frame and gap-aware mixtures.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np

from possession.data import sha256
from scripts.research.spatial_memory.run_external import DEV, TRAIN, dump, isolate, score_parts, torch_setup


PLAN = dict(training_matches=TRAIN, development_matches=[DEV], independent_test_matches=[],
            parent='models/coreorigin-video-v1', fallback='models/coreorigin-robust-v13',
            modes=['all', 'missing', 'long_missing'], weights=[0., .25, .5, .75, 1.],
            long_missing_seconds=2., native_spell_f1_max_decline=.01,
            decoder='Unchanged parent decoder and parent boundary probabilities',
            selection='Maximum mean native/corrupt development spell F1, with native guard; strength zero retains parent',
            provided_match_used=False, head_orientation_used=False,
            limitation='Exploratory selection on the existing external development match; no independent or competition score.')


def gap_gate(memory, names, mode):
    if mode == 'all':
        return np.ones(len(memory), np.float32)
    missing = memory[:, names.index('ball_seen')] < .5
    if mode == 'missing':
        return missing.astype(np.float32)
    if mode == 'long_missing':
        seconds = 10. * (memory[:, names.index('past_anchor_age')] + memory[:, names.index('future_anchor_age')])
        return (missing & (seconds >= PLAN['long_missing_seconds'])).astype(np.float32)
    raise ValueError('Unknown externally declared fusion mode')


def fusion_probability(parent, fallback, memory, names, mode, weight):
    if parent.shape != fallback.shape or parent.shape[1] not in (3, 4):
        raise ValueError('Temporal probability schemas differ')
    if weight == 0:
        return parent.copy()
    if not 0 <= weight <= 1:
        raise ValueError('Invalid frozen mixture weight')
    alpha = weight * gap_gate(memory, names, mode)
    result = parent.copy()
    # Convex probability mixture retains each model's uncertainty and explicitly
    # preserves the parent's boundary head and every frame outside the gate.
    result[:, :3] = (1. - alpha[:, None]) * parent[:, :3] + alpha[:, None] * fallback[:, :3]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    out = args.study / 'temporal_fusion'
    out.mkdir(exist_ok=True)
    if (out / 'selection.json').exists():
        raise ValueError('This external selection is already frozen')
    dump(out / 'plan.json', PLAN)
    denied = isolate(args.study)
    from scripts.run_local import selected_model, symmetric_probability, viterbi_labels
    device = torch_setup(args.device)
    fallback_dir = ROOT / PLAN['fallback']
    fallback, state = selected_model(fallback_dir, device)
    if state['config']['train_matches'] != TRAIN or state['config']['validation_matches'] != [DEV]:
        raise ValueError('Fallback supervision differs from the external-only study')
    step = state['config']['sample_step']
    if step != 2:
        raise ValueError('Expected the verified half-rate robustness model')
    memory_names, baseline, reproduced, data = None, {}, {}, {}
    parent_costs = None
    import torch
    parent_state = torch.load(ROOT / PLAN['parent'] / 'selected.pt', map_location='cpu', weights_only=False)
    parent_costs = parent_state['decoder_costs']
    cache_hashes = {}
    for view in ('native', 'corrupt'):
        raw_path = ROOT / ('data/external/features-v09' + ('-corrupt' if view == 'corrupt' else '')) / (DEV + '.npz')
        meta = json.loads(raw_path.with_suffix('.json').read_text())
        if sha256(raw_path) != meta['sha256']:
            raise ValueError('Fallback external cache changed')
        cache_hashes[str(raw_path.relative_to(ROOT))] = meta['sha256']
        truth_parts, parent_parts, fallback_parts = [], [], []
        with np.load(raw_path, allow_pickle=False) as z:
            if z['names'].tolist() != state['names']:
                raise ValueError('Fallback feature names differ')
            for i in range(2):
                memory_path = args.study / 'cache' / f'{DEV}_{i}_{view}.npz'
                receipt = json.loads(memory_path.with_suffix('.json').read_text())
                if sha256(memory_path) != receipt['sha256']:
                    raise ValueError('External parent/memory cache changed')
                with np.load(memory_path, allow_pickle=False) as h:
                    p, memory, truth = h['probability'], h['memory'], h['truth']
                    memory_names = h['names'].tolist()
                np.testing.assert_array_equal(truth, z[f'truth{i}'])
                q = symmetric_probability(fallback, z[f'x{i}'], state['names'], state['mean'], state['scale'], device)
                fallback_parts.append(np.repeat(viterbi_labels(q, *state['decoder_costs']), step)[:len(truth)])
                expanded = np.repeat(q, step, axis=0)[:len(p)]
                if len(expanded) != len(p):
                    raise ValueError('Half-rate temporal expansion differs from the full-rate clock')
                if expanded.shape[1] != p.shape[1]:
                    raise ValueError('Temporal state/boundary schemas differ')
                data[i, view] = dict(p=p, q=expanded, memory=memory, truth=truth)
                truth_parts.append(truth)
                parent_parts.append(viterbi_labels(p, *parent_costs))
                print(f'External {view} half {i + 1}: robustness inference complete', flush=True)
        baseline[view] = score_parts(truth_parts, parent_parts)
        reproduced[view] = score_parts(truth_parts, fallback_parts)
    expected = {'native':.8374845869297165, 'corrupt':.6870588235294118}
    for view, value in expected.items():
        if abs(reproduced[view]['spell_f1_mean'] - value) > 1e-10:
            raise ValueError('Historical fallback score failed reproduction; do not select a mixture')
    dump(out / 'fallback_reproduction.json', dict(reproduced=reproduced, expected=expected, passed=True,
         fallback_sha256=sha256(fallback_dir / 'selected.pt'), cache_sha256=cache_hashes))
    records = []
    for mode in PLAN['modes']:
        for weight in PLAN['weights']:
            scores = {}
            for view in ('native', 'corrupt'):
                h = [data[i, view] for i in range(2)]
                predicted = [viterbi_labels(fusion_probability(a['p'], a['q'], a['memory'], memory_names, mode, weight),
                                            *parent_costs) for a in h]
                scores[view] = score_parts([a['truth'] for a in h], predicted)
            record = dict(kind='temporal_' + mode, mode=mode, strength=weight, scores=scores,
                          eligible=scores['native']['spell_f1_mean'] >= baseline['native']['spell_f1_mean'] - PLAN['native_spell_f1_max_decline'],
                          mean_spell_f1=float(np.mean([r['spell_f1_mean'] for r in scores.values()])))
            records.append(record); print(json.dumps(record), flush=True)
    best = max([r for r in records if r['eligible']], key=lambda r:(r['mean_spell_f1'], -r['strength'], r['mode'] == 'missing'))
    result = dict(complete=True, correction_family='temporal_fusion', selected=best,
                  parent_scores=baseline, fallback_scores=reproduced, memory_names=memory_names,
                  fallback_dir=PLAN['fallback'], fallback_names=state['names'], fallback_sample_step=step,
                  selected_model_sha256=sha256(fallback_dir / 'selected.pt'),
                  parent_sha256=sha256(ROOT / PLAN['parent'] / 'selected.pt'), decoder_costs=parent_costs,
                  plan=PLAN, source_sha256={name:sha256(ROOT / name) for name in
                    ['possession/spatial_memory.py', 'scripts/research/spatial_memory/run_external.py',
                     'scripts/research/spatial_memory/select_temporal_fusion.py']},
                  data_guard_verified=True, unexpected_competition_access=denied,
                  competition_data_read=False, public_labels_read=False, full_match_competition_score=None)
    dump(out / 'development_results.json', dict(parent=baseline, candidates=records))
    dump(out / 'selection.json', result)
    print('Temporal fusion frozen: ' + json.dumps(best), flush=True)


if __name__ == '__main__':
    main()
