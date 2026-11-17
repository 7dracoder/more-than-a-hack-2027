"""Independently reproduce the frozen external transition-head calibration."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from possession.data import sha256
from possession.uncertain_decode import decode_labels
from possession.observation_boundary import mix_boundaries
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump, score_parts
from scripts.research.spatial_memory.train_real_broadcast_memory import allowed_inputs, STRONG, TRANSFER, VIEWS
from scripts.research.spatial_memory.audit_broadcast_transfer import weak_counts, summarize_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    origin_dir = ROOT / 'models/coreorigin-real-broadcast-v3'
    origin = json.loads((origin_dir / 'selection.json').read_text())
    config = origin['plan']['config']
    denied = restrict_external(args.study, allowed_inputs(config, ROOT / 'data/research/real-broadcast-adaptation-20261114'))
    selection = json.loads((args.study / 'selection.json').read_text())
    plan = selection['plan']
    assert selection['complete'] and not selection['provided_match_used'] and not selection['public_labels_used']
    assert not selection['unexpected_competition_access'] and not selection['model_weights_changed']
    for key, filename in [('model_sha256', 'selected.pt'), ('model_protocol_sha256', 'fixed_protocol.json'),
                          ('preceding_selection_sha256', 'selection.json')]:
        assert sha256(origin_dir / filename) == plan[key]
    fallback_path = ROOT / 'models/coreorigin-continuity-v4/selection.json'
    assert sha256(fallback_path) == plan['fallback_selection_sha256']
    baseline = json.loads(fallback_path.read_text())['selected']
    for path, digest in plan['sources'].items():
        assert sha256(ROOT / path) == digest, path
    freeze = json.loads((args.study / 'probability_freeze.json').read_text())
    for filename, digest in freeze['files'].items():
        assert sha256(args.study / 'probabilities' / filename) == digest
    import torch
    state = torch.load(origin_dir / 'selected.pt', map_location='cpu', weights_only=False)
    strong, weak = [], []
    for match in config['strong_development_matches']:
        for half in (0, 1):
            for view in VIEWS:
                path = STRONG / 'cache' / f'{match}_{half}_{view}_truth.npy'
                assert sha256(path) == origin['plan']['caches'][str(path.relative_to(ROOT))]
                strong.append(dict(view=view, truth=np.load(path)))
    for match in config['weak_development_matches']:
        for half in (1, 2):
            path = TRANSFER / 'features' / f'{match}_{half}_detected_weak.npz'
            assert sha256(path) == origin['plan']['caches'][str(path.relative_to(ROOT))]
            with np.load(path) as data:
                weak.append(dict(truth=data['truth'], ticks=data['ticks'], seen=data['detected_ball_seen']))
    for domain, parts in [('strong', strong), ('weak', weak)]:
        for i, part in enumerate(parts):
            with np.load(args.study / 'probabilities' / f'{domain}_{i}.npz') as data:
                part.update(probability=data['probability'], gate=data['gate'], expert=data['expert'], memory=data['memory'], memory_names=data['memory_names'].tolist())
    reproduced = []
    for expected in selection['candidates']:
        scores = {}
        for view in VIEWS:
            parts = [h for h in strong if h['view'] == view]
            scores[view] = score_parts([h['truth'] for h in parts],
                [decode_labels(mix_boundaries(h['probability'], h['expert'], h['memory'], h['memory_names'], expected['boundary_strength']), h['gate'], expected['penalty'], *state['decoder_costs']) for h in parts])
            for metric in ['spell_f1_mean', 'macro_f1']:
                assert abs(scores[view][metric] - expected['scores'][view][metric]) < 1e-12
        weak_rows = {'all_known':[], 'ball_missing':[]}
        for h in weak:
            labels = decode_labels(mix_boundaries(h['probability'], h['expert'], h['memory'], h['memory_names'], expected['boundary_strength']), h['gate'], expected['penalty'], *state['decoder_costs'])
            for scope, mask in [('all_known', np.ones(len(h['ticks']), bool)), ('ball_missing', ~h['seen'])]:
                weak_rows[scope].append(weak_counts(h['probability'][h['ticks']], labels[h['ticks']], h['truth'], mask))
        weak_scores = {scope:summarize_counts(rows) for scope, rows in weak_rows.items()}
        for scope in weak_scores:
            for metric, value in weak_scores[scope].items():
                if isinstance(value, (int, float)):
                    assert abs(value - expected['weak_scores'][scope][metric]) < 1e-12
        objective = float(np.mean([scores[v]['spell_f1_mean'] for v in VIEWS]))
        assert abs(objective - expected['objective']) < 1e-12
        guards = dict(native=scores['native']['spell_f1_mean'] >= baseline['scores']['native']['spell_f1_mean'] - 1e-12,
            weak_ha=weak_scores['all_known']['weak_ha_accuracy'] >= baseline['weak_scores']['all_known']['weak_ha_accuracy'] - .005,
            weak_missing=weak_scores['ball_missing']['weak_ha_accuracy'] >= baseline['weak_scores']['ball_missing']['weak_ha_accuracy'] - .01)
        assert guards == expected['guards'] and all(guards.values()) == expected['eligible']
        reproduced.append(dict(boundary_strength=expected['boundary_strength'], penalty=expected['penalty'], objective=objective))
    eligible = [r for r in selection['candidates'] if r['eligible']]
    assert max(eligible, key=lambda r:r['objective']) == selection['selected']
    result = dict(complete=True, all_declared_candidates_reproduced=True, frozen_probability_hashes_verified=True,
        external_target_hashes_verified=True, source_hashes_verified=True, weights_unchanged=True,
        candidates=reproduced, selection_sha256=sha256(args.study / 'selection.json'),
        provided_match_used=False, public_labels_used=False, unexpected_competition_access=denied,
        independent_test=False, full_match_spell_f1=None, competition_target_verified=False)
    dump(args.study / 'external_verification.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
