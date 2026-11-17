"""Reproduce the frozen external selection and recheck every training input hash."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from possession.data import sha256
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump, torch_setup
from scripts.research.spatial_memory.train_real_broadcast_memory import (
    allowed_inputs, load_data, evaluate, joint_selection, TRANSFER)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--weak-study', type=Path, default=ROOT / 'data/research/real-broadcast-adaptation-20261114')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    selection = json.loads((args.run / 'selection.json').read_text())
    plan = selection['plan']
    config = plan['config']
    denied = restrict_external(args.run, allowed_inputs(config, args.weak_study))
    if (not selection['complete'] or selection['provided_match_used'] or selection['public_labels_used'] or
            selection['unexpected_competition_access'] or not selection['initial_reproduced']):
        raise ValueError('Need a completed isolated external training run')
    if sha256(args.run / 'selected.pt') != selection['selected_model_sha256']:
        raise ValueError('Selected checkpoint changed')
    for path, digest in plan['sources'].items():
        if sha256(ROOT / path) != digest:
            raise ValueError('Training source changed')
    for path, digest in {**plan['caches'], **plan['reference_receipts_sha256']}.items():
        if sha256(ROOT / path) != digest:
            raise ValueError('Training/development input changed during the run: ' + path)
    initial_dir = ROOT / config['initial_model']
    if (sha256(initial_dir / 'selected.pt') != plan['initial_model_sha256'] or
            sha256(initial_dir / 'selection.json') != plan['initial_selection_sha256'] or
            sha256(args.weak_study / 'weak_preparation.json') != plan['weak_preparation_sha256']):
        raise ValueError('Initial model or weak preparation changed')
    import torch
    from scripts.run_local import selected_model
    device = torch_setup(args.device)
    model, state = selected_model(args.run, device)
    initial = torch.load(initial_dir / 'selected.pt', map_location='cpu', weights_only=False)
    np.testing.assert_array_equal(state['mean'], initial['mean'])
    np.testing.assert_array_equal(state['scale'], initial['scale'])
    if (state['names'] != initial['names'] or state['warm_start_sha256'] != plan['initial_model_sha256'] or
            state['epoch'] != selection['selected']['source_epoch']):
        raise ValueError('Frozen model metadata differs')
    train_matches = config['strong_train_matches'] + config['weak_train_matches']
    if state['config']['train_matches'] != train_matches or len(train_matches) != len(set(train_matches)):
        raise ValueError('Training membership differs')
    _, _, strong, weak, _ = load_data(config, args.weak_study, state['names'])
    strong_scores, weak_scores = evaluate(model, state, strong, weak, device)
    for view, expected in selection['selected']['scores'].items():
        for metric in ('spell_f1_mean', 'macro_f1'):
            if abs(strong_scores[view][metric] - expected[metric]) > 1e-10:
                raise ValueError('Frozen strong development score failed reproduction')
    for scope, expected in selection['selected']['weak_scores'].items():
        for metric in ('weak_ha_accuracy', 'conditional_ha_accuracy', 'conditional_ha_macro_f1', 'dead_prediction_rate_on_weak_ha'):
            if abs(weak_scores[scope][metric] - expected[metric]) > 1e-10:
                raise ValueError('Frozen weak development score failed reproduction')
    baseline = json.loads((args.run / 'initial_reproduction.json').read_text())
    reference = json.loads((TRANSFER / 'results.json').read_text())['summaries']['detected']['natural_v1']
    computed = joint_selection(strong_scores, weak_scores, baseline['baseline'], reference)
    if abs(computed['objective'] - selection['selected']['objective']) > 1e-10:
        raise ValueError('Frozen joint objective differs')
    if selection['selected']['fallback']:
        if state['epoch'] != 0 or any(not torch.equal(state['model'][key].cpu(), value.cpu()) for key, value in initial['model'].items()):
            raise ValueError('Fallback weights differ from the initial external model')
    elif not computed['eligible'] or not all(computed['guards'].values()):
        raise ValueError('Selected adapted weights fail the predeclared domain guards')
    result = dict(complete=True, selected_epoch=state['epoch'], fallback=selection['selected']['fallback'],
        all_input_checksums_verified=True, frozen_normalization_verified=True, selected_scores_reproduced=True,
        selected_model_sha256=selection['selected_model_sha256'], selection_sha256=sha256(args.run / 'selection.json'),
        strong_scores=strong_scores, weak_scores=weak_scores, joint_objective=computed['objective'],
        computed_guards=computed['guards'], fallback_exception=selection['selected']['fallback'],
        provided_match_used=False, public_labels_used=False, unexpected_competition_access=denied,
        competition_target_achieved=False, independent_test=False)
    dump(args.run / 'external_verification.json', result)
    print(json.dumps(dict(verified=True, epoch=state['epoch'], fallback=result['fallback'],
        strong_mean=computed['strong_mean'], joint_objective=computed['objective'])), flush=True)


if __name__ == '__main__':
    main()
