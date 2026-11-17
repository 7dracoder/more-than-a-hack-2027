"""Rehash and reproduce a completed external partial-HAD training selection."""
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
from scripts.research.spatial_memory.train_real_broadcast_memory import allowed_inputs, load_data, VIEWS
from scripts.research.spatial_memory.train_partial_had_memory import evaluate, selection_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    selection = json.loads((args.run / 'selection.json').read_text())
    assert selection['complete'] and selection['initial_reproduced']
    assert not selection['provided_match_used'] and not selection['public_labels_used']
    assert not selection['unexpected_competition_access']
    plan = selection['plan']
    config, inherited = plan['config'], plan['inherited_config']
    weak_study = ROOT / 'data/research/real-broadcast-adaptation-20261114'
    denied = restrict_external(args.run, allowed_inputs(inherited, weak_study) + [ROOT / p for p in plan['caches']])
    for path,digest in {**plan['sources'], **plan['caches']}.items():
        assert sha256(ROOT / path) == digest, path
    initial_dir = ROOT / config['initial_model']
    for key,filename in [('initial_model_sha256','selected.pt'), ('initial_protocol_sha256','fixed_protocol.json'),
                          ('initial_selection_sha256','selection.json')]:
        assert sha256(initial_dir / filename) == plan[key]
    assert sha256(ROOT / config['continuity_model'] / 'selection.json') == plan['continuity_selection_sha256']
    assert sha256(args.run / 'selected.pt') == selection['selected_model_sha256']
    device = torch_setup(args.device)
    import torch
    from scripts.run_local import selected_model
    model,state = selected_model(args.run, device)
    initial = torch.load(initial_dir / 'selected.pt', map_location='cpu', weights_only=False)
    np.testing.assert_array_equal(state['mean'], initial['mean'])
    np.testing.assert_array_equal(state['scale'], initial['scale'])
    assert state['names'] == initial['names'] and state['warm_start_sha256'] == plan['initial_model_sha256']
    expected_membership = inherited['strong_train_matches'] + inherited['weak_train_matches'] + config['partial_training_matches']
    assert state['config']['train_matches'] == expected_membership and len(set(expected_membership)) == 26
    _,_,strong,weak,_ = load_data(inherited, weak_study, state['names'])
    scores,weak_scores = evaluate(model,state,strong,weak,device,config['decoder_penalty'])
    expected = selection['selected']
    for view in VIEWS:
        for metric in ['spell_f1_mean', 'macro_f1']:
            assert abs(scores[view][metric] - expected['scores'][view][metric]) < 1e-10
    for scope in weak_scores:
        for metric in ['weak_ha_accuracy','conditional_ha_accuracy','conditional_ha_macro_f1']:
            assert abs(weak_scores[scope][metric] - expected['weak_scores'][scope][metric]) < 1e-10
    baseline = json.loads((args.run / 'initial_reproduction.json').read_text())['baseline']
    computed = selection_result(scores, weak_scores, baseline)
    assert computed['eligible'] and abs(computed['objective'] - expected['objective']) < 1e-10
    assert state['epoch'] == expected['epoch']
    if expected['fallback']:
        assert state['epoch'] == 0
        assert all(torch.equal(state['model'][k].cpu(), value.cpu()) for k,value in initial['model'].items())
    result = dict(complete=True, selected_scores_reproduced=True, all_input_checksums_verified=True,
        frozen_normalization_verified=True, training_membership_verified=True, selected_epoch=state['epoch'],
        selected_model_sha256=selection['selected_model_sha256'], selection_sha256=sha256(args.run / 'selection.json'),
        fallback=expected['fallback'], scores=scores, weak_scores=weak_scores, objective=computed['objective'],
        guards=computed['guards'], provided_match_used=False, public_labels_used=False,
        unexpected_competition_access=denied, independent_test=False, competition_target_verified=False)
    dump(args.run / 'external_verification.json', result)
    print(json.dumps(dict(complete=True, epoch=state['epoch'], fallback=expected['fallback'], objective=computed['objective'])), flush=True)


if __name__ == '__main__':
    main()
