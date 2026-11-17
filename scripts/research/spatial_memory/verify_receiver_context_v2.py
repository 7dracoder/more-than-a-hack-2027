"""Verify receiver-context projection constraints and external possession scores."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from possession.data import sha256
from possession.feature_extension import extend_state,feature_positions
from possession.frozen_projection import assert_inherited_unchanged
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump,torch_setup
from scripts.research.spatial_memory.train_real_broadcast_memory import allowed_inputs,load_data,VIEWS
from scripts.research.spatial_memory.train_receiver_context_v2 import added_normalization,evaluate,augment_parts
from scripts.research.spatial_memory.train_partial_had_memory import selection_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    selection = json.loads((args.run / 'selection.json').read_text())
    assert selection['complete'] and selection['initial_reproduced']
    assert not selection['provided_match_used'] and not selection['public_labels_used']
    assert not selection['unexpected_competition_access']
    plan = selection['plan']
    config,split = plan['config'],plan['split']
    config_path = ROOT / 'configs/receiver_context_v2_adaptation.json'
    assert sha256(config_path) == plan['config_sha256'] and json.loads(config_path.read_text()) == config
    weak_study = ROOT / 'data/research/real-broadcast-adaptation-20261114'
    denied = restrict_external(args.run,allowed_inputs(split,weak_study)+[ROOT / p for p in plan['caches']])
    for path,digest in {**plan['sources'],**plan['caches']}.items():
        assert sha256(ROOT / path) == digest,path
    initial_dir = ROOT / config['initial_model']
    for key,name in [('initial_model_sha256','selected.pt'),('initial_protocol_sha256','fixed_protocol.json'),('initial_selection_sha256','selection.json')]:
        assert sha256(initial_dir / name) == plan[key]
    assert sha256(ROOT / config['continuity_model'] / 'selection.json') == plan['fallback_selection_sha256']
    assert sha256(args.run / 'selected.pt') == selection['selected_model_sha256']
    assert sha256(args.run / 'added_normalization.npz') == plan['normalization_sha256']
    device = torch_setup(args.device)
    import torch
    from scripts.run_local import selected_model
    model,state = selected_model(args.run,device)
    initial = torch.load(initial_dir / 'selected.pt',map_location='cpu',weights_only=False)
    assert len(state['names']) == 803 and state['names'][:792] == initial['names']
    positions = feature_positions(initial['names'],state['names'])
    np.testing.assert_array_equal(state['mean'][positions],initial['mean'])
    np.testing.assert_array_equal(state['scale'][positions],initial['scale'])
    assert state['warm_start_sha256'] == plan['initial_model_sha256']
    assert state['config']['train_matches'] == split['strong_train_matches']
    assert len(set(state['config']['train_matches'])) == 6
    strong,weak,strong_dev,weak_dev,_ = load_data(split,weak_study,initial['names'])
    studies={ (ROOT/p).parent for p in plan['caches'] if Path(p).name=='preparation.json' and 'receiver-context-v2-20261114' in Path(p).parts }
    assert len(studies)==1
    augment_parts(strong+strong_dev+weak_dev,studies.pop(),{})
    weak=[]
    mean,scale = added_normalization(strong,weak,state['names'],config['normalization_group_weights'])
    np.testing.assert_array_equal(state['mean'][792:803],mean)
    np.testing.assert_array_equal(state['scale'][792:803],scale)
    with np.load(args.run / 'added_normalization.npz') as normalized:
        np.testing.assert_array_equal(normalized['mean'],mean)
        np.testing.assert_array_equal(normalized['scale'],scale)
    extended = extend_state(initial,state['names'],mean,scale)
    added = np.arange(792,803)
    assert_inherited_unchanged(state['model'],extended['model'],added)
    latest = torch.load(args.run / 'latest.pt',map_location='cpu',weights_only=False)
    assert latest['epoch'] == config['epochs'] == 30 and len(latest['history']) == 30
    assert_inherited_unchanged(latest['model'],extended['model'],added)
    groups = latest['optimizer']['param_groups']
    assert len(groups)==1 and len(groups[0]['params'])==1 and groups[0]['weight_decay']==0.
    momentum = latest['optimizer']['state'][groups[0]['params'][0]]
    for key in ('exp_avg','exp_avg_sq'):
        assert torch.count_nonzero(momentum[key][:,positions,:]) == 0
    assert len(added)*state['model']['project.weight'].shape[0] == config['trainable_coefficients'] == 1408
    scores,weak_scores = evaluate(model,state,strong_dev,weak_dev,device)
    expected = selection['selected']
    for view in VIEWS:
        for metric in ('spell_f1_mean','macro_f1'):
            assert abs(scores[view][metric]-expected['scores'][view][metric]) < 1e-10
    for scope in weak_scores:
        for metric in ('weak_ha_accuracy','conditional_ha_accuracy','conditional_ha_macro_f1'):
            assert abs(weak_scores[scope][metric]-expected['weak_scores'][scope][metric]) < 1e-10
    baseline = json.loads((args.run / 'initial_reproduction.json').read_text())['baseline']
    computed = selection_result(scores,weak_scores,baseline)
    assert computed['eligible'] and abs(computed['objective']-expected['objective']) < 1e-10
    assert state['epoch'] == expected['epoch']
    if expected['fallback']:
        assert state['epoch'] == 0
        extended = extend_state(initial,state['names'],mean,scale)
        assert all(torch.equal(state['model'][k].cpu(),value.cpu()) for k,value in extended['model'].items())
    result = dict(complete=True,inherited_tensors_exact_in_selected_and_final=True,inherited_optimizer_moments_zero=True,all_input_checksums_verified=True,old_normalization_unchanged=True,
        new_normalization_recomputed_from_training_only=True,training_membership_verified=True,
        selected_scores_reproduced=True,selected_epoch=state['epoch'],fallback=expected['fallback'],
        selected_model_sha256=selection['selected_model_sha256'],selection_sha256=sha256(args.run / 'selection.json'),
        source_sha256=sha256(Path(__file__)),scores=scores,weak_scores=weak_scores,objective=computed['objective'],
        guards=computed['guards'],provided_match_used=False,public_labels_used=False,
        unexpected_competition_access=denied,independent_test=False,competition_target_verified=False)
    dump(args.run / 'external_verification.json',result)
    print(json.dumps(dict(complete=True,epoch=state['epoch'],fallback=expected['fallback'],objective=computed['objective'])),flush=True)


if __name__ == '__main__':
    main()
