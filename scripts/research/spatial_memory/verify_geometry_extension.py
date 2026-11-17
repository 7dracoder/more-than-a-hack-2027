"""Recompute external input normalization and reproduce a selected expanded expert."""
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
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump,torch_setup
from scripts.research.spatial_memory.train_real_broadcast_memory import allowed_inputs,load_data,VIEWS
from scripts.research.spatial_memory.train_geometry_extension import added_normalization,evaluate
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
    assert len(state['names']) == 1088 and state['names'][:664]+state['names'][960:] == initial['names']
    positions = feature_positions(initial['names'],state['names'])
    np.testing.assert_array_equal(state['mean'][positions],initial['mean'])
    np.testing.assert_array_equal(state['scale'][positions],initial['scale'])
    assert state['warm_start_sha256'] == plan['initial_model_sha256']
    assert state['config']['train_matches'] == split['strong_train_matches']+split['weak_train_matches']
    assert len(set(state['config']['train_matches'])) == 24
    strong,weak,strong_dev,weak_dev,_ = load_data(split,weak_study,initial['names'])
    # Locate new columns from the frozen cache ledger, not a mutable global directory.
    for h in weak:
        name = f"{h['match']}_{h['half']}_detected.npy"
        paths = [ROOT / p for p in plan['caches'] if Path(p).name == name and 'weak_extension' in Path(p).parts]
        assert len(paths) == 1
        h['extra'] = np.load(paths[0],mmap_mode='r')
    mean,scale = added_normalization(strong,weak,state['names'],config['normalization_group_weights'])
    np.testing.assert_array_equal(state['mean'][664:960],mean)
    np.testing.assert_array_equal(state['scale'][664:960],scale)
    with np.load(args.run / 'added_normalization.npz') as normalized:
        np.testing.assert_array_equal(normalized['mean'],mean)
        np.testing.assert_array_equal(normalized['scale'],scale)
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
    result = dict(complete=True,all_input_checksums_verified=True,old_normalization_unchanged=True,
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
