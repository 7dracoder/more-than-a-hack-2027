"""Rebuild every direct weight repair and reproduce all external scores/guards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from possession.data import sha256
from possession.input_alias_repair import repair_state
from scripts.research.spatial_memory.external_development_features import load
from scripts.research.spatial_memory.run_external import dump,torch_setup
from scripts.research.spatial_memory.train_partial_had_memory import evaluate,selection_result
from scripts.research.spatial_memory.train_real_broadcast_memory import VIEWS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    args.study = args.study.resolve()
    selection = json.loads((args.study / 'selection.json').read_text())
    assert selection['complete'] and selection['zero_repair_control_reproduced']
    assert not selection['gradient_training_performed'] and selection['normalization_unchanged']
    assert not selection['provided_match_used'] and not selection['public_labels_used']
    assert not selection['unexpected_competition_access']
    plan = selection['plan']
    base = ROOT / 'models/coreorigin-continuity-v4'
    for key,name in [('initial_model_sha256','selected.pt'),('initial_protocol_sha256','fixed_protocol.json'),('initial_selection_sha256','selection.json')]:
        assert sha256(base / name)==plan[key]
    baseline = json.loads((base / 'selection.json').read_text())
    reference_path = ROOT / 'models/coreorigin-real-broadcast-v3/selection.json'
    assert sha256(reference_path)==plan['reference_selection_sha256']
    reference = json.loads(reference_path.read_text())
    strong,weak,hashes,denied = load(args.study,reference)
    assert hashes==plan['inputs']
    for path,digest in {**plan['sources'],**plan['inputs']}.items():
        assert sha256(ROOT / path)==digest,path
    device = torch_setup(args.device)
    import torch
    from scripts.run_local import selected_model
    initial = torch.load(base / 'selected.pt',map_location='cpu',weights_only=False)
    reproduced = []
    for expected in selection['candidates']:
        directory = args.study / f"candidate_{expected['candidate_index']}"
        assert sha256(directory / 'selected.pt')==expected['model_sha256']
        model,state = selected_model(directory,device)
        rebuilt,receipt = repair_state(initial,expected['repair_strength'])
        assert receipt==expected['repair']==state['input_alias_repair']
        for key,value in rebuilt['model'].items():
            assert torch.equal(value.cpu(),state['model'][key].cpu()),key
        np.testing.assert_array_equal(state['mean'],initial['mean'])
        np.testing.assert_array_equal(state['scale'],initial['scale'])
        fingerprint = hashlib.sha256(json.dumps(dict(plan=plan,strength=expected['repair_strength']),sort_keys=True).encode()).hexdigest()
        assert state['fingerprint']==fingerprint
        scores,weak_scores = evaluate(model,state,strong,weak,device,8.)
        result = selection_result(scores,weak_scores,baseline['selected'])
        result['guards']['strong_macro'] = all(scores[v]['macro_f1']>=baseline['selected']['scores'][v]['macro_f1']-.005 for v in VIEWS)
        result['eligible'] = all(result['guards'].values())
        for key,value in result.items():
            assert value==expected[key],(expected['repair_strength'],key)
        reproduced.append(dict(strength=expected['repair_strength'],objective=result['objective'],eligible=result['eligible']))
    best = max((r for r in selection['candidates'] if r['eligible']),key=lambda r:r['objective'])
    assert best==selection['selected']
    assert sha256(args.study / 'selected.pt')==best['model_sha256']
    result = dict(complete=True,all_repair_tensors_rebuilt_exactly=True,all_candidate_scores_reproduced=True,
        all_guards_recomputed=True,source_and_input_hashes_verified=True,normalization_unchanged=True,
        selected_strength=best['repair_strength'],selected_model_sha256=best['model_sha256'],candidates=reproduced,
        selection_sha256=sha256(args.study / 'selection.json'),verifier_sha256=sha256(Path(__file__)),
        provided_match_used=False,public_labels_used=False,unexpected_competition_access=denied,
        independent_test=False,full_match_spell_f1=None,competition_target_verified=False)
    dump(args.study / 'external_verification.json',result)
    print(json.dumps(dict(complete=True,strength=best['repair_strength'],objective=best['objective'])),flush=True)


if __name__ == '__main__':
    main()
