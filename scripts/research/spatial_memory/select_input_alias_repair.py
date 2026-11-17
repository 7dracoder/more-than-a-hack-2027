"""Test three fixed metadata-derived projection repairs on external fixtures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
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

BASE = ROOT / 'models/coreorigin-continuity-v4'
PROTOCOL = dict(strengths=[0.,.5,1.],decoder_penalty=8.,model_training=False,
    edits='Use frozen base/memory normalization and the observed name-alias mapping. Interpolate their projection preactivations with a closed-form column scale and bias shift. The .5 value is the existing training swap probability; 1 is a full-alias ablation. No coefficients are fitted to match data.',
    objective='Maximum mean strong three-view spell F1 with unchanged native spell F1, strong-view macro decline <=0.005, weak decoded H/A decline <=0.005 overall / <=0.01 missing-ball. Ties retain zero.',
    limitation='Interpolate first-layer preactivations, not complete nonlinear-network outputs. Exploratory reused external development fixtures, not independent or BAR-ATH performance.',
    provided_match_used=False,public_labels_used=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out / 'selection.json').exists():
        raise ValueError('Preserve completed repair study')
    baseline = json.loads((BASE / 'selection.json').read_text())
    reference = json.loads((ROOT / 'models/coreorigin-real-broadcast-v3/selection.json').read_text())
    assert sha256(BASE / 'selected.pt') == baseline['selected_model_sha256'] == reference['selected_model_sha256']
    strong,weak,hashes,denied = load(args.out,reference)
    for h in strong+weak:
        np.testing.assert_array_equal(h['x'][:,0],h['x'][:,960])
        np.testing.assert_array_equal(h['x'][:,140],h['x'][:,1024])
        assert np.isin(h['x'][:,0],[0.,1.]).all() and (h['x'][:,140]==1).all()
    sources = ['scripts/research/spatial_memory/select_input_alias_repair.py',
        'scripts/research/spatial_memory/external_development_features.py',
        'scripts/research/spatial_memory/train_partial_had_memory.py',
        'scripts/research/spatial_memory/external_guard.py','possession/input_alias_repair.py',
        'possession/temporal.py','possession/boundary.py','possession/observation_gate.py',
        'possession/uncertain_decode.py','scripts/run_local.py']
    plan = dict(protocol=PROTOCOL,sources={p:sha256(ROOT / p) for p in sources},inputs=hashes,
        initial_model_sha256=sha256(BASE / 'selected.pt'),initial_protocol_sha256=sha256(BASE / 'fixed_protocol.json'),
        initial_selection_sha256=sha256(BASE / 'selection.json'),reference_selection_sha256=sha256(ROOT / 'models/coreorigin-real-broadcast-v3/selection.json'),
        raw_aliases_identical_on_all_development_inputs=True,provided_match_used=False,public_labels_used=False)
    if (args.out / 'plan.json').exists():
        assert json.loads((args.out / 'plan.json').read_text())==plan
    dump(args.out / 'plan.json',plan)
    import torch
    from scripts.run_local import model_from_state,save_torch
    device = torch_setup(args.device)
    initial = torch.load(BASE / 'selected.pt',map_location='cpu',weights_only=False)
    rows = []
    # Freeze every edited checkpoint before looking at any candidate scores.
    for i,strength in enumerate(PROTOCOL['strengths']):
        state,receipt = repair_state(initial,strength)
        fingerprint = hashlib.sha256(json.dumps(dict(plan=plan,strength=strength),sort_keys=True).encode()).hexdigest()
        state.update(fingerprint=fingerprint,input_alias_repair=receipt,warm_start_sha256=plan['initial_model_sha256'])
        directory = args.out / f'candidate_{i}'
        directory.mkdir(exist_ok=True)
        save_torch(directory / 'selected.pt',state)
        dump(directory / 'fixed_protocol.json',dict(config=state['config'],fingerprint=fingerprint,preparation=plan,repair=receipt))
    for i,strength in enumerate(PROTOCOL['strengths']):
        directory = args.out / f'candidate_{i}'
        state = torch.load(directory / 'selected.pt',map_location='cpu',weights_only=False)
        model = model_from_state(state,device)
        scores,weak_scores = evaluate(model,state,strong,weak,device,8.)
        result = selection_result(scores,weak_scores,baseline['selected'])
        result['guards']['strong_macro'] = all(scores[v]['macro_f1']>=baseline['selected']['scores'][v]['macro_f1']-.005 for v in VIEWS)
        result['eligible'] = all(result['guards'].values())
        row = dict(kind='uncertain_observation_loss',strength=1.,penalty=8.,repair_strength=strength,
            candidate_index=i,repair=state['input_alias_repair'],model_sha256=sha256(directory / 'selected.pt'),**result)
        if strength==0:
            for view in VIEWS:
                assert abs(scores[view]['spell_f1_mean']-baseline['selected']['scores'][view]['spell_f1_mean'])<1e-10
            for scope in weak_scores:
                for metric in ('weak_ha_accuracy','conditional_ha_accuracy','conditional_ha_macro_f1'):
                    assert abs(weak_scores[scope][metric]-baseline['selected']['weak_scores'][scope][metric])<1e-10
        rows.append(row)
        print(json.dumps(row),flush=True)
    best = max((r for r in rows if r['eligible']),key=lambda r:r['objective'])
    for path,digest in {**plan['sources'],**plan['inputs']}.items():
        assert sha256(ROOT / path)==digest,path
    chosen = args.out / f"candidate_{best['candidate_index']}"
    shutil.copyfile(chosen / 'selected.pt',args.out / 'selected.pt')
    shutil.copyfile(chosen / 'fixed_protocol.json',args.out / 'fixed_protocol.json')
    dump(args.out / 'selection.json',dict(complete=True,selected=best,candidates=rows,plan=plan,
        selected_model_sha256=best['model_sha256'],parent_sha256=baseline['parent_sha256'],
        parent_protocol_sha256=baseline['parent_protocol_sha256'],zero_repair_control_reproduced=True,
        gradient_training_performed=False,normalization_unchanged=True,provided_match_used=False,
        public_labels_used=False,unexpected_competition_access=denied,independent_test=False,competition_target_achieved=False))
    print(json.dumps(dict(repair_strength=best['repair_strength'],objective=best['objective'])),flush=True)


if __name__ == '__main__':
    main()
