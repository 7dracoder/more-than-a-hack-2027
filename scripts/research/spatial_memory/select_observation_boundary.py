"""Externally test a clock-corrected observation-loss expert transition head."""
import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from possession.data import sha256
from possession.observation_gate import mix_probabilities, observation_loss
from possession.observation_boundary import mix_boundaries
from possession.uncertain_decode import decode_labels
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump, score_parts, torch_setup
from scripts.research.spatial_memory.train_real_broadcast_memory import allowed_inputs, load_data, VIEWS
from scripts.research.spatial_memory.train_natural_visibility import model_matrix
from scripts.research.spatial_memory.audit_broadcast_transfer import weak_counts, summarize_counts

MODEL = ROOT / 'models/coreorigin-real-broadcast-v3'
FALLBACK = ROOT / 'models/coreorigin-continuity-v4'
PROTOCOL = dict(boundary_strengths=[0., .5, 1.], penalties=[8., 0.],
    clock_conversion='Expert change j covers [(j-1)*2+1,j*2]. Equal per-frame hazard 1-sqrt(1-q[j]). Unbracketed endpoints retain parent.',
    gate='Existing observation-loss gate. State mixture unchanged, expert strength 1.',
    objective='Maximum mean native/masked/mixed strong development spell F1; ties retain v4 fallback (boundary strength 0, penalty 8).',
    native_max_decline=0., weak_ha_max_decline=.005, missing_weak_ha_max_decline=.01,
    models_retrained=False, provided_match_used=False, public_labels_used=False,
    limitation='Exploratory calibration on reused external development fixtures, not independent or competition performance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'selection.json').exists():
        raise ValueError('Preserve the frozen selection')
    origin = json.loads((MODEL / 'selection.json').read_text())
    fallback = json.loads((FALLBACK / 'selection.json').read_text())
    assert origin['complete'] and fallback['complete']
    assert origin['selected_model_sha256'] == fallback['selected_model_sha256'] == sha256(MODEL / 'selected.pt')
    config = origin['plan']['config']
    denied = restrict_external(args.out, allowed_inputs(config, ROOT / 'data/research/real-broadcast-adaptation-20261114'))
    sources = ['scripts/research/spatial_memory/select_observation_boundary.py', 'possession/observation_boundary.py',
        'possession/uncertain_decode.py', 'possession/observation_gate.py',
        'scripts/research/spatial_memory/external_guard.py',
        'scripts/research/spatial_memory/train_real_broadcast_memory.py']
    plan = dict(protocol=PROTOCOL, sources={p:sha256(ROOT / p) for p in sources},
        model_sha256=sha256(MODEL / 'selected.pt'), model_protocol_sha256=sha256(MODEL / 'fixed_protocol.json'),
        preceding_selection_sha256=sha256(MODEL / 'selection.json'), fallback_selection_sha256=sha256(FALLBACK / 'selection.json'))
    if (args.out / 'plan.json').exists():
        assert json.loads((args.out / 'plan.json').read_text()) == plan
    dump(args.out / 'plan.json', plan)
    device = torch_setup(args.device)
    from scripts.run_local import selected_model
    from possession.temporal import symmetric_probability
    model,state = selected_model(MODEL, device)
    _,_,strong,weak,_ = load_data(config, ROOT / 'data/research/real-broadcast-adaptation-20261114', state['names'])
    arrays = []
    directory = args.out / 'probabilities'
    directory.mkdir(exist_ok=True)
    for domain, parts in [('strong', strong), ('weak', weak)]:
        for i,h in enumerate(parts):
            q = symmetric_probability(model, model_matrix(h['x']), state['names'], state['mean'], state['scale'], device)
            p = mix_probabilities(h['parent'], q, h['x'][:,960:], h['memory_names'], 1.)
            gate = np.repeat(observation_loss(h['x'][:,960:], h['memory_names']), 2)[:len(p)]
            np.savez_compressed(directory / f'{domain}_{i}.npz', probability=p, expert=q, gate=gate,
                memory=h['x'][:,960:], memory_names=np.asarray(h['memory_names']))
            arrays.append(dict(domain=domain, part=h, probability=p, expert=q, gate=gate))
            print(f'Frozen boundary study input: {domain} {i + 1}', flush=True)
    dump(args.out / 'probability_freeze.json', dict(files={p.name:sha256(p) for p in directory.glob('*.npz')},
        model_sha256=plan['model_sha256']))
    baseline = fallback['selected']
    rows = []
    for boundary_strength,penalty in itertools.product(PROTOCOL['boundary_strengths'], PROTOCOL['penalties']):
        for h in arrays:
            part = h['part']
            p = mix_boundaries(h['probability'], h['expert'], part['x'][:,960:], part['memory_names'], boundary_strength)
            h['labels'] = decode_labels(p, h['gate'], penalty, *state['decoder_costs'])
        scores = {}
        for view in VIEWS:
            parts = [h for h in arrays if h['domain'] == 'strong' and h['part']['view'] == view]
            scores[view] = score_parts([h['part']['truth'] for h in parts], [h['labels'] for h in parts])
        weak_rows = {'all_known':[], 'ball_missing':[]}
        for h in arrays:
            if h['domain'] != 'weak':
                continue
            part = h['part']
            for scope, mask in [('all_known', np.ones(len(part['ticks']), bool)), ('ball_missing', ~part['seen'])]:
                weak_rows[scope].append(weak_counts(h['probability'][part['ticks']], h['labels'][part['ticks']], part['truth'], mask))
        weak_scores = {scope:summarize_counts(values) for scope,values in weak_rows.items()}
        row = dict(kind='boundary_observation_loss', strength=1., boundary_strength=boundary_strength, penalty=penalty,
            scores=scores, weak_scores=weak_scores, objective=float(np.mean([scores[v]['spell_f1_mean'] for v in VIEWS])))
        if boundary_strength == 0:
            expected = baseline if penalty == 8 else origin['selected']
            for view in VIEWS:
                assert abs(scores[view]['spell_f1_mean'] - expected['scores'][view]['spell_f1_mean']) < 1e-10
            for scope in weak_scores:
                assert abs(weak_scores[scope]['weak_ha_accuracy'] - expected['weak_scores'][scope]['weak_ha_accuracy']) < 1e-10
        row['guards'] = dict(native=scores['native']['spell_f1_mean'] >= baseline['scores']['native']['spell_f1_mean'] - 1e-12,
            weak_ha=weak_scores['all_known']['weak_ha_accuracy'] >= baseline['weak_scores']['all_known']['weak_ha_accuracy'] - PROTOCOL['weak_ha_max_decline'],
            weak_missing=weak_scores['ball_missing']['weak_ha_accuracy'] >= baseline['weak_scores']['ball_missing']['weak_ha_accuracy'] - PROTOCOL['missing_weak_ha_max_decline'])
        row['eligible'] = all(row['guards'].values())
        rows.append(row)
        print(json.dumps(row), flush=True)
    best = max((r for r in rows if r['eligible']), key=lambda r:r['objective'])
    for name,digest in plan['sources'].items():
        assert sha256(ROOT / name) == digest
    dump(args.out / 'selection.json', dict(complete=True, selected=best, candidates=rows, plan=plan,
        selected_model_sha256=plan['model_sha256'], parent_sha256=origin['parent_sha256'],
        parent_protocol_sha256=origin['parent_protocol_sha256'], model_weights_changed=False,
        zero_boundary_controls_reproduced=True, provided_match_used=False, public_labels_used=False,
        unexpected_competition_access=denied, independent_test=False, competition_target_achieved=False))
    print(json.dumps(dict(selected=best['boundary_strength'], penalty=best['penalty'], objective=best['objective'])), flush=True)


if __name__ == '__main__':
    main()
