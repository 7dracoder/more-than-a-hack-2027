"""Calibrate one continuity parameter using external development data only."""
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
from possession.observation_gate import mix_probabilities, observation_loss
from possession.uncertain_decode import decode_labels
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump, score_parts, torch_setup
from scripts.research.spatial_memory.train_natural_visibility import model_matrix
from scripts.research.spatial_memory.train_real_broadcast_memory import allowed_inputs, load_data, VIEWS
from scripts.research.spatial_memory.audit_broadcast_transfer import weak_counts, summarize_counts

MODEL = ROOT / 'models/coreorigin-real-broadcast-v3'
PLAN = dict(penalties=[0., 2., 4., 8.], gate='Existing binary observation-loss gate on the expert input clock.',
    uncertainty='1.5*(1-max(H,A,D)), clipped to [0,1].',
    extra_cost='penalty*gate*uncertainty subtracted from each off-diagonal transition log-score.',
    objective='Maximum mean native/masked/mixed strong development spell F1; ties keep penalty zero.',
    native_max_decline=0., weak_ha_max_decline=.005, missing_weak_ha_max_decline=.01,
    models_retrained=False, provided_match_used=False, public_labels_used=False,
    limitation='Exploratory calibration on already opened external development fixtures; not independent or competition performance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--weak-study', type=Path, default=ROOT / 'data/research/real-broadcast-adaptation-20261114')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    origin = json.loads((MODEL / 'selection.json').read_text())
    config = origin['plan']['config']
    denied = restrict_external(args.out, allowed_inputs(config, args.weak_study))
    if (not origin['complete'] or origin['provided_match_used'] or origin['public_labels_used'] or origin['unexpected_competition_access']):
        raise ValueError('Need an externally frozen model')
    if (args.out / 'selection.json').exists():
        raise ValueError('Decoder selection is already frozen')
    sources = ['scripts/research/spatial_memory/select_uncertain_decode.py', 'possession/uncertain_decode.py',
        'possession/observation_gate.py', 'scripts/research/spatial_memory/external_guard.py',
        'scripts/research/spatial_memory/train_real_broadcast_memory.py']
    freeze = dict(protocol=PLAN, sources={p:sha256(ROOT / p) for p in sources},
        model_sha256=sha256(MODEL / 'selected.pt'), model_protocol_sha256=sha256(MODEL / 'fixed_protocol.json'),
        preceding_selection_sha256=sha256(MODEL / 'selection.json'))
    if freeze['model_sha256'] != origin['selected_model_sha256']:
        raise ValueError('Externally trained weights changed')
    path = args.out / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != freeze:
        raise ValueError('Decoder protocol changed')
    dump(path, freeze)
    device = torch_setup(args.device)
    from scripts.run_local import selected_model
    from possession.temporal import symmetric_probability
    model, state = selected_model(MODEL, device)
    _, _, strong, weak, _ = load_data(config, args.weak_study, state['names'])
    arrays = []
    cache = args.out / 'probabilities'
    cache.mkdir(exist_ok=True)
    for domain, parts in [('strong', strong), ('weak', weak)]:
        for i, h in enumerate(parts):
            q = symmetric_probability(model, model_matrix(h['x']), state['names'], state['mean'], state['scale'], device)
            p = mix_probabilities(h['parent'], q, h['x'][:, 960:], h['memory_names'], 1.)
            gate = np.repeat(observation_loss(h['x'][:, 960:], h['memory_names']), 2)[:len(p)]
            target = cache / f'{domain}_{i}.npz'
            np.savez_compressed(target, probability=p, gate=gate)
            arrays.append(dict(domain=domain, part=h, probability=p, gate=gate))
            print(f'Frozen decoder input: {domain} {i + 1}', flush=True)
    dump(args.out / 'probability_freeze.json', dict(files={p.name:sha256(p) for p in cache.glob('*.npz')},
        model_sha256=freeze['model_sha256']))
    rows, baseline = [], None
    for penalty in PLAN['penalties']:
        scores, weak_rows = {}, {'all_known':[], 'ball_missing':[]}
        for view in VIEWS:
            truth, predicted = [], []
            for h in arrays:
                part = h['part']
                if h['domain'] == 'strong' and part['view'] == view:
                    labels = decode_labels(h['probability'], h['gate'], penalty, *state['decoder_costs'])
                    truth.append(part['truth']); predicted.append(labels)
            scores[view] = score_parts(truth, predicted)
        for h in arrays:
            if h['domain'] != 'weak':
                continue
            part = h['part']
            labels = decode_labels(h['probability'], h['gate'], penalty, *state['decoder_costs'])
            for scope, mask in [('all_known', np.ones(len(part['ticks']), bool)), ('ball_missing', ~part['seen'])]:
                weak_rows[scope].append(weak_counts(h['probability'][part['ticks']], labels[part['ticks']], part['truth'], mask))
        weak_scores = {scope:summarize_counts(values) for scope, values in weak_rows.items()}
        row = dict(kind='uncertain_observation_loss', strength=1., penalty=penalty, scores=scores, weak_scores=weak_scores,
            objective=float(np.mean([scores[v]['spell_f1_mean'] for v in VIEWS])))
        if penalty == 0:
            for view in VIEWS:
                if abs(scores[view]['spell_f1_mean'] - origin['selected']['scores'][view]['spell_f1_mean']) > 1e-10:
                    raise ValueError('Zero-penalty strong control failed reproduction')
            for scope in weak_scores:
                if abs(weak_scores[scope]['weak_ha_accuracy'] - origin['selected']['weak_scores'][scope]['weak_ha_accuracy']) > 1e-10:
                    raise ValueError('Zero-penalty weak control failed reproduction')
            baseline = row
        row['guards'] = dict(native=scores['native']['spell_f1_mean'] >= baseline['scores']['native']['spell_f1_mean'] - 1e-12,
            weak_ha=weak_scores['all_known']['weak_ha_accuracy'] >= baseline['weak_scores']['all_known']['weak_ha_accuracy'] - PLAN['weak_ha_max_decline'],
            weak_missing=weak_scores['ball_missing']['weak_ha_accuracy'] >= baseline['weak_scores']['ball_missing']['weak_ha_accuracy'] - PLAN['missing_weak_ha_max_decline'])
        row['eligible'] = all(row['guards'].values())
        rows.append(row)
        print(json.dumps(row), flush=True)
    best = max((r for r in rows if r['eligible']), key=lambda r:r['objective'])
    for name, digest in freeze['sources'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Executed decoder source changed during calibration')
    dump(args.out / 'selection.json', dict(complete=True, selected=best, candidates=rows, plan=freeze,
        selected_model_sha256=freeze['model_sha256'], parent_sha256=origin['parent_sha256'],
        parent_protocol_sha256=origin['parent_protocol_sha256'], model_weights_changed=False,
        zero_penalty_control_reproduced=True, provided_match_used=False, public_labels_used=False,
        unexpected_competition_access=denied, competition_target_achieved=False))
    print(json.dumps(dict(penalty=best['penalty'], objective=best['objective'])), flush=True)


if __name__ == '__main__':
    main()
