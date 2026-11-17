"""Frozen five-point D-odds calibration on external development predictions."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from possession.data import sha256
from possession.observation_state_calibration import calibrate_dead_odds
from possession.uncertain_decode import decode_labels
from scripts.research.spatial_memory.frozen_development import load, ORIGIN, FALLBACK
from scripts.research.spatial_memory.run_external import dump, score_parts
from scripts.research.spatial_memory.train_real_broadcast_memory import VIEWS
from scripts.research.spatial_memory.audit_broadcast_transfer import weak_counts, summarize_counts

PROTOCOL = dict(dead_log_odds_offsets=[0., -1., -.5, .5, 1.], continuity_penalty=8.,
    calibration='Multiply D probability by exp(offset) only during existing observation-loss gate, then normalize HAD. Conditional H/A odds and parent boundary probabilities unchanged.',
    objective='Maximum mean strong native/masked/mixed spell F1; ties keep unchanged zero-offset fallback.',
    native_max_decline=0., strong_macro_max_decline=.005,
    weak_ha_max_decline=.005, missing_weak_ha_max_decline=.01,
    model_retraining=False, provided_match_used=False, public_labels_used=False,
    limitation='Exploratory calibration on reused external development fixtures, not independent or competition performance.')


def scores(parts, costs, offset):
    predictions, adjusted = [], []
    for h in parts:
        p = calibrate_dead_odds(h['probability'], h['gate'], offset)
        predictions.append(decode_labels(p, h['gate'], PROTOCOL['continuity_penalty'], *costs))
        adjusted.append(p)
    strong = {}
    for view in VIEWS:
        indices = [i for i,h in enumerate(parts) if h['domain'] == 'strong' and h['view'] == view]
        strong[view] = score_parts([parts[i]['truth'] for i in indices], [predictions[i] for i in indices])
    rows = {'all_known':[], 'ball_missing':[]}
    for i,h in enumerate(parts):
        if h['domain'] != 'weak':
            continue
        for scope,mask in [('all_known',np.ones(len(h['ticks']),bool)), ('ball_missing',~h['seen'])]:
            rows[scope].append(weak_counts(adjusted[i][h['ticks']], predictions[i][h['ticks']], h['truth'], mask))
    return strong, {scope:summarize_counts(values) for scope,values in rows.items()}


def make_row(strong, weak, baseline, offset):
    guards = dict(native=strong['native']['spell_f1_mean'] >= baseline['scores']['native']['spell_f1_mean'] - 1e-12,
        strong_macro=all(strong[v]['macro_f1'] >= baseline['scores'][v]['macro_f1'] - PROTOCOL['strong_macro_max_decline'] for v in VIEWS),
        weak_ha=weak['all_known']['weak_ha_accuracy'] >= baseline['weak_scores']['all_known']['weak_ha_accuracy'] - PROTOCOL['weak_ha_max_decline'],
        weak_missing=weak['ball_missing']['weak_ha_accuracy'] >= baseline['weak_scores']['ball_missing']['weak_ha_accuracy'] - PROTOCOL['missing_weak_ha_max_decline'])
    return dict(kind='calibrated_observation_loss', strength=1., penalty=8., dead_offset=offset,
        scores=strong, weak_scores=weak, objective=float(np.mean([strong[v]['spell_f1_mean'] for v in VIEWS])),
        guards=guards, eligible=all(guards.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'selection.json').exists():
        raise ValueError('Preserve completed calibration')
    origin,fallback,parts,hashes,denied = load(args.out)
    sources = ['scripts/research/spatial_memory/select_dead_calibration.py',
        'scripts/research/spatial_memory/frozen_development.py', 'scripts/research/spatial_memory/external_guard.py',
        'scripts/research/spatial_memory/run_external.py', 'scripts/research/spatial_memory/audit_broadcast_transfer.py',
        'possession/observation_state_calibration.py', 'possession/uncertain_decode.py',
        'possession/temporal.py', 'possession/fast_decode.py', 'evaluation/possession_metrics.py']
    plan = dict(protocol=PROTOCOL, sources={p:sha256(ROOT / p) for p in sources}, inputs=hashes,
        model_sha256=sha256(ORIGIN / 'selected.pt'), model_protocol_sha256=sha256(ORIGIN / 'fixed_protocol.json'),
        preceding_selection_sha256=sha256(ORIGIN / 'selection.json'), fallback_selection_sha256=sha256(FALLBACK / 'selection.json'))
    if (args.out / 'plan.json').exists():
        assert json.loads((args.out / 'plan.json').read_text()) == plan
    dump(args.out / 'plan.json', plan)
    import torch
    state = torch.load(ORIGIN / 'selected.pt', map_location='cpu', weights_only=False)
    rows = []
    for offset in PROTOCOL['dead_log_odds_offsets']:
        strong,weak = scores(parts,state['decoder_costs'],offset)
        row = make_row(strong,weak,fallback['selected'],offset)
        if offset == 0:
            for view in VIEWS:
                for metric in ('spell_f1_mean','macro_f1'):
                    assert abs(strong[view][metric] - fallback['selected']['scores'][view][metric]) < 1e-12
            for scope in weak:
                for metric in ('weak_ha_accuracy','conditional_ha_accuracy','conditional_ha_macro_f1'):
                    assert abs(weak[scope][metric] - fallback['selected']['weak_scores'][scope][metric]) < 1e-12
        rows.append(row)
        print(json.dumps(row),flush=True)
    best = max((r for r in rows if r['eligible']),key=lambda r:r['objective'])
    for path,digest in {**plan['sources'], **plan['inputs']}.items():
        assert sha256(ROOT / path) == digest,path
    dump(args.out / 'selection.json',dict(complete=True,selected=best,candidates=rows,plan=plan,
        selected_model_sha256=plan['model_sha256'],parent_sha256=origin['parent_sha256'],
        parent_protocol_sha256=origin['parent_protocol_sha256'],model_weights_changed=False,
        zero_offset_control_reproduced=True,provided_match_used=False,public_labels_used=False,
        unexpected_competition_access=denied,independent_test=False,competition_target_achieved=False))
    print(json.dumps(dict(offset=best['dead_offset'],objective=best['objective'])),flush=True)


if __name__ == '__main__':
    main()
