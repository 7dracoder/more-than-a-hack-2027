"""One-factor external comparison: replace the old missing-ball expert.

The prior gate, strength, decoder and parent boundary probabilities stay fixed.
The new gate is evaluated on the expert's explicit half-rate input clock.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from possession.data import sha256
from scripts.research.spatial_memory.run_external import DEV, dump, isolate, torch_setup, score_parts
from scripts.research.spatial_memory.train_natural_visibility import model_matrix
from scripts.research.spatial_memory.select_temporal_fusion import gap_gate


def protocol(study):
    prior_path = ROOT / 'models/coreorigin-temporal-fusion-v1/selection.json'
    prior = json.loads(prior_path.read_text())
    assert prior['selected']['mode'] == 'long_missing' and prior['selected']['strength'] == .75
    plan = dict(match=DEV, views=['native', 'masked', 'mixed'],
        variants=['development_selected_checkpoint', 'fixed_final_epoch_checkpoint'],
        gate_mode='long_missing', missing_seconds=2., gate_sample_step=2, strength=.75,
        boundary='Unchanged parent boundary probabilities', decoder_costs=prior['decoder_costs'],
        baseline='Winner of the preceding frozen external comparison',
        selection='Largest mean native/masked/mixed spell F1 with parent native guard; ties retain baseline.',
        native_spell_f1_max_decline=.01, parameters_tuned=False,
        provided_match_used=False, public_labels_used=False,
        prior_selection_sha256=sha256(prior_path),
        training_plan_sha256=sha256(study / 'training_plan.json'),
        preceding_comparison_plan_sha256=sha256(study / 'comparison_plan.json'),
        source_sha256=sha256(Path(__file__)),
        limitation='Exploratory external development only, not independent or competition verification.')
    out = study / 'memory_replacement'
    out.mkdir(exist_ok=True)
    path = out / 'plan.json'
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError('Frozen replacement protocol changed')
    else:
        dump(path, plan)
    return out, plan


def replacement_probability(parent, expert, memory, memory_names):
    gate = np.repeat(gap_gate(memory, memory_names, 'long_missing'), 2)[:len(parent)]
    expanded = np.repeat(expert, 2, axis=0)[:len(parent)]
    if expanded.shape != parent.shape:
        raise ValueError('Expert and parent clocks differ')
    alpha = .75 * gate[:, None]
    result = parent.copy()
    result[:, :3] = (1. - alpha) * parent[:, :3] + alpha * expanded[:, :3]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    denied = isolate(args.study)
    out, plan = protocol(args.study)
    if args.freeze_only:
        print('Expert-replacement protocol frozen before final training results.', flush=True)
        return
    if (out / 'selection.json').exists():
        raise ValueError('Replacement comparison is already frozen')
    previous_path = args.study / 'final_external_selection.json'
    previous = json.loads(previous_path.read_text())
    if (not previous['complete'] or previous['provided_match_used'] or previous['public_labels_used'] or
            previous['unexpected_competition_access'] or not previous['selected_score_reproduced']):
        raise ValueError('Need the completed isolated external comparison')
    training_selection = json.loads((args.study / 'selection.json').read_text())
    if not training_selection['complete']:
        raise ValueError('Training is incomplete')
    device = torch_setup(args.device)
    import torch
    from scripts.run_local import model_from_state, save_torch
    from possession.temporal import symmetric_probability, viterbi_labels
    selected_path, latest_path = args.study / 'selected.pt', args.study / 'latest.pt'
    if not selected_path.exists() or sha256(selected_path) != training_selection['selected_sha256']:
        raise ValueError('Missing or changed development-selected checkpoint')
    metadata = torch.load(selected_path, map_location='cpu', weights_only=False)
    latest = torch.load(latest_path, map_location='cpu', weights_only=False)
    if latest['epoch'] != metadata['config']['epochs']:
        raise ValueError('The final epoch checkpoint is incomplete')
    hashes = dict(development_selected_checkpoint=sha256(selected_path),
        fixed_final_epoch_checkpoint=sha256(latest_path))
    dump(out / 'checkpoint_freeze.json', dict(checkpoint_sha256=hashes,
        preceding_selection_sha256=sha256(previous_path), plan_sha256=sha256(out / 'plan.json')))
    parts = []
    for i in (0, 1):
        for view in plan['views']:
            path = args.study / 'cache' / f'{DEV}_{i}_{view}.npy'
            record = json.loads(path.with_suffix('.json').read_text())
            if sha256(path) != record['sha256']:
                raise ValueError('External feature cache changed')
            for name, digest in record['auxiliary_sha256'].items():
                if sha256(path.parent / name) != digest:
                    raise ValueError('External cached targets/probabilities changed')
            names = record['names'][:664] + record['names'][960:]
            if names != metadata['names']:
                raise ValueError('Expert feature schema differs')
            parts.append(dict(view=view, x=np.load(path, mmap_mode='r'), memory_names=record['names'][960:],
                truth=np.load(path.with_name(path.stem + '_truth.npy')),
                parent=np.load(path.with_name(path.stem + '_parent.npy'))))
    rows = [previous['selected']]
    best_model = None
    best = rows[0]
    parent_native = next(r for r in previous['candidates'] if r['kind'] == 'parent')['scores']['native']['spell_f1_mean']
    for kind in plan['variants']:
        state = dict(metadata)
        if kind == 'fixed_final_epoch_checkpoint':
            state['model'] = latest['model']
            state['epoch'] = latest['epoch']
        model = model_from_state(state, device)
        predictions = {view:[] for view in plan['views']}
        truths = {view:[] for view in plan['views']}
        for part in parts:
            q = symmetric_probability(model, model_matrix(part['x']), state['names'], state['mean'], state['scale'], device)
            p = replacement_probability(part['parent'], q, part['x'][:, 960:], part['memory_names'])
            predictions[part['view']].append(viterbi_labels(p, *plan['decoder_costs']))
            truths[part['view']].append(part['truth'])
        scores = {view:score_parts(truths[view], predictions[view]) for view in plan['views']}
        row = dict(kind='memory_replacement_' + kind, scores=scores,
            objective=float(np.mean([scores[v]['spell_f1_mean'] for v in plan['views']])),
            eligible=scores['native']['spell_f1_mean'] >= parent_native - plan['native_spell_f1_max_decline'],
            source_checkpoint_sha256=hashes[kind], source_epoch=int(state['epoch']))
        rows.append(row)
        print(json.dumps(row), flush=True)
        if row['eligible'] and row['objective'] > best['objective'] + 1e-10:
            best = row
            best_model = dict(state)
        del model
    selected_hash = None
    if best_model is not None:
        fingerprint = hashlib.sha256(json.dumps(dict(plan=plan, hashes=hashes), sort_keys=True).encode()).hexdigest()
        best_model['fingerprint'] = fingerprint
        best_model['validation'] = best
        save_torch(out / 'selected.pt', best_model)
        dump(out / 'fixed_protocol.json', dict(config=best_model['config'], fingerprint=fingerprint,
            adaptation=False, source_checkpoint_sha256=best['source_checkpoint_sha256'], plan=plan))
        selected_hash = sha256(out / 'selected.pt')
    dump(out / 'selection.json', dict(complete=True, selected=best, candidates=rows,
        selected_model_sha256=selected_hash, preceding_selection_sha256=sha256(previous_path),
        checkpoint_freeze_sha256=sha256(out / 'checkpoint_freeze.json'), plan=plan,
        provided_match_used=False, public_labels_used=False, unexpected_competition_access=denied,
        competition_target_achieved=False))
    print(json.dumps(dict(selected=best['kind'], objective=best['objective'])), flush=True)


if __name__ == '__main__':
    main()
