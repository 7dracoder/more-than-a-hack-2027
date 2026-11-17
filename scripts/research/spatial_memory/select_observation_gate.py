"""Select a camera-loss gate on the existing external development fixture only."""
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
from possession.observation_gate import mix_probabilities
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import DEV, dump, torch_setup, score_parts
from scripts.research.spatial_memory.train_natural_visibility import model_matrix

STUDY = ROOT / 'data/research/natural-visibility-20261114'
INCUMBENT = ROOT / 'models/coreorigin-natural-visibility-v1'
PLAN = dict(development_matches=[DEV], independent_test_matches=[],
    views=['native', 'masked', 'mixed'], checkpoints=['selected', 'final_epoch'],
    strengths=[.25, .5, .75, 1.], sample_step=2,
    gate='Ball absent OR fewer than eleven observed players on either team, on the expert input clock.',
    boundary='Unchanged parent calibrated change probabilities on the full 25 Hz clock.',
    selection='Largest mean native/masked/mixed spell F1 with parent native guard; ties retain incumbent.',
    native_spell_f1_max_decline=.01, additional_training=False,
    provided_match_used=False, public_labels_used=False,
    limitation='Repeated exploratory development data, not independent-test or competition performance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    allowed = [STUDY / 'latest.pt', STUDY / 'memory_replacement/checkpoint_freeze.json']
    for i in (0, 1):
        for view in PLAN['views']:
            stem = STUDY / 'cache' / f'{DEV}_{i}_{view}'
            allowed += [stem.with_suffix('.npy'), stem.with_suffix('.json'),
                stem.with_name(stem.name + '_truth.npy'), stem.with_name(stem.name + '_parent.npy')]
    denied = restrict_external(args.out, allowed)
    if (args.out / 'selection.json').exists():
        raise ValueError('Selection is already frozen')
    preparation = json.loads((INCUMBENT / 'training_plan.json').read_text())
    preceding = json.loads((INCUMBENT / 'memory_replacement/selection.json').read_text())
    origin = json.loads((STUDY / 'memory_replacement/checkpoint_freeze.json').read_text())
    if (preceding['provided_match_used'] or preceding['public_labels_used'] or
            preceding['unexpected_competition_access'] or not preceding['complete']):
        raise ValueError('Incumbent lacks isolated external provenance')
    source_paths = ['possession/observation_gate.py', 'scripts/research/spatial_memory/external_guard.py',
        'scripts/research/spatial_memory/select_observation_gate.py']
    freeze = dict(plan=PLAN, sources={p:sha256(ROOT / p) for p in source_paths},
        incumbent_selection_sha256=sha256(INCUMBENT / 'memory_replacement/selection.json'),
        training_plan_sha256=sha256(INCUMBENT / 'training_plan.json'),
        checkpoint_sha256=dict(selected=sha256(INCUMBENT / 'selected.pt'), final_epoch=sha256(STUDY / 'latest.pt')))
    if (freeze['checkpoint_sha256']['selected'] != origin['checkpoint_sha256']['development_selected_checkpoint'] or
            freeze['checkpoint_sha256']['final_epoch'] != origin['checkpoint_sha256']['fixed_final_epoch_checkpoint']):
        raise ValueError('Previously frozen external checkpoints changed')
    plan_path = args.out / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != freeze:
        raise ValueError('Study protocol changed')
    dump(plan_path, freeze)
    device = torch_setup(args.device)
    import torch
    from scripts.run_local import model_from_state, save_torch
    from possession.temporal import symmetric_probability, viterbi_labels
    metadata = torch.load(INCUMBENT / 'selected.pt', map_location='cpu', weights_only=False)
    latest = torch.load(STUDY / 'latest.pt', map_location='cpu', weights_only=False)
    if latest['epoch'] != metadata['config']['epochs'] or metadata['config']['sample_step'] != 2:
        raise ValueError('Unexpected training endpoint or input clock')
    costs = metadata['decoder_costs']
    parts = []
    for i in (0, 1):
        for view in PLAN['views']:
            path = STUDY / 'cache' / f'{DEV}_{i}_{view}.npy'
            record = json.loads(path.with_suffix('.json').read_text())
            if record['fingerprint']['match'] != DEV or record['fingerprint']['preparation']['plan']['provided_match_used']:
                raise ValueError('Not an isolated external cache')
            if sha256(path) != record['sha256']:
                raise ValueError('External feature cache changed')
            for name, digest in record['auxiliary_sha256'].items():
                if sha256(path.parent / name) != digest:
                    raise ValueError('External auxiliary cache changed')
            if record['names'][:664] + record['names'][960:] != metadata['names']:
                raise ValueError('Expert feature schema changed')
            parts.append(dict(view=view, half=i, x=np.load(path, mmap_mode='r'), names=record['names'][960:],
                parent=np.load(path.with_name(path.stem + '_parent.npy')),
                truth=np.load(path.with_name(path.stem + '_truth.npy'))))
    rows = [dict(preceding['selected'], checkpoint=None, strength=None)]
    best = rows[0]
    parent_native = None
    best_model = None
    cache = args.out / 'probabilities'
    cache.mkdir(exist_ok=True)
    for checkpoint in PLAN['checkpoints']:
        state = dict(metadata)
        if checkpoint == 'final_epoch':
            state.update(model=latest['model'], epoch=latest['epoch'])
        model = model_from_state(state, device)
        probabilities = []
        for h in parts:
            q = symmetric_probability(model, model_matrix(h['x']), state['names'], state['mean'], state['scale'], device)
            np.save(cache / f'{checkpoint}_{h["half"]}_{h["view"]}.npy', q)
            probabilities.append(q)
        if checkpoint == 'selected':
            # Reproduce the old whole-frame mixture before comparing a new gate.
            for view in PLAN['views']:
                truth, pred, base = [], [], []
                for h, q in zip(parts, probabilities):
                    if h['view'] != view:
                        continue
                    p = .75 * h['parent'] + .25 * np.repeat(q, 2, axis=0)[:len(h['truth'])]
                    truth.append(h['truth']); pred.append(viterbi_labels(p, *costs))
                    base.append(viterbi_labels(h['parent'], *costs))
                reproduced = score_parts(truth, pred)
                if abs(reproduced['spell_f1_mean'] - preceding['selected']['scores'][view]['spell_f1_mean']) > 1e-10:
                    raise ValueError('Incumbent score failed exact reproduction')
                if view == 'native':
                    parent_native = score_parts(truth, base)['spell_f1_mean']
        for strength in PLAN['strengths']:
            scores = {}
            for view in PLAN['views']:
                truth, pred = [], []
                for h, q in zip(parts, probabilities):
                    if h['view'] != view:
                        continue
                    p = mix_probabilities(h['parent'], q, h['x'][:, 960:], h['names'], strength)
                    truth.append(h['truth']); pred.append(viterbi_labels(p, *costs))
                scores[view] = score_parts(truth, pred)
            row = dict(kind='observation_loss', checkpoint=checkpoint, strength=strength, scores=scores,
                source_epoch=int(state['epoch']), objective=float(np.mean([scores[v]['spell_f1_mean'] for v in PLAN['views']])),
                eligible=scores['native']['spell_f1_mean'] >= parent_native - PLAN['native_spell_f1_max_decline'])
            rows.append(row)
            print(json.dumps(row), flush=True)
            if row['eligible'] and row['objective'] > best['objective'] + 1e-10:
                best, best_model = row, dict(state)
        del model
    selected_hash = None
    fingerprint = hashlib.sha256(json.dumps(freeze, sort_keys=True).encode()).hexdigest()
    if best_model is not None:
        best_model.update(fingerprint=fingerprint, validation=best)
        save_torch(args.out / 'selected.pt', best_model)
        dump(args.out / 'fixed_protocol.json', dict(config=best_model['config'], fingerprint=fingerprint,
            external_only=True, additional_training=False, source_checkpoint_sha256=freeze['checkpoint_sha256'][best['checkpoint']]))
        selected_hash = sha256(args.out / 'selected.pt')
    for path, expected in freeze['sources'].items():
        if sha256(ROOT / path) != expected:
            raise ValueError('Executed source changed during selection')
    dump(args.out / 'selection.json', dict(complete=True, selected=best, candidates=rows, plan=freeze,
        selected_model_sha256=selected_hash, parent_sha256=preparation['comparison_sha256'],
        parent_protocol_sha256=sha256(ROOT / 'models/coreorigin-video-v1/fixed_protocol.json'),
        incumbent_reproduced=True, native_guard_reference=parent_native,
        provided_match_used=False, public_labels_used=False, unexpected_competition_access=denied,
        competition_target_achieved=False))
    print(json.dumps(dict(selected=best['kind'], objective=best['objective'])), flush=True)


if __name__ == '__main__':
    main()
