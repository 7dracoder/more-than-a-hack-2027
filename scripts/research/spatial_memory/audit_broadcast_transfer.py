"""Frozen transfer audit on real SkillCorner detected-only camera observations.

Provider possession groups are weak H/A targets, not dense HAD truth. No spell
score, dead-ball accuracy, parameter selection or competition score is computed.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from possession.data import sha256
from possession.skillcorner import load_match, resample_coordinates
from possession.features_v25 import build_features
from possession.spatial_memory import build_features as memory_features
from possession.observation_gate import mix_probabilities
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump, torch_setup

GATED = ROOT / 'data/research/observation-gate-20261114'
PLAN = dict(matches=['2016236', '2017461'], views=['native', 'detected'],
    models=['parent', 'natural_v1', 'observation_gate'], source_fps=10, model_fps=25,
    scoring_fps=5, scoring='Only exact shared 10/25 Hz ticks; one label every 0.2 seconds, no duplicated interpolated labels.',
    player_speed_limit=15., ball_speed_limit=80.,
    strata=['all_known', 'ball_observed', 'ball_missing'],
    label_policy='Provider home/away group only; unknown stays X, never D.',
    metrics='Weak H/A frame accuracy, conditional H/A accuracy/macro F1/NLL, D prediction rate on weak H/A.',
    selection=False, training=False, provided_match_used=False, public_labels_used=False,
    limitation='Existing weak-label validation fixtures, previously opened during encoder research. No independent dense HAD truth or BAR-ATH score. Native includes provider-inferred coordinates; detected-only is actual detection flags plus adjacent finite interpolation.')


def weak_counts(probability, predicted, truth, mask):
    known = np.isin(truth, ['H', 'A']) & mask
    y = np.where(truth[known] == 'A', 1, 0)
    p = probability[known, :2].astype(np.float64)
    denominator = p.sum(1, keepdims=True)
    p = np.divide(p, denominator, out=np.full_like(p, .5), where=denominator > 0)
    conditional = p.argmax(1)
    confusion = np.zeros((2, 2), np.int64)
    np.add.at(confusion, (y, conditional), 1)
    return dict(n=int(known.sum()), frame_correct=int((predicted[known] == truth[known]).sum()),
        dead_predictions=int((predicted[known] == 'D').sum()),
        conditional_correct=int((conditional == y).sum()), confusion=confusion.tolist(),
        conditional_nll_sum=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-8, 1.)).sum()))


def summarize_counts(rows):
    count = sum(r['n'] for r in rows)
    c = np.sum([np.asarray(r['confusion']) for r in rows], axis=0)
    f1 = [2. * c[i, i] / max(1, c[i].sum() + c[:, i].sum()) for i in (0, 1)]
    return dict(frames=count, weak_ha_accuracy=sum(r['frame_correct'] for r in rows) / count if count else None,
        conditional_ha_accuracy=sum(r['conditional_correct'] for r in rows) / count if count else None,
        conditional_ha_macro_f1=float(np.mean(f1)) if count else None,
        conditional_ha_nll=sum(r['conditional_nll_sum'] for r in rows) / count if count else None,
        dead_prediction_rate_on_weak_ha=sum(r['dead_predictions'] for r in rows) / count if count else None,
        conditional_ha_confusion=c.tolist())


def prepare(args, freeze):
    cache = args.out / 'features'
    cache.mkdir(exist_ok=True)
    for match in PLAN['matches']:
        directory = ROOT / 'data/external/skillcorner-raw' / match
        source = json.loads((directory / 'source.json').read_text(encoding='utf-8'))
        if sha256(directory / 'metadata.json') != source['metadata_sha256'] or sha256(directory / 'tracking.jsonl') != source['tracking_sha256']:
            raise ValueError('External source differs from its pin')
        metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
        pitch = np.array([metadata['pitch_length'], metadata['pitch_width']], np.float32)
        halves, teams = load_match(metadata, directory / 'tracking.jsonl')
        for period in (1, 2):
            half = halves[period]
            detected_ball = half['detected'][1]
            for view in PLAN['views']:
                path = cache / f'{match}_{period}_{view}.npy'
                sidecar = path.with_suffix('.json')
                expected = dict(source=source, freeze=freeze, match=match, period=period, view=view)
                if path.exists() and sidecar.exists():
                    receipt = json.loads(sidecar.read_text())
                    if receipt['fingerprint'] != expected or sha256(path) != receipt['sha256']:
                        raise ValueError('Incompatible external transfer cache')
                    for name, digest in receipt['auxiliary_sha256'].items():
                        if sha256(cache / name) != digest:
                            raise ValueError('Transfer target/timeline cache changed')
                    print('Reused ' + path.name, flush=True)
                    continue
                start = time.monotonic()
                p, b = half[view]
                p, low = resample_coordinates(p + pitch / 2., speed_limit=PLAN['player_speed_limit'])
                b, ball_low = resample_coordinates(b + pitch / 2., speed_limit=PLAN['ball_speed_limit'])
                np.testing.assert_array_equal(low, ball_low)
                base, names = build_features(p, teams, b, pitch, step=1)
                memory, memory_names = memory_features(p, teams, b, pitch)
                matrix = np.concatenate([base, memory], axis=1)
                del base, memory, p, b
                if matrix.shape[1] != 1088:
                    raise ValueError('Unexpected transfer feature count')
                part = path.with_suffix('.partial.npy')
                np.save(part, matrix)
                part.replace(path)
                ticks = np.arange(0, len(matrix), 5)
                raw_ticks = low[ticks]
                np.testing.assert_array_equal(raw_ticks, np.arange(len(ticks)) * 2)
                target_path = path.with_name(path.stem + '_weak.npz')
                np.savez_compressed(target_path, ticks=ticks, truth=half['labels'][raw_ticks],
                    detected_ball_seen=np.isfinite(detected_ball[raw_ticks]).all(1))
                dump(sidecar, dict(fingerprint=expected, sha256=sha256(path),
                    auxiliary_sha256={target_path.name:sha256(target_path)}, names=names + memory_names,
                    frames=len(matrix), source_first=half['first'], source_last=half['last'],
                    seconds=round(time.monotonic() - start, 1)))
                print(f'Prepared real broadcast transfer {path.name}: {time.monotonic() - start:.1f}s', flush=True)
                del matrix
    dump(args.out / 'preparation.json', dict(complete=True, freeze=freeze, source_frames_to_model='10 to 25 Hz, score exact 5 Hz intersection'))


def evaluate(args, freeze, denied):
    import torch
    from scripts.run_local import selected_model, symmetric_probability, viterbi_labels
    if not json.loads((args.out / 'preparation.json').read_text())['complete']:
        raise ValueError('Transfer preparation is incomplete')
    device = torch_setup(args.device)
    parent, parent_state = selected_model(ROOT / 'models/coreorigin-video-v1', device)
    natural, natural_state = selected_model(ROOT / 'models/coreorigin-natural-visibility-v1', device)
    expert, expert_state = selected_model(GATED, device)
    strength = freeze['gated_selection']['selected']['strength']
    probabilities_out = args.out / 'predictions'
    probabilities_out.mkdir(exist_ok=True)
    for match in PLAN['matches']:
        for period in (1, 2):
            for view in PLAN['views']:
                path = args.out / 'features' / f'{match}_{period}_{view}.npy'
                receipt = json.loads(path.with_suffix('.json').read_text())
                if sha256(path) != receipt['sha256']:
                    raise ValueError('Transfer features changed')
                matrix = np.load(path, mmap_mode='r')
                names = receipt['names']
                if names[:960] != parent_state['names'] or names[:664] + names[960:] != expert_state['names'] or expert_state['names'] != natural_state['names']:
                    raise ValueError('Frozen model input schemas disagree')
                p = symmetric_probability(parent, matrix[:, :960], parent_state['names'], parent_state['mean'], parent_state['scale'], device)
                x = np.concatenate([matrix[::2, :664], matrix[::2, 960:]], axis=1)
                q = symmetric_probability(natural, x, natural_state['names'], natural_state['mean'], natural_state['scale'], device)
                r = symmetric_probability(expert, x, expert_state['names'], expert_state['mean'], expert_state['scale'], device)
                combined = .75 * p + .25 * np.repeat(q, 2, axis=0)[:len(p)]
                gated = mix_probabilities(p, r, matrix[::2, 960:], names[960:], strength)
                outputs = dict(parent=p, natural_v1=combined, observation_gate=gated)
                target = probabilities_out / path.with_suffix('.npz').name
                arrays = {}
                for kind, prob in outputs.items():
                    arrays[kind] = prob
                    arrays[kind + '_labels'] = viterbi_labels(prob, *parent_state['decoder_costs'])
                np.savez_compressed(target, **arrays)
                print(f'Frozen real broadcast predictions: {match}, half {period}, {view}', flush=True)
                del matrix, x, q, r, p, combined, gated, arrays, outputs
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    # All predictions are frozen before opening any target cache for metrics.
    dump(args.out / 'prediction_freeze.json', dict(files={p.name:sha256(p) for p in probabilities_out.glob('*.npz')},
        checkpoint_sha256=freeze['models'], model_selection_performed=False))
    rows = []
    for match in PLAN['matches']:
        for period in (1, 2):
            for view in PLAN['views']:
                stem = f'{match}_{period}_{view}'
                receipt = json.loads((args.out / 'features' / (stem + '.json')).read_text())
                target_path = args.out / 'features' / (stem + '_weak.npz')
                if sha256(target_path) != receipt['auxiliary_sha256'][target_path.name]:
                    raise ValueError('Weak target cache changed')
                with np.load(target_path) as target, np.load(probabilities_out / (stem + '.npz')) as prediction:
                    ticks, truth, seen = target['ticks'], target['truth'], target['detected_ball_seen']
                    for kind in PLAN['models']:
                        for scope, mask in [('all_known', np.ones(len(ticks), bool)), ('ball_observed', seen), ('ball_missing', ~seen)]:
                            counts = weak_counts(prediction[kind][ticks], prediction[kind + '_labels'][ticks], truth, mask)
                            rows.append(dict(match=match, period=period, view=view, model=kind, scope=scope, counts=counts))
    summaries = {view:{kind:{scope:summarize_counts([r['counts'] for r in rows if r['view'] == view and r['model'] == kind and r['scope'] == scope])
        for scope in PLAN['strata']} for kind in PLAN['models']} for view in PLAN['views']}
    dump(args.out / 'results.json', dict(complete=True, plan=PLAN, summaries=summaries, per_half_counts=rows,
        model_selection_performed=False, training_performed=False, dense_had_truth_available=False,
        competition_target_achieved=False, provided_match_used=False, public_labels_used=False,
        unexpected_competition_access=denied))
    print(json.dumps(summaries), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare', 'evaluate'], required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    denied = restrict_external(args.out, [GATED / name for name in ('selection.json', 'selected.pt', 'fixed_protocol.json')])
    selection = json.loads((GATED / 'selection.json').read_text())
    if (not selection['complete'] or selection['selected']['kind'] != 'observation_loss' or
            selection['provided_match_used'] or selection['public_labels_used'] or selection['unexpected_competition_access']):
        raise ValueError('Need an isolated, externally frozen observation-gate candidate')
    models = {name:sha256(path) for name, path in dict(parent=ROOT / 'models/coreorigin-video-v1/selected.pt',
        natural_v1=ROOT / 'models/coreorigin-natural-visibility-v1/selected.pt', observation_gate=GATED / 'selected.pt').items()}
    if models['observation_gate'] != selection['selected_model_sha256'] or models['parent'] != selection['parent_sha256']:
        raise ValueError('Frozen gated model members changed')
    config = json.loads((ROOT / 'configs/skillcorner_encoder_pretrain.json').read_text())
    if PLAN['matches'] != config['validation_matches'] or set(PLAN['matches']) & set(config['train_matches']):
        raise ValueError('Weak-label validation split changed')
    source_files = ['scripts/research/spatial_memory/audit_broadcast_transfer.py',
        'scripts/research/spatial_memory/external_guard.py', 'possession/skillcorner.py',
        'possession/features_v25.py', 'possession/spatial_memory.py', 'possession/observation_gate.py']
    freeze = dict(plan=PLAN, models=models, sources={p:sha256(ROOT / p) for p in source_files},
        gated_selection=selection)
    freeze_path = args.out / 'freeze.json'
    if freeze_path.exists() and json.loads(freeze_path.read_text()) != freeze:
        raise ValueError('Frozen transfer audit changed')
    dump(freeze_path, freeze)
    if (args.out / 'results.json').exists():
        raise ValueError('Audit is complete; do not reopen it adaptively')
    if args.stage == 'prepare':
        prepare(args, freeze)
    else:
        evaluate(args, freeze, denied)


if __name__ == '__main__':
    main()
