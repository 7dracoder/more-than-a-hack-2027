"""External-only temporal adaptation to independently measured camera loss.

All training/selection reads are isolated from competition data and public truth.
The development match is already used development data, never an independent test.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np

from possession.data import sha256
from scripts.research.spatial_memory.run_external import TRAIN, DEV, dump, isolate, score_parts, torch_setup

PLAN = dict(train_matches=TRAIN, validation_matches=[DEV], test_matches=[], development_only=True,
    provided_match_used=False, public_labels_used=False, sample_step=2, fps=25,
    feature_version='natural_visibility_memory', architecture='boundary_hybrid',
    hidden_channels=128, dropout=.2, boundary_positive_weight=10.,
    views=['native', 'masked', 'mixed'], training_view_probability=[.5, .25, .25],
    initialization='models/coreorigin-robust-v13', comparison='models/coreorigin-video-v1',
    seed=20321115, epochs=50, batches_per_epoch=80, batch_size=16, sequence_length=512,
    learning_rate=.0002, weight_decay=.001, boundary_auxiliary_weight=.3,
    run_weight_power=.5, validation_interval_epochs=5,
    mixture_weights=[0., .25, .5, .75, 1.], native_spell_f1_max_decline=.01,
    decoder='Frozen comparison decoder on the original 25 Hz frame grid',
    selection='Maximum mean spell F1 of three external development views, with native guard; comparison at strength zero remains eligible.',
    head_orientation_used=False, gaze_measured=False,
    limitation='Repeated external development match; synthetic views are not natural held-out match truth or competition performance.')


def sources():
    files = ['scripts/research/spatial_memory/train_natural_visibility.py',
        'possession/external_visibility.py', 'possession/features_v08.py', 'possession/features_v09.py',
        'possession/features_v13.py', 'possession/features_v25.py', 'possession/spatial_memory.py',
        'possession/external.py', 'possession/temporal.py', 'possession/boundary.py',
        'possession/recurrent.py', 'scripts/run_local.py']
    return {name:sha256(ROOT / name) for name in files}


def seed_for(match, half, view):
    return int(hashlib.sha256(f'natural-camera-v1/{match}/{half}/{view}'.encode()).hexdigest()[:8], 16)


def model_matrix(x):
    # Keep the frozen robustness model's 664-feature prefix and add 128 memory
    # features. The remaining 296 features are cached for the comparison only.
    return np.concatenate([x[:, :664], x[:, 960:]], axis=1)


def prepare(out, device):
    from possession.external import read_sportec
    from possession.external_visibility import degrade
    from possession.features_v25 import build_features
    from possession.spatial_memory import build_features as memory_features
    from scripts.run_local import selected_model, symmetric_probability

    profile_path = out / 'external_visibility.json'
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    if (profile['provided_match_used'] or profile['possession_labels_used'] or
            profile['validation_matches_used'] or not profile['data_guard_verified']):
        raise ValueError('Need an independently fitted external training-only visibility profile')
    folder = ROOT / 'data/external/sportec'
    pins = {x['name']:x['sha256'] for x in json.loads((folder / 'download_manifest.json').read_text())['files']}
    comparison, state = selected_model(ROOT / PLAN['comparison'], device)
    cache = out / 'cache'
    cache.mkdir(exist_ok=True)
    receipt = dict(plan=PLAN, sources=sources(), profile_sha256=sha256(profile_path),
        initialization_sha256=sha256(ROOT / PLAN['initialization'] / 'selected.pt'),
        comparison_sha256=sha256(ROOT / PLAN['comparison'] / 'selected.pt'))
    plan_path = out / 'training_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != receipt:
        raise ValueError('Preparation changed; use a new study directory')
    dump(plan_path, receipt)
    for match in TRAIN + [DEV]:
        paths = [next(folder.glob('*' + kind + '*' + match + '.xml')) for kind in ('matchinformation', 'positions')]
        inputs = {p.name:sha256(p) for p in paths}
        if any(pins.get(k) != v for k, v in inputs.items()):
            raise ValueError('External source differs from published pin')
        halves = read_sportec(*paths)
        for i, half in enumerate(halves):
            for view in PLAN['views']:
                path = cache / f'{match}_{i}_{view}.npy'
                meta_path = path.with_suffix('.json')
                expected = dict(preparation=receipt, inputs=inputs, match=match, half=i,
                    view=view, seed=seed_for(match, i, view))
                if path.exists() and meta_path.exists():
                    old = json.loads(meta_path.read_text())
                    if old['fingerprint'] != expected or sha256(path) != old['sha256']:
                        raise ValueError('Cache changed; refusing reuse')
                    for key, digest in old['auxiliary_sha256'].items():
                        if sha256(cache / key) != digest:
                            raise ValueError('Auxiliary cache changed')
                    print('Reused ' + path.name, flush=True)
                    continue
                started = time.monotonic()
                p, b = half.players, half.ball
                if view != 'native':
                    p, b = degrade(p, half.teams, b, half.pitch, profile, expected['seed'],
                        mode='none' if view == 'masked' else 'mixed')
                features, names = build_features(p, half.teams, b, half.pitch, step=1)
                memory, memory_names = memory_features(p, half.teams, b, half.pitch)
                if names != state['names'] or len(memory_names) != 128 or features.shape[1] != 960:
                    raise ValueError('Feature schema differs from the frozen comparison')
                stem = path.stem
                auxiliary = {}
                if match == DEV:
                    probability = symmetric_probability(comparison, features, names,
                        state['mean'], state['scale'], device)
                    pp = cache / (stem + '_parent.npy')
                    np.save(pp, probability)
                    auxiliary[pp.name] = sha256(pp)
                    del probability
                x = np.concatenate([features[::2], memory[::2]], axis=1)
                del features, memory
                part = path.with_suffix('.partial.npy')
                np.save(part, x)
                part.replace(path)
                tp = cache / (stem + '_truth.npy')
                np.save(tp, half.labels)
                auxiliary[tp.name] = sha256(tp)
                dump(meta_path, dict(fingerprint=expected, sha256=sha256(path), auxiliary_sha256=auxiliary,
                    names=names + memory_names, rows=len(x),
                    ball_missing_fraction=float((~np.isfinite(b).all(1)).mean()),
                    visible_players_per_team=[float(np.isfinite(p[:, half.teams == side]).all(-1).sum(1).mean()) for side in (0, 1)],
                    seconds=round(time.monotonic() - started, 2)))
                print(f'Prepared {path.name}: {len(x)} steps, {time.monotonic() - started:.1f}s', flush=True)
                del x
    dump(out / 'preparation.json', dict(complete=True, fingerprint=receipt,
        provided_match_used=False, public_labels_used=False))


def load_parts(out, matches):
    from possession.temporal import training_weights
    result, names = [], None
    for match in matches:
        for half in (0, 1):
            for view in PLAN['views']:
                path = out / 'cache' / f'{match}_{half}_{view}.npy'
                meta = json.loads(path.with_suffix('.json').read_text())
                if sha256(path) != meta['sha256']:
                    raise ValueError('Changed prepared features')
                for key, digest in meta['auxiliary_sha256'].items():
                    if sha256(path.parent / key) != digest:
                        raise ValueError('Changed prepared targets or comparison probabilities')
                current = meta['names'][:664] + meta['names'][960:]
                if names is not None and current != names:
                    raise ValueError('Inconsistent schemas')
                names = current
                truth = np.load(path.with_name(path.stem + '_truth.npy'), allow_pickle=False)
                y = np.array([{'H':0, 'A':1, 'D':2}.get(t, -100) for t in truth[::2]], np.int64)
                result.append(dict(match=match, half=half, view=view,
                    x=np.load(path, mmap_mode='r', allow_pickle=False), truth=truth, y=y,
                    w=training_weights(truth[::2], PLAN['run_weight_power']),
                    parent=np.load(path.with_name(path.stem + '_parent.npy')) if match == DEV else None))
    return result, names


def selection_scores(parts, probabilities, costs, baseline):
    from possession.temporal import viterbi_labels
    candidates = []
    for strength in PLAN['mixture_weights']:
        scores = {}
        for view in PLAN['views']:
            truth, predicted = [], []
            for part, q in zip(parts, probabilities):
                if part['view'] != view:
                    continue
                expanded = np.repeat(q, 2, axis=0)[:len(part['truth'])]
                p = (1. - strength) * part['parent'] + strength * expanded
                truth.append(part['truth'])
                predicted.append(viterbi_labels(p, *costs))
            scores[view] = score_parts(truth, predicted)
        eligible = scores['native']['spell_f1_mean'] >= baseline['native']['spell_f1_mean'] - PLAN['native_spell_f1_max_decline']
        candidates.append(dict(strength=strength, scores=scores, eligible=eligible,
            objective=float(np.mean([scores[v]['spell_f1_mean'] for v in PLAN['views']]))))
    return candidates


def train(out, device, denied):
    import torch
    from possession.temporal import make_model, normalized, frame_loss, symmetric_probability
    from possession.external import swap_features
    from possession.boundary import boundary_loss
    from scripts.run_local import save_torch

    prepared = json.loads((out / 'preparation.json').read_text())
    if not prepared['complete'] or prepared['fingerprint']['sources'] != sources():
        raise ValueError('Prepared study source differs')
    torch.manual_seed(PLAN['seed'])
    rng = np.random.default_rng(PLAN['seed'])
    training, names = load_parts(out, TRAIN)
    development, other_names = load_parts(out, [DEV])
    assert names == other_names
    initial = torch.load(ROOT / PLAN['initialization'] / 'selected.pt', map_location='cpu', weights_only=False)
    if names[:664] != initial['names'] or initial['config']['boundary_positive_weight'] != PLAN['boundary_positive_weight']:
        raise ValueError('Warm start schema differs')
    count, total, squares = 0, np.zeros(128), np.zeros(128)
    for part in training:
        values = np.asarray(part['x'][::5, 960:], np.float64)
        count += len(values); total += values.sum(0); squares += (values ** 2).sum(0)
    from possession.spatial_memory import symmetric_features
    extra_names = names[664:]
    em, es = total / count, squares / count
    em = (em + symmetric_features(em[None], extra_names)[0]) / 2.
    es = (es + symmetric_features(es[None], extra_names)[0]) / 2.
    mean = np.r_[initial['mean'], em].astype(np.float32)
    scale = np.r_[initial['scale'], np.sqrt(np.maximum(es - em ** 2, 1e-4))].astype(np.float32)
    model = make_model(len(names), PLAN).to(device)
    weights = initial['model']
    weights['project.weight'] = torch.cat([weights['project.weight'], torch.zeros(128, 128, 1)], dim=1)
    model.load_state_dict(weights)
    del weights, initial
    optimizer = torch.optim.AdamW(model.parameters(), lr=PLAN['learning_rate'], weight_decay=PLAN['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, PLAN['epochs'], eta_min=PLAN['learning_rate'] / 10.)
    parent_state = torch.load(ROOT / PLAN['comparison'] / 'selected.pt', map_location='cpu', weights_only=False)
    costs = parent_state['decoder_costs']
    del parent_state
    from possession.temporal import viterbi_labels
    baseline = {view:score_parts([h['truth'] for h in development if h['view'] == view],
        [viterbi_labels(h['parent'], *costs) for h in development if h['view'] == view]) for view in PLAN['views']}
    dump(out / 'external_comparison.json', dict(scores=baseline, competition_performance=False))
    baseline_objective = float(np.mean([baseline[v]['spell_f1_mean'] for v in PLAN['views']]))
    best = dict(epoch=0, strength=0., scores=baseline, objective=baseline_objective, eligible=True)
    history, start = [], 1
    latest = out / 'latest.pt'
    fingerprint = hashlib.sha256(json.dumps(prepared, sort_keys=True).encode()).hexdigest()
    dump(out / 'fixed_protocol.json', dict(config=PLAN, fingerprint=fingerprint,
        preparation=prepared, baseline=baseline, mean_statistics_matches=TRAIN))
    if latest.exists():
        checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
        if checkpoint['fingerprint'] != fingerprint:
            raise ValueError('Cannot resume changed protocol')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best, history, start = checkpoint['best'], checkpoint['history'], checkpoint['epoch'] + 1
        rng.bit_generator.state = checkpoint['rng']
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    mapping = swap_features(np.arange(len(names)), names).astype(int)
    view_groups = [[j for j, h in enumerate(training) if h['view'] == view] for view in PLAN['views']]
    for epoch in range(start, PLAN['epochs'] + 1):
        started = time.monotonic()
        model.train(); losses = []
        for _ in range(PLAN['batches_per_epoch']):
            bx, by, bw = [], [], []
            for _ in range(PLAN['batch_size']):
                group = int(rng.choice(3, p=PLAN['training_view_probability']))
                h = training[int(rng.choice(view_groups[group]))]
                a = int(rng.integers(0, len(h['x']) - PLAN['sequence_length'] + 1))
                b = a + PLAN['sequence_length']
                x = normalized(model_matrix(h['x'][a:b]), mean, scale)
                y = h['y'][a:b].copy()
                if rng.random() < .5:
                    x = x[:, mapping]
                    y = np.where(y == 0, 1, np.where(y == 1, 0, y))
                bx.append(x.T); by.append(y); bw.append(h['w'][a:b])
            tx = torch.as_tensor(np.stack(bx), device=device)
            ty = torch.as_tensor(np.stack(by), device=device)
            tw = torch.as_tensor(np.stack(bw), device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, change = model.forward_with_boundary(tx)
            loss = (frame_loss(logits, ty) * tw).sum() / tw.sum().clamp_min(1.)
            loss = loss + PLAN['boundary_auxiliary_weight'] * boundary_loss(change, ty, model.positive_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite external training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        row = dict(epoch=epoch, loss=float(np.mean(losses)), seconds=round(time.monotonic() - started, 1))
        if epoch == 1 or epoch % PLAN['validation_interval_epochs'] == 0:
            probabilities = [symmetric_probability(model, model_matrix(h['x']), names, mean, scale, device) for h in development]
            candidates = selection_scores(development, probabilities, costs, baseline)
            chosen = max((c for c in candidates if c['eligible']), key=lambda c:c['objective'])
            row.update(selection=chosen, candidates=candidates)
            if chosen['objective'] > best['objective'] + 1e-10:
                best = dict(epoch=epoch, **chosen)
                save_torch(out / 'selected.pt', dict(model=model.state_dict(), mean=mean, scale=scale,
                    names=names, config=PLAN, decoder_costs=costs, validation=chosen,
                    epoch=epoch, fingerprint=fingerprint))
            del probabilities
        history.append(row)
        scheduler.step()
        save_torch(latest, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), best=best, history=history, epoch=epoch,
            rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            fingerprint=fingerprint))
        dump(out / 'history.json', history)
        print(json.dumps(row), flush=True)
    dump(out / 'selection.json', dict(complete=True, best=best,
        comparison_objective=baseline_objective, absolute_external_gain=best['objective'] - baseline_objective,
        selected_sha256=sha256(out / 'selected.pt') if best['strength'] > 0 else None,
        protocol_sha256=sha256(out / 'fixed_protocol.json'),
        unexpected_competition_access=denied, provided_match_used=False, public_labels_used=False,
        competition_target_achieved=False, limitation=PLAN['limitation']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare', 'train'], required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    denied = isolate(args.out)
    device = torch_setup(args.device)
    if args.stage == 'prepare':
        prepare(args.out, device)
    else:
        train(args.out, device, denied)


if __name__ == '__main__':
    main()
