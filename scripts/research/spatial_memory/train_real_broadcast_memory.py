"""Mix strong external HAD with real broadcast conditional H/A supervision.

All supplied-match data/predictions and public/private possession truth are
blocked. Previously opened external development fixtures remain development.
"""
import argparse
import hashlib
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
from possession.observation_gate import mix_probabilities
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump, score_parts, torch_setup
from scripts.research.spatial_memory.train_natural_visibility import model_matrix
from scripts.research.spatial_memory.audit_broadcast_transfer import weak_counts, summarize_counts

STRONG = ROOT / 'data/research/natural-visibility-20261114'
TRANSFER = ROOT / 'data/research/broadcast-transfer-20261114'
VIEWS = ['native', 'masked', 'mixed']


def joint_selection(strong_scores, weak_scores, initial, reference):
    strong_mean = float(np.mean([strong_scores[v]['spell_f1_mean'] for v in VIEWS]))
    weak = weak_scores['all_known']
    missing = weak_scores['ball_missing']
    objective = (strong_mean + weak['conditional_ha_macro_f1']) / 2.
    guards = dict(
        native=strong_scores['native']['spell_f1_mean'] >= initial['strong_scores']['native']['spell_f1_mean'] - .01,
        strong_mean=strong_mean >= initial['strong_mean'] - .01,
        weak_ha=weak['weak_ha_accuracy'] >= reference['all_known']['weak_ha_accuracy'] - .005,
        weak_missing_ha=missing['conditional_ha_accuracy'] >= reference['ball_missing']['conditional_ha_accuracy'] - .005)
    return dict(strong_scores=strong_scores, weak_scores=weak_scores, strong_mean=strong_mean,
        objective=float(objective), guards=guards, eligible=all(guards.values()))


def allowed_inputs(config, weak_study):
    paths = [weak_study / 'weak_preparation.json', weak_study / 'weak_preparation_plan.json',
        TRANSFER / 'prediction_freeze.json', TRANSFER / 'freeze.json', TRANSFER / 'results.json']
    for match in config['strong_train_matches'] + config['strong_development_matches']:
        for half in (0, 1):
            for view in VIEWS:
                stem = STRONG / 'cache' / f'{match}_{half}_{view}'
                paths += [stem.with_suffix('.npy'), stem.with_suffix('.json'), stem.with_name(stem.name + '_truth.npy')]
                if match in config['strong_development_matches']:
                    paths.append(stem.with_name(stem.name + '_parent.npy'))
    for match in config['weak_train_matches']:
        for period in (1, 2):
            stem = weak_study / 'weak_features' / f'{match}_{period}_detected'
            paths += [stem.with_suffix('.npy'), stem.with_suffix('.json'), stem.with_name(stem.name + '_weak.npy')]
    for match in config['weak_development_matches']:
        for period in (1, 2):
            stem = f'{match}_{period}_detected'
            paths += [TRANSFER / 'features' / (stem + suffix) for suffix in ('.npy', '.json', '_weak.npz')]
            paths.append(TRANSFER / 'predictions' / (stem + '.npz'))
    return paths


def label_codes(labels):
    if set(np.unique(labels)) - set('HADX'):
        raise ValueError('Unknown external target')
    return np.array([{'H':0, 'A':1, 'D':2}.get(t, -100) for t in labels], np.int64)


def load_data(config, weak_study, names):
    from possession.temporal import training_weights
    strong_train, weak_train, strong_dev, weak_dev, hashes = [], [], [], [], {}
    for match in config['strong_train_matches'] + config['strong_development_matches']:
        for half in (0, 1):
            for view in VIEWS:
                path = STRONG / 'cache' / f'{match}_{half}_{view}.npy'
                record = json.loads(path.with_suffix('.json').read_text())
                provenance = record['fingerprint']
                if (provenance['match'] != match or provenance['preparation']['plan']['provided_match_used'] or
                        provenance['preparation']['plan']['public_labels_used'] or sha256(path) != record['sha256']):
                    raise ValueError('Strong external cache provenance changed')
                for filename, digest in record['auxiliary_sha256'].items():
                    if sha256(path.parent / filename) != digest:
                        raise ValueError('Strong auxiliary cache changed')
                    hashes[str((path.parent / filename).relative_to(ROOT))] = digest
                if record['names'][:664] + record['names'][960:] != names:
                    raise ValueError('Strong input schema differs')
                x = np.load(path, mmap_mode='r')
                truth = np.load(path.with_name(path.stem + '_truth.npy'))
                part = dict(match=match, half=half, view=view, x=x, y=label_codes(truth[::2]),
                    w=training_weights(truth[::2], .5), memory_names=record['names'][960:])
                hashes[str(path.relative_to(ROOT))] = record['sha256']
                hashes[str(path.with_suffix('.json').relative_to(ROOT))] = sha256(path.with_suffix('.json'))
                if match in config['strong_train_matches']:
                    strong_train.append(part)
                else:
                    part.update(truth=truth, parent=np.load(path.with_name(path.stem + '_parent.npy')))
                    strong_dev.append(part)
    preparation = json.loads((weak_study / 'weak_preparation.json').read_text())
    if (not preparation['complete'] or preparation['provided_match_used'] or preparation['public_labels_used'] or
            preparation['unexpected_competition_access'] or preparation['validation_matches_used']):
        raise ValueError('Need completed isolated weak preparation')
    for path, digest in preparation['fingerprint']['sources'].items():
        if sha256(ROOT / path) != digest:
            raise ValueError('Weak feature producer changed')
    for match in config['weak_train_matches']:
        for period in (1, 2):
            path = weak_study / 'weak_features' / f'{match}_{period}_detected.npy'
            record = json.loads(path.with_suffix('.json').read_text())
            if record['names'] != names or record['fingerprint']['preparation'] != preparation['fingerprint']:
                raise ValueError('Weak training schema or provenance differs')
            target = path.with_name(path.stem + '_weak.npy')
            if sha256(path) != record['sha256'] or sha256(target) != record['target_sha256']:
                raise ValueError('Weak training cache changed')
            labels = np.load(target)
            if set(np.unique(labels)) - set('HAX'):
                raise ValueError('Weak targets cannot contain D')
            weak_train.append(dict(match=match, half=period, x=np.load(path, mmap_mode='r'),
                y=label_codes(labels), w=np.isin(labels, ['H', 'A']).astype(np.float32)))
            hashes[str(path.relative_to(ROOT))] = record['sha256']
            hashes[str(target.relative_to(ROOT))] = record['target_sha256']
            hashes[str(path.with_suffix('.json').relative_to(ROOT))] = sha256(path.with_suffix('.json'))
    prediction_freeze = json.loads((TRANSFER / 'prediction_freeze.json').read_text())
    for match in config['weak_development_matches']:
        for period in (1, 2):
            stem = f'{match}_{period}_detected'
            path = TRANSFER / 'features' / (stem + '.npy')
            record = json.loads(path.with_suffix('.json').read_text())
            if sha256(path) != record['sha256'] or record['names'][:664] + record['names'][960:] != names:
                raise ValueError('Weak development features changed')
            target = path.with_name(stem + '_weak.npz')
            prediction = TRANSFER / 'predictions' / (stem + '.npz')
            if (sha256(target) != record['auxiliary_sha256'][target.name] or
                    sha256(prediction) != prediction_freeze['files'][prediction.name]):
                raise ValueError('Weak development truth/parent predictions changed')
            with np.load(target) as y, np.load(prediction) as p:
                weak_dev.append(dict(match=match, half=period, x=np.load(path, mmap_mode='r')[::2],
                    memory_names=record['names'][960:], ticks=y['ticks'], truth=y['truth'],
                    seen=y['detected_ball_seen'], parent=p['parent']))
            hashes[str(path.relative_to(ROOT))] = record['sha256']
            hashes[str(path.with_suffix('.json').relative_to(ROOT))] = sha256(path.with_suffix('.json'))
            hashes[str(target.relative_to(ROOT))] = record['auxiliary_sha256'][target.name]
            hashes[str(prediction.relative_to(ROOT))] = prediction_freeze['files'][prediction.name]
    return strong_train, weak_train, strong_dev, weak_dev, hashes


def evaluate(model, state, strong, weak, device):
    from possession.temporal import symmetric_probability, viterbi_labels
    scores = {}
    for view in VIEWS:
        truth, pred = [], []
        for h in strong:
            if h['view'] != view:
                continue
            q = symmetric_probability(model, model_matrix(h['x']), state['names'], state['mean'], state['scale'], device)
            p = mix_probabilities(h['parent'], q, h['x'][:, 960:], h['memory_names'], 1.)
            truth.append(h['truth']); pred.append(viterbi_labels(p, *state['decoder_costs']))
        scores[view] = score_parts(truth, pred)
    weak_rows = {'all_known':[], 'ball_missing':[]}
    for h in weak:
        q = symmetric_probability(model, model_matrix(h['x']), state['names'], state['mean'], state['scale'], device)
        p = mix_probabilities(h['parent'], q, h['x'][:, 960:], h['memory_names'], 1.)
        labels = viterbi_labels(p, *state['decoder_costs'])
        for scope, mask in [('all_known', np.ones(len(h['ticks']), bool)), ('ball_missing', ~h['seen'])]:
            weak_rows[scope].append(weak_counts(p[h['ticks']], labels[h['ticks']], h['truth'], mask))
    return scores, {scope:summarize_counts(rows) for scope, rows in weak_rows.items()}


def batch(rng, config, strong, weak, mean, scale, mapping):
    from possession.temporal import normalized
    weak_count = round(config['batch_size'] * config['weak_batch_fraction'])
    strong_count = config['batch_size'] - weak_count
    if min(weak_count, strong_count) <= 0:
        raise ValueError('Need both strong and weak examples')
    grouped = [[h for h in strong if h['view'] == view] for view in VIEWS]
    bx, by, bw = [], [], []
    native = config['strong_native_view_probability']
    for i in range(config['batch_size']):
        if i < strong_count:
            group = int(rng.choice(3, p=[native, (1. - native) / 2., (1. - native) / 2.]))
            h = grouped[group][int(rng.integers(len(grouped[group])))]
        else:
            h = weak[int(rng.integers(len(weak)))]
        a = int(rng.integers(0, len(h['x']) - config['sequence_length'] + 1))
        b = a + config['sequence_length']
        raw = h['x'][a:b]
        x = normalized(model_matrix(raw) if raw.shape[1] == 1088 else raw, mean, scale)
        y = h['y'][a:b].copy()
        if rng.random() < .5:
            x = x[:, mapping]
            y = np.where(y == 0, 1, np.where(y == 1, 0, y))
        bx.append(x.T); by.append(y); bw.append(h['w'][a:b])
    return np.stack(bx), np.stack(by), np.stack(bw), strong_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--weak-study', type=Path, default=ROOT / 'data/research/real-broadcast-adaptation-20261114')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/real_broadcast_memory_adaptation.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config['provided_match_used'] or config['public_labels_used'] or config['independent_test_matches']:
        raise ValueError('This is external development only')
    if set(config['weak_train_matches']) & set(config['weak_development_matches']) or set(config['strong_train_matches']) & set(config['strong_development_matches']):
        raise ValueError('Training and development overlap')
    args.out.mkdir(parents=True, exist_ok=True)
    denied = restrict_external(args.out, allowed_inputs(config, args.weak_study))
    device = torch_setup(args.device)
    import torch
    from possession.temporal import frame_loss
    from possession.weak_supervision import conditional_ha_loss
    from possession.boundary import boundary_loss
    from possession.external import swap_features
    from scripts.run_local import model_from_state, save_torch
    initial_dir = ROOT / config['initial_model']
    initial_selection = json.loads((initial_dir / 'selection.json').read_text())
    initial_hash = sha256(initial_dir / 'selected.pt')
    if (initial_hash != initial_selection['selected_model_sha256'] or initial_selection['provided_match_used'] or
            initial_selection['public_labels_used'] or initial_selection['unexpected_competition_access']):
        raise ValueError('Initial external checkpoint changed')
    parent_dir = ROOT / 'models/coreorigin-video-v1'
    if (sha256(parent_dir / 'selected.pt') != initial_selection['parent_sha256'] or
            sha256(parent_dir / 'fixed_protocol.json') != initial_selection['parent_protocol_sha256']):
        raise ValueError('Frozen parent member changed')
    state = torch.load(initial_dir / 'selected.pt', map_location='cpu', weights_only=False)
    strong, weak, strong_dev, weak_dev, caches = load_data(config, args.weak_study, state['names'])
    if len(weak) != 36:
        raise ValueError('Expected all eighteen external weak training matches')
    source_files = ['scripts/research/spatial_memory/train_real_broadcast_memory.py',
        'scripts/research/spatial_memory/external_guard.py', 'possession/weak_supervision.py',
        'possession/observation_gate.py', 'scripts/research/spatial_memory/audit_broadcast_transfer.py']
    freeze = dict(config=config, config_sha256=sha256(args.config), initial_model_sha256=initial_hash,
        initial_selection_sha256=sha256(initial_dir / 'selection.json'),
        reference_receipts_sha256={str((TRANSFER / name).relative_to(ROOT)):sha256(TRANSFER / name)
            for name in ('results.json', 'freeze.json', 'prediction_freeze.json')},
        weak_preparation_sha256=sha256(args.weak_study / 'weak_preparation.json'),
        sources={p:sha256(ROOT / p) for p in source_files}, caches=caches,
        weak_macro_definition='Conditional H/A macro F1 on exact 5 Hz detected-only weak development ticks.',
        normalization='Unchanged initial external training statistics', additional_training=True,
        runtime=dict(torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda, device=str(device)),
        provided_match_used=False, public_labels_used=False)
    plan_path = args.out / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != freeze:
        raise ValueError('Training protocol changed; use a new run directory')
    dump(plan_path, freeze)
    fingerprint = hashlib.sha256(json.dumps(freeze, sort_keys=True).encode()).hexdigest()
    torch.manual_seed(config['seed']); rng = np.random.default_rng(config['seed'])
    model = model_from_state(state, device)
    source_result = json.loads((TRANSFER / 'results.json').read_text())
    reference = source_result['summaries']['detected']['natural_v1']
    initial_strong, initial_weak = evaluate(model, state, strong_dev, weak_dev, device)
    for view in VIEWS:
        if abs(initial_strong[view]['spell_f1_mean'] - initial_selection['selected']['scores'][view]['spell_f1_mean']) > 1e-10:
            raise ValueError('Initial strong development scores failed reproduction')
    expected_weak = source_result['summaries']['detected']['observation_gate']
    for scope in ('all_known', 'ball_missing'):
        for metric in ('weak_ha_accuracy', 'conditional_ha_accuracy', 'conditional_ha_macro_f1'):
            if abs(initial_weak[scope][metric] - expected_weak[scope][metric]) > 1e-10:
                raise ValueError('Initial real broadcast scores failed reproduction')
    initial = dict(strong_scores=initial_strong, weak_scores=initial_weak,
        strong_mean=float(np.mean([initial_strong[v]['spell_f1_mean'] for v in VIEWS])))
    initial['objective'] = (initial['strong_mean'] + initial_weak['all_known']['conditional_ha_macro_f1']) / 2.
    best = dict(epoch=0, **initial, eligible=True, fallback=True)
    dump(args.out / 'initial_reproduction.json', dict(passed=True, baseline=best, weak_reference=reference,
        independent_test=False, provided_match_used=False))
    print(json.dumps(dict(initial_reproduced=True, strong_mean=initial['strong_mean'],
        weak_ha_accuracy=initial_weak['all_known']['weak_ha_accuracy'], joint_objective=initial['objective'])), flush=True)
    network_config = dict(state['config'])
    network_config.update(epochs=config['epochs'], learning_rate=config['learning_rate'],
        train_matches=config['strong_train_matches'] + config['weak_train_matches'],
        validation_matches=config['strong_development_matches'] + config['weak_development_matches'],
        real_broadcast_adaptation=config, training_phase='strong_HAD_and_conditional_weak_HA')
    dump(args.out / 'fixed_protocol.json', dict(config=network_config, fingerprint=fingerprint,
        preparation=freeze, baseline=best, weak_reference=reference))

    def snapshot(epoch, result):
        return dict(model=model.state_dict(), names=state['names'], mean=state['mean'], scale=state['scale'],
            config=network_config, decoder_costs=state['decoder_costs'], epoch=epoch, validation=result,
            fingerprint=fingerprint, warm_start_sha256=initial_hash, warm_start_epoch=state['epoch'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], eta_min=config['learning_rate'] / 10.)
    mapping = swap_features(np.arange(len(state['names'])), state['names']).astype(int)
    latest_path = args.out / 'latest.pt'
    history, start = [], 1
    if latest_path.exists():
        latest = torch.load(latest_path, map_location='cpu', weights_only=False)
        if latest['fingerprint'] != fingerprint or sha256(args.out / 'selected.pt') != latest['selected_sha256']:
            raise ValueError('Saved training state or selected checkpoint changed')
        model.load_state_dict(latest['model']); optimizer.load_state_dict(latest['optimizer'])
        scheduler.load_state_dict(latest['scheduler'])
        rng.bit_generator.state = latest['rng']; torch.set_rng_state(latest['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(latest['cuda_rng'])
        best, history, start = latest['best'], latest['history'], latest['epoch'] + 1
    else:
        save_torch(args.out / 'selected.pt', snapshot(0, best))
    for epoch in range(start, config['epochs'] + 1):
        started = time.monotonic(); model.train(); losses = []
        for _ in range(1 if args.smoke else config['batches_per_epoch']):
            bx, by, bw, nstrong = batch(rng, config, strong, weak, state['mean'], state['scale'], mapping)
            tx, ty, tw = [torch.as_tensor(x, device=device) for x in (bx, by, bw)]
            optimizer.zero_grad(set_to_none=True)
            logits, boundary = model.forward_with_boundary(tx)
            strong_loss = (frame_loss(logits[:nstrong], ty[:nstrong]) * tw[:nstrong]).sum() / tw[:nstrong].sum().clamp_min(1.)
            weak_loss = conditional_ha_loss(logits[nstrong:], ty[nstrong:], tw[nstrong:])
            loss = strong_loss + config['weak_loss_weight'] * weak_loss
            loss = loss + config['strong_boundary_auxiliary_weight'] * boundary_loss(boundary[:nstrong], ty[:nstrong], model.positive_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite mixed-domain training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach()))
        if args.smoke:
            dump(args.out / 'gpu_preflight.json', dict(passed=True, loss=losses[-1],
                strong_loss=float(strong_loss.detach()), conditional_weak_loss=float(weak_loss.detach()),
                batch_size=config['batch_size'], sequence_length=config['sequence_length'],
                device=str(device), trained_weights_saved=False, provided_match_used=False))
            print('Mixed strong/weak GPU optimizer preflight passed; updated weights discarded.', flush=True)
            return
        row = dict(epoch=epoch, loss=float(np.mean(losses)), seconds=round(time.monotonic() - started, 1))
        if epoch == 1 or epoch % config['validation_interval_epochs'] == 0 or epoch == config['epochs']:
            strong_scores, weak_scores = evaluate(model, state, strong_dev, weak_dev, device)
            result = joint_selection(strong_scores, weak_scores, initial, reference)
            row['validation'] = result
            if result['eligible'] and result['objective'] > best['objective'] + 1e-10:
                best = dict(epoch=epoch, **result, fallback=False)
                save_torch(args.out / 'selected.pt', snapshot(epoch, best))
        history.append(row); scheduler.step()
        selected_hash = sha256(args.out / 'selected.pt')
        save_torch(latest_path, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), best=best, history=history, epoch=epoch, fingerprint=fingerprint,
            selected_sha256=selected_hash, rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None))
        dump(args.out / 'history.json', history)
        print(json.dumps(row), flush=True)
    for path, digest in freeze['sources'].items():
        if sha256(ROOT / path) != digest:
            raise ValueError('Executed source changed during training')
    dump(args.out / 'selection.json', dict(complete=True, selected=dict(kind='observation_loss', strength=1.,
        source_epoch=best['epoch'], scores=best['strong_scores'], weak_scores=best['weak_scores'],
        objective=best['objective'], eligible=best['eligible'], fallback=best['fallback']), best=best,
        selected_model_sha256=sha256(args.out / 'selected.pt'), parent_sha256=initial_selection['parent_sha256'],
        parent_protocol_sha256=initial_selection['parent_protocol_sha256'], plan=freeze,
        initial_reproduced=True, initial_joint_objective=initial['objective'],
        provided_match_used=False, public_labels_used=False, unexpected_competition_access=denied,
        competition_target_achieved=False, independent_test=False))


if __name__ == '__main__':
    main()
