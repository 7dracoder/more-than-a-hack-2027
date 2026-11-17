"""Fixed external partial-HAD study with strong/weak replay and frozen decoder."""
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
from possession.observation_gate import mix_probabilities, observation_loss
from possession.uncertain_decode import decode_labels
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump, score_parts, torch_setup
from scripts.research.spatial_memory.train_real_broadcast_memory import allowed_inputs, load_data, label_codes, VIEWS
from scripts.research.spatial_memory.train_natural_visibility import model_matrix
from scripts.research.spatial_memory.audit_broadcast_transfer import weak_counts, summarize_counts


def selection_result(scores, weak, baseline):
    objective = float(np.mean([scores[v]['spell_f1_mean'] for v in VIEWS]))
    guards = dict(native=scores['native']['spell_f1_mean'] >= baseline['scores']['native']['spell_f1_mean'] - 1e-12,
        weak_ha=weak['all_known']['weak_ha_accuracy'] >= baseline['weak_scores']['all_known']['weak_ha_accuracy'] - .005,
        weak_missing=weak['ball_missing']['weak_ha_accuracy'] >= baseline['weak_scores']['ball_missing']['weak_ha_accuracy'] - .01)
    return dict(scores=scores, weak_scores=weak, objective=objective, guards=guards, eligible=all(guards.values()))


def evaluate(model, state, strong, weak, device, penalty):
    from possession.temporal import symmetric_probability
    scores = {}
    for view in VIEWS:
        truth, predicted = [], []
        for h in strong:
            if h['view'] != view:
                continue
            q = symmetric_probability(model, model_matrix(h['x']), state['names'], state['mean'], state['scale'], device)
            p = mix_probabilities(h['parent'], q, h['x'][:, 960:], h['memory_names'], 1.)
            gate = np.repeat(observation_loss(h['x'][:, 960:], h['memory_names']), 2)[:len(p)]
            truth.append(h['truth'])
            predicted.append(decode_labels(p, gate, penalty, *state['decoder_costs']))
        scores[view] = score_parts(truth, predicted)
    weak_rows = {'all_known':[], 'ball_missing':[]}
    for h in weak:
        q = symmetric_probability(model, model_matrix(h['x']), state['names'], state['mean'], state['scale'], device)
        p = mix_probabilities(h['parent'], q, h['x'][:, 960:], h['memory_names'], 1.)
        gate = np.repeat(observation_loss(h['x'][:, 960:], h['memory_names']), 2)[:len(p)]
        labels = decode_labels(p, gate, penalty, *state['decoder_costs'])
        for scope, mask in [('all_known', np.ones(len(h['ticks']), bool)), ('ball_missing', ~h['seen'])]:
            weak_rows[scope].append(weak_counts(p[h['ticks']], labels[h['ticks']], h['truth'], mask))
    return scores, {scope:summarize_counts(rows) for scope,rows in weak_rows.items()}


def batch(rng, config, strong, weak, partial, state, mapping):
    from possession.temporal import normalized
    ns, nw, npart = config['batch_domain_counts']
    assert ns + nw + npart == config['batch_size'] and min(ns, nw, npart) > 0
    grouped = [[h for h in strong if h['view'] == view] for view in VIEWS]
    bx, by, bw = [], [], []
    for i in range(config['batch_size']):
        if i < ns:
            native = config['strong_native_view_probability']
            group = int(rng.choice(3, p=[native, (1-native)/2, (1-native)/2]))
            choices = grouped[group]
        elif i < ns + nw:
            choices = weak
        else:
            choices = partial
        h = choices[int(rng.integers(len(choices)))]
        first = int(rng.integers(0, len(h['x']) - config['sequence_length'] + 1))
        last = first + config['sequence_length']
        raw = h['x'][first:last]
        x = normalized(model_matrix(raw) if raw.shape[1] == 1088 else raw, state['mean'], state['scale'])
        y = h['y'][first:last].copy()
        if rng.random() < .5:
            x = x[:, mapping]
            y = np.where(y == 0, 1, np.where(y == 1, 0, y))
        bx.append(x.T); by.append(y); bw.append(h['w'][first:last])
    return np.stack(bx), np.stack(by), np.stack(bw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--partial-study', type=Path, default=ROOT / 'data/research/partial-had-adaptation-20261114')
    parser.add_argument('--weak-study', type=Path, default=ROOT / 'data/research/real-broadcast-adaptation-20261114')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config_path = ROOT / 'configs/partial_had_memory_adaptation.json'
    config = json.loads(config_path.read_text())
    assert not config['provided_match_used'] and not config['public_labels_used'] and not config['independent_test_matches']
    assert config['partial_training_matches'] == ['METRICA1', 'METRICA2'] and config['decoder_penalty'] == 8.
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'selection.json').exists():
        raise ValueError('Preserve the completed study')
    initial_dir = ROOT / config['initial_model']
    origin = json.loads((initial_dir / 'selection.json').read_text())
    inherited = origin['plan']['config']
    partial_files = [args.partial_study / 'partial_preparation.json', args.partial_study / 'preparation_plan.json']
    for number in (1, 2):
        for half in (0, 1):
            for view in config['partial_views']:
                path = args.partial_study / 'partial_features' / f'METRICA{number}_{half}_{view}.npy'
                partial_files += [path, path.with_suffix('.json'), path.with_name(path.stem + '_truth.npy')]
    denied = restrict_external(args.out, allowed_inputs(inherited, args.weak_study) + partial_files)
    # Do not access the diagnostic study, nor even raw Game 3 during this study.
    def no_game3(event, values):
        if event == 'open' and values and isinstance(values[0], (str, bytes)):
            if 'sample_game_3' in str(values[0]).lower():
                raise PermissionError('Training excludes Metrica Game 3')
    sys.addaudithook(no_game3)
    device = torch_setup(args.device)
    import torch
    from possession.temporal import frame_loss, training_weights
    from possession.weak_supervision import conditional_ha_loss
    from possession.boundary import boundary_loss
    from possession.external import swap_features
    from scripts.run_local import model_from_state, save_torch
    assert origin['complete'] and not origin['provided_match_used'] and not origin['public_labels_used']
    assert sha256(initial_dir / 'selected.pt') == origin['selected_model_sha256']
    continuity = json.loads((ROOT / config['continuity_model'] / 'selection.json').read_text())
    assert continuity['selected_model_sha256'] == origin['selected_model_sha256']
    assert continuity['selected']['penalty'] == config['decoder_penalty']
    state = torch.load(initial_dir / 'selected.pt', map_location='cpu', weights_only=False)
    strong, weak, strong_dev, weak_dev, caches = load_data(inherited, args.weak_study, state['names'])
    preparation = json.loads((args.partial_study / 'partial_preparation.json').read_text())
    assert preparation['complete'] and not preparation['provided_match_used'] and not preparation['public_labels_used']
    assert not preparation['validation_matches_used'] and not preparation['unexpected_competition_access']
    for path,digest in preparation['plan']['sources'].items():
        assert sha256(ROOT / path) == digest
    partial = []
    for path in partial_files:
        caches[str(path.relative_to(ROOT))] = sha256(path)
        if path.suffix != '.json' or path.name in ['partial_preparation.json', 'preparation_plan.json']:
            continue
        record = json.loads(path.read_text())
        data_path = path.with_suffix('.npy')
        target_path = path.with_name(path.stem + '_truth.npy')
        assert sha256(data_path) == record['sha256'] and sha256(target_path) == record['target_sha256']
        assert record['names'] == state['names'] and record['fingerprint']['plan'] == preparation['plan']
        truth = np.load(target_path)
        partial.append(dict(x=np.load(data_path, mmap_mode='r'), y=label_codes(truth), w=training_weights(truth, .5)))
    assert len(partial) == 8
    source_names = ['scripts/research/spatial_memory/train_partial_had_memory.py',
        'scripts/research/spatial_memory/train_real_broadcast_memory.py',
        'scripts/research/spatial_memory/train_natural_visibility.py',
        'scripts/research/spatial_memory/external_guard.py', 'possession/uncertain_decode.py',
        'possession/observation_gate.py', 'possession/weak_supervision.py',
        'possession/temporal.py', 'possession/boundary.py', 'possession/external.py',
        'scripts/run_local.py',
        'scripts/research/spatial_memory/audit_broadcast_transfer.py']
    plan = dict(config=config, inherited_config=inherited, config_sha256=sha256(config_path),
        sources={path:sha256(ROOT / path) for path in source_names}, caches=caches,
        initial_model_sha256=sha256(initial_dir / 'selected.pt'),
        initial_protocol_sha256=sha256(initial_dir / 'fixed_protocol.json'),
        initial_selection_sha256=sha256(initial_dir / 'selection.json'),
        continuity_selection_sha256=sha256(ROOT / config['continuity_model'] / 'selection.json'),
        provided_match_used=False, public_labels_used=False, independent_test=False)
    if (args.out / 'plan.json').exists():
        assert json.loads((args.out / 'plan.json').read_text()) == plan
    dump(args.out / 'plan.json', plan)
    fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    torch.manual_seed(config['seed']); rng = np.random.default_rng(config['seed'])
    model = model_from_state(state, device)
    scores, weak_scores = evaluate(model, state, strong_dev, weak_dev, device, config['decoder_penalty'])
    for view in VIEWS:
        assert abs(scores[view]['spell_f1_mean'] - continuity['selected']['scores'][view]['spell_f1_mean']) < 1e-10
    for scope in weak_scores:
        assert abs(weak_scores[scope]['weak_ha_accuracy'] - continuity['selected']['weak_scores'][scope]['weak_ha_accuracy']) < 1e-10
    baseline = dict(scores=scores, weak_scores=weak_scores)
    best = dict(epoch=0, fallback=True, **selection_result(scores, weak_scores, baseline))
    dump(args.out / 'initial_reproduction.json', dict(passed=True, baseline=best))
    print(json.dumps(dict(initial_reproduced=True, strong_mean=best['objective'])), flush=True)
    network_config = dict(state['config'])
    network_config.update(train_matches=inherited['strong_train_matches'] + inherited['weak_train_matches'] + config['partial_training_matches'],
        partial_had_adaptation=config, training_phase='strong_weak_partial_HAD', epochs=config['epochs'])
    dump(args.out / 'fixed_protocol.json', dict(config=network_config, fingerprint=fingerprint, preparation=plan, baseline=best))

    def snapshot(epoch, result):
        return dict(model=model.state_dict(), names=state['names'], mean=state['mean'], scale=state['scale'],
            config=network_config, decoder_costs=state['decoder_costs'], epoch=epoch, validation=result,
            fingerprint=fingerprint, warm_start_sha256=plan['initial_model_sha256'], warm_start_epoch=state['epoch'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], eta_min=config['learning_rate']/10.)
    mapping = swap_features(np.arange(len(state['names'])), state['names']).astype(int)
    history, start = [], 1
    if (args.out / 'latest.pt').exists():
        latest = torch.load(args.out / 'latest.pt', map_location='cpu', weights_only=False)
        assert latest['fingerprint'] == fingerprint and latest['selected_sha256'] == sha256(args.out / 'selected.pt')
        model.load_state_dict(latest['model']); optimizer.load_state_dict(latest['optimizer']); scheduler.load_state_dict(latest['scheduler'])
        best, history, start = latest['best'], latest['history'], latest['epoch'] + 1
        rng.bit_generator.state = latest['rng']; torch.set_rng_state(latest['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(latest['cuda_rng'])
    elif not args.smoke:
        save_torch(args.out / 'selected.pt', snapshot(0, best))
    ns, nw, _ = config['batch_domain_counts']
    for epoch in range(start, config['epochs'] + 1):
        started = time.monotonic(); model.train(); losses = []
        for _ in range(1 if args.smoke else config['batches_per_epoch']):
            bx,by,bw = batch(rng, config, strong, weak, partial, state, mapping)
            tx,ty,tw = [torch.as_tensor(a, device=device) for a in (bx,by,bw)]
            optimizer.zero_grad(set_to_none=True)
            logits, boundary = model.forward_with_boundary(tx)
            strong_loss = (frame_loss(logits[:ns], ty[:ns]) * tw[:ns]).sum() / tw[:ns].sum().clamp_min(1.)
            weak_loss = conditional_ha_loss(logits[ns:ns+nw], ty[ns:ns+nw], tw[ns:ns+nw])
            partial_loss = (frame_loss(logits[ns+nw:], ty[ns+nw:]) * tw[ns+nw:]).sum() / tw[ns+nw:].sum().clamp_min(1.)
            loss = strong_loss + config['weak_loss_weight'] * weak_loss + config['partial_loss_weight'] * partial_loss
            loss += config['strong_boundary_auxiliary_weight'] * boundary_loss(boundary[:ns], ty[:ns], model.positive_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite partial-HAD training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach()))
        if args.smoke:
            dump(args.out / 'gpu_preflight.json', dict(passed=True, loss=losses[-1],
                trained_weights_saved=False, provided_match_used=False))
            print('Partial-HAD GPU optimizer preflight passed; updated weights discarded.', flush=True)
            return
        row = dict(epoch=epoch, loss=float(np.mean(losses)), seconds=time.monotonic()-started)
        if epoch == 1 or epoch % config['validation_interval_epochs'] == 0 or epoch == config['epochs']:
            scores, weak_scores = evaluate(model, state, strong_dev, weak_dev, device, config['decoder_penalty'])
            result = selection_result(scores, weak_scores, baseline)
            row['validation'] = result
            if result['eligible'] and result['objective'] > best['objective'] + 1e-10:
                best = dict(epoch=epoch, fallback=False, **result)
                save_torch(args.out / 'selected.pt', snapshot(epoch, best))
        history.append(row); scheduler.step()
        save_torch(args.out / 'latest.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            epoch=epoch, best=best, history=history, fingerprint=fingerprint, selected_sha256=sha256(args.out / 'selected.pt'),
            rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None))
        dump(args.out / 'history.json', history)
        print(json.dumps(row), flush=True)
    for path,digest in {**plan['sources'], **plan['caches']}.items():
        assert sha256(ROOT / path) == digest
    dump(args.out / 'selection.json', dict(complete=True, selected=dict(kind='uncertain_observation_loss', strength=1.,
        penalty=config['decoder_penalty'], source_epoch=best['epoch'], **best), plan=plan,
        selected_model_sha256=sha256(args.out / 'selected.pt'), parent_sha256=origin['parent_sha256'],
        parent_protocol_sha256=origin['parent_protocol_sha256'], initial_reproduced=True,
        provided_match_used=False, public_labels_used=False, unexpected_competition_access=denied,
        independent_test=False, competition_target_achieved=False))


if __name__ == '__main__':
    main()
