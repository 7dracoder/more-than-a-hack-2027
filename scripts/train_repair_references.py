"""Three fixed-epoch reference fits, each excluding two repair-study matches."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from evaluation import possession_metrics as pm
from possession.boundary import boundary_loss
from possession.data import sha256
from possession.mapped_features import mapped_normalized
from possession.sampling import short_spell_anchors, sample_crop_start
from possession.temporal import (make_model, make_weight_average, symmetric_normalization,
                                symmetric_probability, training_weights, frame_loss)
from scripts.run_local import load_halves, save_json, save_torch, selected_model


def validate_plan(plan):
    matches = plan['training_matches']
    pairs = plan['excluded_pairs']
    if len(matches) != 6 or len(set(matches)) != 6 or len(pairs) != 3:
        raise ValueError('Expected six distinct training matches in three excluded pairs')
    if sorted(sum(pairs, [])) != sorted(matches) or any(len(p) != 2 for p in pairs):
        raise ValueError('Each training match must appear in exactly one excluded pair')
    if 'J03WPY' in matches or any('BAR' in m or 'ATH' in m for m in matches):
        raise ValueError('Development/challenge matches cannot enter repair training')


def train_reference(plan, pair, out, status):
    train = [m for m in plan['training_matches'] if m not in pair]
    config = dict(plan['reference'], train_matches=train, validation_matches=[], test_matches=[],
                  development_only=True, feature_version='v13', features_dir='data/external/features-v13',
                  sample_step=1, fps=25)
    if out.exists():
        raise ValueError('Reference output already exists; preserve it: ' + str(out))
    out.mkdir(parents=True)
    sources = ['scripts/train_repair_references.py', 'scripts/run_local.py', 'possession/temporal.py',
               'possession/boundary.py', 'possession/recurrent.py', 'possession/sampling.py',
               'possession/mapped_features.py', 'possession/external.py', 'possession/features_v13.py',
               'evaluation/possession_metrics.py']
    record = dict(config=config, sources={p: sha256(ROOT/p) for p in sources},
                  excluded_matches=pair, selection_rule='fixed_final_epoch_ema_no_validation',
                  training_caches={v: {m: sha256(ROOT/v/(m+'.npz')) for m in train}
                                   for v in ('data/external/features-v13', 'data/external/features-v13-corrupt')})
    record['fingerprint'] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    save_json(out/'fixed_protocol.json', record)
    # Capture inspectable sources before optimization; all required producers stay frozen.
    import subprocess
    subprocess.run([sys.executable, 'scripts/snapshot_local_run.py', '--run', str(out)], cwd=ROOT, check=True)
    torch.manual_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])
    maps = ROOT/'data/external/mapped-features'
    native, names = load_halves(train, ROOT/config['features_dir'], maps)
    augmented, other = load_halves(train, ROOT/'data/external/features-v13-corrupt', maps)
    if names != other:
        raise ValueError('Reference native/corrupted feature schemas differ')
    for a, b in zip(native, augmented):
        if not np.array_equal(a['y'], b['y']):
            raise ValueError('Corruption altered training labels')
    bank = native + augmented
    mean, scale = symmetric_normalization([h['x'] for h in bank], names)
    tensors = []
    for half in bank:
        labels = pm.label_codes(half['y']).astype(np.int64)
        labels[labels == pm.X] = -100
        tensors.append((mapped_normalized(half.pop('x'), mean, scale, maps), labels,
                        training_weights(half['y'], .5), short_spell_anchors(labels, 25)))
    del bank, native, augmented, half
    model = make_model(len(names), config).cuda()
    average = make_weight_average(model, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], eta_min=1e-5)
    mapping = torch.tensor([names.index('away_'+n[5:] if n.startswith('home_') else
                                       'home_'+n[5:] if n.startswith('away_') else n) for n in names], device='cuda')
    size, length = config['batch_size'], config['sequence_length']
    steps = max(16, math.ceil(sum(len(t[0]) for t in tensors)/(size*length)))
    history = []
    for epoch in range(1, config['epochs']+1):
        started = time.monotonic(); model.train(); losses = []
        for _ in range(steps):
            xs, ys, ws = [], [], []
            for _ in range(size):
                view = 0 if rng.random() < .9 else 8
                x, y, w, anchors = tensors[view+int(rng.integers(8))]
                first = sample_crop_start(rng, len(x), length, anchors, .5)
                xs.append(x[first:first+length].T); ys.append(y[first:first+length]); ws.append(w[first:first+length])
            bx = torch.as_tensor(np.stack(xs), device='cuda')
            by = torch.as_tensor(np.stack(ys), device='cuda')
            bw = torch.as_tensor(np.stack(ws), device='cuda')
            if rng.random() < .5:
                bx = bx[:, mapping]; by = torch.where(by == 0, 1, torch.where(by == 1, 0, by))
            optimizer.zero_grad(set_to_none=True)
            logits, edge = model.forward_with_boundary(bx)
            loss = (frame_loss(logits, by)*bw).sum()/bw.sum().clamp_min(1.)
            loss = loss + .3*boundary_loss(edge, by, 20.)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite reference loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); average.update_parameters(model); losses.append(float(loss.detach()))
        scheduler.step()
        entry = dict(epoch=epoch, loss=float(np.mean(losses)), seconds=round(time.monotonic()-started, 2))
        history.append(entry); save_json(out/'history.json', history)
        save_json(status, dict(stage='reference_training', excluded_matches=pair, train_matches=train,
                               epoch=epoch, epochs=config['epochs'], output=str(out)))
        print(json.dumps(dict(reference=out.name, **entry)), flush=True)
    save_torch(out/'selected.pt', dict(model=average.module.state_dict(), names=names, mean=mean, scale=scale,
                config=config, epoch=config['epochs'], weights_kind='fixed_final_ema',
                decoder_costs=plan['decoder_costs'], fingerprint=record['fingerprint']))
    save_json(out/'selection.json', dict(epoch=config['epochs'], checkpoint_sha256=sha256(out/'selected.pt'),
                selection_rule='fixed_final_epoch_ema_no_validation', optimization_matches=train,
                excluded_matches=pair, no_development_targets_read=True))
    del model, average, optimizer, tensors
    gc.collect(); torch.cuda.empty_cache()
    # Excluded targets are first loaded after the reference weights are frozen.
    model, state = selected_model(out, 'cuda')
    for match in pair:
        halves, current = load_halves([match], ROOT/config['features_dir'], maps)
        assert current == names
        arrays = {f'p{i}': symmetric_probability(model, h['x'], names, mean, scale, 'cuda')
                  for i, h in enumerate(halves)}
        target = out/(match+'-predictions.npz')
        np.savez_compressed(target, **arrays)
        save_json(target.with_suffix('.json'), dict(match=match, reference_train_matches=train,
                  reference_excluded_matches=pair, checkpoint_sha256=sha256(out/'selected.pt'),
                  sha256=sha256(target), probability_targets_read=False,
                  input_cache_sha256=sha256(ROOT/config['features_dir']/(match+'.npz'))))
        del halves
    del model; gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT/'configs/transition_repair_v1.json')
    args = parser.parse_args(); plan = json.loads(args.plan.read_text()); validate_plan(plan)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the local reference fits')
    status = ROOT/plan['output']/'reference_status.json'; status.parent.mkdir(parents=True, exist_ok=True)
    for i, pair in enumerate(plan['excluded_pairs']):
        train_reference(plan, pair, status.parent/('reference-'+str(i)), status)
    save_json(status, dict(stage='references_complete', pairs=plan['excluded_pairs'],
                           plan_sha256=sha256(args.plan), no_development_targets_read=True))


if __name__ == '__main__':
    main()
