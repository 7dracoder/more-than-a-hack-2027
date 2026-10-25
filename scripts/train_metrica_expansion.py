"""A matched control/additional-data comparison, followed by one sealed partial-label audit."""
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from evaluation import possession_metrics as pm
from possession.boundary import boundary_loss
from possession.data import sha256
from possession.evaluation_guard import reserve_test_cohort
from possession.mapped_features import mapped_member, mapped_normalized
from possession.metrica_samples import load_tracking, load_events, partial_labels, controlled
from possession.sampling import short_spell_anchors, sample_crop_start
from possession.temporal import (make_model, make_weight_average, symmetric_probability,
                                training_weights, frame_loss, viterbi_labels)
from scripts.run_local import load_halves, selected_model, save_json, save_torch, score_probabilities
from scripts.snapshot_local_run import verify_source_archive

PLAN = ROOT/'configs/metrica_expansion_plan.json'


def validate_plan(plan):
    strong = plan['strong_training_matches']; extra = plan['additional_partial_label_training_matches']
    if (len(strong) != 6 or len(set(strong)) != 6 or extra != ['METRICA1', 'METRICA2'] or
            plan['reserved_partial_label_evaluation_match'] != 'METRICA3' or
            any(m in ('J03WPY', 'METRICA3') or 'BAR' in m or 'ATH' in m for m in strong) or
            not np.isfinite(plan['extra_crop_probability']) or not 0 < plan['extra_crop_probability'] <= .5):
        raise ValueError('Additional-data split/proportion violates the sealed protocol')


def extra_halves(match, corrupt=False, targets=True):
    if targets and match == 'METRICA3':
        raise ValueError('Held-out targets cannot enter a training loader')
    folder = ROOT/('data/external/metrica-features-v13'+('-corrupt' if corrupt else ''))
    path = folder/(match+'.npz'); meta = json.loads(path.with_suffix('.json').read_text())
    if sha256(path) != meta['sha256'] or any(sha256(ROOT/p) != h for p, h in meta['sources'].items()):
        raise ValueError('Additional-data cache/source changed')
    if meta['targets_present'] != targets:
        raise ValueError('Additional-data target role changed')
    result = []
    with np.load(path, allow_pickle=False) as data:
        names = data['names'].tolist()
        for i in range(2):
            row = dict(match=match, half=i, x=mapped_member(path, f'x{i}.npy', ROOT/'data/external/mapped-features', meta['sha256']))
            if targets:
                row['y'] = data[f'y{i}']; row['truth'] = row['y']
            result.append(row)
    return result, names


def train_case(plan, extra, status):
    parent = ROOT/plan['parent']; parent_selection = json.loads((parent/'selection.json').read_text())
    if sha256(parent/'selected.pt') != parent_selection['checkpoint_sha256']:
        raise ValueError('Frozen parent checkpoint changed')
    verify_source_archive(parent, json.loads((parent/'fixed_protocol.json').read_text()))
    state = torch.load(parent/'selected.pt', map_location='cpu', weights_only=False)
    name = 'additional-data' if extra else 'control'; out = ROOT/plan['output']/name
    if out.exists():
        raise ValueError('Preserve existing experimental outputs: '+str(out))
    out.mkdir(parents=True)
    config = dict(version='metrica-expansion-'+name, seed=2045, architecture='residual_boundary',
        hidden_channels=128, dropout=.2, residual_channels=64, residual_dilations=[1, 2, 4, 8, 16, 32],
        residual_dropout=.1, teacher_model_config=state['config'], teacher_feature_count=len(state['names']),
        boundary_positive_weight=20., residual_teacher_context=False,
        train_matches=plan['strong_training_matches']+(['METRICA1', 'METRICA2'] if extra else []),
        strong_training_matches=plan['strong_training_matches'],
        validation_matches=['J03WPY'], test_matches=[], development_only=True, feature_version='v13',
        features_dir='data/external/features-v13', augmentation_features_dir='data/external/features-v13-corrupt',
        sample_step=1, fps=25, cache_inference_features=True, epochs=40,
        additional_partial_label_matches=['METRICA1', 'METRICA2'] if extra else [],
        additional_partial_label_features='data/external/metrica-features-v13',
        additional_crop_probability=plan['extra_crop_probability'] if extra else 0.,
        additional_label_kind='event_derived_partial_not_dense_truth',
        training_runner='scripts/train_metrica_expansion.py', target_spell_f1=.9,
        learning_rate=.001, weight_decay=.001, batch_size=8, sequence_length=1024, ema_decay=.995)
    config_path = ROOT/f'configs/metrica_expansion_{name}.json'; save_json(config_path, config)
    sources = ['scripts/train_metrica_expansion.py', 'possession/metrica_samples.py', 'scripts/prepare_metrica_samples.py',
               'scripts/run_local.py', 'possession/temporal.py', 'possession/boundary.py', 'possession/residual.py',
               'possession/recurrent.py', 'possession/sampling.py', 'possession/mapped_features.py', 'evaluation/possession_metrics.py']
    record = dict(config=config, sources={p: sha256(ROOT/p) for p in sources},
        parent_checkpoint_sha256=sha256(parent/'selected.pt'), data_plan_sha256=sha256(PLAN),
        metrica_data_audit_sha256=sha256(ROOT/'data/local-v08/metrica_data_audit.json'),
        strong_caches={v: {m: sha256(ROOT/v/(m+'.npz')) for m in plan['strong_training_matches']+['J03WPY']}
                       for v in ('data/external/features-v13', 'data/external/features-v13-corrupt')},
        additional_caches={v: {m: sha256(ROOT/v/(m+'.npz')) for m in config['additional_partial_label_matches']}
                          for v in ('data/external/metrica-features-v13', 'data/external/metrica-features-v13-corrupt')},
        held_out_targets_read=False, normalization_rule='Frozen parent statistics only; no new-data/held-out normalization',
        comparison='Same initialization, seed, optimizer schedule and number of updates; extra-data treatment replaces 20 percent of sampled crops.')
    record['fingerprint'] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    save_json(out/'fixed_protocol.json', record)
    subprocess.run([sys.executable, 'scripts/snapshot_local_run.py', '--run', str(out)], cwd=ROOT, check=True)
    for filename in ('baseline.joblib', 'baseline_training.json', 'baseline_validation.json'):
        shutil.copyfile(parent/filename, out/filename)
    torch.manual_seed(config['seed']); rng = np.random.default_rng(config['seed'])
    model = make_model(len(state['names']), config).cuda(); model.teacher.load_state_dict(state['model'], strict=True)
    mean, scale, names = state['mean'], state['scale'], state['names']
    maps = ROOT/'data/external/mapped-features'; banks = []
    for corrupt in (False, True):
        halves, schema = load_halves(config['strong_training_matches'], ROOT/('data/external/features-v13'+('-corrupt' if corrupt else '')), maps)
        if schema != names:
            raise ValueError('Strong schema changed')
        banks.append(halves)
    strong_half_count = len(banks[0]); strong_frames = sum(len(h['x']) for bank in banks for h in bank)
    if extra:
        for i, corrupt in enumerate((False, True)):
            for match in config['additional_partial_label_matches']:
                halves, schema = extra_halves(match, corrupt)
                if schema != names:
                    raise ValueError('Additional feature schema differs')
                banks[i].extend(halves)
    tensors = []
    for bank in banks:
        group = []
        for half in bank:
            labels = pm.label_codes(half['y']).astype(np.int64); labels[labels == pm.X] = -100
            group.append((mapped_normalized(half.pop('x'), mean, scale, maps), labels,
                          training_weights(half['y'], .5), short_spell_anchors(labels, 25), np.flatnonzero(labels >= 0)))
        tensors.append(group)
    del banks, bank, half; gc.collect()
    validation, schema = load_halves(['J03WPY'], ROOT/'data/external/features-v13', maps)
    if schema != names:
        raise ValueError('Development schema changed')
    initial = [symmetric_probability(model, h['x'], names, mean, scale, 'cuda') for h in validation]
    score = score_probabilities(validation, initial, state['decoder_costs'], 1)
    if score['spell_f1_mean'] != parent_selection['validation']['spell_f1_mean']:
        raise ValueError('Zero-correction control differs from frozen parent')
    def save_selected(module, epoch, kind, score, costs):
        save_torch(out/'selected.pt', dict(model=module.state_dict(), mean=mean, scale=scale, names=names, config=config,
                   fingerprint=record['fingerprint'], epoch=epoch, weights_kind=kind, validation=score, decoder_costs=costs))
    save_selected(model, 0, 'warm_initial', score, state['decoder_costs'])
    best = score['spell_f1_mean']; history = [dict(epoch=0, validation=score, weights_kind='warm_initial')]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 40, eta_min=1e-5)
    average = make_weight_average(model, config)
    mapping = torch.as_tensor([names.index('away_'+n[5:] if n.startswith('home_') else 'home_'+n[5:] if n.startswith('away_') else n)
                               for n in names], device='cuda')
    steps = max(16, math.ceil(strong_frames/(8*1024)))
    save_json(out/'sampling_protocol.json', dict(steps_per_epoch=steps, strong_half_count=strong_half_count,
             training_normalization_matches=state['config']['train_matches'], extra_crop_probability=config['additional_crop_probability'],
             held_out_targets_read=False, metrica_sampling='Known training-frame anchors; unknown frames remain ignored'))
    for epoch in range(1, 41):
        started = time.monotonic(); model.train(); losses = []; extra_crops = 0
        for _ in range(steps):
            xs, ys, ws = [], [], []
            for _ in range(8):
                group = 0 if rng.random() < .9 else 1
                use_extra = rng.random() < config['additional_crop_probability']
                index = (strong_half_count+int(rng.integers(len(tensors[group])-strong_half_count))
                         if use_extra else int(rng.integers(strong_half_count)))
                x, y, w, anchors, known = tensors[group][index]
                if use_extra:
                    anchor = int(known[int(rng.integers(len(known)))])
                    first = int(np.clip(anchor-int(rng.integers(128, 897)), 0, len(x)-1024)); extra_crops += 1
                else:
                    first = sample_crop_start(rng, len(x), 1024, anchors, .5)
                xs.append(x[first:first+1024].T); ys.append(y[first:first+1024]); ws.append(w[first:first+1024])
            bx = torch.as_tensor(np.stack(xs), device='cuda'); by = torch.as_tensor(np.stack(ys), device='cuda')
            bw = torch.as_tensor(np.stack(ws), device='cuda'); flip = torch.as_tensor(rng.random(8) < .5, device='cuda')
            bx[flip] = bx[flip][:, mapping]; by[flip] = torch.where(by[flip] == 0, 1, torch.where(by[flip] == 1, 0, by[flip]))
            optimizer.zero_grad(set_to_none=True); logits, edge = model.forward_with_boundary(bx)
            loss = (frame_loss(logits, by)*bw).sum()/bw.sum().clamp_min(1.) + .3*boundary_loss(edge, by, 20.)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite expansion loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); average.update_parameters(model); losses.append(float(loss.detach()))
        entry = dict(epoch=epoch, loss=float(np.mean(losses)), seconds=round(time.monotonic()-started, 2), extra_crops=extra_crops)
        if epoch == 1 or epoch % 3 == 0 or epoch == 40:
            candidates = []
            for kind, module in (('raw', model), ('ema', average.module)):
                probabilities = [symmetric_probability(module, h['x'], names, mean, scale, 'cuda') for h in validation]
                for costs in plan['decoder_costs']:
                    result = score_probabilities(validation, probabilities, costs, 1)
                    candidates.append(dict(kind=kind, costs=costs, score=result))
            chosen = max(candidates, key=lambda r: r['score']['spell_f1_mean'])
            entry.update(validation=chosen['score'], weights_kind=chosen['kind'], decoder_costs=chosen['costs'])
            if chosen['score']['spell_f1_mean'] > best:
                best = chosen['score']['spell_f1_mean']
                save_selected(model if chosen['kind'] == 'raw' else average.module, epoch, chosen['kind'], chosen['score'], chosen['costs'])
        scheduler.step(); history.append(entry); save_json(out/'history.json', history)
        save_torch(out/'latest.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(), ema_model=average.state_dict(),
                   epoch=epoch, history=history, numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all(), lr_scheduler=scheduler.state_dict()))
        save_json(status, dict(stage='training', case=name, epoch=epoch, epochs=40, best_development_score=best))
        print(json.dumps(dict(case=name, **entry)), flush=True)
    for key, value in model.teacher.state_dict().items():
        if not torch.equal(value.cpu(), state['model'][key].cpu()):
            raise ValueError('Frozen teacher changed during training')
    selected = torch.load(out/'selected.pt', map_location='cpu', weights_only=False)
    save_json(out/'selection.json', dict(epoch=selected['epoch'], weights_kind=selected['weights_kind'], validation=selected['validation'],
        decoder_costs=selected['decoder_costs'], checkpoint_sha256=sha256(out/'selected.pt'), baseline_sha256=sha256(out/'baseline.joblib'),
        held_out_opened=False, official_competition_score=None))
    del model, average, optimizer, tensors, validation, initial, state, selected; gc.collect(); torch.cuda.empty_cache()
    return out, config_path


def held_out(plan, cases, directory):
    raw = ROOT/'data/external/metrica-raw'; report_path = directory/'held_out_partial_label_agreement.json'
    if report_path.exists():
        raise ValueError('Held-out evaluation already opened; preserve it')
    manifest = json.loads((raw/'download_manifest.json').read_text())
    event_hash = next(r['sha256'] for r in manifest['files'] if r['path'].endswith('Sample_Game_3_events.json'))
    model_hashes = {name: sha256(path/'selected.pt') for name, path in cases.items()}
    combined = hashlib.sha256(json.dumps(model_hashes, sort_keys=True).encode()).hexdigest()
    reserve_test_cohort(ROOT/'data/local-v08/metrica_holdout_registry.json', {'METRICA3': event_hash}, combined, directory)
    save_json(directory/'held_out_open_receipt.json', dict(models_frozen_before_opening=model_hashes,
              label_policy_sha256=sha256(ROOT/'possession/metrica_samples.py'), target_role='One held-out event-derived partial-label audit, never full-spell ground truth.'))
    tracking, roster = load_tracking(raw, 3); events = load_events(raw, 3, allow_held_out=True)
    alignment = []
    for half in tracking:
        for event in events:
            if event.period == int(half.period) and controlled(event, roster) and np.isfinite(event.xy).all():
                i = event.start-half.frames[0]
                if 0 <= i < len(half.ball) and np.isfinite(half.ball[i]).all():
                    alignment.append(float(np.linalg.norm(half.ball[i]-np.array(event.xy)*half.pitch)))
    if len(alignment) < 100 or np.median(alignment) > 2. or np.percentile(alignment, 95) > 5.:
        save_json(report_path, dict(status='inconclusive_alignment_failed', source='METRICA3', models=model_hashes,
                  no_scores_claimed=True, targets_opened_once=True))
        return
    targets = [partial_labels(h.frames, int(h.period), events, roster)[0] for h in tracking]
    feature_halves, _ = extra_halves('METRICA3', targets=False)
    truth = np.concatenate([pm.label_codes(y) for y in targets]); cm = {}; scores = {}
    for name, path in cases.items():
        model, state = selected_model(path, 'cuda'); outputs = []
        for half in feature_halves:
            p = symmetric_probability(model, half['x'], state['names'], state['mean'], state['scale'], 'cuda')
            outputs.append(pm.label_codes(viterbi_labels(p, *state['decoder_costs'])))
        matrix = pm.scored_confusion(pm.confusion(truth, np.concatenate(outputs))); cm[name] = matrix.tolist()
        scores[name] = dict(macro_HA_f1=float(np.mean([pm.per_class_prf(matrix, k)[2] for k in (0, 1)])),
            macro_HAD_f1=pm.macro_f1(matrix), known_frame_accuracy=pm.accuracy(matrix),
            alive_dead_f1=pm.alive_dead_f1(matrix), per_class={k: pm.per_class_prf(matrix, i) for i, k in enumerate('HAD')})
        del model, state; gc.collect(); torch.cuda.empty_cache()
    report = dict(match='METRICA3', scores=scores, confusion=cm, known_frames=int((truth != pm.X).sum()),
          total_frames=len(truth), coverage=float((truth != pm.X).mean()), label_counts={k: int((truth == i).sum()) for i, k in enumerate('HADX')},
          models=model_hashes, policy_sha256=sha256(ROOT/'possession/metrica_samples.py'),
          event_alignment_metres=dict(median=float(np.median(alignment)), p95=float(np.percentile(alignment, 95))),
          scope='Held-out agreement with event-derived partial labels only. No dense spell F1, human-verified match accuracy, or official score is claimed.',
          used_for_training_or_model_selection=False, official_competition_score=None)
    save_json(report_path, report); return report


def main():
    plan = json.loads(PLAN.read_text()); validate_plan(plan)
    directory = ROOT/plan['output']; directory.mkdir(parents=True, exist_ok=True)
    status = directory/'status.json'; save_json(status, dict(stage='waiting_for_data_audit'))
    deadline = time.monotonic()+14400
    while not (ROOT/'data/local-v08/metrica_data_audit.json').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Metrica preparation did not finish')
        time.sleep(5)
    audit = json.loads((ROOT/'data/local-v08/metrica_data_audit.json').read_text())
    if audit['held_out_labels_parsed'] or audit['training_matches'] != ['METRICA1', 'METRICA2']:
        raise ValueError('Data split audit failed')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    cases = {}; configs = {}
    for extra in (False, True):
        out, config = train_case(plan, extra, status); cases[out.name] = out; configs[out.name] = config
    selections = {name: json.loads((path/'selection.json').read_text()) for name, path in cases.items()}
    winner = max(selections, key=lambda name: selections[name]['validation']['spell_f1_mean'])
    save_json(directory/'frozen_development_comparison.json', dict(winner=winner, selections=selections,
             parent_score=plan['parent_score'], held_out_targets_read=False,
             data_gain=selections['additional-data']['validation']['spell_f1_mean']-selections['control']['validation']['spell_f1_mean']))
    save_json(status, dict(stage='held_out_partial_label_audit', winner_frozen=winner))
    held_out(plan, {'parent': ROOT/plan['parent'], **cases}, directory)
    for name, path in cases.items():
        commands = [(['scripts/audit_local_errors.py', '--run', str(path), '--out', str(path/'fresh-audit')], 'fresh-audit.log'),
                    (['scripts/evaluate_gap_stress.py', '--run', str(path)], 'gap-stress.log')]
        for args, log in commands:
            with (path/log).open('w') as stream:
                subprocess.run([sys.executable, *args], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        fresh = json.loads((path/'fresh-audit/audit.json').read_text())['scores']['spell_f1_mean']
        if fresh != selections[name]['validation']['spell_f1_mean']:
            raise ValueError('Fresh development audit disagrees')
        save_json(status, dict(stage='two_full_exports', case=name, development_score=fresh))
        for number in (1, 2):
            with (path/f'export-{number}.log').open('w') as stream:
                subprocess.run([sys.executable, 'scripts/run_local.py', '--infer', '--config', str(configs[name]), '--out', str(path)],
                               cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            csv = path/'candidate/CoreOrigin.csv'
            if number == 1:
                shutil.copyfile(csv, path/'candidate/CoreOrigin.first-pass.csv')
            elif sha256(csv) != sha256(path/'candidate/CoreOrigin.first-pass.csv'):
                raise ValueError('Repeated exports differ')
        save_json(path/'reproducibility.json', dict(two_complete_passes_byte_identical=True, rows=199319,
                  sha256=sha256(csv), selected_sha256=sha256(path/'selected.pt'), official_competition_score=None))
    save_json(status, dict(stage='complete_and_stopped', selections={k: v['validation']['spell_f1_mean'] for k, v in selections.items()},
        winner=winner, parent_score=plan['parent_score'],
        data_gain=selections['additional-data']['validation']['spell_f1_mean']-selections['control']['validation']['spell_f1_mean'],
        held_out_scope='Event-derived partial-label agreement only', two_complete_exports_per_case=True, official_competition_score=None))


if __name__ == '__main__':
    main()
