"""One bounded, cross-fitted segment-edit study and two full on-device exports."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from evaluation import possession_metrics as pm
from possession.data import sha256, slot_teams
from possession.temporal import symmetric_probability, viterbi_labels
from possession.transition_repair import (Edit, RepairMLP, propose, edit_features, supervision,
                                         apply_edits, choose_edits, fast_spell_score)
from scripts.run_local import load_halves, selected_model, save_json, save_torch
from scripts.train_repair_references import validate_plan


def packet(probability, half, names, config, targets=False):
    labels = pm.label_codes(viterbi_labels(probability, *config['decoder_costs']))
    edits = propose(probability, labels, config['candidate'])
    x = edit_features(probability, labels, half['x'], names, edits)
    result = dict(x=x, edits=edits, labels=labels, truth=pm.label_codes(half['truth']))
    if targets:
        result['gain'] = supervision(result['truth'], labels, edits)
    return result


def fit(rows, config):
    x = np.concatenate([row['x'] for row in rows])
    y = np.concatenate([row['gain'] for row in rows])
    known = np.isfinite(y); x, y = x[known], y[known]
    if len(x) < 100 or not np.isfinite(x).all():
        raise ValueError('Too few valid repair examples')
    mean = x.mean(0); scale = np.maximum(x.std(0), .001)
    x = np.clip((x-mean)/scale, -10, 10).astype(np.float32)
    torch.manual_seed(config['seed'])
    model = RepairMLP(x.shape[1], config['hidden_channels']).cuda()
    bx = torch.as_tensor(x, device='cuda'); by = torch.as_tensor(y, device='cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    rng = np.random.default_rng(config['seed']); history = []
    for epoch in range(config['epochs']):
        model.train(); losses = []
        order = rng.permutation(len(x))
        for first in range(0, len(order), config['batch_size']):
            index = torch.as_tensor(order[first:first+config['batch_size']], device='cuda')
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.smooth_l1_loss(model(bx[index]), by[index], beta=.25)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite repair loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    model.eval()
    return model, mean, scale, dict(examples=len(x), positive=int((y > 1e-6).sum()),
                                  negative=int((y < -1e-6).sum()), final_loss=history[-1], history=history)


def predict(model, mean, scale, row):
    if not len(row['x']):
        return np.empty(0, np.float32)
    values = np.clip((row['x']-mean)/scale, -10, 10).astype(np.float32)
    with torch.inference_mode():
        return np.concatenate([model(torch.as_tensor(values[a:a+4096], device='cuda')).cpu().numpy()
                               for a in range(0, len(values), 4096)])


def score_rows(rows, predictions):
    truth = np.concatenate([np.r_[r['truth'], pm.X] for r in rows])
    labels = np.concatenate([np.r_[p, pm.X] for p in predictions])
    f1, _ = fast_spell_score(truth, labels)
    cm = pm.scored_confusion(pm.confusion(truth, labels))
    graded = pm.valid_runs(truth)
    gt = pm.possession_spells(truth, graded); pred = pm.possession_spells(labels, graded)
    flags = pm.matched_gt_flags(pm.spell_overlaps(gt, pred), len(gt), .5)
    short = [i for i, (a, b, _) in enumerate(gt) if b-a < 25]
    return dict(spell_f1_mean=f1, macro_f1=pm.macro_f1(cm), alive_dead_f1=pm.alive_dead_f1(cm),
                n_gt_spells=len(gt), n_pred_spells=len(pred), short_gt=len(short),
                short_matched=sum(bool(flags[i]) for i in short))


def corrected(row, gain, threshold, enabled=True):
    edits = choose_edits(row['edits'], gain, threshold) if enabled else []
    return apply_edits(row['labels'], edits), edits


def prepare_training(plan, directory):
    datasets = {}; names = None
    maps = ROOT/'data/external/mapped-features'
    for i, pair in enumerate(plan['excluded_pairs']):
        reference = directory/('reference-'+str(i))
        selection = json.loads((reference/'selection.json').read_text())
        train = [m for m in plan['training_matches'] if m not in pair]
        if selection['optimization_matches'] != train or selection['excluded_matches'] != pair:
            raise ValueError('Reference fold membership changed')
        if sha256(reference/'selected.pt') != selection['checkpoint_sha256']:
            raise ValueError('Reference checkpoint changed')
        for match in pair:
            input_path = reference/(match+'-predictions.npz')
            receipt = json.loads(input_path.with_suffix('.json').read_text())
            if (receipt['reference_train_matches'] != train or receipt['reference_excluded_matches'] != pair or
                    receipt['sha256'] != sha256(input_path) or receipt['checkpoint_sha256'] != selection['checkpoint_sha256']):
                raise ValueError('Out-of-fold prediction provenance changed')
            halves, current = load_halves([match], ROOT/'data/external/features-v13', maps)
            if names is None:
                names = current
            if current != names:
                raise ValueError('Repair feature schemas differ')
            rows = []
            with np.load(input_path, allow_pickle=False) as stored:
                for half_index, half in enumerate(halves):
                    started = time.monotonic()
                    row = packet(stored['p'+str(half_index)], half, names, plan, targets=True)
                    rows.append(row)
                    target = directory/'repair'/f'{match}-{half_index}-examples.npz'
                    np.savez_compressed(target, x=row['x'], gain=row['gain'],
                                        edits=np.array([[e.start, e.stop, e.target, ('shift', 'insert', 'delete').index(e.kind)]
                                                        for e in row['edits']], dtype=np.int32).reshape(-1, 4))
                    print(json.dumps(dict(stage='repair_examples', match=match, half=half_index,
                                          candidates=len(row['edits']), positive=int((row['gain'] > 1e-6).sum()),
                                          seconds=round(time.monotonic()-started, 2))), flush=True)
            datasets[match] = rows
            del halves; gc.collect()
    return datasets, names


def evaluate_development(plan, repair, model, mean, scale, enabled, threshold, feature_names):
    parent = ROOT/plan['parent_run']; teacher, state = selected_model(parent, 'cuda')
    if state['names'] != feature_names or state['decoder_costs'] != plan['decoder_costs']:
        raise ValueError('Production teacher schema/decoder differs')
    halves, names = load_halves(plan['development_matches'], ROOT/'data/external/features-v13', ROOT/'data/external/mapped-features')
    rows = []; predictions = []; all_edits = []; arrays = {}
    for i, half in enumerate(halves):
        p = symmetric_probability(teacher, half['x'], names, state['mean'], state['scale'], 'cuda')
        row = packet(p, half, names, plan)
        gains = predict(model, mean, scale, row)
        changed, edits = corrected(row, gains, threshold, enabled)
        rows.append(row); predictions.append(changed); all_edits.append([e.__dict__ for e in edits])
        arrays.update({f'p{i}': p, f'base{i}': row['labels'], f'prediction{i}': changed, f'truth{i}': row['truth']})
    baseline = score_rows(rows, [r['labels'] for r in rows]); result = score_rows(rows, predictions)
    truth = np.concatenate([np.r_[r['truth'], pm.X] for r in rows])
    output = np.concatenate([np.r_[p, pm.X] for p in predictions])
    official = pm.evaluate(truth, output)
    if official['spell_f1_mean'] != result['spell_f1_mean']:
        raise ValueError('Final official scoring differs from helper')
    expected = json.loads((parent/'selection.json').read_text())['validation']['spell_f1_mean']
    if baseline['spell_f1_mean'] != expected:
        raise ValueError('Fresh teacher control does not reproduce selected development score')
    report = dict(scope='Repeatedly used development match only; not independent/official evidence.',
                  baseline=baseline, repaired=result, edits=all_edits, enabled=enabled, threshold=threshold,
                  parent_checkpoint_sha256=sha256(parent/'selected.pt'), official_scorer_sha256=sha256(ROOT/'evaluation/possession_metrics.py'),
                  improved=result['spell_f1_mean'] > baseline['spell_f1_mean'], official_competition_score=None)
    save_json(repair/'development_audit.json', report)
    np.savez_compressed(repair/'development_probabilities.npz', **arrays)
    del teacher, halves; gc.collect(); torch.cuda.empty_cache()
    return report


def export(plan, repair, model, mean, scale, enabled, threshold, feature_names, number):
    """Read automatic tactical coordinates only; run both halves afresh on CUDA."""
    from games import GAME, GAMES, asset_paths, half_video_blocks
    from possession.features_v13 import build_features
    from possession.inference_cache import cached_block
    import scipy
    paths = asset_paths(data_dir=str(ROOT/'data'))
    hashes = {k: sha256(paths[k]) for k in ('atd_raw', 'atd_meta')}
    n = GAMES[GAME]['n_video_frames']; pitch = GAMES[GAME]['pitch']; teams = slot_teams(paths['atd_meta'])
    players = np.full((n, len(teams), 2), np.nan, np.float32); ball = np.full((n, 2), np.nan, np.float32)
    count = 0
    with open(paths['atd_raw']) as stream:
        for i, line in enumerate(stream):
            fields = line.strip().split(':')
            if len(fields) != 3 or int(fields[0]) != i+1 or i >= n:
                raise ValueError('Tactical timeline changed')
            row = np.fromstring(fields[1].replace(';', ','), sep=','); point = np.fromstring(fields[2], sep=',')
            if row.size != len(teams)*3 or point.size < 2:
                raise ValueError('Malformed tactical tracking')
            players[i] = row.reshape(-1, 3)[:, :2]*pitch; ball[i] = point[:2]*pitch; count += 1
    if count != n or hashes != {k: sha256(paths[k]) for k in hashes}:
        raise ValueError('Incomplete or unstable tactical input')
    teacher, state = selected_model(ROOT/plan['parent_run'], 'cuda')
    base = np.full(n, 'X', dtype='<U1'); labels = base.copy(); receipts = []; edits_total = 0
    sources = {p: sha256(ROOT/p) for p in ('scripts/run_transition_repair.py', 'possession/features_v13.py',
              'possession/features_v09.py', 'possession/features_v08.py', 'possession/external.py',
              'possession/data.py', 'possession/inference_cache.py', 'games.py')}
    for _, first, last in half_video_blocks(GAME):
        key = dict(study='transition-repair-v1', inputs=hashes, sources=sources, first=int(first), last=int(last),
                   pitch=[float(v) for v in pitch], sample_step=1, numpy=np.__version__, scipy=scipy.__version__)
        x, names, receipt = cached_block(ROOT/'data/inference-features', key,
            lambda: build_features(players[first:last+1], teams, ball[first:last+1], pitch, step=1), last-first+1)
        if names != feature_names or names != state['names']:
            raise ValueError('Tactical feature schema differs')
        p = symmetric_probability(teacher, x, names, state['mean'], state['scale'], 'cuda')
        decoded = pm.label_codes(viterbi_labels(p, *plan['decoder_costs']))
        edits = propose(p, decoded, plan['candidate'])
        row = dict(x=edit_features(p, decoded, x, names, edits), edits=edits, labels=decoded)
        changed, selected = corrected(row, predict(model, mean, scale, row), threshold, enabled)
        base[first:last+1] = np.array(pm.LABELS)[decoded]
        labels[first:last+1] = np.array(pm.LABELS)[changed]
        receipts.append(receipt); edits_total += len(selected)
    folder = repair/'candidate'; folder.mkdir(exist_ok=True)
    path = folder/f'CoreOrigin.pass-{number}.csv'
    frame = pd.DataFrame(dict(frame=np.arange(n), label=labels))
    failure = pm.validate_submission(frame, n)
    if failure:
        raise ValueError(failure)
    frame.to_csv(path, index=False)
    control = folder/f'control.pass-{number}.csv'
    pd.DataFrame(dict(frame=np.arange(n), label=base)).to_csv(control, index=False)
    expected = json.loads((ROOT/plan['parent_run']/'reproducibility.json').read_text())['sha256']
    if sha256(control) != expected or hashes != {k: sha256(paths[k]) for k in hashes}:
        raise ValueError('Full tactical control/input no longer matches frozen parent')
    save_json(folder/f'manifest.pass-{number}.json', dict(rows=n, sha256=sha256(path), control_sha256=expected,
        selected_sha256=sha256(repair/'selected.pt'), parent_sha256=sha256(ROOT/plan['parent_run']/'selected.pt'),
        raw_inputs=hashes, coordinate_feature_cache=receipts, applied_edits=edits_total,
        device=torch.cuda.get_device_name(), dtype='float32', public_labels_read=False, official_competition_score=None))
    del teacher, players, ball; gc.collect(); torch.cuda.empty_cache()
    print(json.dumps(dict(stage='full_export', pass_number=number, rows=n, applied_edits=edits_total, sha256=sha256(path))), flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT/'configs/transition_repair_v1.json')
    args = parser.parse_args(); plan = json.loads(args.plan.read_text()); validate_plan(plan)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    directory = ROOT/plan['output']; status = directory/'status.json'; deadline = time.monotonic()+14400
    def stage(name, **values):
        save_json(status, dict(stage=name, **values)); print(json.dumps(dict(stage=name, **values)), flush=True)
    stage('waiting_for_references')
    while True:
        reference_status = directory/'reference_status.json'
        if reference_status.exists() and json.loads(reference_status.read_text())['stage'] == 'references_complete':
            break
        if time.monotonic() > deadline:
            raise TimeoutError('Reference fits did not complete')
        time.sleep(5)
    repair = directory/'repair'; repair.mkdir(exist_ok=False)
    sources = {p: sha256(ROOT/p) for p in ('scripts/run_transition_repair.py', 'scripts/train_repair_references.py',
               'possession/transition_repair.py', 'scripts/run_local.py', 'possession/temporal.py',
               'evaluation/possession_metrics.py')}
    fixed = dict(config=plan, sources=sources, plan_sha256=sha256(args.plan),
                 parent_checkpoint_sha256=sha256(ROOT/plan['parent_run']/'selected.pt'))
    fixed['fingerprint'] = hashlib.sha256(json.dumps(fixed, sort_keys=True).encode()).hexdigest()
    save_json(repair/'fixed_protocol.json', fixed)
    subprocess.run([sys.executable, 'scripts/snapshot_local_run.py', '--run', str(repair)], cwd=ROOT, check=True)
    stage('preparing_repair_training')
    datasets, names = prepare_training(plan, directory)
    folds = []
    for pair in plan['excluded_pairs']:
        for fit_match, check_match in (pair, pair[::-1]):
            stage('cross_check', fit_match=fit_match, check_match=check_match)
            model, mean, scale, receipt = fit(datasets[fit_match], plan['repair'])
            rows = datasets[check_match]; gain = [predict(model, mean, scale, r) for r in rows]
            baseline = score_rows(rows, [r['labels'] for r in rows])
            scores = {str(t): score_rows(rows, [corrected(r, g, t)[0] for r, g in zip(rows, gain)])
                      for t in plan['repair']['thresholds']}
            folds.append(dict(fit_match=fit_match, check_match=check_match,
                              excluded_reference_pair=pair, baseline=baseline, thresholds=scores, training=receipt))
            save_json(repair/'cross_checks.json', folds)
            del model; gc.collect(); torch.cuda.empty_cache()
    controls = [f['baseline']['spell_f1_mean'] for f in folds]
    candidates = []
    for threshold in plan['repair']['thresholds']:
        gains = [f['thresholds'][str(threshold)]['spell_f1_mean']-control for f, control in zip(folds, controls)]
        candidates.append(dict(threshold=threshold, mean_gain=float(np.mean(gains)),
                               positive_folds=sum(g > 1e-9 for g in gains), fold_gains=gains))
    eligible = [r for r in candidates if r['mean_gain'] > 1e-9 and r['positive_folds'] >= plan['repair']['minimum_positive_folds']]
    chosen = max(eligible, key=lambda r: (r['mean_gain'], r['threshold'])) if eligible else None
    enabled = chosen is not None; threshold = chosen['threshold'] if enabled else 1.
    selection = dict(enabled=enabled, threshold=threshold, candidates=candidates,
                     rule='Positive mean spell-F1 gain and at least four of six positive excluded-match cross-checks; otherwise identity.',
                     development_targets_used_for_threshold=False, independent_test=False)
    save_json(repair/'cross_check_selection.json', selection)
    stage('fitting_final_repair', enabled=enabled, threshold=threshold)
    all_rows = sum((datasets[m] for m in plan['training_matches']), [])
    model, mean, scale, receipt = fit(all_rows, plan['repair'])
    save_torch(repair/'selected.pt', dict(model=model.state_dict(), mean=mean, scale=scale,
        input_features=len(mean), hidden_channels=plan['repair']['hidden_channels'], enabled=enabled, threshold=threshold,
        fingerprint=fixed['fingerprint'], feature_names=names, config=plan))
    save_json(repair/'training_receipt.json', receipt)
    del datasets, all_rows; gc.collect()
    stage('development_evaluation', enabled=enabled)
    report = evaluate_development(plan, repair, model, mean, scale, enabled, threshold, names)
    stage('end_to_end_exports', development_score=report['repaired']['spell_f1_mean'], improved=report['improved'])
    first = export(plan, repair, model, mean, scale, enabled, threshold, names, 1)
    second = export(plan, repair, model, mean, scale, enabled, threshold, names, 2)
    if sha256(first) != sha256(second):
        raise ValueError('Two complete exports differ')
    shutil.copyfile(second, repair/'candidate/CoreOrigin.csv')
    save_json(repair/'reproducibility.json', dict(two_complete_passes_byte_identical=True, rows=199319,
              sha256=sha256(first), selected_sha256=sha256(repair/'selected.pt'), official_competition_score=None))
    if sources != {p: sha256(ROOT/p) for p in sources}:
        raise ValueError('Study sources changed during fitting/evaluation')
    stage('complete_and_stopped', development_baseline=report['baseline']['spell_f1_mean'],
          development_score=report['repaired']['spell_f1_mean'], improved=report['improved'],
          cross_check_enabled=enabled, two_complete_passes_byte_identical=True,
          selected_sha256=sha256(repair/'selected.pt'), csv_sha256=sha256(first),
          scope='Development only; no independent or official score. No further runs queued.')


if __name__ == '__main__':
    main()
