"""External-only paired study of longer spatial memory and passing options.

The frozen externally trained temporal model supplies base margins. Two matched
XGBoost fits compare probability-only correction with spatial-memory correction.
BAR-ATH assets and all public possession labels are denied by an I/O guard.
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
from possession.spatial_memory import build_features, symmetric_features


TRAIN = ['J03WMX', 'J03WN1', 'J03WOH', 'J03WOY', 'J03WQQ', 'J03WR9']
DEV = 'J03WPY'
PLAN = dict(train_matches=TRAIN, development_matches=[DEV], independent_test_matches=[],
            views=['native', 'corrupt'], parent='models/coreorigin-video-v1',
            training_stride=5, training_boundary_radius_frames=12,
            correction_strengths=[0., .25, .5, 1.],
            native_spell_f1_max_decline=.01,
            selection='Maximum mean native/corrupt development spell F1, subject to native guard; parent at strength zero is always eligible.',
            fits=['probability_control', 'spatial_memory'],
            parameters=dict(n_estimators=240, max_depth=4, learning_rate=.035,
                            min_child_weight=30, reg_lambda=20., subsample=.8,
                            colsample_bytree=1., max_bin=128, random_state=20321114),
            head_orientation_used=False, gaze_measured=False,
            provided_match_data_used=False, public_labels_used=False,
            limitation='Existing development match, not an independent test or competition score.')


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def isolate(out):
    data, external, truth = (ROOT / 'data').resolve(), (ROOT / 'data/external').resolve(), (ROOT / 'evaluation/gt').resolve()
    out = out.resolve()
    if not out.is_relative_to(data) or out.is_relative_to(external):
        raise ValueError('Study output must be a separate ignored data/ directory')
    denied = []
    def guard(event, values):
        if event == 'open' and values and isinstance(values[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(values[0])).resolve()
            if path.is_relative_to(truth) or (path.is_relative_to(data) and
                    not path.is_relative_to(external) and not path.is_relative_to(out)):
                denied.append(str(path))
                raise PermissionError('External study cannot read competition assets, predictions or labels')
    sys.addaudithook(guard)
    try:
        open(ROOT / 'data/tactical/BAR-ATH_tactical_atd.txt', 'rb')
        raise AssertionError('Isolation failed')
    except PermissionError:
        pass
    denied.clear()
    return denied


def torch_setup(device):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    return torch.device(device)


def prepare(out, device):
    import torch
    from possession.external import read_sportec
    from scripts.prepare_external_v08 import corrupt_tracking
    from scripts.run_local import selected_model, symmetric_probability

    device = torch_setup(device)
    parent = ROOT / PLAN['parent']
    model, state = selected_model(parent, device)
    fixed = json.loads((parent / 'fixed_protocol.json').read_text())
    folder = ROOT / 'data/external/sportec'
    pins = {item['name']:item['sha256'] for item in json.loads((folder / 'download_manifest.json').read_text())['files']}
    sources = {name:sha256(ROOT / name) for name in
               ['possession/spatial_memory.py', 'scripts/research/spatial_memory/run_external.py',
                'scripts/prepare_external_v08.py', 'scripts/prepare_external_gaps.py',
                'possession/external.py', 'possession/temporal.py', 'scripts/run_local.py']}
    cache = out / 'cache'
    cache.mkdir(exist_ok=True)
    for match in TRAIN + [DEV]:
        metadata = next(folder.glob('*matchinformation*' + match + '.xml'))
        tracking = next(folder.glob('*positions*' + match + '.xml'))
        inputs = {path.name:sha256(path) for path in (metadata, tracking)}
        if any(pins.get(name) != value for name, value in inputs.items()):
            raise ValueError('External raw input differs from pinned source')
        expected = {}
        for view in PLAN['views']:
            path = ROOT / ('data/external/features-v25' + ('-corrupt' if view == 'corrupt' else '')) / (match + '.npz')
            actual = sha256(path)
            expected_hash = fixed['augmentation_caches' if view == 'corrupt' else 'external_caches'].get(match)
            if expected_hash is None and match == DEV and view == 'corrupt':
                # Parent training pins augmentation for training matches only.
                # Development augmentation has its own existing provenance sidecar.
                cache_receipt = json.loads(path.with_suffix('.json').read_text())
                expected_hash = cache_receipt['sha256']
                provenance = cache_receipt['provenance']
                for key, filename in [('v25_feature_source_sha256', 'possession/features_v25.py'),
                                      ('v25_preparation_source_sha256', 'scripts/prepare_external_v25.py'),
                                      ('preparation_source_sha256', 'scripts/prepare_external_v08.py'),
                                      ('gap_mask_source_sha256', 'scripts/prepare_external_gaps.py')]:
                    if provenance[key] != sha256(ROOT / filename):
                        raise ValueError('Development augmentation source differs')
            if expected_hash != actual:
                raise ValueError('External feature cache differs from the frozen parent')
            expected[view] = (path, actual)
        halves = read_sportec(metadata, tracking)
        for i, half in enumerate(halves):
            for view in PLAN['views']:
                target = cache / f'{match}_{i}_{view}.npz'
                receipt = target.with_suffix('.json')
                fingerprint = dict(inputs=inputs, sources=sources, parent_sha256=sha256(parent / 'selected.pt'),
                                   base_cache_sha256=expected[view][1], device=str(device), torch=torch.__version__)
                if target.exists() and receipt.exists():
                    old = json.loads(receipt.read_text())
                    if old['fingerprint'] == fingerprint and old['sha256'] == sha256(target):
                        print(f'Reused {target.name}', flush=True)
                        continue
                    raise ValueError('Incompatible study cache; use a new study directory')
                start = time.monotonic()
                seed = int(hashlib.sha256((match + '/' + str(i) + '/v08').encode()).hexdigest()[:8], 16)
                players, ball = corrupt_tracking(half.players, half.ball, seed) if view == 'corrupt' else (half.players, half.ball)
                memory, names = build_features(players, half.teams, ball, half.pitch)
                with np.load(expected[view][0], allow_pickle=False) as stored:
                    if stored['names'].tolist() != state['names']:
                        raise ValueError('Parent feature schema differs')
                    np.testing.assert_array_equal(stored[f'truth{i}'], half.labels)
                    np.testing.assert_array_equal(stored[f'frames{i}'], half.frames)
                    matrix = stored[f'x{i}']
                    p = symmetric_probability(model, matrix, state['names'], state['mean'], state['scale'], device)
                    del matrix
                part = target.with_suffix('.partial.npz')
                np.savez_compressed(part, memory=memory.astype(np.float32), names=np.array(names),
                                    probability=p, truth=half.labels)
                part.replace(target)
                dump(receipt, dict(fingerprint=fingerprint, sha256=sha256(target), rows=len(p),
                                   train=match in TRAIN, view=view, seconds=time.monotonic() - start))
                print(f'Prepared {target.name}: {len(p)} frames, {time.monotonic() - start:.1f}s', flush=True)
                del memory, p
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    dump(out / 'preparation.json', dict(complete=True, plan=PLAN, sources=sources,
                                       parent_sha256=sha256(parent / 'selected.pt'), competition_data_read=False))


def input_matrix(probability, memory, kind):
    context = probability if probability.shape[1] == 4 else np.column_stack([probability, np.zeros(len(probability))])
    return context.astype(np.float32) if kind == 'probability_control' else np.column_stack([context, memory]).astype(np.float32)


def swap_matrix(matrix, names, kind):
    result = matrix.copy()
    result[:, :2] = matrix[:, [1, 0]]
    if kind != 'probability_control':
        result[:, 4:] = symmetric_features(matrix[:, 4:], names)
    return result


def corrected_probability(model, probability, memory, names, kind):
    x = input_matrix(probability, memory, kind)
    margin = np.log(np.clip(probability[:, :3], 1e-6, 1.))
    a = model.predict_proba(x, base_margin=margin)
    b = model.predict_proba(swap_matrix(x, names, kind), base_margin=margin[:, [1, 0, 2]])[:, [1, 0, 2]]
    return ((a + b) / 2.).astype(np.float32)


def blend(probability, corrected, strength):
    if strength == 0:
        return probability.copy()
    logits = ((1. - strength) * np.log(np.clip(probability[:, :3], 1e-6, 1.)) +
              strength * np.log(np.clip(corrected, 1e-6, 1.)))
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(1, keepdims=True)
    return np.column_stack([p, probability[:, 3:]]).astype(np.float32)


def score_parts(truth, prediction):
    from evaluation import possession_metrics as pm
    t = np.concatenate([part for y in truth for part in (y, np.array(['X']))])
    p = np.concatenate([part for y in prediction for part in (y, np.array(['X']))])
    r = pm.evaluate(pm.label_codes(t), pm.label_codes(p), bootstrap_n=0)
    return {k:r[k] for k in ('spell_f1_mean', 'macro_f1', 'n_gt_spells', 'n_pred_spells')}


def fit(out, device, denied):
    import xgboost as xgb
    from possession.temporal import training_weights, viterbi_labels

    parent = ROOT / PLAN['parent']
    # Load metadata only, without allocating a second temporal model.
    import torch
    state = torch.load(parent / 'selected.pt', map_location='cpu', weights_only=False)
    preparation = json.loads((out / 'preparation.json').read_text())
    if not preparation['complete'] or preparation['plan'] != PLAN:
        raise ValueError('Study preparation/protocol differs')
    cache = out / 'cache'
    for path in cache.glob('*.npz'):
        receipt = json.loads(path.with_suffix('.json').read_text())
        if sha256(path) != receipt['sha256']:
            raise ValueError('Study cache changed')
    training, margins, labels, weights = [], [], [], []
    names = None
    for match in TRAIN:
        for i in range(2):
            for view in PLAN['views']:
                with np.load(cache / f'{match}_{i}_{view}.npz', allow_pickle=False) as z:
                    truth, p, m = z['truth'], z['probability'], z['memory']
                    if names is not None and names != z['names'].tolist():
                        raise ValueError('Memory feature schema differs')
                    names = z['names'].tolist()
                    keep = np.arange(len(truth)) % PLAN['training_stride'] == 0
                    changes = np.flatnonzero(truth[1:] != truth[:-1]) + 1
                    for boundary in changes:
                        r = PLAN['training_boundary_radius_frames']
                        keep[max(0, boundary - r):boundary + r + 1] = True
                    keep &= np.isin(truth, ['H', 'A', 'D'])
                    y = np.array([{'H':0, 'A':1, 'D':2}.get(value, -1) for value in truth], np.int32)
                    w = training_weights(truth, .5).astype(np.float32)
                    training.append(input_matrix(p[keep], m[keep], 'spatial_memory'))
                    labels.append(y[keep]); margins.append(np.log(np.clip(p[keep, :3], 1e-6, 1.)))
                    weights.append(w[keep])
    x, y, margin, weight = np.concatenate(training), np.concatenate(labels), np.concatenate(margins), np.concatenate(weights)
    del training, labels, margins, weights
    # Team-symmetric training uses identical sampled rows for both matched fits.
    x = np.concatenate([x, swap_matrix(x, names, 'spatial_memory')])
    y = np.concatenate([y, np.where(y == 0, 1, np.where(y == 1, 0, y))])
    margin = np.concatenate([margin, margin[:, [1, 0, 2]]])
    weight = np.tile(weight, 2); weight /= weight.mean()
    dump(out / 'training_receipt.json', dict(rows=len(y), features=x.shape[1], names=names,
         labels={str(k):int((y == k).sum()) for k in (0, 1, 2)}, plan=PLAN,
         data_guard_verified=True, parent_sha256=sha256(parent / 'selected.pt'),
         xgboost=xgb.__version__, numpy=np.__version__))
    dev = {(i, view):dict(np.load(cache / f'{DEV}_{i}_{view}.npz', allow_pickle=False))
           for i in range(2) for view in PLAN['views']}
    baselines = {}
    for view in PLAN['views']:
        halves = [dev[i, view] for i in range(2)]
        baselines[view] = score_parts([h['truth'] for h in halves],
                                    [viterbi_labels(h['probability'], *state['decoder_costs']) for h in halves])
    records = []
    for kind in PLAN['fits']:
        model = xgb.XGBClassifier(**PLAN['parameters'], tree_method='hist', device=device,
                                   n_jobs=4, objective='multi:softprob', num_class=3,
                                   eval_metric='mlogloss')
        features = x[:, :4] if kind == 'probability_control' else x
        print(f'Fitting {kind}: {len(y)} external rows', flush=True)
        model.fit(features, y, sample_weight=weight, base_margin=margin, verbose=False)
        model.set_params(device='cpu')
        model.save_model(out / f'{kind}.ubj')
        corrected = {(i, view):corrected_probability(model, h['probability'], h['memory'], names, kind)
                     for (i, view), h in dev.items()}
        for strength in PLAN['correction_strengths']:
            scores = {}
            for view in PLAN['views']:
                halves = [dev[i, view] for i in range(2)]
                preds = [viterbi_labels(blend(h['probability'], corrected[i, view], strength),
                                        *state['decoder_costs']) for i, h in enumerate(halves)]
                scores[view] = score_parts([h['truth'] for h in halves], preds)
            eligible = scores['native']['spell_f1_mean'] >= baselines['native']['spell_f1_mean'] - PLAN['native_spell_f1_max_decline']
            entry = dict(kind=kind, strength=strength, scores=scores, eligible=eligible,
                         mean_spell_f1=float(np.mean([r['spell_f1_mean'] for r in scores.values()])))
            records.append(entry)
            print(json.dumps(entry), flush=True)
        dump(out / 'development_results.json', dict(parent=baselines, candidates=records))
    eligible = [r for r in records if r['eligible']]
    best = max(eligible, key=lambda r:(r['mean_spell_f1'], -r['strength'], r['kind'] == 'spatial_memory'))
    selection = dict(complete=True, selected=best, parent_scores=baselines,
                     selected_model_sha256=sha256(out / (best['kind'] + '.ubj')),
                     parent_sha256=sha256(parent / 'selected.pt'), memory_names=names,
                     decoder_costs=state['decoder_costs'], plan=PLAN,
                     preparation_sha256=sha256(out / 'preparation.json'),
                     source_sha256={name:sha256(ROOT / name) for name in
                         ['possession/spatial_memory.py', 'scripts/research/spatial_memory/run_external.py']},
                     data_guard_verified=True, unexpected_competition_access=denied,
                     competition_data_read=False, public_labels_read=False,
                     full_match_competition_score=None)
    dump(out / 'selection.json', selection)
    print('External selection frozen: ' + json.dumps(best), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare', 'fit'], required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plan_path = args.out / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != PLAN:
        raise ValueError('Study protocol changed; use a new output directory')
    dump(plan_path, PLAN)
    denied = isolate(args.out)
    if args.stage == 'prepare':
        prepare(args.out, args.device)
    else:
        if (args.out / 'selection.json').exists():
            raise ValueError('This bounded study already completed; do not refit silently')
        fit(args.out, args.device, denied)


if __name__ == '__main__':
    main()
