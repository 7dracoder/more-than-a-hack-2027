"""GPU gradient boosting on external contact features; development-only selection."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import xgboost as xgb
from evaluation import possession_metrics as pm
from possession.data import sha256
from possession.external import swap_features
from possession.temporal import ClassifierAdapter, symmetric_probability, training_weights
from scripts.run_local import load_halves, provenance, save_json, save_torch, score_probabilities, validate_splits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/local_v08_contacts.json')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=600)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--stride', type=int, default=3)
    p.add_argument('--robust', action='store_true')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    params = dict(n_estimators=args.iterations, max_depth=args.depth, learning_rate=.05,
                  min_child_weight=10., reg_lambda=10., subsample=.85, colsample_bytree=.85,
                  n_jobs=2, tree_method='hist', device='cuda', objective='multi:softprob',
                  eval_metric='mlogloss', random_state=config['seed'])
    config.update(architecture='external_xgb', xgb_params=params, classifier_loss_weight_power=.5,
                  classifier_training_stride=args.stride, xgboost_version=xgb.__version__)
    if args.robust:
        config['augmentation_features_dir'] = 'data/external/features-v08-corrupt'
    validate_splits(config)
    save_json(args.out/'config.json', config)
    record = provenance(config, config['train_matches']+config['validation_matches'])
    record['xgb_training_source_sha256'] = sha256(Path(__file__))
    record['fingerprint'] = hashlib.sha256(json.dumps({k:v for k,v in record.items() if k != 'fingerprint'}, sort_keys=True).encode()).hexdigest()
    save_json(args.out/'fixed_protocol.json', record)
    save_json(args.out/'search_plan.json', dict(costs=config['development_decoder_costs'],
              independent_test_opened=False, target_spell_f1=.9, evaluation_scope='Fixed external development match'))
    native, names = load_halves(config['train_matches'], ROOT/config['features_dir'])
    training = list(native)
    if args.robust:
        augmented, extra_names = load_halves(config['train_matches'], ROOT/config['augmentation_features_dir'])
        if extra_names != names or any(not np.array_equal(a['truth'], b['truth']) for a,b in zip(native, augmented)):
            raise ValueError('Corrupted view changed schema or supervision')
        training += augmented
    matrices, labels, weights = [], [], []
    for half in training:
        rows = np.arange(0, len(half['y']), args.stride)
        rows = rows[half['y'][rows] != 'X']
        matrices.append(np.clip(half['x'][rows], -10, 10))
        labels.append(pm.label_codes(half['y'][rows]).astype(np.int64))
        weights.append(training_weights(half['y'], .5)[rows])
    x, y, w = np.concatenate(matrices), np.concatenate(labels), np.concatenate(weights)
    del matrices, labels, weights, training, native
    x = np.concatenate([x, swap_features(x, names)])
    y = np.concatenate([y, np.array([1,0,2])[y]])
    w = np.concatenate([w, w])
    print(json.dumps(dict(rows=len(y), features=len(names), params=params, xgboost=xgb.__version__)), flush=True)
    started = time.monotonic()
    classifier = xgb.XGBClassifier(**params)
    classifier.fit(x, y, sample_weight=w, eval_set=[(x,y)], verbose=100)
    actual_device = json.loads(classifier.get_booster().save_config())['learner']['generic_param']['device']
    if actual_device != 'cuda:0':
        raise RuntimeError('GPU experiment silently fell back to '+actual_device)
    save_json(args.out/'environment.json', dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
              xgboost=xgb.__version__, build=xgb.build_info(), training_device=actual_device, fit_seconds=time.monotonic()-started))
    # CPU inference avoids device mismatch copies in the shared torch adapter.
    classifier.set_params(device='cpu')
    classifier.save_model(args.out/'classifier.ubj')
    del x,y,w
    validation, current_names = load_halves(config['validation_matches'], ROOT/config['features_dir'])
    if names != current_names:
        raise ValueError('Feature schema changed')
    mean, scale = np.zeros(len(names), np.float32), np.ones(len(names), np.float32)
    model = ClassifierAdapter(classifier)
    probabilities = [symmetric_probability(model,h['x'],names,mean,scale,'cpu') for h in validation]
    results = []
    for costs in config['development_decoder_costs']:
        score = score_probabilities(validation, probabilities, costs, config['sample_step'])
        results.append(dict(costs=costs, score=score))
        save_json(args.out/'search_results.json', results)
        print(json.dumps(results[-1]), flush=True)
    chosen = max(results, key=lambda r:r['score']['spell_f1_mean'])
    state = dict(classifier=classifier, config=config, mean=mean, scale=scale, names=names,
                  validation=chosen['score'], decoder_costs=chosen['costs'], fingerprint=record['fingerprint'], epoch=args.iterations)
    save_torch(args.out/'selected.pt', state)
    save_json(args.out/'selection.json', dict(validation=chosen['score'], decoder_costs=chosen['costs'],
              checkpoint_sha256=sha256(args.out/'selected.pt'), test_opened=False, official_competition_score=None,
              target_0_90_met_on_development=chosen['score']['spell_f1_mean'] >= .9))


if __name__ == '__main__':
    main()
