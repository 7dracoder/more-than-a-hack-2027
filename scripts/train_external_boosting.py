"""Fit a contact-feature tree alternative using external training matches only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from evaluation import possession_metrics as pm
from possession.external import swap_features
from possession.temporal import ClassifierAdapter, symmetric_probability, training_weights
from possession.data import sha256
from scripts.run_local import load_halves, provenance, save_json, save_torch, score_probabilities, validate_splits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/local_v08_contacts.json')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=400)
    p.add_argument('--leaves', type=int, default=15)
    p.add_argument('--stride', type=int, default=3)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    config.update(architecture='external_boosted', classifier_iterations=args.iterations,
                  classifier_max_leaf_nodes=args.leaves, classifier_training_stride=args.stride,
                  classifier_learning_rate=.07, classifier_l2=10., classifier_min_samples_leaf=35,
                  boundary_weight=1., classifier_loss_weight_power=.5)
    validate_splits(config)
    save_json(args.out/'config.json', config)
    record = provenance(config, config['train_matches']+config['validation_matches'])
    record['training_source_sha256'] = sha256(Path(__file__))
    record['fingerprint'] = hashlib.sha256(json.dumps({k:v for k,v in record.items() if k != 'fingerprint'}, sort_keys=True).encode()).hexdigest()
    save_json(args.out/'fixed_protocol.json', record)
    save_json(args.out/'search_plan.json', dict(costs=config['development_decoder_costs'],
              supervision='Four external training matches only', independent_test_opened=False))
    training, names = load_halves(config['train_matches'], ROOT/config['features_dir'])
    x, y, w = [], [], []
    for half in training:
        rows = np.arange(0, len(half['y']), args.stride)
        rows = rows[half['y'][rows] != 'X']
        x.append(np.clip(half['x'][rows], -10, 10))
        y.append(pm.label_codes(half['y'][rows]).astype(np.int64))
        w.append(training_weights(half['y'], .5)[rows])
    x, y, w = np.concatenate(x), np.concatenate(y), np.concatenate(w)
    x = np.concatenate([x, swap_features(x, names)])
    y = np.concatenate([y, np.array([1, 0, 2])[y]])
    w = np.concatenate([w, w])
    print(json.dumps(dict(training_rows=len(y), features=len(names), iterations=args.iterations, leaves=args.leaves)), flush=True)
    classifier = HistGradientBoostingClassifier(max_iter=args.iterations, max_leaf_nodes=args.leaves,
                  learning_rate=.07, l2_regularization=10., min_samples_leaf=35,
                  early_stopping=False, random_state=config['seed'], verbose=0)
    with threadpool_limits(limits=2):
        classifier.fit(x, y, sample_weight=w)
    del x, y, w
    print('External training finished; evaluating fixed development plan', flush=True)
    validation, val_names = load_halves(config['validation_matches'], ROOT/config['features_dir'])
    if names != val_names:
        raise ValueError('Feature schema differs')
    mean, scale = np.zeros(len(names), np.float32), np.ones(len(names), np.float32)
    model = ClassifierAdapter(classifier)
    probabilities = [symmetric_probability(model, h['x'], names, mean, scale, 'cpu') for h in validation]
    results = []
    for costs in config['development_decoder_costs']:
        score = score_probabilities(validation, probabilities, costs, config['sample_step'])
        results.append(dict(costs=costs, score=score))
        save_json(args.out/'search_results.json', results)
        print(json.dumps(results[-1]), flush=True)
    chosen = max(results, key=lambda r: r['score']['spell_f1_mean'])
    state = dict(classifier=classifier, config=config, mean=mean, scale=scale, names=names,
                 decoder_costs=chosen['costs'], validation=chosen['score'], fingerprint=record['fingerprint'], epoch='fixed boosting iterations')
    save_torch(args.out/'selected.pt', state)
    save_json(args.out/'selection.json', dict(validation=chosen['score'], decoder_costs=chosen['costs'],
              checkpoint_sha256=sha256(args.out/'selected.pt'), test_opened=False,
              official_competition_score=None, target_0_90_met_on_development=chosen['score']['spell_f1_mean'] >= .9))


if __name__ == '__main__':
    main()
