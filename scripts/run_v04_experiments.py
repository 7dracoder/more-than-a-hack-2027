"""BlindSpot v0.4 experiments: contact, ball-gap, out-of-play and team-shape features.

Compares the v0.3.1 feature set against v0.3.1 plus each new feature group, under the same two
held-out protocols used in AUDIT.md (whole-run purged folds and equal-duration folds, 300-frame
embargo). Every number is from models that did not see the scored fold's labels.

Results are appended to EXPERIMENTS_v04.md (aggregate scores only, no match data).

    python scripts/run_v04_experiments.py                    # all experiments (~10-15 min CPU)
    python scripts/run_v04_experiments.py contacts shape     # selected experiments
    python scripts/run_v04_experiments.py --candidate all    # export CoreOrigin.csv with a group set

Groups: baseline, contacts, asym, gap_out, ballmotion, shape, all, contacts_team_only,
        baseline_outplay, team_only_outplay, team_only_outplay_snap   (v0.5, possession/structure.py)
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from games import asset_paths
from possession.data import sha256, slot_teams
from possession.metric_features import extract_metric, design, swap_teams
from possession.event_features import FEATURE_VERSION, build, swap
from possession.team_fusion import decode_team_only, team_only_inputs
from possession import structure
from possession.champion import (classifier_probability, decode_factorized, fit_classifier,
                                 purged_folds, fixed_time_folds)
from scripts.run_champion import evaluate, pooled
from scripts.run_experiment import save_submission
from evaluation import possession_metrics as pm

CONFIG_PATH = ROOT / 'configs/champion_v031.json'
CACHE = ROOT / 'data/experiments/cache'
RESULTS_MD = ROOT / 'EXPERIMENTS_v04.md'
OUTPUT = ROOT / 'data/experiments/blindspot-v04'
GT_PATH = ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'

GROUPS = {
    'baseline': [],
    'contacts': ['last_contact', 'next_contact', 'since_contact', 'until_contact', 'bracket'],
    'asym': ['contact_past', 'contact_future', 'margin_'],
    'gap_out': ['gap_len', 'since_seen', 'until_seen', 'out_', 'since_out', 'until_out', 'edge_'],
    'ballmotion': ['ball_still', 'since_ball', 'until_ball', 'ball_speed_'],
    'shape': ['adv_', 'spread_', 'width_', 'ball_adv', 'fwd_vel', 'near_', 'press_'],
}
GROUPS['all'] = sorted({p for v in GROUPS.values() for p in v})
GROUPS['contacts_team_only'] = GROUPS['contacts']
# v0.5 structural groups (possession/structure.py). Predeclared before scoring; constants fixed.
GROUPS['baseline_outplay'] = []
GROUPS['team_only_outplay'] = GROUPS['contacts']
GROUPS['team_only_outplay_snap'] = GROUPS['contacts']
STRUCTURAL = {'baseline_outplay': ('baseline', False), 'team_only_outplay': ('contacts', False),
              'team_only_outplay_snap': ('contacts', True)}


def raw_tracking():
    """Raw positions as a hash-checked numpy cache (ignored under data/)."""
    paths = asset_paths(data_dir=str(ROOT / 'data'))
    digest = sha256(paths['atd_raw'])
    file, meta = CACHE / 'raw-tracking-v4.npz', CACHE / 'raw-tracking-v4.json'
    if file.exists() and meta.exists():
        info = json.loads(meta.read_text())
        if info.get('atd_sha256') == digest and info.get('cache_sha256') == sha256(file):
            with np.load(file) as z:
                return z['players'], z['ball']
    print('Reading raw tracking (one-off, a few minutes)...', flush=True)
    n = 199319
    players = np.full((n, 46, 3), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    with open(paths['atd_raw']) as stream:
        for i, line in enumerate(stream):
            frame_id, p, b = line.strip().split(':')
            if int(frame_id) != i + 1:
                raise ValueError('Tracking frame order mismatch')
            players[i] = np.fromstring(p.replace(';', ','), sep=',').reshape(46, 3)
            ball[i] = np.fromstring(b, sep=',')[:3]
    if i + 1 != n:
        raise ValueError('Tracking frame count mismatch')
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(file, players=players, ball=ball)
    meta.write_text(json.dumps({'atd_sha256': digest, 'cache_sha256': sha256(file)}, indent=2))
    return players, ball


def event_matrix():
    source = sha256(ROOT / 'possession/event_features.py')
    file, meta = CACHE / 'event-features-v4.npz', CACHE / 'event-features-v4.json'
    atd = sha256(asset_paths(data_dir=str(ROOT / 'data'))['atd_raw'])
    inputs = {'tracking': atd,
              'metadata': sha256(asset_paths(data_dir=str(ROOT / 'data'))['atd_meta']),
              'game_constants': sha256(ROOT / 'games.py')}
    if file.exists() and meta.exists():
        info = json.loads(meta.read_text())
        if info.get('source_sha256') == source and info.get('inputs') == inputs and info.get('cache_sha256') == sha256(file):
            with np.load(file) as z:
                rules = {k: (tuple(v) if isinstance(v, list) else v) for k, v in info['rules'].items()}
                return z['matrix'], list(info['names']), rules
    players, ball = raw_tracking()
    started = time.time()
    matrix, names, rules, metas = build(players, ball, slot_teams(asset_paths(data_dir=str(ROOT / 'data'))['atd_meta']))
    print(f'Built {len(names)} event features in {time.time()-started:.0f}s; attack direction {metas}', flush=True)
    np.savez(file, matrix=matrix)
    meta.write_text(json.dumps({'feature_version': FEATURE_VERSION, 'source_sha256': source, 'atd_sha256': atd,
                                'inputs': inputs, 'cache_sha256': sha256(file), 'names': names, 'rules': rules,
                                'attack_direction': {str(k): v for k, v in metas.items()}}, indent=2))
    return matrix, names, rules


class Setup:
    def __init__(self, output=OUTPUT):
        self.output = Path(output)
        self.probability_cache = {}
        self.last_prediction_audit = {}
        self.config = json.loads(CONFIG_PATH.read_text())
        self.arrays, feature_meta = extract_metric(ROOT / 'data', CACHE, self.config)
        self.base, self.base_names = design(self.arrays, self.config)
        self.events, self.event_names, self.rules = event_matrix()
        self.truth = pd.read_csv(GT_PATH)['label'].to_numpy()
        self.valid = self.truth != 'X'
        sources = sorted((ROOT / 'possession').glob('*.py')) + [Path(__file__), ROOT / 'games.py',
                    ROOT / 'scripts/run_champion.py', ROOT / 'evaluation/possession_metrics.py']
        self.provenance = {'source_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources},
                           'input_sha256': feature_meta['input_sha256'], 'public_truth_sha256': sha256(GT_PATH),
                           'config': self.config, 'groups': GROUPS,
                           'team_only_plan': 'One fixed experiment: baseline P(D), contacts P(H/A | live), identical training masks; no new hyperparameter search.',
                           'scope': 'Same inspected public development sample; offline raw features can span whole halves. The 300-frame embargo excludes training labels, not all shared raw context. Feature-group selection reuses this sample.'}
        self.fingerprint = hashlib.sha256(json.dumps(self.provenance, sort_keys=True).encode()).hexdigest()

    def matrix(self, group):
        prefixes = GROUPS[group]
        keep = [i for i, name in enumerate(self.event_names) if any(name.startswith(p) for p in prefixes)]
        names = self.base_names + [self.event_names[i] for i in keep]
        k = len(self.base_names)
        rules = self.rules

        def team_swap(matrix, _names):
            return np.column_stack([swap_teams(matrix[:, :k], self.base_names),
                                    swap(matrix[:, k:], names[k:], rules)])
        return np.column_stack([self.base, self.events[:, keep]]), names, team_swap

    def decode(self, probability):
        p = probability.clip(1e-6, 1.)
        return decode_factorized(self.arrays, dict(self.config, history_samples=0), p[:, 2],
                                 np.log(p[:, :2] / p[:, :2].sum(1)[:, None]))[0]

    def probability(self, group, train_mask):
        key = (group, hashlib.sha256(train_mask.tobytes()).hexdigest())
        if key in self.probability_cache:
            return self.probability_cache[key]
        matrix, names, team_swap = self.matrix(group)
        model, audit = fit_classifier(matrix, names, self.truth, train_mask, self.config, swap=team_swap)
        result = classifier_probability(model, matrix, names, swap=team_swap), len(names), audit
        self.probability_cache[key] = result
        return result

    def structural(self):
        if not hasattr(self, '_structural'):
            hard, soft = structure.out_of_play_hold(self.arrays['ball'])
            self._structural = hard, soft, structure.contact_series(self.events, self.event_names)
        return self._structural

    def predict(self, group, train_mask):
        if group in STRUCTURAL:
            team_group, snap = STRUCTURAL[group]
            hard, soft, contacts = self.structural()
            baseline, n_base, base_audit = self.probability('baseline', train_mask)
            team_source, n_team, team_audit = self.probability(team_group, train_mask)
            dead, team = team_only_inputs(baseline, team_source)
            dead = structure.apply_dead_prior(dead, hard, soft)
            labels = decode_factorized(self.arrays, dict(self.config, history_samples=0), dead, team)[0]
            moved = 0
            if snap:
                labels, moved = structure.snap_transitions(labels, contacts)
            self.last_prediction_audit = {'kind': 'structural_v05', 'team_source': team_group, 'snap': snap,
                'baseline_model': base_audit, 'team_model': team_audit,
                'out_of_play_hard_frames': int(hard.sum()), 'out_of_play_soft_frames': int(soft.sum()),
                'transitions_snapped': int(moved),
                'constants': {k: getattr(structure, k) for k in ('OUT_MARGIN_M', 'KICK_SPEED_MPS', 'HOLD_CAP_FRAMES',
                              'HOLD_LOG_ODDS', 'HARD_DEAD_PROBABILITY', 'SNAP_FRAMES')}}
            return labels, n_team
        if group == 'contacts_team_only':
            baseline, n_base, base_audit = self.probability('baseline', train_mask)
            contacts, n_contact, contact_audit = self.probability('contacts', train_mask)
            labels = decode_team_only(self.arrays, dict(self.config, history_samples=0), baseline, contacts)[0]
            reference = self.decode(baseline)
            if not np.array_equal(labels == 'D', reference == 'D') or not np.array_equal(labels == 'X', reference == 'X'):
                raise ValueError('Team-only fusion changed baseline dead-ball or ungraded mask')
            self.last_prediction_audit = {'kind': 'baseline_dead_contacts_team',
                'baseline_features': n_base, 'contacts_features': n_contact, 'unique_features': n_contact,
                'baseline_model': base_audit, 'contacts_model': contact_audit,
                'identical_training_mask': base_audit['training_mask_sha256'] == contact_audit['training_mask_sha256'],
                'baseline_dead_mask_unchanged': True, 'baseline_ungraded_mask_unchanged': True,
                'dead_mask_sha256': hashlib.sha256((labels == 'D').tobytes()).hexdigest(),
                'baseline_dead_probability_sha256': hashlib.sha256(baseline[:, 2].tobytes()).hexdigest()}
            return labels, n_contact
        probability, n_features, audit = self.probability(group, train_mask)
        self.last_prediction_audit = {'kind': 'single_symmetric_classifier', 'model': audit}
        return self.decode(probability), n_features

    def held_out(self, group):
        result = {}
        out = self.output / group
        out.mkdir(parents=True, exist_ok=True)
        for protocol in ('whole_run', 'fixed_time'):
            folds = (purged_folds(self.truth, 4, 300, 130) if protocol == 'whole_run'
                     else fixed_time_folds(self.valid))
            scores = []
            audits = []
            for train, test, info in folds:
                labels, n_features = self.predict(group, train)
                scores.append(evaluate(self.truth, labels, test))
                file = out / f"{protocol}_fold_{info['fold']}.csv"
                digest = save_submission(file, labels)
                if group == 'baseline':
                    reference = (ROOT / f"data/experiments/blindspot-v031/supervised/fold_{info['fold']}_predictions.csv"
                                 if protocol == 'whole_run' else ROOT / f"data/experiments/blindspot-v031/audit/fixed_time_audit_revision_{info['fold']}.csv")
                    if reference.exists() and sha256(reference) != digest:
                        raise ValueError(f'Baseline predictions differ from frozen reference: {protocol} fold {info["fold"]}')
                audits.append(dict(info, train_mask_sha256=hashlib.sha256(train.tobytes()).hexdigest(),
                                   test_mask_sha256=hashlib.sha256(test.tobytes()).hexdigest(),
                                   prediction_sha256=digest, model_audit=self.last_prediction_audit, score=scores[-1]))
                print(f"{group}: {protocol} fold {info['fold']} spellF1={scores[-1]['spell_f1_mean']:.4f}", flush=True)
            p = pooled(scores)
            result[protocol] = {'spell': p['spell_f1_mean'], 'macro': p['macro_f1'], 'dead': p['dead_f1'],
                                'folds': [s['spell_f1_mean'] for s in scores]}
            (out / f'{protocol}_folds.json').write_text(json.dumps(audits, indent=2), encoding='utf-8')
        result['n_features'] = n_features
        return result


def append_markdown(rows, note):
    if not RESULTS_MD.exists():
        RESULTS_MD.write_text(HEADER, encoding='utf-8')
    stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    lines = [f'\n### Run {stamp}\n', note + '\n' if note else '',
             '| Group | Features | Whole-run spellF1 | Fixed-time spellF1 | Whole-run macroF1 | Fixed-time macroF1 | Whole-run deadF1 | Fixed-time deadF1 | Fixed-time folds |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for group, r in rows:
        w, f = r['whole_run'], r['fixed_time']
        lines.append(f"| {group} | {r['n_features']} | {w['spell']:.4f} | {f['spell']:.4f} | "
                     f"{w['macro']:.4f} | {f['macro']:.4f} | {w['dead']:.4f} | {f['dead']:.4f} | "
                     f"{', '.join(f'{x:.3f}' for x in f['folds'])} |")
    with open(RESULTS_MD, 'a', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


HEADER = """# BlindSpot v0.4 experiment log

Generated by `python scripts/run_v04_experiments.py`. Aggregate public-window scores only.

How to read this:
* Every score comes from models that did not see the labels of the fold being scored
  (whole-run purged folds and equal-duration folds, both with a 300-frame embargo).
* `baseline` is the v0.3.1 feature set (68 features) and must reproduce
  whole-run 0.4457 / fixed-time 0.6200 spellF1. If it does not, the harness is broken.
* Only 17 public possessions exist: one possession moves spellF1 by about 0.06. Treat a group as
  better only if it improves BOTH protocols, by clearly more than that.
"""


def export_candidate(setup, group):
    labels, n_features = setup.predict(group, setup.valid)
    out = setup.output / group / 'candidate'
    out.mkdir(parents=True, exist_ok=True)
    digest = save_submission(out / 'CoreOrigin.csv', labels)
    table = pm.read_label_csv(out / 'CoreOrigin.csv')
    reason = pm.validate_submission(table, 199319)
    if reason is not None:
        raise ValueError('Invalid candidate: ' + reason)
    (out / 'manifest.json').write_text(json.dumps({'group': group, 'sha256': digest,
        'n_features': n_features, 'fingerprint': setup.fingerprint, 'provenance': setup.provenance,
        'model_audit': setup.last_prediction_audit,
        'scope': 'Fitted on all public labels; public score is training fit, not validation.'}, indent=2), encoding='utf-8')
    codes = pm.label_codes(table)
    runs = pm.valid_runs(codes)
    n_spells = len(pm.possession_spells(codes, runs))
    print(f'Candidate {out / "CoreOrigin.csv"}: valid={reason is None}, sha256={digest}, features={n_features}, '
          f'possessions in file={n_spells}')
    with open(RESULTS_MD, 'a', encoding='utf-8') as handle:
        handle.write(f"\nExported candidate `{group}`: `{out.as_posix()}/CoreOrigin.csv`, "
                     f"valid={reason is None}, SHA256 `{digest}`, {n_spells} possessions in file. "
                     "Fitted on all public labels; its public score is training fit, not validation.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('groups', nargs='*', default=list(GROUPS), help='experiments to run')
    parser.add_argument('--candidate', help='export a full-public-trained CoreOrigin.csv using this group')
    parser.add_argument('--note', default='', help='free-text note written above the results table')
    parser.add_argument('--out', type=Path, default=OUTPUT, help='experiment output directory')
    args = parser.parse_args()
    unknown = [g for g in args.groups + ([args.candidate] if args.candidate else []) if g not in GROUPS]
    if unknown:
        parser.error(f'unknown group(s) {unknown}; choose from {list(GROUPS)}')
    setup = Setup(args.out)
    if args.candidate:
        export_candidate(setup, args.candidate)
        return
    rows = []
    groups = ['baseline'] + [g for g in args.groups if g != 'baseline']
    setup.output.mkdir(parents=True, exist_ok=True)
    (setup.output / 'run_provenance.json').write_text(json.dumps(setup.provenance, indent=2), encoding='utf-8')
    for group in groups:
        started = time.time()
        checkpoint = setup.output / group / 'scores.json'
        cached = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
        if cached.get('fingerprint') == setup.fingerprint and all(
                (setup.output / group / name).exists() and sha256(setup.output / group / name) == digest
                for name, digest in cached.get('artifacts_sha256', {}).items()) and cached.get('artifacts_sha256'):
            r = cached['scores']
            print('Verified completed group; reused:', group, flush=True)
        else:
            r = setup.held_out(group)
            if group == 'baseline':
                verify_baseline(r)
            artifacts = {f.name: sha256(f) for f in checkpoint.parent.iterdir() if f.suffix in ('.csv', '.json') and f.name != 'scores.json'}
            checkpoint.write_text(json.dumps({'fingerprint': setup.fingerprint, 'scores': r,
                                              'artifacts_sha256': artifacts}, indent=2), encoding='utf-8')
        if group == 'baseline':
            verify_baseline(r)
        rows.append((group, r))
        print(f"{group:11s} features={r['n_features']:3d} whole-run={r['whole_run']['spell']:.4f} "
              f"fixed-time={r['fixed_time']['spell']:.4f} ({time.time()-started:.0f}s)", flush=True)
    append_markdown(rows, args.note)
    (setup.output / 'scores.json').write_text(json.dumps(dict(rows), indent=2), encoding='utf-8')
    print(f'Results appended to {RESULTS_MD}')


def verify_baseline(result):
    expected = {'whole_run': 0.4457142857142856, 'fixed_time': 0.62}
    if result['n_features'] != 68 or any(not np.isclose(result[k]['spell'], v, rtol=0, atol=1e-10)
                                           for k, v in expected.items()):
        raise ValueError('Baseline does not reproduce frozen v0.3.1; stop before comparing feature groups')


if __name__ == '__main__':
    main()
