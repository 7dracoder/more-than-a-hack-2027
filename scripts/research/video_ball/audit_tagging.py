"""Compare supplied automatic phase cues with frozen predictions, without truth.

This is a label-free input diagnostic. It does not fit, score, select or alter
any possession model. Offense/defense tags describing the same interval and
possessor are counted once, rather than treated as independent corroboration.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from games import GAMES, asset_paths
from starter.load_smart_tagging import load as load_tagging
from scripts.research.video_ball.audit_inputs import digest
from scripts.research.video_ball.tracking import load_tracking, play_mask

# These are phase descriptions of ball ownership. Spatial occupancy, goals and
# set-piece windows are deliberately not interpreted as live possession states.
OFFENSIVE = {'BUILD UP', 'PROGRESSION', 'CREATING CHANCES', 'ATTACKING TRANSITION', 'LONG BALL'}
DEFENSIVE = {'DEFENDING IN ATTACKING THIRD', 'DEFENDING IN MIDDLE THIRD',
             'DEFENDING IN DEFENSIVE THIRD', 'DEFENSIVE TRANSITION'}
TEAM = {'barcelona': 0, 'atletic club': 1, 'athletic club': 1}


def phase_evidence(events, n):
    intervals = {}
    unknown_teams = Counter()
    for event in events:
        code = (event['code'] or '').strip().upper()
        if code not in OFFENSIVE | DEFENSIVE:
            continue
        name = (event['tags'].get('Team') or '').strip().lower()
        if name not in TEAM:
            unknown_teams[name] += 1
            continue
        owner = TEAM[name] if code in OFFENSIVE else 1 - TEAM[name]
        start, stop = max(0, event['start_frame']), min(n, event['end_frame'])
        if stop <= start:
            continue
        intervals.setdefault((start, stop, owner), set()).add(code)
    differences = np.zeros((n + 1, 2), np.int32)
    for start, stop, owner in intervals:
        differences[start, owner] += 1
        differences[stop, owner] -= 1
    count = np.cumsum(differences[:-1], axis=0)
    # -1 denotes no unambiguous cue; it is never interpreted as a label.
    owner = np.full(n, -1, np.int8)
    owner[(count[:, 0] > 0) & (count[:, 1] == 0)] = 0
    owner[(count[:, 1] > 0) & (count[:, 0] == 0)] = 1
    return owner, count, intervals, dict(unknown_teams)


def fraction(a, mask):
    return float(np.asarray(a)[mask].mean()) if mask.any() else None


def compare(labels, players, ball, teams, events):
    n = len(labels)
    active = play_mask(n)
    seen = np.isfinite(ball).all(1)
    owner, count, intervals, unknown = phase_evidence(events, n)
    predicted = np.where(labels == 'H', 0, np.where(labels == 'A', 1, -1))
    conflict = (count > 0).all(1)
    live = np.isin(labels, ['H', 'A'])
    distances = np.linalg.norm(players - ball[:, None], axis=-1)
    distances = np.where(np.isfinite(distances), distances, np.inf)
    nearest = np.stack([distances[:, teams == t].min(1) for t in (0, 1)], axis=1)
    usable = np.isfinite(nearest).all(1) & (nearest[:, 0] != nearest[:, 1]) & seen
    geometric = nearest.argmin(1)
    scopes = {}
    for name, mask in [('all', active), ('ball_observed', active & seen), ('ball_missing', active & ~seen)]:
        agreed_cue = mask & (owner >= 0)
        comparison = agreed_cue & live
        disagree = comparison & (owner != predicted)
        proximity_check = disagree & usable
        scopes[name] = {
            'frames': int(mask.sum()),
            'unambiguous_phase_frames': int(agreed_cue.sum()),
            'conflicting_phase_frames': int((mask & conflict).sum()),
            'model_live_and_phase_frames': int(comparison.sum()),
            'phase_model_disagreement_frames': int(disagree.sum()),
            'phase_model_agreement_fraction': fraction(owner == predicted, comparison),
            'model_dead_inside_phase_frames': int((agreed_cue & (labels == 'D')).sum()),
            'disagreement_nearest_player_comparable_frames': int(proximity_check.sum()),
            'disagreement_nearest_player_supports_phase_fraction': fraction(geometric == owner, proximity_check),
        }
    boundaries = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, n]
    missing_live_runs = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if labels[start] not in ('H', 'A') or not active[start:stop].all() or seen[start:stop].any():
            continue
        cues = owner[start:stop]
        known = cues[cues >= 0]
        missing_live_runs.append({
            'start_frame': int(start), 'stop_frame_exclusive': int(stop),
            'predicted_side': str(labels[start]), 'frames': int(stop - start),
            'phase_coverage_fraction': float((cues >= 0).mean()),
            'phase_conflict_frames': int(conflict[start:stop].sum()),
            'unique_phase_sides': sorted(set('H' if x == 0 else 'A' for x in known)),
        })
    return {
        'purpose': 'Compare automatic inputs; not accuracy, training or model selection',
        'possession_labels_read': False, 'candidate_modified': False, 'possession_score': None,
        'canonical_phase_intervals': len(intervals),
        'raw_phase_intervals': sum(len(v) for v in intervals.values()),
        'duplicate_semantic_phase_intervals': sum(len(v) - 1 for v in intervals.values()),
        'unknown_phase_teams': unknown, 'scopes': scopes,
        'entirely_ball_missing_live_runs': missing_live_runs,
        'limitations': [
            'Automatic tags are noisy phase intervals, not possession truth.',
            'Offense/defense pairs with identical intervals are duplicate evidence.',
            'Nearest-player support does not establish controlled possession.',
            'Disagreement cannot determine which automatic input is correct.',
            'No setting or candidate may be selected using public BAR-ATH possession labels.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--candidate', type=Path, default=ROOT / 'data/bar-ath-check-20261103/submission/CoreOrigin.csv')
    parser.add_argument('--out', type=Path, default=ROOT / 'data/research/target080/tagging_audit.json')
    args = parser.parse_args()
    candidate_hash = digest(args.candidate)
    table = pd.read_csv(args.candidate)
    n = GAMES['BAR-ATH']['n_video_frames']
    if list(table.columns) != ['frame', 'label'] or not np.array_equal(table['frame'], np.arange(n)):
        raise ValueError('Candidate must use the complete tactical video timeline')
    if not table['label'].isin(['H', 'A', 'D', 'X']).all():
        raise ValueError('Invalid candidate labels')
    players, ball, teams = load_tracking(args.data)
    events = load_tagging(data_dir=str(args.data))
    result = compare(table['label'].to_numpy(), players, ball, teams, events)
    paths = asset_paths(data_dir=str(args.data))
    result['input_sha256'] = {k: digest(paths[k]) for k in ('atd_raw', 'atd_meta', 'smart_tagging')}
    result['candidate_sha256'] = candidate_hash
    result['source_sha256'] = digest(__file__)
    result['tracking_source_sha256'] = digest(Path(__file__).with_name('tracking.py'))
    if digest(args.candidate) != candidate_hash:
        raise ValueError('Candidate changed during audit')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    part = args.out.with_suffix('.json.partial')
    part.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    part.replace(args.out)
    print(json.dumps({k: v for k, v in result.items() if k != 'entirely_ball_missing_live_runs'}, indent=2))


if __name__ == '__main__':
    main()
