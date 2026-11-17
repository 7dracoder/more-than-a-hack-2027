"""Bounded external-only diagnostic of fixed two-view ball fusion.

Uses J03WPY and synthetic independent camera losses, never BAR-ATH statistics,
video or labels. This is not an estimate of competition performance. No model
fit, parameter sweep or candidate selection is performed.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from possession.data import sha256
from scripts.research.broadcast.dual_feed import PROTOCOL, fuse_ball, paired_cost, write_json


BENCHMARK = dict(match='J03WPY', seeds=[713, 714], block_frames=125,
                 primary_missing_probability=.4, secondary_missing_probability=.4,
                 secondary_ball_noise_std_m=.25, secondary_player_noise_std_m=.25,
                 selection_performed=False, provided_match_statistics_used=False,
                 limitation='Artificial independent losses; real cameras may fail together.')


def synthetic_views(players, teams, ball, seed):
    """Coordinate-only simulation; possession labels do not drive either sensor."""
    rng = np.random.default_rng(seed)
    n, width = len(ball), BENCHMARK['block_frames']
    blocks = (n + width - 1) // width
    primary = ball.copy()
    primary_missing = np.repeat(rng.random(blocks) < BENCHMARK['primary_missing_probability'], width)[:n]
    primary[primary_missing] = np.nan
    secondary = ball + rng.normal(0., BENCHMARK['secondary_ball_noise_std_m'], ball.shape).astype(np.float32)
    secondary_missing = np.repeat(rng.random(blocks) < BENCHMARK['secondary_missing_probability'], width)[:n]
    secondary[secondary_missing] = np.nan
    observed_players = players + rng.normal(0., BENCHMARK['secondary_player_noise_std_m'], players.shape).astype(np.float32)
    accepted = np.zeros(n, bool)
    for first in range(0, n, 1000):
        stop = min(n, first + 1000)
        cost, count = paired_cost(observed_players[first:stop], teams, players[first:stop], teams,
                                  PROTOCOL['chamfer_clip_m'])
        accepted[first:stop] = ((count >= PROTOCOL['min_visible_players']) &
                                (cost < PROTOCOL['max_player_cost_m']))
    return primary, dict(ball=secondary, accepted_view=accepted)


def evaluate(truths, predictions):
    from evaluation import possession_metrics as pm
    # Explicit unknown separators prevent spells from crossing halftime.
    truth = np.concatenate([part for a in truths for part in (a, np.array(['X']))])
    predicted = np.concatenate([part for a in predictions for part in (a, np.array(['X']))])
    result = pm.evaluate(pm.label_codes(truth), pm.label_codes(predicted), bootstrap_n=0)
    return {key:result[key] for key in ('spell_f1_mean', 'macro_f1', 'n_gt_spells', 'n_pred_spells')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'protocol.json', dict(benchmark=BENCHMARK, fusion=PROTOCOL))
    external = (ROOT / 'data/external').resolve()
    data = (ROOT / 'data').resolve()
    out = args.out.resolve()
    public_truth = (ROOT / 'evaluation/gt').resolve()
    blocked = []

    def guard(event, values):
        if event == 'open' and values and isinstance(values[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(values[0])).resolve()
            if path.is_relative_to(public_truth) or (
                    path.is_relative_to(data) and not path.is_relative_to(external) and not path.is_relative_to(out)):
                blocked.append(str(path))
                raise PermissionError('External benchmark may not access competition data or labels')

    sys.addaudithook(guard)
    # Verify the isolation guard before any model or benchmark input is loaded.
    try:
        open(ROOT / 'data/tactical/BAR-ATH_tactical_atd.txt', 'rb')
        raise AssertionError('Isolation guard failed')
    except PermissionError:
        pass
    blocked.clear()
    import torch
    from possession.external import read_sportec
    from possession.features_v25 import build_features
    from scripts.run_local import selected_model, symmetric_probability, viterbi_labels

    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    device = torch.device('cpu')
    parent = ROOT / 'models/coreorigin-video-v1'
    model, state = selected_model(parent, device)
    motion_path = parent / 'motion.json'
    motion = json.loads(motion_path.read_text(encoding='utf-8'))
    if motion['provided_match_data_used'] or BENCHMARK['match'] in state['config']['train_matches']:
        raise ValueError('Benchmark cannot use a training match or target-derived motion fit')
    folder = external / 'sportec'
    metadata = next(folder.glob('*matchinformation*' + BENCHMARK['match'] + '.xml'))
    tracking = next(folder.glob('*positions*' + BENCHMARK['match'] + '.xml'))
    pins = {item['name']:item['sha256'] for item in json.loads((folder / 'download_manifest.json').read_text())['files']}
    for path in (metadata, tracking):
        if pins.get(path.name) != sha256(path):
            raise ValueError('External match differs from its verified source manifest')
    halves = read_sportec(metadata, tracking)
    print('Verified external development match loaded; competition-data access denied', flush=True)
    results = []
    for seed in BENCHMARK['seeds']:
        predictions = dict(primary=[], fused=[])
        truths, counts = [], []
        for i, half in enumerate(halves):
            primary, signal = synthetic_views(half.players, half.teams, half.ball, seed + 1000 * i)
            fused, _, _ = fuse_ball(primary, signal, [(0, len(primary))], motion['max_speed_mps'])
            counts.append(dict(primary_missing=int((~np.isfinite(primary).all(1)).sum()),
                               fused_missing=int((~np.isfinite(fused).all(1)).sum())))
            for name, ball in (('primary', primary), ('fused', fused)):
                matrix, names = build_features(half.players, half.teams, ball, half.pitch, step=1)
                if names != state['names']:
                    raise ValueError('External feature schema differs from the frozen model')
                probability = symmetric_probability(model, matrix, names, state['mean'], state['scale'], device)
                predictions[name].append(viterbi_labels(probability, *state['decoder_costs'])[:len(half.labels)])
                del matrix, probability
                print(f'External seed {seed}, half {i + 1}, {name}: complete', flush=True)
            truths.append(half.labels)
        scores = {name:evaluate(truths, pred) for name, pred in predictions.items()}
        results.append(dict(seed=seed, scores=scores, missing_frames=counts,
                            spell_f1_gain=scores['fused']['spell_f1_mean'] - scores['primary']['spell_f1_mean']))
        write_json(args.out / f'seed-{seed}.json', results[-1])
        print(json.dumps(results[-1]), flush=True)
    result = dict(benchmark=BENCHMARK, results=results, complete=True,
                  mean_spell_f1_gain=float(np.mean([item['spell_f1_gain'] for item in results])),
                  parent_sha256=sha256(parent / 'selected.pt'), motion_sha256=sha256(motion_path),
                  input_sha256={path.name:sha256(path) for path in (metadata, tracking)},
                  source_sha256=sha256(__file__), fusion_source_sha256=sha256(ROOT / 'scripts/research/broadcast/dual_feed.py'),
                  isolation_guard_verified=True, unexpected_competition_data_access=blocked,
                  provided_match_data_read=False, training_performed=False, selection_performed=False,
                  full_match_competition_f1=None,
                  limitation='Two artificial seeds on the existing development match; no independent test or BAR-ATH score.')
    write_json(args.out / 'results.json', result)
    print('External diagnostic finished; no competition candidate selected', flush=True)


if __name__ == '__main__':
    main()
