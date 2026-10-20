"""Explicit SI-unit features, retaining the historical v1 cache for reproduction."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET
import warnings

import numpy as np

from games import GAME, GAMES, asset_paths, half_video_blocks
from possession.data import extract, sha256, slot_teams, rolling_mean, load_cached_arrays


def speed_scale(metadata_path):
    sensors = [item for item in ET.parse(metadata_path).iter('Sensor') if item.get('id') == 'speed']
    units = {(item.findtext('Unit') or '').strip().lower() for sensor in sensors for item in sensor.iter('Channel')}
    if units in ({'kph'}, {'km/h'}):
        return 1 / 3.6, sorted(units)
    if units in ({'m/s'}, {'mps'}):
        return 1., sorted(units)
    raise ValueError(f'Unsupported/ambiguous player speed units: {units}')


def metric_reduction(players, teams, scale, max_speed=15., stationary_threshold=.5):
    """Pure, slot-permutation-invariant speed reductions AFTER unit conversion."""
    xy = players[..., :2]
    speed = players[..., 2] * scale
    known = np.isfinite(xy).all(-1) & np.isfinite(speed) & (speed >= 0) & (speed <= max_speed)
    values = np.where(known, speed, np.nan)
    count = known.sum(-1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        quantiles = np.nanquantile(values, [.2, .5, .8], axis=-1).T
        team_speed = np.stack([np.nanmedian(np.where(known & (teams == side), speed, np.nan), axis=-1) for side in (0, 1)], axis=-1)
    stationary = np.divide(((values < stationary_threshold) & known).sum(-1), count,
                           out=np.full(np.shape(count), np.nan), where=count > 0)
    running = np.divide(((values > 2.) & known).sum(-1), count,
                        out=np.full(np.shape(count), np.nan), where=count > 0)
    return quantiles, stationary, running, team_speed


def trajectory_speed(xy, fps=25., pitch=(105.3, 68.), span=6, max_speed=15.):
    """Centered displacement speed in metres/second with continuous visibility.

    Reject windows crossing a missing point or an impossible adjacent-frame jump.
    Track slots are only assumed continuous within this short 0.24-second window.
    """
    xy = np.asarray(xy, dtype=float)
    if span < 2 or span % 2 or len(xy) <= span:
        raise ValueError('Need an even span and enough trajectory frames')
    out = np.full(xy.shape[:-1], np.nan)
    invalid = ~np.isfinite(xy).all(-1)
    prefix = np.concatenate([np.zeros((1, xy.shape[1]), dtype=int), np.cumsum(invalid, axis=0)])
    complete = (prefix[span+1:] - prefix[:-span-1]) == 0
    jumps = np.linalg.norm(np.diff(xy, axis=0) * pitch, axis=-1) * fps > max_speed
    jump_prefix = np.concatenate([np.zeros((1, xy.shape[1]), dtype=int), np.cumsum(jumps, axis=0)])
    continuous = (jump_prefix[span:] - jump_prefix[:-span]) == 0
    speed = np.linalg.norm((xy[span:] - xy[:-span]) * pitch, axis=-1) * fps / span
    speed = np.where(complete & continuous & (speed <= max_speed), speed, np.nan)
    out[span//2:-span//2] = speed
    return out


def restart_features(path, n):
    """Keep phase disagreements explicit; restart ownership is a separate weak cue."""
    phase_delta = np.zeros((n + 1, 2))
    owner_delta = np.zeros(n + 1)
    restart_delta = np.zeros(n + 1)
    kinds = ('throw', 'free', 'corner', 'goal', 'kick off')
    type_delta = np.zeros((n + 1, len(kinds)))
    offensive = {'build up', 'progression', 'creating chances', 'attacking transition', 'long ball',
                 'ball in final third', 'ball in the box', 'players in the box'}
    defensive = {'defending in attacking third', 'defending in middle third', 'defending in defensive third', 'defensive transition'}
    for item in ET.parse(path).iter('instance'):
        a, b = [max(0, min(n, round(float(item.findtext(key)) * 25))) for key in ('start', 'end')]
        if b <= a:
            continue
        code = (item.findtext('code') or '').lower().strip()
        tags = {(x.findtext('group') or '').lower().strip(): (x.findtext('text') or '').lower().strip() for x in item.findall('label')}
        name = tags.get('team', '')
        sign = 1 if name == 'barcelona' else -1 if name in ('athletic club', 'atletic club') else 0
        if code == 'set pieces':
            restart_delta[a] += 1
            restart_delta[b] -= 1
            owner_delta[a] += sign
            owner_delta[b] -= sign
            kind = tags.get('type', '')
            for k, word in enumerate(kinds):
                if word in kind:
                    type_delta[a, k] += 1
                    type_delta[b, k] -= 1
        elif code in offensive | defensive and sign:
            if code in defensive:
                sign *= -1
            side = 0 if sign > 0 else 1
            phase_delta[a, side] += 1
            phase_delta[b, side] -= 1
    phase = np.cumsum(phase_delta[:-1], axis=0)
    both = (phase > 0).all(1)
    # Bounded normalized vote, not possession ground truth.
    support = (phase[:, 0] - phase[:, 1]) / np.maximum(1., phase.sum(1))
    return {'phase_support': support, 'phase_conflict': both.astype(float),
            'restart_owner': np.clip(np.cumsum(owner_delta[:-1]), -1, 1),
            'restart_types': np.cumsum(type_delta[:-1], axis=0).clip(0, 1),
            'restart': np.cumsum(restart_delta[:-1]) > 0}


def extract_metric(data_dir, cache_dir, config):
    base, base_meta = extract(data_dir, cache_dir)
    paths = asset_paths(data_dir=str(data_dir))
    declared_scale, units = speed_scale(paths['atd_meta'])
    fingerprint = {'inputs': base_meta['input_sha256'], 'extractor_sha256': sha256(__file__),
                   'base_feature_fingerprint': base_meta.get('fingerprint'),
                   'game_constants_sha256': sha256(Path(__file__).resolve().parents[1] / 'games.py'),
                   'speed_source': 'centered_six_frame_displacement_mps', 'declared_scale_to_mps': declared_scale,
                   'max_player_speed_mps': config['max_player_speed_mps'],
                   'stationary_threshold_mps': config['stationary_threshold_mps']}
    cache_dir = Path(cache_dir)
    file, meta_file = cache_dir / 'metric-features-v3.npz', cache_dir / 'metric-features-v3.json'
    cached = load_cached_arrays(file, meta_file, fingerprint)
    if cached is not None:
        return cached
    n = len(base['ball'])
    teams = slot_teams(paths['atd_meta'])
    quantiles = np.full((n, 3), np.nan, dtype=np.float32)
    team_speed = np.full((n, 2), np.nan, dtype=np.float32)
    stationary = np.full(n, np.nan, dtype=np.float32)
    running = stationary.copy()
    players_all = np.full((n, len(teams), 3), np.nan, dtype=np.float32)
    with open(paths['atd_raw']) as stream:
        count = 0
        for count, line in enumerate(stream, 1):
            frame_id, raw, _ = line.strip().split(':')
            if int(frame_id) != count:
                raise ValueError('Tracking frame order mismatch')
            players = np.fromstring(raw.replace(';', ','), sep=',')
            if players.size != len(teams) * 3:
                raise ValueError('Unexpected track dimensions')
            players_all[count - 1] = players.reshape(-1, 3)
    if count != n:
        raise ValueError('Tracking frame count mismatch')
    provided = players_all[:, :, 2].copy()
    derived = np.full_like(provided, np.nan)
    for _, first, last in half_video_blocks(GAME):
        derived[first:last+1] = trajectory_speed(players_all[first:last+1, :, :2], max_speed=config['max_player_speed_mps'])
    sampled = np.arange(0, n, 25)
    v, raw = derived[sampled], provided[sampled]
    valid = np.isfinite(v) & np.isfinite(raw) & (v > .5) & (v < 12) & (raw > .5)
    v, raw = v[valid], raw[valid]
    unit_audit = {'declared_metadata_units': units, 'sampled_pairs': len(v),
                  'median_reported_over_trajectory_mps': float(np.median(raw/v)),
                  'reported_vs_trajectory_correlation': float(np.corrcoef(raw, v)[0, 1]),
                  'median_absolute_error_if_raw_mps': float(np.median(abs(raw-v))),
                  'median_absolute_error_if_metadata_kph': float(np.median(abs(raw*declared_scale-v))),
                  'decision': 'Derive speed from normalized positions, pitch dimensions and frame timing; metadata scaling is not trusted.',
                  'caution': 'Position-derived speed also inherits tracking noise and slot reassignment risk; short windows with gaps/jumps are rejected.'}
    players_all[:, :, 2] = derived
    for start in range(0, n, 2048):
        stop = min(n, start+2048)
        q, s, r, t = metric_reduction(players_all[start:stop], teams, 1.,
                                     config['max_player_speed_mps'], config['stationary_threshold_mps'])
        quantiles[start:stop], stationary[start:stop], running[start:stop], team_speed[start:stop] = q, s, r, t
    result = dict(base)
    result.update(player_speed_quantiles=quantiles, median_speed=quantiles[:, 1], stationary=stationary,
                  running_fraction=running, team_speed=team_speed)
    result.update(restart_features(paths['smart_tagging'], n))
    np.savez_compressed(file, **result)
    meta = {'feature_version': 3, 'fingerprint': fingerprint, 'cache_sha256': sha256(file),
            'input_sha256': base_meta['input_sha256'], 'source_speed_units': units, 'output_speed_units': 'm/s',
            'unit_audit': unit_audit,
            'warning': 'Metadata says kph, while reported values empirically resemble trajectory m/s. The initial metadata-scaling experiment is retained as a pilot, not a proven unit correction.'}
    meta_file.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    return result, meta


def design(arrays, config):
    """Team-symmetric, finite physical features; temporal context stays within halves."""
    n = len(arrays['ball'])
    columns = {}
    def put(name, values):
        columns[name] = np.nan_to_num(values, nan=0., posinf=0., neginf=0.)
    for _, start, last in half_video_blocks(GAME):
        stop = last + 1
        ball = arrays['ball'][start:stop]
        seen = np.isfinite(ball).all(1)
        nearest = np.nan_to_num(arrays['nearest'][start:stop], nan=30.).clip(0, 30)
        velocity = np.linalg.norm(np.diff(ball, axis=0, prepend=ball[:1]) * GAMES[GAME]['pitch'], axis=1) * 25
        contiguous = seen & np.r_[False, seen[:-1]]
        velocity = np.where(contiguous & (velocity < 60), velocity, np.nan)
        source = {'speed20': arrays['player_speed_quantiles'][start:stop, 0],
                  'speed50': arrays['player_speed_quantiles'][start:stop, 1],
                  'speed80': arrays['player_speed_quantiles'][start:stop, 2],
                  'stationary': arrays['stationary'][start:stop], 'running': arrays['running_fraction'][start:stop],
                  'ball_seen': seen.astype(float), 'ball_speed': velocity,
                  'ball_outside': (seen & ((ball < 0) | (ball > 1)).any(1)).astype(float),
                  'nearest_home': nearest[:, 0] / 10, 'nearest_away': nearest[:, 1] / 10,
                  'control_margin': (nearest[:, 1] - nearest[:, 0]) / (nearest.sum(1) + .5),
                  'phase': arrays['phase_support'][start:stop],
                  'phase_conflict': arrays['phase_conflict'][start:stop],
                  'restart': arrays['restart'][start:stop].astype(float),
                  'restart_owner': arrays['restart_owner'][start:stop],
                  'count_total': arrays['counts'][start:stop].sum(1) / 22.,
                  'count_difference': (arrays['counts'][start:stop, 0] - arrays['counts'][start:stop, 1]) / 11.,
                  'team_speed_difference': arrays['team_speed'][start:stop, 0] - arrays['team_speed'][start:stop, 1]}
        if config.get('include_observation_masks', False):
            source.update(speed_observed=np.isfinite(source['speed50']).astype(float),
                          ball_speed_observed=np.isfinite(velocity).astype(float),
                          team_speed_observed=np.isfinite(arrays['team_speed'][start:stop]).all(1).astype(float))
        for name, values in source.items():
            for width in config['feature_windows_frames']:
                key = f'{name}_{width}'
                if key not in columns:
                    columns[key] = np.zeros(n, dtype=np.float32)
                columns[key][start:stop] = np.nan_to_num(rolling_mean(values, width), nan=0.)
        for k in range(arrays['restart_types'].shape[1]):
            key = f'restart_type_{k}'
            if key not in columns:
                columns[key] = np.zeros(n, dtype=np.float32)
            columns[key][start:stop] = arrays['restart_types'][start:stop, k]
    names = list(columns)
    matrix = np.column_stack([columns[key] for key in names])
    return matrix, names


def swap_teams(matrix, names):
    swapped = matrix.copy()
    for index, name in enumerate(names):
        if name.startswith('nearest_home_'):
            other = names.index(name.replace('nearest_home_', 'nearest_away_'))
            swapped[:, index] = matrix[:, other]
        elif name.startswith('nearest_away_'):
            other = names.index(name.replace('nearest_away_', 'nearest_home_'))
            swapped[:, index] = matrix[:, other]
        elif name.startswith(('control_margin_', 'phase_', 'restart_owner_', 'count_difference_', 'team_speed_difference_')) and not name.startswith('phase_conflict_'):
            swapped[:, index] *= -1
    return swapped
