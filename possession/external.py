"""External-match adapters with a strict boundary between coordinates and labels.

Source: Bassek et al. 2025, doi:10.1038/s41597-025-04505-y, CC BY 4.0.
Only x/y coordinates, frame counters and team membership enter features.
BallStatus/BallPossession are supervision, never input features.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET

import numpy as np
from scipy.ndimage import uniform_filter1d

from possession.data import sha256


@dataclass
class TrackingHalf:
    match_id: str
    period: str
    frames: np.ndarray
    players: np.ndarray   # metres, shape (frames, slots, 2)
    teams: np.ndarray     # 0 home / 1 away
    ball: np.ndarray      # metres, shape (frames, 2)
    pitch: tuple
    labels: np.ndarray   # deliberately separate from build_features arguments


def external_labels(status, possession):
    status, possession = np.asarray(status), np.asarray(possession)
    result = np.full(len(status), 'X', dtype='<U1')
    result[status == '0'] = 'D'
    result[(status == '1') & (possession == '1')] = 'H'
    result[(status == '1') & (possession == '2')] = 'A'
    return result


def read_sportec(metadata, tracking):
    root = ET.parse(metadata).getroot()
    general = next(root.iter('General'))
    match = general.get('MatchId').split('-')[-1]
    sides = {general.get('HomeTeamId'): 0, general.get('GuestTeamId'): 1}
    environment = next(root.iter('Environment'))
    pitch = (float(environment.get('PitchX')), float(environment.get('PitchY')))
    periods = {}
    # Frames are grouped by tracked object, so clear complete FrameSets only.
    for _, item in ET.iterparse(tracking, events=('end',)):
        if item.tag != 'FrameSet':
            continue
        team, person, period = item.get('TeamId', ''), item.get('PersonId', ''), item.get('GameSection')
        is_ball = team.lower() == 'ball' or person.lower() == 'ball'
        if period in ('firstHalf', 'secondHalf') and (team in sides or is_ball):
            rows = list(item.iter('Frame'))
            ids = np.fromiter((int(x.get('N')) for x in rows), dtype=np.int64)
            xy = np.array([(float(x.get('X', 'nan')), float(x.get('Y', 'nan'))) for x in rows], dtype=np.float32)
            data = dict(ids=ids, xy=xy, side=sides.get(team, -1))
            if is_ball:
                data['labels'] = external_labels([x.get('BallStatus', '') for x in rows],
                                               [x.get('BallPossession', '') for x in rows])
            periods.setdefault(period, {})['ball' if is_ball else person] = data
        item.clear()
    halves = []
    for period, objects in sorted(periods.items()):
        if 'ball' not in objects:
            raise ValueError('No ball FrameSet: ' + match + '/' + period)
        observed = objects.pop('ball')
        frames = np.arange(observed['ids'].min(), observed['ids'].max() + 1)
        players = np.full((len(frames), len(objects), 2), np.nan, dtype=np.float32)
        teams = np.empty(len(objects), np.int8)
        for slot, (_, player) in enumerate(sorted(objects.items())):
            ids = player['ids'] - frames[0]
            keep = (ids >= 0) & (ids < len(frames))
            players[ids[keep], slot] = player['xy'][keep]
            teams[slot] = player['side']
        ball = np.full((len(frames), 2), np.nan, np.float32)
        labels = np.full(len(frames), 'X', dtype='<U1')
        ids = observed['ids'] - frames[0]
        ball[ids], labels[ids] = observed['xy'], observed['labels']
        # Source uses a centred pitch. Translate to the shared [0, pitch] convention.
        players += np.asarray(pitch, np.float32) / 2
        ball += np.asarray(pitch, np.float32) / 2
        halves.append(TrackingHalf(match, period, frames, players, teams, ball, pitch, labels))
    return halves


def finite_mean(values, radius):
    values = np.asarray(values, np.float32)
    known = np.isfinite(values)
    size = radius * 2 + 1
    sums = uniform_filter1d(np.where(known, values, 0.), size, axis=0, mode='nearest')
    mass = uniform_filter1d(known.astype(np.float32), size, axis=0, mode='nearest')
    return np.divide(sums, mass, out=np.full_like(sums, np.nan), where=mass > 0)


def displacement_speed(xy, fps=25., max_speed=15., span=6):
    xy = np.asarray(xy)
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    if len(xy) <= span:
        return out
    finite = np.isfinite(xy).all(-1)
    jumps = np.linalg.norm(np.diff(xy, axis=0), axis=-1) * fps
    bad = ~finite
    prefix = np.concatenate([np.zeros_like(bad[:1], dtype=int), np.cumsum(bad, axis=0)])
    complete = prefix[span+1:] == prefix[:-span-1]
    jump_prefix = np.concatenate([np.zeros_like(jumps[:1], dtype=int),
                                  np.cumsum((jumps > max_speed) | ~np.isfinite(jumps), axis=0)])
    continuous = jump_prefix[span:] == jump_prefix[:-span]
    speed = np.linalg.norm(xy[span:] - xy[:-span], axis=-1) * fps / span
    valid = complete & continuous & (speed <= max_speed)
    out[span//2:-span//2] = np.where(valid, speed, np.nan)
    return out


def build_features(players, teams, ball, pitch, fps=25., step=5):
    """Label-free, slot-invariant feature extraction; call separately per half.

    Coordinates are metres with pitch origin at one corner. No provider speed,
    Z-height, event log, match id, timestamps, labels, or state flags are accepted.
    Exactly the same function is used for external training and BAR-ATH inference.
    """
    players, ball = np.asarray(players), np.asarray(ball)
    n = len(ball)
    if players.shape[0] != n or players.shape[1] != len(teams):
        raise ValueError('Tracking dimensions disagree')
    columns = {}
    seen = np.isfinite(ball).all(-1)
    velocity = displacement_speed(players, fps)
    ball_speed = displacement_speed(ball, fps, max_speed=60.)
    columns['ball_seen'] = seen.astype(float)
    columns['ball_speed'] = ball_speed / 20.
    outside = np.maximum.reduce([-ball[:, 0], ball[:, 0] - pitch[0], -ball[:, 1], ball[:, 1] - pitch[1]])
    columns['ball_outside'] = outside.clip(-10, 10) / 10.
    columns['ball_edge_distance'] = np.minimum.reduce([ball[:, 0], pitch[0] - ball[:, 0], ball[:, 1], pitch[1] - ball[:, 1]]) / 34.
    nearest = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for side, prefix in ((0, 'home'), (1, 'away')):
            xy, speed = players[:, teams == side], velocity[:, teams == side]
            distance = np.linalg.norm(xy - ball[:, None, :], axis=-1)
            ordered = np.sort(np.where(np.isfinite(distance), distance, np.inf), axis=1)
            for rank in range(3):
                columns[f'{prefix}_distance{rank+1}'] = np.where(np.isfinite(ordered[:, rank]), np.minimum(ordered[:, rank], 30.) / 10., np.nan)
            nearest.append(ordered[:, 0])
            columns[f'{prefix}_observed'] = np.isfinite(xy).all(-1).sum(1) / 11.
            columns[f'{prefix}_near_ball'] = (distance < 3.).sum(1) / 3.
            columns[f'{prefix}_contact'] = (ordered[:, 0] < 1.).astype(float) * seen
            columns[f'{prefix}_speed20'] = np.nanquantile(speed, .2, axis=1) / 5.
            columns[f'{prefix}_speed50'] = np.nanquantile(speed, .5, axis=1) / 5.
            columns[f'{prefix}_speed80'] = np.nanquantile(speed, .8, axis=1) / 5.
            count = np.isfinite(speed).sum(1)
            columns[f'{prefix}_stationary'] = np.divide((speed < .5).sum(1), count, out=np.full(n, np.nan), where=count > 0)
            columns[f'{prefix}_running'] = np.divide((speed > 2.).sum(1), count, out=np.full(n, np.nan), where=count > 0)
            columns[f'{prefix}_speed_observed'] = count / 11.
            centroid = np.nanmean(xy, axis=1)
            columns[f'{prefix}_centroid_ball'] = np.linalg.norm(centroid - ball, axis=1) / 50.
            columns[f'{prefix}_spread'] = np.sqrt(np.nanmean(np.sum((xy - centroid[:, None]) ** 2, axis=-1), axis=1)) / 30.
    # A last reliable contact persists through flight/missing ball; it is evidence,
    # not a ground-truth label. Simultaneous/equidistant contacts remain ambiguous.
    distance = np.stack(nearest, axis=1)
    owner = np.where(distance[:, 0] < distance[:, 1], 0, 1)
    valid_contact = seen & (distance.min(1) < 1.) & (np.abs(distance[:, 0] - distance[:, 1]) > .25)
    last = np.maximum.accumulate(np.where(valid_contact, np.arange(n), -1))
    age = np.where(last >= 0, np.minimum((np.arange(n) - last) / fps, 30.), 30.)
    columns['contact_age'] = age / 30.
    for side, prefix in ((0, 'home'), (1, 'away')):
        columns[f'{prefix}_last_contact'] = ((last >= 0) & (owner[np.maximum(last, 0)] == side)).astype(float)
    # Symmetric temporal context, using coordinates only. No half-boundary crossing.
    base = list(columns.items())
    for name, value in base:
        for radius in (12, 50, 125):
            columns[f'{name}_mean{radius}'] = finite_mean(value, radius)
    names = list(columns)
    values = np.stack([columns[name][::step] for name in names], axis=1)
    known = np.isfinite(values)
    names += [name + '_known' for name in names]
    values = np.concatenate([np.where(known, values, 0.), known.astype(np.float32)], axis=1)
    return values.astype(np.float32), names


def swap_features(matrix, names):
    index = {name: i for i, name in enumerate(names)}
    mapping = []
    for name in names:
        other = 'away_' + name[5:] if name.startswith('home_') else 'home_' + name[5:] if name.startswith('away_') else name
        mapping.append(index[other])
    return matrix[..., mapping]


def prepare_match(folder, match, cache_dir, step=5):
    metadata = next(folder.glob('*matchinformation*' + match + '.xml'))
    tracking = next(folder.glob('*positions*' + match + '.xml'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (match + '.npz')
    provenance = dict(metadata_sha256=sha256(metadata), tracking_sha256=sha256(tracking),
                      source_sha256=sha256(Path(__file__)), step=step)
    meta_path = path.with_suffix('.json')
    if path.exists() and meta_path.exists():
        record = json.loads(meta_path.read_text())
        if record.get('provenance') == provenance and record.get('sha256') == sha256(path):
            return path
    output, records = {}, []
    for i, half in enumerate(read_sportec(metadata, tracking)):
        x, names = build_features(half.players, half.teams, half.ball, half.pitch, step=step)
        output.update({f'x{i}': x, f'y{i}': half.labels[::step], f'truth{i}': half.labels, f'frames{i}': half.frames})
        records.append(dict(period=half.period, frames=len(half.frames),
                            labels={s: int((half.labels == s).sum()) for s in 'HADX'}))
    output['names'] = np.array(names)
    np.savez_compressed(path, **output)
    meta_path.write_text(json.dumps(dict(provenance=provenance, sha256=sha256(path), halves=records), indent=2))
    return path
