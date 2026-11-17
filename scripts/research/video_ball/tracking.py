"""Read supplied automatic tracking on its one-based video timeline; no labels."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from games import GAMES, asset_paths, half_video_blocks
from possession.data import slot_teams


def load_tracking(data_dir):
    paths = asset_paths(data_dir=str(data_dir))
    teams = slot_teams(paths['atd_meta'])
    n = GAMES['BAR-ATH']['n_video_frames']
    players = np.full((n, len(teams), 2), np.nan, np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    count = 0
    with open(paths['atd_raw'], encoding='utf-8') as stream:
        for i, line in enumerate(stream):
            frame, people, tracked_ball = line.strip().split(':')
            if i >= n or int(frame) != i + 1:
                raise ValueError('ATD must contain exactly one-based sequential video frames')
            values = np.fromstring(people.replace(';', ','), sep=',')
            b = np.fromstring(tracked_ball, sep=',')
            if values.size != len(teams) * 3 or b.size != 3:
                raise ValueError(f'Invalid tracking record at video frame {i}')
            players[i] = values.reshape(-1, 3)[:, :2]
            ball[i] = b[:2]
            count += 1
    if count != n:
        raise ValueError(f'Expected {n} tracking rows, found {count}')
    pitch = np.asarray(GAMES['BAR-ATH']['pitch'], np.float32)
    return players * pitch, ball * pitch, teams


def play_mask(n):
    mask = np.zeros(n, bool)
    for _, start, stop in half_video_blocks('BAR-ATH'):
        mask[start:stop + 1] = True
    return mask


def runs(mask):
    edges = np.diff(np.r_[0, np.asarray(mask, np.int8), 0])
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def project_points(matrices, xy, pitch):
    """Metre pitch points -> image pixels, preserving invalid transforms as NaN."""
    xy = np.asarray(xy, float) / np.asarray(pitch, float)
    homogeneous = np.concatenate([xy, np.ones((*xy.shape[:-1], 1))], axis=-1)
    q = np.einsum('...ij,...j->...i', matrices, homogeneous)
    ok = np.isfinite(q).all(-1) & (np.abs(q[..., 2]) > np.finfo(float).eps)
    return np.divide(q[..., :2], q[..., 2, None],
                     out=np.full(q.shape[:-1] + (2,), np.nan), where=ok[..., None])


def bracketed_gaps(ball):
    """Missing runs with finite anchors inside the same half only; stops exclusive."""
    seen = np.isfinite(ball).all(1)
    result = []
    for half, first, last in half_video_blocks('BAR-ATH'):
        starts, stops = runs(~seen[first:last + 1])
        for start, stop in zip(starts + first, stops + first):
            if start > first and stop <= last:
                result.append((half, int(start), int(stop)))
    return result


def linear_gap(ball, start, stop):
    """Screening baseline only; does not claim the ball follows a straight path."""
    t = np.arange(1, stop - start + 1, dtype=float) / (stop - start + 1)
    return ball[start - 1] * (1 - t[:, None]) + ball[stop] * t[:, None]
