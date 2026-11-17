"""Coordinate-only input adapter for the pinned external PathCRF checkpoint.

No ball, possession, event, video or provider phase fields enter this adapter.
It abstains when a complete 22-player window cannot be formed. Outside nodes
are spatial boundary states; they must never be interpreted as dead-ball truth.
"""
import numpy as np
from scipy.signal import savgol_filter


def finite_runs(mask):
    change = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1))


def player_features(players, pitch, fps=25.):
    """Match the author's six physical channels on contiguous observations.

    Rescale coordinates to the checkpoint's fixed 105 x 68 metre field. Never
    interpolate across absent observations or a half boundary. Samples with
    impossible speed remain invalid rather than becoming a learned zero.
    """
    players = np.asarray(players, np.float32)
    pitch = np.asarray(pitch, np.float32)
    if players.ndim != 3 or players.shape[-1] != 2 or pitch.shape != (2,) or np.any(pitch <= 0) or fps <= 0:
        raise ValueError('Need one half of metre coordinates and a positive pitch/fps')
    xy = players * (np.array([105., 68.], np.float32) / pitch)
    result = np.full((*xy.shape[:2], 6), np.nan, np.float32)
    result[:, :, :2] = xy
    for slot in range(xy.shape[1]):
        valid = np.isfinite(xy[:, slot]).all(1)
        speed = np.r_[0., np.linalg.norm(np.diff(xy[:, slot], axis=0), axis=1) * fps]
        valid &= ~((speed > 12.) & np.isfinite(speed))
        for a, b in finite_runs(valid):
            if b - a < 16:
                continue
            velocity = savgol_filter(np.diff(xy[a:b, slot], axis=0) * fps, 15, 2, axis=0)
            acceleration = savgol_filter(np.diff(velocity, axis=0) * fps, 9, 2, axis=0)
            result[a + 1:b, slot, 2:4] = velocity
            result[a, slot, 2:4] = velocity[0]
            result[a + 1:b, slot, 4] = np.linalg.norm(velocity, axis=1)
            result[a, slot, 4] = result[a + 1, slot, 4]
            result[a + 1:b - 1, slot, 5] = np.linalg.norm(acceleration, axis=1)
            result[[a, b - 1], slot, 5] = 0.
    return result


def windows(features, teams, *, sample_step=5, window_steps=50, stride_steps=25):
    """Yield complete windows, ordered left team, right team, four boundaries.

    Player identities/jersey numbers are unnecessary. Team labels returned with
    each window map checkpoint source nodes back to H/A. Roster changes and
    incomplete tracking cause abstention rather than undocumented padding.
    """
    teams = np.asarray(teams)
    if features.ndim != 3 or features.shape[2] != 6 or len(teams) != features.shape[1] or not np.isin(teams, [0, 1]).all():
        raise ValueError('Need physical player features and aligned binary teams')
    small = features[::sample_step]
    for first in range(0, len(small) - window_steps + 1, stride_steps):
        part = small[first:first + window_steps]
        seen = np.isfinite(part).all(-1)
        # Any partially visible extra player indicates an unresolved roster
        # transition; don't silently choose eleven based on future outcomes.
        active = seen.any(0)
        groups = [np.flatnonzero(active & (teams == side)) for side in (0, 1)]
        if any(len(g) != 11 or not seen[:, g].all() for g in groups):
            continue
        keepers = [g[np.argmax(np.abs(part[:, g, 0].mean(0) - 52.5))] for g in groups]
        left = int(part[:, keepers[1], 0].mean() < part[:, keepers[0], 0].mean())
        order = np.r_[groups[left], groups[1 - left]]
        input = np.zeros((window_steps, 26, 8), np.float32)
        input[:, :11, 0] = -1.
        input[:, 11:22, 0] = 1.
        input[:, :22, 1] = np.isin(order, keepers)[None]
        input[:, :22, 2:] = part[:, order]
        input[:, 22:, 2:4] = np.array([[0., 34.], [105., 34.], [52.5, 0.], [52.5, 68.]])
        yield first, input, left


def source_teams(source_nodes, left_team):
    nodes = np.asarray(source_nodes)
    return np.where(nodes < 11, left_team, np.where(nodes < 22, 1 - left_team, -1))
