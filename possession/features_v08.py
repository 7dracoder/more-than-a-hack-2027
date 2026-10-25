"""Coordinate-only, two-sided contact and relative-motion evidence.

Offline inference can see the complete half. Contact events are geometric
observations, not possession labels; the feature API accepts no supervision.
"""
import warnings
import numpy as np
from possession.external import build_features as base_features, finite_mean


def velocity(xy, fps=25., span=4, max_speed=60.):
    xy = np.asarray(xy)
    result = np.full_like(xy, np.nan)
    if len(xy) <= span:
        return result
    valid = np.isfinite(xy).all(-1)
    bad = np.concatenate([np.zeros_like(valid[:1], dtype=int), np.cumsum(~valid, axis=0)])
    continuous = bad[span+1:] == bad[:-span-1]
    value = (xy[span:] - xy[:-span]) * fps / span
    continuous &= np.linalg.norm(value, axis=-1) < max_speed
    result[span//2:-span//2] = np.where(continuous[..., None], value, np.nan)
    return result


def contact_context(distance, radius, fps=25.):
    """A contact requires a nearby player and an unambiguous distance margin."""
    n = len(distance)
    tick = np.arange(n)
    known = np.isfinite(distance).all(1)
    difference = np.zeros(n)
    np.subtract(distance[:, 1], distance[:, 0], out=difference, where=known)
    reliable = known & (distance.min(1) < radius) & (np.abs(difference) > .25)
    owner = (difference < 0).astype(np.int8)
    previous = np.maximum.accumulate(np.where(reliable, tick, -1))
    following = np.minimum.accumulate(np.where(reliable, tick, n)[::-1])[::-1]
    result = {}
    for direction, indices, exists in [('past', previous, previous >= 0), ('future', following, following < n)]:
        safe = np.clip(indices, 0, n-1)
        age = np.where(exists, np.minimum(np.abs(tick-indices)/fps, 30.), 30.)
        result[f'{direction}_age'] = age / 30.
        for side, prefix in [(0, 'home'), (1, 'away')]:
            match = exists & (owner[safe] == side)
            result[f'{prefix}_{direction}_owner'] = match.astype(float)
            result[f'{prefix}_{direction}_recent'] = match * np.exp(-age/2.)
    return result


def build_features(players, teams, ball, pitch, fps=25., step=2):
    base, names = base_features(players, teams, ball, pitch, fps=fps, step=step)
    players, ball, teams = np.asarray(players), np.asarray(ball), np.asarray(teams)
    n = len(ball)
    columns, distances = {}, []
    ball_v = velocity(ball, fps)
    player_v = velocity(players, fps, max_speed=15.)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for side, prefix in [(0, 'home'), (1, 'away')]:
            xy, pv = players[:, teams == side], player_v[:, teams == side]
            delta = xy-ball[:, None, :]
            d = np.linalg.norm(delta, axis=-1)
            nearest = np.where(np.isfinite(d), d, np.inf).argmin(1)
            close = d[np.arange(n), nearest]
            distances.append(np.where(np.isfinite(close), close, np.inf))
            relative = pv[np.arange(n), nearest]-ball_v
            direction = delta[np.arange(n), nearest] / np.maximum(close[:, None], .1)
            columns[f'{prefix}_relative_speed'] = np.linalg.norm(relative, axis=-1).clip(0, 60)/20.
            columns[f'{prefix}_closing_speed'] = np.sum(relative*direction, axis=-1).clip(-40, 40)/20.
            columns[f'{prefix}_ball_toward'] = np.sum(ball_v*direction, axis=-1).clip(-40, 40)/20.
            columns[f'{prefix}_nearest_speed'] = np.linalg.norm(pv[np.arange(n), nearest], axis=-1)/10.
            for radius in (1., 2., 3., 5.):
                columns[f'{prefix}_proximity{radius:g}'] = np.exp(-close/radius)
    distances = np.stack(distances, axis=1)
    for radius in (1., 2., 3.):
        for key, value in contact_context(distances, radius, fps).items():
            columns[f'{key}_r{radius:g}'] = value
    # Short smoothing windows preserve evidence for sub-second state runs.
    original = list(columns.items())
    for key, value in original:
        if 'age_' in key or '_owner_' in key:
            continue
        for radius in (5, 15):
            columns[f'{key}_mean{radius}'] = finite_mean(value, radius)
    extra_names = list(columns)
    extra = np.stack([columns[key][::step] for key in extra_names], axis=1)
    known = np.isfinite(extra)
    extra = np.concatenate([np.where(known, extra, 0.), known.astype(float)], axis=1)
    return np.concatenate([base, extra.astype(np.float32)], axis=1), names+extra_names+[key+'_known' for key in extra_names]
