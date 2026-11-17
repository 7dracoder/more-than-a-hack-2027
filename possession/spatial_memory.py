"""Identity-free spatial memory and all-direction passing options.

Coordinates are observations, not gaze, peripheral vision or player intent.
No labels enter this API. Each call must contain exactly one continuous half.
Past/future anchors are permitted because this is offline possession recovery.
"""
import numpy as np


def nearest(distance, rank=0):
    if not distance.shape[1]:
        return np.full(len(distance), np.nan)
    ordered = np.sort(np.where(np.isfinite(distance), distance, np.inf), axis=1)
    value = ordered[:, min(rank, ordered.shape[1] - 1)]
    return np.where(np.isfinite(value), value, np.nan)


def shifted(values, offset):
    result = np.full_like(values, np.nan)
    if offset == 0:
        return values.copy()
    if abs(offset) >= len(values):
        return result
    if offset > 0:
        result[offset:] = values[:-offset]
    else:
        result[:offset] = values[-offset:]
    return result


def passing_options(players, teams, origin):
    """Aggregate candidate receivers in every direction; no assumed facing cone.

    Team sets, rather than slot identities, permit tracker fragmentation and
    identity changes. Lane distance considers opponents inside the pass segment;
    receiver clearance separately accounts for pressure at its endpoint.
    """
    result = {}
    for side, prefix in ((0, 'home'), (1, 'away')):
        receivers, opponents = players[:, teams == side], players[:, teams != side]
        delta = receivers - origin[:, None, :]
        distance = np.linalg.norm(delta, axis=-1)
        if not receivers.shape[1]:
            for name in ('receiver_distance', 'receiver_clearance', 'lane_clearance', 'options'):
                result[f'{prefix}_current_{name}'] = np.full(len(players), np.nan)
            result[f'{prefix}_current_receivers_known'] = np.zeros(len(players))
            continue
        length2 = np.sum(delta ** 2, axis=-1)
        if opponents.shape[1]:
            relative = opponents[:, None, :, :] - origin[:, None, None, :]
            projection = np.divide(np.sum(relative * delta[:, :, None, :], axis=-1),
                                   length2[:, :, None],
                                   out=np.full((len(players), receivers.shape[1], opponents.shape[1]), np.nan),
                                   where=length2[:, :, None] > 1e-6)
            lane = np.linalg.norm(relative - projection[..., None] * delta[:, :, None, :], axis=-1)
            lane = np.where((projection > .1) & (projection < .9) & np.isfinite(lane), lane, np.inf).min(-1)
            receiver_pressure = np.linalg.norm(receivers[:, :, None, :] - opponents[:, None, :, :], axis=-1)
            clearance = np.where(np.isfinite(receiver_pressure), receiver_pressure, np.inf).min(-1)
            # Infinite means there is no observed blocker, not a known empty pitch.
            known_opponents = np.isfinite(opponents).all(-1).sum(1) > 0
            lane = np.where(known_opponents[:, None], np.minimum(lane, 20.), np.nan)
            clearance = np.where(known_opponents[:, None], np.minimum(clearance, 20.), np.nan)
        else:
            lane = clearance = np.full(distance.shape, np.nan)
        # Exclude the carrier's own immediate location from receiver options.
        eligible = np.isfinite(distance) & (distance > 2.) & np.isfinite(origin).all(1)[:, None]
        candidates = np.where(eligible, distance, np.nan)
        closest = nearest(candidates)
        tied = eligible & (np.abs(distance - closest[:, None]) <= 1e-5)
        count = tied.sum(1)
        def pool(values):
            good = tied & np.isfinite(values)
            return np.divide(np.where(good, values, 0.).sum(1), good.sum(1),
                             out=np.full(len(players), np.nan), where=good.sum(1) > 0)
        result[f'{prefix}_current_receiver_distance'] = closest / 10.
        result[f'{prefix}_current_receiver_clearance'] = pool(clearance) / 10.
        result[f'{prefix}_current_lane_clearance'] = pool(lane) / 10.
        score = np.exp(-distance / 20.) * np.minimum(lane, clearance).clip(0., 10.) / 10.
        result[f'{prefix}_current_options'] = np.where(eligible & np.isfinite(score), score, 0.).sum(1) / 11.
        result[f'{prefix}_current_receivers_known'] = (count > 0).astype(float)
    return result


def build_features(players, teams, ball, pitch, fps=25.):
    players, ball = np.asarray(players, np.float32), np.asarray(ball, np.float32)
    teams = np.asarray(teams)
    n = len(ball)
    if (players.ndim != 3 or players.shape[-1] != 2 or ball.shape != (len(players), 2)
            or teams.shape != (players.shape[1],) or not np.isin(teams, [0, 1]).all()
            or n == 0 or not np.isfinite(fps) or fps <= 0):
        raise ValueError('Need one half of aligned coordinates and binary team membership')
    if len(pitch) != 2 or not np.isfinite(pitch).all() or min(pitch) <= 0:
        raise ValueError('Invalid pitch dimensions')
    tick = np.arange(n)
    seen = np.isfinite(ball).all(1)
    previous = np.maximum.accumulate(np.where(seen, tick, -1))
    following = np.minimum.accumulate(np.where(seen, tick, n)[::-1])[::-1]
    anchors = {}
    columns = {'ball_seen': seen.astype(float)}
    for direction, index, exists in (('past', previous, previous >= 0),
                                      ('future', following, following < n)):
        safe = np.clip(index, 0, n - 1)
        point = np.where(exists[:, None], ball[safe], np.nan)
        age = np.where(exists, np.abs(tick - index) / fps, 60.)
        anchors[direction] = point
        columns[f'{direction}_anchor_known'] = exists.astype(float)
        columns[f'{direction}_anchor_age'] = np.minimum(age, 60.) / 10.
        columns[f'{direction}_anchor_decay'] = np.where(exists, np.exp(-age / 4.), 0.)
        for side, prefix in ((0, 'home'), (1, 'away')):
            xy = players[:, teams == side]
            at_anchor = players[safe][:, teams == side]
            columns[f'{prefix}_{direction}_anchor_distance'] = nearest(np.linalg.norm(at_anchor - point[:, None], axis=-1)) / 10.
            columns[f'{prefix}_{direction}_distance_now'] = nearest(np.linalg.norm(xy - point[:, None], axis=-1)) / 10.
    displacement = np.linalg.norm(anchors['future'] - anchors['past'], axis=-1)
    age_sum = columns['past_anchor_age'] + columns['future_anchor_age']
    columns['anchor_displacement'] = displacement / 10.
    columns['anchor_implied_speed'] = np.divide(displacement, age_sum * 10.,
                                               out=np.zeros(n), where=age_sum > 0) / 20.
    columns['anchor_time_fraction'] = np.divide(columns['past_anchor_age'], age_sum,
                                               out=np.zeros(n), where=age_sum > 0)
    # Do not fill unknown coordinates into the old model. This remembered point
    # is only a reference for features whose age/availability are explicit.
    origin = np.where(seen[:, None], ball, anchors['past'])
    columns.update(passing_options(players, teams, origin))
    for side, prefix in ((0, 'home'), (1, 'away')):
        xy = players[:, teams == side]
        known = np.isfinite(xy).all(-1)
        count = known.sum(1)
        center = np.divide(np.where(known[..., None], xy, 0.).sum(1), count[:, None],
                           out=np.full((n, 2), np.nan), where=count[:, None] > 0)
        columns[f'{prefix}_current_visible'] = count / 11.
        columns[f'{prefix}_current_ball_distance'] = nearest(np.linalg.norm(xy - ball[:, None], axis=-1)) / 10.
        for seconds in (1, 2, 4, 8):
            lag = int(round(seconds * fps))
            historical = shifted(center, lag)
            columns[f'{prefix}_memory{seconds}_centroid_displacement'] = np.linalg.norm(center - historical, axis=-1) / 10.
            old_players = shifted(xy, lag)
            old_distance = np.linalg.norm(old_players - origin[:, None], axis=-1)
            current_distance = np.linalg.norm(xy - origin[:, None], axis=-1)
            columns[f'{prefix}_memory{seconds}_nearest_to_origin'] = nearest(old_distance) / 10.
            columns[f'{prefix}_memory{seconds}_approach_to_origin'] = (nearest(old_distance) - nearest(current_distance)) / (10. * seconds)
            change = ((np.isfinite(current_distance) & (current_distance < 10.)).sum(1) -
                      (np.isfinite(old_distance) & (old_distance < 10.)).sum(1)) / 11.
            historical_known = (tick >= lag) & np.isfinite(origin).all(1) & np.isfinite(old_players).all(-1).any(1)
            columns[f'{prefix}_memory{seconds}_near_origin_change'] = np.where(historical_known, change, np.nan)
    # Explicit missing flags keep absent tracking distinct from real zeros.
    names = list(columns)
    matrix = np.column_stack([columns[name] for name in names]).astype(np.float32)
    known = np.isfinite(matrix)
    matrix = np.where(known, matrix, 0.).clip(-20., 20.)
    result = np.concatenate([matrix, known.astype(np.float32)], axis=1)
    return result, names + [name + '_known' for name in names]


def symmetric_features(matrix, names):
    mapping = [names.index('away_' + name[5:] if name.startswith('home_') else
                           'home_' + name[5:] if name.startswith('away_') else name)
               for name in names]
    return matrix[:, mapping]
