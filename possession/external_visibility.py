"""Camera-loss modelling fitted exclusively to external tracking observations.

No possession, event, match-identity or supplied-match statistics enter the
feature/simulation APIs. Coordinate conventions are metres from a pitch corner.
"""
import warnings

import numpy as np


BINS = dict(crowd=[1., 2., 4.], speed=[2., 8., 15.],
            edge=[0., 3., 10.], player_speed=[1., 2.])


def runs(mask):
    edges = np.diff(np.r_[0, np.asarray(mask, np.int8), 0])
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def context_bins(players, ball, pitch, fps):
    distance = np.linalg.norm(players - ball[:, None], axis=-1)
    crowd = (distance < 3.).sum(1).astype(float)
    speed = np.full(len(ball), np.nan)
    speed[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1) * fps
    edge = np.minimum.reduce([ball[:, 0], pitch[0] - ball[:, 0], ball[:, 1], pitch[1] - ball[:, 1]])
    span = max(1, round(.2 * fps))
    movement = np.full(players.shape[:2], np.nan)
    movement[span:] = np.linalg.norm(players[span:] - players[:-span], axis=-1) * fps / span
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        player_speed = np.nanmedian(np.where(movement < 12., movement, np.nan), axis=1)
    index = np.zeros(len(ball), np.int64)
    for values, cuts in zip((crowd, speed, edge, player_speed), BINS.values()):
        b = np.digitize(np.nan_to_num(values), cuts)
        b = np.where(np.isfinite(values), b, len(cuts) + 1)
        index = index * (len(cuts) + 2) + b
    return index


def summarize(native_players, native_ball, detected_players, detected_ball, teams, pitch, fps=10.):
    n = len(native_ball)
    if (native_players.shape != detected_players.shape or native_ball.shape != detected_ball.shape or
            native_players.shape[0] != n or native_ball.shape != (n, 2) or
            native_players.ndim != 3 or native_players.shape[-1] != 2 or fps <= 0):
        raise ValueError('Need aligned native/detected external coordinates')
    seen = np.isfinite(detected_ball).all(1)
    bins = context_bins(native_players, native_ball, pitch, fps)
    lost = seen[:-1] & ~seen[1:]
    hazards = {}
    for key in np.unique(bins[:-1][seen[:-1]]):
        at = seen[:-1] & (bins[:-1] == key)
        hazards[str(int(key))] = [int(at.sum()), int((at & lost).sum())]
    first, stop = runs(~seen)
    # End-censored absences do not reveal a complete natural gap duration.
    bracketed = (first > 0) & (stop < n)
    gaps = ((stop - first)[bracketed] / fps).tolist()
    present = np.isfinite(detected_players).all(-1)
    fragments = []
    for slot in range(present.shape[1]):
        a, b = runs(present[:, slot])
        fragments.extend(((b - a) / fps).tolist())
    return dict(frames=n, fps=fps, seen=int(seen.sum()), lost=int(lost.sum()),
                hazards=hazards, gap_seconds=gaps, fragment_seconds=fragments,
                player_frames_per_team=[int(present[:, teams == side].sum()) for side in (0, 1)])


def fit_profile(summaries, training_matches):
    if not summaries:
        raise ValueError('Need external training observations')
    frames, missing, visible_seconds, losses = 0, 0, 0., 0
    hazards, gaps, fragments = {}, [], []
    player_counts = np.zeros(2)
    for item in summaries:
        frames += item['frames']; missing += item['frames'] - item['seen']
        visible_seconds += item['seen'] / item['fps']; losses += item['lost']
        player_counts += item['player_frames_per_team']
        gaps.extend(item['gap_seconds']); fragments.extend(item['fragment_seconds'])
        for key, (count, lost) in item['hazards'].items():
            seconds, events = hazards.get(key, (0., 0))
            hazards[key] = (seconds + count / item['fps'], events + lost)
    if frames <= 0 or visible_seconds <= 0 or not gaps or not fragments:
        raise ValueError('External observations cannot identify a visibility model')
    base = losses / visible_seconds
    # Fifty seconds of global-rate shrinkage, fixed before application.
    rates = {key:(lost + 50. * base) / (seconds + 50.) for key, (seconds, lost) in hazards.items()}
    return dict(kind='External-only natural camera-loss model', training_matches=training_matches,
                bins=BINS, rate_units='loss events per visible second', base_rate=base, rates=rates,
                gap_seconds=gaps, fragment_seconds=fragments,
                missing_fraction=missing / frames, visible_players_per_team=(player_counts / frames).tolist(),
                observed_frames=frames, natural_gap_count=len(gaps),
                possession_labels_used=False, provided_match_used=False)


def sample_mask(hazard, lengths, multiplier, seed):
    rng = np.random.default_rng(seed)
    uniform = rng.random(len(hazard))
    duration = rng.choice(lengths, len(hazard))
    missing = np.zeros(len(hazard), bool)
    i = 0
    while i < len(hazard):
        if uniform[i] < min(1., hazard[i] * multiplier):
            stop = min(len(hazard), i + 1 + int(duration[i]))
            missing[i + 1:stop] = True
            i = stop
        else:
            i += 1
    return missing


def remove_stationary_cue(players, ball, mode, seed):
    """External augmentation: hide/follow a player over bit-identical ball runs.

    This removes the source-specific parked-ball cue without consulting labels.
    It is not a correction or transformation applied to the provided match.
    """
    if mode not in ('none', 'hide', 'carry', 'mixed'):
        raise ValueError('Unknown external stationary-cue augmentation')
    result = ball.copy()
    if mode == 'none':
        return result
    same = np.r_[False, np.isfinite(ball[1:]).all(1) & np.isfinite(ball[:-1]).all(1) & (np.diff(ball, axis=0) == 0).all(1)]
    start, stop = runs(same)
    rng = np.random.default_rng(seed)
    for a, b in zip(start - 1, stop):
        if b - a < 5:
            continue
        if mode == 'hide' or (mode == 'mixed' and rng.random() < .5):
            result[a:b] = np.nan
            continue
        distance = np.linalg.norm(players[a] - ball[a], axis=1)
        distance = np.where(np.isfinite(distance), distance, np.inf)
        if not np.isfinite(distance).any():
            result[a:b] = np.nan
        else:
            slot = int(distance.argmin())
            path = players[a:b, slot]
            result[a:b] = path + rng.normal(0., .3, path.shape).astype(np.float32)
    return result


def degrade(players, teams, ball, pitch, profile, seed, mode='mixed', fps=25.):
    """Generate an external view; profile provenance must be external only."""
    if profile['provided_match_used'] or profile['possession_labels_used']:
        raise ValueError('Visibility fitting must use external observations only')
    if not 0 <= profile['missing_fraction'] < 1 or fps <= 0:
        raise ValueError('Invalid visibility model')
    result_players = players.copy()
    result_ball = remove_stationary_cue(players, ball, mode, seed + 31)
    index = context_bins(players, ball, pitch, fps)
    rates = np.array([profile['rates'].get(str(int(k)), profile['base_rate']) for k in index])
    hazard = -np.expm1(-rates / fps)
    prior = ~np.isfinite(result_ball).all(1)
    hazard[prior] = 0.
    lengths = np.maximum(1, np.rint(np.asarray(profile['gap_seconds']) * fps).astype(int))
    target = profile['missing_fraction']
    low, high = 0., 100.
    for _ in range(14):
        middle = (low + high) / 2
        mask = prior | sample_mask(hazard, lengths, middle, seed)
        if mask.mean() < target:
            low = middle
        else:
            high = middle
    result_ball[prior | sample_mask(hazard, lengths, (low + high) / 2, seed)] = np.nan
    rng = np.random.default_rng(seed + 73)
    fragments = np.maximum(1, np.rint(np.asarray(profile['fragment_seconds']) * fps).astype(int))
    present = np.isfinite(players).all(-1)
    for side in (0, 1):
        slots = np.flatnonzero(teams == side)
        count = present[:, slots].sum(1)
        available = float(np.median(count[count > 0])) if (count > 0).any() else 0.
        if available <= 0:
            continue
        absent = np.clip(1. - profile['visible_players_per_team'][side] / available, 0., .95)
        mean_absent = absent / max(1. - absent, 1e-6) * float(np.mean(fragments))
        if absent == 0:
            continue
        for slot in slots:
            known = np.flatnonzero(present[:, slot])
            if not len(known):
                continue
            i = int(known[0] + rng.choice(fragments))
            while i < known[-1]:
                length = max(1, int(rng.exponential(mean_absent)))
                result_players[i:i + length, slot] = np.nan
                i += length + int(rng.choice(fragments))
    return result_players, result_ball
