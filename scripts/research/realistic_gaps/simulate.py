"""Degrade clean external tracking to the BAR-ATH degradation profile (label-free).

Ball: gaps start with the per-context loss hazard measured on BAR-ATH (crowding, ball speed,
distance to touchline, collective player speed), last for durations resampled from BAR-ATH's own
gap lengths, and a single multiplier is solved so the missing fraction matches BAR-ATH.
Players: each track alternates BAR-ATH-length visible fragments with short absences so the
visible players per team match BAR-ATH. Labels are never read here.
"""
import numpy as np

from degradation_profile import bin_index, context

BACKOFF = 500.0


def load_profile(profile):
    halves = list(profile.values())
    joint, n, lost = {}, 0, 0
    for h in halves:
        for k, (m, l) in h["loss_hazard_joint_bins"].items():
            a, b = joint.get(int(k), (0, 0))
            joint[int(k)] = (a + m, b + l)
            n, lost = n + m, lost + l
    base = lost / n
    rates = {k: (l + BACKOFF * base) / (m + BACKOFF) for k, (m, l) in joint.items()}
    return dict(rates=rates, base=base,
                gaps=np.concatenate([h["ball_gap_frames"] for h in halves]),
                fragments=np.concatenate([h["player_fragment_frames"] for h in halves]),
                missing=float(np.mean([h["ball_missing_fraction"] for h in halves])),
                per_team=float(np.mean([np.mean(list(h["objects_per_frame"].values())) for h in halves])))


def _ball_mask(hazard, gaps, mult, seed):
    rng = np.random.default_rng(seed)
    u = rng.random(len(hazard))
    lengths = rng.choice(gaps, size=len(hazard))
    missing = np.zeros(len(hazard), bool)
    t = 0
    while t < len(hazard):
        if u[t] < hazard[t] * mult:
            missing[t + 1:t + 1 + lengths[t]] = True
            t += 1 + lengths[t]
        else:
            t += 1
    return missing


def ball_mask(players, ball, pitch, prof, seed, target=None, max_gap=None):
    target = prof["missing"] if target is None else target
    gaps = prof["gaps"] if max_gap is None else prof["gaps"][prof["gaps"] <= max_gap]
    b = bin_index(context(players, ball, np.asarray(pitch, float)))
    hazard = np.array([prof["rates"].get(int(k), prof["base"]) for k in b])
    hazard[~np.isfinite(ball).all(1)] = 0.0
    lo, hi = 0.0, 50.0
    for _ in range(14):
        mid = (lo + hi) / 2
        if _ball_mask(hazard, gaps, mid, seed).mean() < target:
            lo = mid
        else:
            hi = mid
    return _ball_mask(hazard, gaps, (lo + hi) / 2, seed)


def player_mask(players, teams, prof, seed):
    rng = np.random.default_rng(seed + 1)
    present = np.isfinite(players).all(-1)
    on_pitch = present.sum(1)
    per_team = np.median(on_pitch[on_pitch > 0]) / 2
    absent = max(0.0, 1 - prof["per_team"] / per_team)
    mean_absent = absent / max(1 - absent, 1e-9) * prof["fragments"].mean()
    drop = np.zeros_like(present)
    for s in range(players.shape[1]):
        idx = np.flatnonzero(present[:, s])
        if not len(idx):
            continue
        t = idx[0] + int(rng.choice(prof["fragments"]))
        while t < idx[-1]:
            gap = max(1, int(rng.exponential(mean_absent)))
            drop[t:t + gap, s] = True
            t += gap + int(rng.choice(prof["fragments"]))
    return drop & present


def frozen_runs(ball, min_frames=5):
    """Stretches where the ball coordinate is bit-identical frame to frame (DFL parks it at stoppages)."""
    same = np.r_[False, (np.diff(ball, axis=0) == 0).all(1)]
    edges = np.diff(np.r_[0, same.astype(np.int8), 0])
    starts, stops = np.flatnonzero(edges == 1) - 1, np.flatnonzero(edges == -1)
    keep = stops - starts >= min_frames
    return list(zip(np.maximum(starts[keep], 0), stops[keep]))


def unfreeze(players, ball, mode, seed):
    """Remove the parked-ball cue BAR-ATH lacks: hide the ball, or let it follow the nearest player."""
    rng = np.random.default_rng(seed + 2)
    ball = ball.copy()
    for a, z in frozen_runs(ball):
        if mode == "hide" or (mode == "mixed" and rng.random() < 0.5):
            ball[a:z] = np.nan
            continue
        d = np.linalg.norm(players[a:z] - ball[a:z, None, :], axis=-1)
        d = np.where(np.isfinite(d), d, np.inf)
        carrier = d[0].argmin()
        path = players[a:z, carrier].copy()
        ok = np.isfinite(path).all(1)
        ball[a:z] = np.where(ok[:, None], path + rng.normal(0, 0.3, path.shape), np.nan)
    return ball


def stoppage_mask(labels, q, seed, jitter=38, min_fraction=1.0):
    """Hide the ball over a fraction q of labelled DFL stoppages, from near the whistle for a random
    U(min_fraction, 1) share of each stoppage; both endpoints jittered by up to 1.5 s."""
    rng = np.random.default_rng(seed + 3)
    edges = np.diff(np.r_[0, (labels == "D").astype(np.int8), 0])
    mask = np.zeros(len(labels), bool)
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if rng.random() < q:
            length = round(rng.uniform(min_fraction, 1.0) * (b - a))
            a = max(0, a + rng.integers(-jitter, jitter + 1))
            mask[a:max(a, a + length + rng.integers(-jitter, jitter + 1))] = True
    return mask


def degrade(players, teams, ball, pitch, prof, seed, unfreeze_mode=None, labels=None, stoppage_q=0.0,
            stoppage_fraction=1.0, max_alive_gap=None):
    """stoppage_q > 0 ties ball loss to DFL stoppages (external labels); the rest is filled by play-time
    gaps, optionally capped at max_alive_gap frames."""
    players, ball = players.copy(), ball.copy()
    if unfreeze_mode:
        ball = unfreeze(players, ball, unfreeze_mode, seed)
    target, max_gap = None, None
    if stoppage_q:
        ball[stoppage_mask(labels, stoppage_q, seed, min_fraction=stoppage_fraction)[:len(ball)]] = np.nan
        target = max(0.0, prof["missing"] - (~np.isfinite(ball).all(1)).mean())
        max_gap = max_alive_gap
    ball[ball_mask(players, ball, pitch, prof, seed, target, max_gap)] = np.nan
    players[player_mask(players, teams, prof, seed)] = np.nan
    return players, ball
