"""v0.5 structural rules: out-of-play stoppage prior and contact-aligned team boundaries.

Both rules come from the Laws of the Game and the tracking geometry, not from fitting labels.
Their constants are fixed here, before scoring, and are not tuned:

* Out-of-play hold (IFAB Law 9: the ball is out of play when it has wholly crossed a boundary
  line). A frame where the observed ball is more than OUT_MARGIN_M beyond a line is dead. The
  stoppage then continues, as soft evidence, until the ball is next observed inside the pitch
  (by OUT_MARGIN_M) and kicked (speed above KICK_SPEED_MPS), capped at HOLD_CAP_FRAMES.
  Missing ball never starts a hold. The margin absorbs tracking error at the line.
* Contact snapping. A decoded H<->A change is moved to the nearest frame, within SNAP_FRAMES,
  where a player of the new team is within 1 m of the ball and nearer than any opponent.
  Changes are never moved across a dead-ball frame, and nothing is created or deleted.
"""

import numpy as np

from games import GAME, GAMES, half_video_blocks

PITCH = np.asarray(GAMES[GAME]['pitch'], dtype=float)
OUT_MARGIN_M = 1.0
KICK_SPEED_MPS = 5.0
HOLD_CAP_FRAMES = 1500        # 60 s
HOLD_LOG_ODDS = 2.0           # soft dead evidence added during the hold, per frame
HARD_DEAD_PROBABILITY = .999
SNAP_FRAMES = 25              # 1 s


def out_of_play_hold(ball, fps=25):
    """(hard, soft) boolean masks over all frames; computed independently per half."""
    ball = np.asarray(ball, dtype=float)[:, :2]
    n = len(ball)
    hard = np.zeros(n, dtype=bool)
    soft = np.zeros(n, dtype=bool)
    for _, start, last in half_video_blocks(GAME):
        b = ball[start:last + 1] * PITCH
        seen = np.isfinite(b).all(1)
        beyond = np.max(np.stack([-b[:, 0], b[:, 0] - PITCH[0], -b[:, 1], b[:, 1] - PITCH[1]]), axis=0)
        out = seen & (np.nan_to_num(beyond, nan=-99.) > OUT_MARGIN_M)
        inside = seen & (np.nan_to_num(beyond, nan=99.) < -OUT_MARGIN_M)
        step = np.r_[np.nan, np.linalg.norm(np.diff(b, axis=0), axis=1) * fps]
        kicked = inside & np.r_[False, seen[:-1]] & (np.nan_to_num(step) > KICK_SPEED_MPS) & (step < 60)
        h = np.zeros(len(b), dtype=bool)
        s = np.zeros(len(b), dtype=bool)
        i = 0
        while i < len(b):
            if not out[i]:
                i += 1
                continue
            j = i
            while j < len(b) and j - i < HOLD_CAP_FRAMES and not kicked[j]:
                j += 1
            s[i:j] = True
            i = max(j, i + 1)
        h[out] = True
        hard[start:last + 1], soft[start:last + 1] = h, s & ~h
    return hard, soft


def apply_dead_prior(p_dead, hard, soft):
    p = np.asarray(p_dead, dtype=float).clip(1e-6, 1 - 1e-6).copy()
    logit = np.log(p / (1 - p))
    logit[soft] += HOLD_LOG_ODDS
    p = 1 / (1 + np.exp(-logit))
    p[hard] = np.maximum(p[hard], HARD_DEAD_PROBABILITY)
    return p


def contact_series(events, names):
    """+1 home contact, -1 away contact, 0 none, from the v0.4 1 m contact features."""
    last = events[:, names.index('last_contact_1m')]
    since = events[:, names.index('since_contact_1m')]
    return np.where(since == 0, np.sign(last), 0.).astype(np.int8)


def snap_transitions(labels, contacts, window=SNAP_FRAMES):
    """Move each H<->A change to the nearest new-team contact within the window."""
    labels = np.asarray(labels).copy()
    original = labels.copy()
    moved = 0
    changes = np.flatnonzero((original[1:] != original[:-1])) + 1
    positions = changes.copy()
    for index, t in enumerate(changes):
        old, new = original[t - 1], original[t]
        if {old, new} != {'H', 'A'}:
            continue
        sign = 1 if new == 'H' else -1
        lo, hi = max(0, t - window), min(len(labels), t + window + 1)
        # Stay inside the live run that contains this change.
        live = np.isin(original[lo:hi], ['H', 'A'])
        left = t - lo - 1
        while left >= 0 and live[left]:
            left -= 1
        right = t - lo
        while right < len(live) and live[right]:
            right += 1
        candidates = np.flatnonzero(contacts[lo:hi] == sign) + lo
        candidates = candidates[(candidates > lo + left + 1) & (candidates < lo + right)]
        if not len(candidates):
            continue
        target = int(candidates[np.argmin(np.abs(candidates - t))])
        # Do not swallow another change: only move within this change's two neighbouring spells.
        # Earlier boundaries may already have moved. Comparing only with their
        # original positions lets neighbouring moves cross and erase a spell.
        previous = positions[index - 1] if index else 0
        following = positions[index + 1] if index + 1 < len(positions) else len(labels)
        if target <= previous or target >= following:
            continue
        if target < t:
            labels[target:t] = new
        elif target > t:
            labels[t:target] = old
        moved += target != t
        positions[index] = target
    # This is boundary relocation only: enforce its contract on real tracks too.
    before_runs = original[np.r_[True, original[1:] != original[:-1]]]
    after_runs = labels[np.r_[True, labels[1:] != labels[:-1]]]
    if (not np.array_equal(before_runs, after_runs)
            or not np.array_equal(original == 'D', labels == 'D')
            or not np.array_equal(original == 'X', labels == 'X')):
        raise RuntimeError('Contact snapping changed ordered spells or D/X masks')
    return labels, moved
