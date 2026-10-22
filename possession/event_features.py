"""v0.4 contact, ball-gap, out-of-play and team-shape features.

All features are computed from the automatic tactical tracking only (never from labels) and
independently within each half. Every feature is declared with a team-swap rule so the classifier
can keep the exact home/away symmetry used since v0.3:

    'sym'  unchanged when the two teams are exchanged
    'anti' negated when the two teams are exchanged
    ('pair', other)  exchanged with another column

Design notes (see TECHNICAL_REPORT / CoreOrigin.md):
* The public ground truth changes owner around ball contacts, not around a fixed distance rule;
  the automatic positions are often 1-3 m off. Contacts are therefore summarised as the team of
  the most recent and next close approach, with time since/until, rather than a hard rule.
* Missing ball is never converted to a position. Gap-length and last/next observed states are
  explicit features, so an occlusion inside a possession can be bridged by its two ends.
* Possession also changes team shape: the team in possession spreads out and moves forward.
  Those collective features exist even when the ball is missing. Attack direction is estimated
  per half from where each team's players stand on average (no labels, no event data).
"""

import warnings

import numpy as np

from games import GAME, GAMES, half_video_blocks

PITCH = np.asarray(GAMES[GAME]['pitch'], dtype=float)
FEATURE_VERSION = 4


def _ffill_index(mask):
    """Index of the most recent True at or before each position (-1 if none)."""
    idx = np.where(mask, np.arange(len(mask)), -1)
    return np.maximum.accumulate(idx)


def _bfill_index(mask):
    """Index of the next True at or after each position (len if none)."""
    n = len(mask)
    idx = np.where(mask, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1]


def _since_until(mask, cap):
    n = len(mask)
    i = np.arange(n)
    last = _ffill_index(mask)
    nxt = _bfill_index(mask)
    since = np.where(last >= 0, i - last, cap).clip(0, cap)
    until = np.where(nxt < n, nxt - i, cap).clip(0, cap)
    return last, nxt, since.astype(float), until.astype(float)


def _window(values, width, side):
    """Finite-only mean over the past (t-width..t-1), future (t+1..t+width) or centre."""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    s = np.r_[0., np.cumsum(np.where(ok, v, 0.))]
    c = np.r_[0, np.cumsum(ok)]
    i = np.arange(len(v))
    if side == 'past':
        lo, hi = np.maximum(0, i - width), i
    elif side == 'future':
        lo, hi = np.minimum(len(v), i + 1), np.minimum(len(v), i + 1 + width)
    else:
        lo, hi = np.maximum(0, i - width), np.minimum(len(v), i + width + 1)
    cnt = c[hi] - c[lo]
    return np.divide(s[hi] - s[lo], cnt, out=np.zeros(len(v)), where=cnt > 0)


def half_features(players, ball, teams, fps=25):
    """Features for one half. players (n, slots, >=2) normalised, ball (n, >=2) normalised."""
    n = len(ball)
    xy = players[:, :, :2].astype(float) * PITCH
    b = ball[:, :2].astype(float) * PITCH
    seen = np.isfinite(b).all(1)
    home, away = teams == 0, teams == 1
    pvis = np.isfinite(xy).all(-1)
    f = {}

    # ---- player/ball proximity --------------------------------------------------------------
    d = np.linalg.norm(xy - b[:, None, :], axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    dh = np.sort(d[:, home], axis=1)
    da = np.sort(d[:, away], axis=1)
    h1, h2, a1, a2 = dh[:, 0], dh[:, 1], da[:, 0], da[:, 1]
    cap = 30.
    H1, A1 = np.minimum(h1, cap), np.minimum(a1, cap)
    within = lambda dist, r: (dist < r).sum(1)
    f['near_home2'] = (np.minimum(h2, cap), ('pair', 'near_away2'))
    f['near_away2'] = (np.minimum(a2, cap), ('pair', 'near_home2'))
    f['press_diff_3m'] = (np.where(seen, within(d[:, home], 3.) - within(d[:, away], 3.), 0.), 'anti')
    f['press_total_5m'] = (np.where(seen, within(d, 5.), 0.), 'sym')

    contact_all = {}
    for r in (1.0, 2.0):
        c = np.where(seen & (H1 < r) & (H1 < A1), 1., np.where(seen & (A1 < r) & (A1 < H1), -1., 0.))
        contact_all[r] = c
        last, nxt, since, until = _since_until(c != 0, 500)
        lt = np.where(last >= 0, c[np.maximum(last, 0)], 0.)
        nt = np.where(nxt < n, c[np.minimum(nxt, n - 1)], 0.)
        tag = f'{r:.0f}m'
        f[f'last_contact_{tag}'] = (lt, 'anti')
        f[f'next_contact_{tag}'] = (nt, 'anti')
        f[f'since_contact_{tag}'] = (np.log1p(since), 'sym')
        f[f'until_contact_{tag}'] = (np.log1p(until), 'sym')
        f[f'bracket_{tag}'] = (lt * np.exp(-since / 50.) + nt * np.exp(-until / 50.), 'anti')
        for w in (25, 75):
            f[f'contact_past_{tag}_{w}'] = (_window(np.where(seen, c, np.nan), w, 'past'), 'anti')
            f[f'contact_future_{tag}_{w}'] = (_window(np.where(seen, c, np.nan), w, 'future'), 'anti')
    margin = np.where(seen, (A1 - H1) / (A1 + H1 + .5), np.nan)
    for w in (12, 50):
        f[f'margin_past_{w}'] = (_window(margin, w, 'past'), 'anti')
        f[f'margin_future_{w}'] = (_window(margin, w, 'future'), 'anti')

    # ---- ball gaps --------------------------------------------------------------------------
    last_seen, next_seen, since_seen, until_seen = _since_until(seen, 750)
    gap = since_seen + until_seen
    f['gap_len'] = (np.where(seen, 0., np.log1p(gap)), 'sym')
    f['since_seen'] = (np.log1p(since_seen), 'sym')
    f['until_seen'] = (np.log1p(until_seen), 'sym')

    # ---- out of play geometry ---------------------------------------------------------------
    out = np.max(np.stack([-b[:, 0], b[:, 0] - PITCH[0], -b[:, 1], b[:, 1] - PITCH[1]]), axis=0)
    out = np.where(seen, out, np.nan)
    edge = np.where(seen, -out, np.nan)  # distance inside the nearest line (negative outside)
    lo = np.where(last_seen >= 0, out[np.maximum(last_seen, 0)], np.nan)
    no = np.where(next_seen < n, out[np.minimum(next_seen, n - 1)], np.nan)
    f['out_now'] = (np.nan_to_num(np.clip(out, -10, 5), nan=-10.), 'sym')
    f['out_last_seen'] = (np.nan_to_num(np.clip(lo, -10, 5), nan=-10.), 'sym')
    f['out_next_seen'] = (np.nan_to_num(np.clip(no, -10, 5), nan=-10.), 'sym')
    outside = np.nan_to_num(out, nan=-1.) > .3
    _, _, since_out, until_out = _since_until(outside, 1500)
    f['since_out'] = (np.log1p(since_out), 'sym')
    f['until_out'] = (np.log1p(until_out), 'sym')
    f['edge_dist'] = (np.nan_to_num(np.clip(edge, -5, 34), nan=-5.), 'sym')

    # ---- ball motion: placed ball and restarts ----------------------------------------------
    step = np.linalg.norm(np.diff(b, axis=0, prepend=b[:1]), axis=1) * fps
    contiguous = seen & np.r_[False, seen[:-1]]
    speed = np.where(contiguous & (step < 60), step, np.nan)
    still = np.nan_to_num(speed, nan=99.) < .7
    moving = np.nan_to_num(speed, nan=0.) > 4.
    run_start = _ffill_index(~still)
    still_run = np.where(still, np.arange(n) - run_start, 0).astype(float)
    f['ball_still_run'] = (np.log1p(still_run), 'sym')
    _, _, since_moving, until_moving = _since_until(moving, 750)
    f['since_ball_moving'] = (np.log1p(since_moving), 'sym')
    f['until_ball_moving'] = (np.log1p(until_moving), 'sym')
    f['ball_speed_past_25'] = (_window(speed, 25, 'past'), 'sym')
    f['ball_speed_future_25'] = (_window(speed, 25, 'future'), 'sym')

    # ---- team shape and attack direction ----------------------------------------------------
    def team_stats(mask):
        p = np.where(pvis[:, mask, None], xy[:, mask], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            cx = np.nanmean(p[:, :, 0], 1)
            cy = np.nanmean(p[:, :, 1], 1)
            sx = np.nanstd(p[:, :, 0], 1)
            sy = np.nanstd(p[:, :, 1], 1)
        return cx, cy, sx, sy
    hcx, hcy, hsx, hsy = team_stats(home)
    acx, acy, asx, asy = team_stats(away)
    # Attack direction: a team stands, on average, nearer the goal it defends.
    direction_h = 1. if np.nanmean(hcx) < np.nanmean(acx) else -1.
    direction_a = -direction_h
    centre = PITCH[0] / 2
    adv_h = direction_h * (hcx - centre)
    adv_a = direction_a * (acx - centre)
    ball_adv = np.where(seen, direction_h * (b[:, 0] - centre), np.nan)
    for w in (25, 75):
        f[f'adv_diff_{w}'] = (_window(adv_h - adv_a, w, 'centre'), 'anti')
        f[f'adv_sum_{w}'] = (_window(adv_h + adv_a, w, 'centre'), 'sym')
        f[f'spread_diff_{w}'] = (_window(hsx * hsy - asx * asy, w, 'centre') / 100., 'anti')
        f[f'spread_sum_{w}'] = (_window(hsx * hsy + asx * asy, w, 'centre') / 100., 'sym')
        f[f'width_diff_{w}'] = (_window(hsy - asy, w, 'centre'), 'anti')
        f[f'ball_adv_{w}'] = (_window(ball_adv, w, 'centre'), 'anti')
    # Oriented collective velocity (centroid motion across 12 frames; gaps excluded).
    vh = direction_h * (np.r_[hcx[12:], np.full(12, np.nan)] - hcx) * fps / 12
    va = direction_a * (np.r_[acx[12:], np.full(12, np.nan)] - acx) * fps / 12
    vh = np.where(np.abs(vh) < 10, vh, np.nan)
    va = np.where(np.abs(va) < 10, va, np.nan)
    for w in (25, 75):
        f[f'fwd_vel_diff_{w}'] = (_window(vh - va, w, 'centre'), 'anti')
        f[f'fwd_vel_sum_{w}'] = (_window(vh + va, w, 'centre'), 'sym')
    meta = {'direction_home': direction_h}
    return f, meta


def build(players, ball, teams):
    """Full-length matrix; frames outside the two halves are zero (never scored)."""
    n = len(ball)
    columns, rules, metas = {}, {}, {}
    for half, start, last in half_video_blocks(GAME):
        stop = last + 1
        feats, meta = half_features(players[start:stop], ball[start:stop], teams)
        metas[half] = meta
        for name, (values, rule) in feats.items():
            if name not in columns:
                columns[name] = np.zeros(n, dtype=np.float32)
                rules[name] = rule
            columns[name][start:stop] = np.nan_to_num(values, nan=0., posinf=0., neginf=0.)
    names = list(columns)
    return np.column_stack([columns[k] for k in names]), names, rules, metas


def swap(matrix, names, rules):
    out = matrix.copy()
    for i, name in enumerate(names):
        rule = rules[name]
        if rule == 'anti':
            out[:, i] = -matrix[:, i]
        elif isinstance(rule, tuple) and rule[0] == 'pair':
            out[:, i] = matrix[:, names.index(rule[1])]
    return out
