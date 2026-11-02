"""Unsupervised two-class latent model (stoppage vs play) over label-free per-frame indicators.

Indicators on graded tactical frames covered by the broadcast:
  ball:  tactical ball missing (0/1)
  view:  broadcast live (0), close-up / no homography (1), other moment or too few players (2)
  speed: median 1-s player displacement < 1.0, 1.0-1.6, >= 1.6 m/s (0/1/2)
EM assumes conditional independence given the class; it never reads labels.
"""
import numpy as np


def indicators(signal, tactical_players, tactical_ball):
    seen = np.isfinite(tactical_ball).all(1)
    live = signal["valid"] & (signal["players"] >= 6) & (signal["cost"] < 1.5)
    view = np.where(live, 0, np.where(~signal["valid"], 1, 2))
    step = np.linalg.norm(tactical_players[25:] - tactical_players[:-25], axis=2)
    step = np.r_[np.full((25, tactical_players.shape[1]), np.nan), step]
    with np.errstate(all="ignore"):
        speed = np.nanmedian(np.where(step < 12, step, np.nan), 1)
    speed_bin = np.digitize(np.nan_to_num(speed, nan=1.3), [1.0, 1.6])
    return (~seen).astype(int), view, speed_bin


def fit(columns, sizes, iterations=200, seed=0):
    """columns: list of int arrays; sizes: number of categories per column. Returns prior, conditionals, posterior."""
    rng = np.random.default_rng(seed)
    n = len(columns[0])
    post = np.clip(0.5 + 0.3 * (columns[0] - 0.5) + rng.normal(0, 0.01, n), 0.01, 0.99)  # class 1 = stoppage
    for _ in range(iterations):
        prior = post.mean()
        cond = [np.array([[np.average(c == k, weights=w) for k in range(s)] for w in (1 - post, post)])
                for c, s in zip(columns, sizes)]
        log1 = np.log(prior) + sum(np.log(p[1][c]) for p, c in zip(cond, columns))
        log0 = np.log(1 - prior) + sum(np.log(p[0][c]) for p, c in zip(cond, columns))
        post = 1 / (1 + np.exp(log0 - log1))
    return prior, cond, post
