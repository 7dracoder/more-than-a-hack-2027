"""Small deterministic supervised heads with explicit temporal provenance.

Only this opt-in module reads public labels for fitting. The base pipeline does not.
These models are development experiments with tiny, single-match supervision.
"""

import numpy as np

from possession.data import rolling_mean
from possession.model import evidence, sigmoid
from games import GAME, half_video_blocks

DEAD_NAMES = ["bias", "median_player_speed", "stationary_fraction", "ball_missing",
              "missing_fraction_2s", "outside_pitch", "distance_nearest", "phase_strength",
              "restart_window", "camera_activity", "raw_dead_logit"]


def design(arrays, config):
    n = len(arrays["ball"])
    dead = np.zeros((n, len(DEAD_NAMES)))
    team = np.zeros((n, 2))
    for _, start, last in half_video_blocks(GAME):
        stop = last + 1
        _, parts = evidence(arrays, start, stop, config)
        ball = arrays["ball"][start:stop]
        missing = ~np.isfinite(ball).all(1)
        near = np.min(np.nan_to_num(arrays["nearest"][start:stop], nan=30.), axis=1)
        speed = np.nan_to_num(rolling_mean(arrays["median_speed"][start:stop], 25), nan=1.)
        stationary = np.nan_to_num(rolling_mean(arrays["stationary"][start:stop], 25), nan=.5)
        d = parts["dead_probability"]
        dead[start:stop] = np.column_stack([
            np.ones(stop - start), np.clip(speed, 0, 5) / 5., stationary, missing,
            rolling_mean(missing.astype(float), 50),
            (((ball < -.015) | (ball > 1.015)).any(1) & ~missing),
            np.clip(near, 0, 15) / 15., np.abs(arrays["tag"][start:stop]),
            arrays["restart"][start:stop], parts["camera_activity"], np.log(d / (1. - d)) / 4.])
        team[start:stop] = np.column_stack([parts["geometry"], arrays["tag"][start:stop]])
    return dead, team


def fit_logistic(x, y, balanced=False, nonnegative=False, epochs=400):
    if len(np.unique(y)) != 2:
        raise ValueError("Training block must contain both classes for each calibration head")
    weights = np.zeros(x.shape[1])
    first, second = np.zeros_like(weights), np.zeros_like(weights)
    importance = np.ones(len(y))
    if balanced:
        importance = np.where(y > .5, .5 / y.mean(), .5 / (1. - y.mean()))
    for epoch in range(1, epochs + 1):
        error = (sigmoid(x @ weights) - y) * importance
        regularization = weights.copy()
        if not nonnegative:
            regularization[0] = 0.
        gradient = x.T @ error / len(y) + .02 * regularization
        first = .9 * first + .1 * gradient
        second = .999 * second + .001 * gradient ** 2
        weights -= .05 * (first / (1. - .9 ** epoch)) / (np.sqrt(second / (1. - .999 ** epoch)) + 1e-8)
        if nonnegative:
            weights = np.clip(weights, 0., 8.)
    return weights


def fit(arrays, config, labels, start, stop, train_mask=None):
    dead, team = design(arrays, config)
    indices = np.arange(start, stop)
    if train_mask is not None:
        indices = indices[np.asarray(train_mask, dtype=bool)[indices]]
    if len(indices) == 0:
        raise ValueError("No training frames")
    if not np.isin(labels[indices], ["H", "A", "D"]).all():
        raise ValueError("Cannot train on unlabeled/X frames")
    alive = np.isin(labels[indices], ["H", "A"])
    dead_weights = fit_logistic(dead[indices], (labels[indices] == "D").astype(float), balanced=True)
    team_weights = fit_logistic(team[indices][alive], (labels[indices][alive] == "H").astype(float), nonnegative=True)
    return {"kind": "regularized_logistic_two_head", "train_first_frame": start,
            "train_stop_exclusive": stop, "n_train_frames": len(indices),
            "dead_feature_names": DEAD_NAMES, "team_feature_names": ["geometry", "tag"],
            "dead_weights": dead_weights.tolist(), "team_weights": team_weights.tolist(),
            "epochs": 400, "l2": .02, "class_balanced_dead": True,
            "warning": "Tiny one-match public training block; not independent full-match validation"}


def scores(arrays, config, model):
    dead, team = design(arrays, config)
    p_dead = np.clip(sigmoid(dead @ np.array(model["dead_weights"])), .01, .99)
    p_home = sigmoid(team @ np.array(model["team_weights"]))
    return np.log(np.clip(np.column_stack([(1. - p_dead) * p_home,
                                          (1. - p_dead) * (1. - p_home), p_dead]), 1e-6, 1.))
