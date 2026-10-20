"""Fixed, bounded temporal stoppage proposal. No possession labels enter here."""

import numpy as np

from games import GAME, half_video_blocks
from possession.data import rolling_mean
from possession.model import evidence


def scores(arrays, config):
    """Retain team odds; combine multi-second inactivity and observed ball cues.

    Missing ball alone is neutral. A motionless observed ball only contributes
    when player inactivity agrees. This fixed candidate is evaluated separately.
    """
    result = np.zeros((len(arrays['ball']), 3))
    for _, start, last in half_video_blocks(GAME):
        stop = last + 1
        base, _ = evidence(arrays, start, stop, config)
        p = np.exp(base)
        ball = arrays['ball'][start:stop]
        seen = np.isfinite(ball).all(1)
        continuous = seen & np.r_[False, seen[:-1]]
        velocity = np.linalg.norm(np.diff(ball, axis=0, prepend=ball[:1]) * [105.3, 68.], axis=1) * 25
        still = np.nan_to_num(rolling_mean(np.where(continuous, velocity < .8, np.nan), 37), nan=0.)
        outside = seen & ((ball < -.008) | (ball > 1.008)).any(1)
        slow = np.nan_to_num(rolling_mean(arrays['median_speed'][start:stop], 37), nan=2.)
        stationary = np.nan_to_num(rolling_mean(arrays['stationary'][start:stop], 37), nan=0.)
        inactive = np.clip((1.3 - slow) / .9, 0, 1) * np.clip((stationary - .25) / .45, 0, 1)
        outside_context = np.nan_to_num(rolling_mean(outside.astype(float), 12))
        # Fixed before evaluating this candidate; no score-driven grid search.
        correction = 2.0 * inactive + 1.5 * inactive * still + 2.0 * outside_context
        correction = np.clip(correction, 0., 4.)
        result[start:stop] = base
        result[start:stop, 2] += correction
        result[start:stop] -= np.logaddexp.reduce(result[start:stop], axis=1)[:, None]
    return result
