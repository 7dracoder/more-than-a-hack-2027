"""Observation-aware continuity penalty for uncertain external model emissions."""
import numpy as np


def decode_labels(probability, availability_loss, penalty, team_cost=0., dead_cost=2., change_weight=1., change_bias=0.):
    from possession.temporal import viterbi_labels
    p = np.asarray(probability, dtype=np.float64)
    gate = np.asarray(availability_loss, dtype=np.float64)
    if (p.ndim != 2 or p.shape[1] != 4 or not len(p) or gate.shape != (len(p),) or
            not np.isfinite(gate).all() or not np.isin(gate, [0., 1.]).all() or
            not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or
            not np.allclose(p[:, :3].sum(1), 1., atol=1e-5)):
        raise ValueError('Need valid HAD/change probabilities and binary observation availability')
    if (not np.isfinite([penalty, team_cost, dead_cost, change_weight, change_bias]).all() or
            min(penalty, team_cost, dead_cost, change_weight) < 0):
        raise ValueError('Invalid continuity or transition parameter')
    if penalty == 0 or not gate.any():
        return viterbi_labels(p, team_cost, dead_cost, change_weight, change_bias)
    change = p[:, 3].clip(1e-6, 1. - 1e-6)
    odds = (np.log(change) - np.log1p(-change) + change_bias) * change_weight
    uncertainty = np.clip(1.5 * (1. - p[:, :3].max(1)), 0., 1.)
    odds = odds - penalty * gate * uncertainty
    emission = np.log(np.maximum(p[:, :3], 1e-9))
    transition = np.array([[0., -team_cost, -dead_cost], [-team_cost, 0., -dead_cost], [-dead_cost, -dead_cost, 0.]])
    from possession.fast_decode import path
    return np.array(['H', 'A', 'D'])[path(emission, transition, odds)]
