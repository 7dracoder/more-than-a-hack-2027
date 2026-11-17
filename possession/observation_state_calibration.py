"""Observation-loss calibration of D odds, preserving conditional H/A odds."""
import numpy as np


def calibrate_dead_odds(probability, availability_loss, offset):
    p = np.asarray(probability)
    gate = np.asarray(availability_loss)
    if (p.ndim != 2 or p.shape[1] != 4 or not len(p) or gate.shape != (len(p),) or
            not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or
            not np.allclose(p[:, :3].sum(1), 1., atol=1e-5) or
            not np.isin(gate, [0, 1]).all() or not np.isfinite(offset) or abs(offset) > 20):
        raise ValueError('Need valid HAD/change probabilities, binary gate and bounded finite offset')
    result = p.copy()
    if offset == 0 or not gate.any():
        return result
    active = gate.astype(bool)
    q = p[active, :3].astype(np.float64)
    q[:, 2] *= np.exp(offset)
    q /= q.sum(1, keepdims=True)
    result[active, :3] = q
    return result
