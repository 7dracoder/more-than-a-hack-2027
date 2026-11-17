"""Blend an external expert's transition head after explicit clock conversion."""
import numpy as np
from possession.observation_gate import observation_loss


def mix_boundaries(probability, expert, memory, names, strength, step=2):
    p, q = np.asarray(probability), np.asarray(expert)
    if (p.ndim != 2 or q.ndim != 2 or p.shape[1] != 4 or q.shape[1] != 4 or
            not isinstance(step, int) or step < 1 or not 0 <= strength <= 1 or
            len(q) != (len(p) + step - 1) // step or len(memory) != len(q)):
        raise ValueError('Expert boundary clock or strength differs')
    if not np.isfinite(q[:, 3]).all() or np.any((q[:, 3] < 0) | (q[:, 3] > 1)):
        raise ValueError('Invalid expert transition probabilities')
    result = p.copy()
    if strength == 0:
        return result
    # Expert index j describes a change from source (j-1)*step to j*step.
    # Equal hazards across that interval give 1 - product(1-h) = q[j].
    index = (np.arange(len(p)) + step - 1) // step
    bracketed = (index > 0) & (index < len(q))
    safe_index = np.minimum(index, len(q) - 1)
    hazard = -np.expm1(np.log1p(-q[safe_index, 3].astype(np.float64).clip(0, 1-1e-12)) / step)
    gate = np.repeat(observation_loss(memory, names), step)[:len(p)]
    alpha = strength * gate * bracketed
    result[:, 3] = (1-alpha) * p[:, 3] + alpha * hazard
    return result
