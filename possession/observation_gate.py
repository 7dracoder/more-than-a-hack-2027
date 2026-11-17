"""Gate an external camera-loss expert using input availability, not labels.

Nominal coverage is eleven observed players per team and an observed ball.
This is a finite-coordinate availability rule, not confidence in true tracking.
"""
import numpy as np


def observation_loss(memory, names):
    values = np.asarray(memory)
    if values.ndim != 2 or values.shape[1] != len(names) or len(names) != len(set(names)):
        raise ValueError('Memory schema differs')
    keys = ['ball_seen', 'home_current_visible', 'away_current_visible']
    fields = values[:, [names.index(name) for name in keys]]
    if not np.isfinite(fields).all():
        raise ValueError('Availability fields must be finite')
    return ((fields[:, 0] < .5) | (fields[:, 1] < 1.) | (fields[:, 2] < 1.)).astype(np.float32)


def mix_probabilities(parent, expert, memory, names, strength, step=2):
    parent, expert = np.asarray(parent), np.asarray(expert)
    if (parent.ndim != 2 or expert.ndim != 2 or parent.shape[1] != 4 or expert.shape[1] != 4 or
            not isinstance(step, int) or step < 1 or len(expert) != (len(parent) + step - 1) // step or
            len(memory) != len(expert) or not 0 <= strength <= 1):
        raise ValueError('Incompatible probability clocks or strength')
    gate = np.repeat(observation_loss(memory, names), step)[:len(parent)]
    q = np.repeat(expert[:, :3], step, axis=0)[:len(parent)]
    alpha = strength * gate[:, None]
    result = parent.copy()
    result[:, :3] = (1. - alpha) * parent[:, :3] + alpha * q
    # The parent's calibrated change probabilities retain their original clock.
    return result
