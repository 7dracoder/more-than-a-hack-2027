"""Exact small-state dynamic programming and reproducible alternative histories."""

import numpy as np


def logsumexp(x, axis=None):
    maximum = np.max(x, axis=axis, keepdims=True)
    result = maximum + np.log(np.exp(x - maximum).sum(axis=axis, keepdims=True))
    return result.squeeze(axis=axis) if axis is not None else result.squeeze()


def infer(emissions, transition, samples=12, seed=0, temperature=1.):
    """Viterbi path, exact model marginals, and backward posterior path samples.

    Potentials need not be normalized probabilities. Marginals are conditional on
    these hand-designed potentials, not calibrated confidence in real possession.
    All paths are constrained only by supplied transition scores, not hidden truth.
    """
    emissions = np.asarray(emissions, dtype=float) / temperature
    transition = np.asarray(transition, dtype=float) / temperature
    n, states = emissions.shape
    if n == 0 or not np.isfinite(emissions).all() or not np.isfinite(transition).all():
        raise ValueError("Expected nonempty finite score arrays")
    forward = np.empty_like(emissions)
    best = emissions[0].copy()
    forward[0] = emissions[0] - logsumexp(emissions[0])
    back = np.zeros((n, states), dtype=np.int16)
    for t in range(1, n):
        candidates = best[:, None] + transition
        back[t] = np.argmax(candidates, axis=0)
        best = candidates[back[t], np.arange(states)] + emissions[t]
        best -= best.max()
        f = emissions[t] + logsumexp(forward[t - 1, :, None] + transition, axis=0)
        forward[t] = f - logsumexp(f)
    path = np.empty(n, dtype=np.int8)
    path[-1] = np.argmax(best)
    for t in range(n - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    backward = np.zeros_like(emissions)
    for t in range(n - 2, -1, -1):
        b = logsumexp(transition + emissions[t + 1] + backward[t + 1], axis=1)
        backward[t] = b - logsumexp(b)
    marginal_log = forward + backward
    posterior = np.exp(marginal_log - logsumexp(marginal_log, axis=1)[:, None])
    rng = np.random.default_rng(seed)
    alternatives = np.empty((samples, n), dtype=np.int8)
    for k in range(samples):
        alternatives[k, -1] = rng.choice(states, p=np.exp(forward[-1]))
        for t in range(n - 2, -1, -1):
            weights = forward[t] + transition[:, alternatives[k, t + 1]]
            probabilities = np.exp(weights - logsumexp(weights))
            alternatives[k, t] = rng.choice(states, p=probabilities)
    return path, posterior, alternatives


def transition_costs(team_cost=6., stoppage_cost=4.):
    return np.array([[0., -team_cost, -stoppage_cost],
                     [-team_cost, 0., -stoppage_cost],
                     [-stoppage_cost, -stoppage_cost, 0.]])
