"""Fixed fusion: baseline live/dead evidence, contacts conditional team evidence."""
import numpy as np

from possession.champion import decode_factorized


def team_only_inputs(baseline, contacts):
    """Return separate decoder inputs; contact dead probability is never used.

    Keeping the two inputs separate avoids changing team odds when a tiny
    baseline live probability would floor both entries of a synthetic joint PDF.
    """
    baseline, contacts = np.asarray(baseline), np.asarray(contacts)
    if baseline.ndim != 2 or baseline.shape[1] != 3 or contacts.shape != baseline.shape:
        raise ValueError('Expected matching (frames, 3) H/A/D probability arrays')
    for values in (baseline, contacts):
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError('Probabilities must be finite and within [0, 1]')
        if not np.allclose(values.sum(1), 1., atol=1e-6):
            raise ValueError('Probability rows must sum to one')
    team = contacts[:, :2].clip(1e-6, 1.)
    return baseline[:, 2].clip(1e-6, 1.), np.log(team / team.sum(1)[:, None])


def decode_team_only(arrays, config, baseline, contacts):
    dead, team = team_only_inputs(baseline, contacts)
    return decode_factorized(arrays, config, dead, team)
