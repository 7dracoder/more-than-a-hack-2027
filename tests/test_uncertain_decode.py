import itertools
import numpy as np
from possession.temporal import viterbi_labels
from possession.uncertain_decode import decode_labels


def test_zero_penalty_and_complete_observations_preserve_original_labels_exactly():
    rng = np.random.default_rng(747)
    p = np.column_stack([rng.dirichlet([1., 1., 1.], 100), rng.random(100)])
    original = viterbi_labels(p, 0., 2., 1.)
    np.testing.assert_array_equal(decode_labels(p, np.ones(100), 0.), original)
    np.testing.assert_array_equal(decode_labels(p, np.zeros(100), 8.), original)


def test_compiled_path_matches_exhaustive_time_varying_objective():
    p = np.array([[.7, .2, .1, .3], [.4, .5, .1, .6], [.2, .7, .1, .8], [.1, .2, .7, .2]])
    gate = np.array([0., 1., 1., 0.])
    penalty, team, dead, weight = 2., .3, .8, .7
    logp = np.log(p[:, :3])
    odds = np.log(p[:, 3] / (1. - p[:, 3])) * weight - penalty * gate * 1.5 * (1. - p[:, :3].max(1))
    transition = np.array([[0., -team, -dead], [-team, 0., -dead], [-dead, -dead, 0.]])
    def objective(path):
        score = sum(logp[t, x] for t, x in enumerate(path))
        return score + sum(transition[path[t - 1], path[t]] + (odds[t] if path[t - 1] != path[t] else 0.) for t in range(1, len(path)))
    best = max(itertools.product(range(3), repeat=4), key=objective)
    np.testing.assert_array_equal(decode_labels(p, gate, penalty, team, dead, weight), np.array(['H', 'A', 'D'])[list(best)])


def test_uncertain_short_flip_is_suppressed_but_confident_change_remains():
    p = np.tile([.9, .05, .05, .5], (50, 1))
    p[15:18, :3] = [.44, .5, .06]
    p[30:, :3] = [.02, .97, .01]
    baseline = decode_labels(p, np.ones(50), 0.)
    stabilized = decode_labels(p, np.ones(50), 4.)
    assert (baseline[15:18] == 'A').all()
    assert (stabilized[15:18] == 'H').all()
    assert (stabilized[30:] == 'A').all()
