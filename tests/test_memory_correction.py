import numpy as np

from scripts.research.spatial_memory.run_external import (
    blend, corrected_probability, input_matrix, swap_matrix,
)


class AsymmetricCorrector:
    def predict_proba(self, matrix, base_margin):
        # Deliberately asymmetric: the wrapper must remove home/away bias.
        logits = base_margin.copy()
        logits[:, 0] += .3 * matrix[:, 4] + .2
        logits[:, 1] += .1 * matrix[:, 5]
        logits -= logits.max(1, keepdims=True)
        result = np.exp(logits)
        return result / result.sum(1, keepdims=True)


def test_correction_is_team_equivariant_even_for_asymmetric_learner():
    rng = np.random.default_rng(25)
    p = rng.uniform(.1, 1., (20, 4)).astype(np.float32)
    p[:, :3] /= p[:, :3].sum(1, keepdims=True)
    memory = rng.normal(size=(20, 3)).astype(np.float32)
    names = ['home_memory', 'away_memory', 'shared']
    a = corrected_probability(AsymmetricCorrector(), p, memory, names, 'spatial_memory')
    b = corrected_probability(AsymmetricCorrector(), p[:, [1, 0, 2, 3]],
                              memory[:, [1, 0, 2]], names, 'spatial_memory')
    np.testing.assert_allclose(a, b[:, [1, 0, 2]], atol=1e-7)
    np.testing.assert_allclose(a.sum(1), 1., atol=1e-7)


def test_strength_zero_is_exact_parent_and_boundary_is_preserved():
    p = np.array([[.2, .3, .5, .17], [.1, .8, .1, .96]], np.float32)
    correction = np.array([[.6, .2, .2], [.2, .7, .1]], np.float32)
    np.testing.assert_array_equal(blend(p, correction, 0.), p)
    for strength in (.25, .5, 1.):
        mixed = blend(p, correction, strength)
        np.testing.assert_allclose(mixed[:, :3].sum(1), 1., atol=1e-7)
        np.testing.assert_array_equal(mixed[:, 3], p[:, 3])


def test_control_and_memory_learner_share_exact_parent_context():
    p = np.array([[.2, .3, .5, .1]], np.float32)
    memory = np.array([[1., 2.]], np.float32)
    control = input_matrix(p, memory, 'probability_control')
    treatment = input_matrix(p, memory, 'spatial_memory')
    np.testing.assert_array_equal(control, treatment[:, :4])
    np.testing.assert_array_equal(swap_matrix(control, [], 'probability_control'), p[:, [1, 0, 2, 3]])
