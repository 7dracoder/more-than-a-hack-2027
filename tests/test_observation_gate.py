import numpy as np
import pytest
from possession.observation_gate import observation_loss, mix_probabilities


NAMES = ['ball_seen', 'home_current_visible', 'away_current_visible']


def test_visible_ball_with_missing_player_activates_the_expert():
    x = np.array([[1., 1., 1.], [1., 10 / 11, 1.], [1., 1., 9 / 11], [0., 1., 1.]], np.float32)
    np.testing.assert_array_equal(observation_loss(x, NAMES), [0., 1., 1., 1.])
    np.testing.assert_array_equal(observation_loss(x[:, [0, 2, 1]], NAMES), observation_loss(x, NAMES))


def test_gate_preserves_complete_inputs_and_boundary_without_mutating_inputs():
    p = np.tile([.8, .1, .1, .03], (5, 1)).astype(np.float32)
    q = np.tile([.1, .8, .1, .98], (3, 1)).astype(np.float32)
    x = np.array([[1., 1., 1.], [1., .5, 1.], [0., .5, .5]], np.float32)
    before = p.copy()
    result = mix_probabilities(p, q, x, NAMES, .75)
    np.testing.assert_array_equal(p, before)
    np.testing.assert_array_equal(result[:2], p[:2])
    np.testing.assert_array_equal(result[:, 3], p[:, 3])
    assert (result[2:, 1] > result[2:, 0]).all()
    np.testing.assert_allclose(result[:, :3].sum(1), 1.)
    np.testing.assert_array_equal(mix_probabilities(p, q, x, NAMES, 0.), p)


def test_clock_mismatch_is_rejected_instead_of_silently_truncating():
    with pytest.raises(ValueError, match='clocks'):
        mix_probabilities(np.zeros((9, 4)), np.zeros((3, 4)), np.zeros((3, 3)), NAMES, .5)
