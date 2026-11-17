import numpy as np
from scripts.research.spatial_memory.compare_memory_replacement import replacement_probability


def test_missing_expert_preserves_observations_parent_boundaries_and_inputs():
    parent = np.tile([.8, .1, .1, .07], (7, 1)).astype(np.float32)
    expert = np.tile([.1, .8, .1, .95], (4, 1)).astype(np.float32)
    memory = np.array([[1., .3, .3], [0., .3, .3], [0., .01, .01], [1., 0., 0.]], np.float32)
    original = parent.copy()
    actual = replacement_probability(parent, expert, memory,
        ['ball_seen', 'past_anchor_age', 'future_anchor_age'])
    np.testing.assert_array_equal(parent, original)
    np.testing.assert_array_equal(actual[:, 3], parent[:, 3])
    np.testing.assert_array_equal(actual[[0, 1, 4, 5, 6]], parent[[0, 1, 4, 5, 6]])
    assert (actual[2:4, 1] > actual[2:4, 0]).all()
    np.testing.assert_allclose(actual[:, :3].sum(1), 1.)


def test_fully_observed_external_view_remains_exactly_the_parent():
    rng = np.random.default_rng(53)
    parent = rng.random((9, 4)).astype(np.float32)
    expert = rng.random((5, 4)).astype(np.float32)
    memory = np.ones((5, 3), np.float32)
    actual = replacement_probability(parent, expert, memory,
        ['ball_seen', 'past_anchor_age', 'future_anchor_age'])
    np.testing.assert_array_equal(actual, parent)
