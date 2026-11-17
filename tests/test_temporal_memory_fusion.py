import numpy as np

from scripts.research.spatial_memory.select_temporal_fusion import gap_gate, fusion_probability


def inputs():
    parent = np.array([[.8, .1, .1, .2], [.3, .2, .5, .9], [.1, .7, .2, .4]], np.float32)
    fallback = np.array([[.1, .8, .1, .7], [.7, .2, .1, .4], [.3, .2, .5, .1]], np.float32)
    names = ['ball_seen', 'past_anchor_age', 'future_anchor_age']
    memory = np.array([[1, 0, 0], [0, .03, .04], [0, .15, .2]], np.float32)
    return parent, fallback, memory, names


def test_temporal_gate_uses_missingness_and_elapsed_time_without_labels():
    _, _, memory, names = inputs()
    np.testing.assert_array_equal(gap_gate(memory, names, 'all'), [1, 1, 1])
    np.testing.assert_array_equal(gap_gate(memory, names, 'missing'), [0, 1, 1])
    np.testing.assert_array_equal(gap_gate(memory, names, 'long_missing'), [0, 0, 1])


def test_mixture_keeps_observed_frames_and_parent_boundary_head_exact():
    parent, fallback, memory, names = inputs()
    combined = fusion_probability(parent, fallback, memory, names, 'long_missing', .75)
    np.testing.assert_array_equal(combined[:2], parent[:2])
    np.testing.assert_array_equal(combined[:, 3], parent[:, 3])
    np.testing.assert_allclose(combined[:, :3].sum(1), 1., atol=1e-7)
    np.testing.assert_allclose(combined[2, :3], .25 * parent[2, :3] + .75 * fallback[2, :3])
    np.testing.assert_array_equal(fusion_probability(parent, fallback, memory, names, 'all', 0), parent)


def test_temporal_mixture_commutes_with_team_renaming():
    parent, fallback, memory, names = inputs()
    a = fusion_probability(parent, fallback, memory, names, 'missing', .5)
    b = fusion_probability(parent[:, [1, 0, 2, 3]], fallback[:, [1, 0, 2, 3]], memory, names, 'missing', .5)
    np.testing.assert_array_equal(a, b[:, [1, 0, 2, 3]])
