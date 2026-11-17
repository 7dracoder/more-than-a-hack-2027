import numpy as np

from scripts.research.spatial_memory.train_natural_visibility import model_matrix, selection_scores
from scripts.research.spatial_memory.run_external import score_parts


def test_model_input_keeps_teacher_prefix_and_memory_without_trajectory_columns():
    x = np.arange(4 * 1088, dtype=np.float32).reshape(4, 1088)
    actual = model_matrix(x)
    assert actual.shape == (4, 792)
    np.testing.assert_array_equal(actual[:, :664], x[:, :664])
    np.testing.assert_array_equal(actual[:, 664:], x[:, 960:])


def test_native_guard_rejects_gain_obtained_by_destroying_observed_ball_performance():
    truth = np.array(list('H' * 250 + 'A' * 250))
    good = np.zeros((len(truth), 4), np.float32)
    good[:, :3] = .001
    good[np.arange(len(truth)), np.where(truth == 'H', 0, 1)] = .998
    bad = good[:, [1, 0, 2, 3]]
    parts, probabilities = [], []
    baseline = {}
    for view in ('native', 'masked', 'mixed'):
        parent = good if view == 'native' else bad
        parts.append(dict(view=view, truth=truth, parent=parent))
        probabilities.append((bad if view == 'native' else good)[::2])
        baseline[view] = score_parts([truth], [truth if view == 'native' else np.where(truth == 'H', 'A', 'H')])
    result = selection_scores(parts, probabilities, (0., 0., 0.), baseline)
    assert next(r for r in result if r['strength'] == 0.)['eligible']
    full = next(r for r in result if r['strength'] == 1.)
    assert full['objective'] > result[0]['objective']
    assert not full['eligible']
