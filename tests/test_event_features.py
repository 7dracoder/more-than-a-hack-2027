"""Invariants for the user-added v0.4 tracking features and experiment gate."""
import unittest
from unittest.mock import patch

import numpy as np

from possession.event_features import _window, half_features, build, swap
from scripts.run_v04_experiments import verify_baseline


def tracking(n=32):
    teams = np.array([0, 0, 1, 1])
    players = np.zeros((n, 4, 3), dtype=float)
    players[:, :, :2] = [[.2, .3], [.3, .6], [.7, .3], [.8, .6]]
    players[:, :, 0] += np.arange(n)[:, None] * .0001
    ball = np.full((n, 3), .3)
    ball[:, :2] = players[:, 0, :2]
    ball[n // 3: n // 2, :2] = np.nan
    ball[n // 2:, :2] = players[n // 2:, 2, :2]
    return players, ball, teams


class EventFeatureTests(unittest.TestCase):
    def test_declared_team_swap_matches_recomputed_features(self):
        players, ball, teams = tracking()
        original, _ = half_features(players, ball, teams)
        exchanged, _ = half_features(players, ball, 1 - teams)
        names = list(original)
        matrix = np.column_stack([original[k][0] for k in names])
        rules = {k: original[k][1] for k in names}
        expected = np.column_stack([exchanged[k][0] for k in names])
        np.testing.assert_allclose(swap(matrix, names, rules), expected, atol=1e-6)
        np.testing.assert_allclose(swap(swap(matrix, names, rules), names, rules), matrix)

    def test_joint_slot_permutation_preserves_features(self):
        players, ball, teams = tracking()
        first, _ = half_features(players, ball, teams)
        order = [3, 0, 2, 1]
        other, _ = half_features(players[:, order], ball, teams[order])
        for name in first:
            np.testing.assert_allclose(first[name][0], other[name][0], atol=1e-6, err_msg=name)

    def test_missing_ball_never_creates_contacts(self):
        players, ball, teams = tracking()
        ball[:] = np.nan
        features, _ = half_features(players, ball, teams)
        for name in ('last_contact_1m', 'next_contact_1m', 'bracket_1m', 'ball_still_run'):
            np.testing.assert_array_equal(features[name][0], 0)
        self.assertTrue(np.isfinite(np.column_stack([v[0] for v in features.values()])).all())

    def test_features_do_not_cross_halftime(self):
        players, ball, teams = tracking(72)
        blocks = [(1, 4, 35), (2, 40, 71)]
        with patch('possession.event_features.half_video_blocks', return_value=blocks):
            original, names, _, _ = build(players, ball, teams)
            players[:36, :, :2] = .9
            ball[:36, :2] = -.5
            changed, _, _, _ = build(players, ball, teams)
        np.testing.assert_allclose(original[40:], changed[40:])
        np.testing.assert_array_equal(original[:4], 0)
        np.testing.assert_array_equal(original[36:40], 0)
        self.assertGreater(len(names), 0)

    def test_directional_windows_exclude_current_frame_and_ignore_nan(self):
        values = [1., np.nan, 3.]
        np.testing.assert_array_equal(_window(values, 1, 'past'), [0, 1, 0])
        np.testing.assert_array_equal(_window(values, 1, 'future'), [0, 3, 0])

    def test_mismatching_reference_score_stops_experiments(self):
        good = {'n_features': 68, 'whole_run': {'spell': 0.4457142857142856},
                'fixed_time': {'spell': .62}}
        verify_baseline(good)
        with self.assertRaises(ValueError):
            verify_baseline(dict(good, fixed_time={'spell': .8}))


if __name__ == '__main__':
    unittest.main()
