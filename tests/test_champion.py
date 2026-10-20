"""Unit semantics, team symmetry, purging and strict fitting-label isolation."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from possession.metric_features import metric_reduction, speed_scale, restart_features, swap_teams, trajectory_speed
from possession.champion import purged_folds, decode_factorized, fit_classifier, classifier_probability, training_rows
from test_possession import toy, CONFIG


class UnitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def test_trajectory_speed_uses_pitch_and_frame_timing_and_rejects_gaps(self):
        xy = np.zeros((30, 2, 2))
        xy[:, :, 0] = np.arange(30)[:, None] * (2. / 25. / 105.3)
        velocity = trajectory_speed(xy)
        np.testing.assert_allclose(velocity[3:-3], 2., atol=1e-12)
        xy[12, 0] = np.nan
        xy[14:, 1, 0] += .3
        velocity = trajectory_speed(xy)
        self.assertTrue(np.isnan(velocity[9:16, 0]).all())
        self.assertTrue(np.isnan(velocity[11:17, 1]).all())

    def test_speed_conversion_precedes_filtering_and_stationarity(self):
        players = np.array([[.1, .2, .72], [.3, .2, 3.6], [.7, .8, 36.]])
        q, stationary, running, _ = metric_reduction(players, np.array([0, 1, 0]), 1/3.6)
        self.assertAlmostEqual(q[1], 1.)
        self.assertAlmostEqual(stationary, 1/3)
        self.assertAlmostEqual(running, 1/3)
        permutation = [2, 0, 1]
        other = metric_reduction(players[permutation], np.array([0, 1, 0])[permutation], 1/3.6)
        for a, b in zip((q, stationary, running), other[:3]):
            np.testing.assert_allclose(a, b)

    def test_unknown_speed_units_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=self.folder) as temporary:
            path = Path(temporary) / 'metadata.xml'
            path.write_text('<root><Sensor id="speed"><Channel><Unit>miles per hour</Unit></Channel></Sensor></root>')
            with self.assertRaises(ValueError):
                speed_scale(path)
            path.write_text('<root><Sensor id="speed"><Channel><Unit>kph</Unit></Channel></Sensor></root>')
            self.assertAlmostEqual(speed_scale(path)[0], 1/3.6)

    def test_restart_ownership_is_not_dropped_and_unknown_team_is_neutral(self):
        with tempfile.TemporaryDirectory(dir=self.folder) as temporary:
            path = Path(temporary) / 'tags.xml'
            path.write_text('<file><instance><start>0</start><end>1</end><code>SET PIECES</code><label><group>Team</group><text>Barcelona</text></label><label><group>Type</group><text>Throw In</text></label></instance><instance><start>1</start><end>2</end><code>SET PIECES</code><label><group>Team</group><text>N/A</text></label></instance></file>')
            result = restart_features(path, 60)
            self.assertTrue((result['restart_owner'][:25] == 1).all())
            self.assertTrue((result['restart_owner'][25:] == 0).all())
            self.assertTrue((result['restart_types'][:25, 0] == 1).all())
            self.assertTrue(result['restart'][:50].all())


class SplitTests(unittest.TestCase):
    def test_every_run_is_whole_and_context_does_not_cross_into_training(self):
        labels = np.array(list('XX' + 'HHHAAADDD' * 6 + 'XXX'))
        folds = purged_folds(labels, n_folds=4, embargo=4, context=2)
        coverage = np.zeros(len(labels), dtype=int)
        for train, test, info in folds:
            coverage += test
            self.assertFalse((train & test).any())
            first, stop = info['test_first'], info['test_stop']
            self.assertFalse(train[max(0, first-4):stop+4].any())
            for a in range(2, len(labels)-3, 3):
                self.assertIn(int(train[a:a+3].sum()), (0, 3))
                self.assertIn(int(test[a:a+3].sum()), (0, 3))
        np.testing.assert_array_equal(coverage, (labels != 'X').astype(int))

    def test_fit_rows_and_weights_ignore_excluded_labels(self):
        labels = np.array(list('HHHAAADDDHHHAAADDD'))
        mask = np.zeros(len(labels), dtype=bool)
        mask[3:12] = True
        altered = labels.copy()
        altered[~mask] = 'X'
        for a, b in zip(training_rows(labels, mask, 2), training_rows(altered, mask, 2)):
            np.testing.assert_array_equal(a, b)

    def test_fitted_predictions_ignore_excluded_labels(self):
        labels = np.resize(np.array(['H', 'A', 'D']), 90)
        mask = np.zeros(90, dtype=bool)
        mask[:60] = True
        altered = labels.copy()
        altered[~mask] = 'X'
        x = np.random.default_rng(7).normal(size=(90, 3))
        names = ['speed50_50', 'control_margin_12', 'phase_conflict_12']
        config = {'classifier_training_stride': 1, 'classifier_iterations': 5, 'classifier_learning_rate': .05,
                  'classifier_max_leaf_nodes': 3, 'classifier_l2': 10., 'classifier_min_samples_leaf': 5, 'seed': 7}
        a, _ = fit_classifier(x, names, labels, mask, config)
        b, _ = fit_classifier(x, names, altered, mask, config)
        np.testing.assert_array_equal(classifier_probability(a, x, names), classifier_probability(b, x, names))
        p = classifier_probability(a, x, names)
        swapped = classifier_probability(a, swap_teams(x, names), names)
        np.testing.assert_allclose(p[:, [1, 0, 2]], swapped, atol=1e-15)


class FactorizedTests(unittest.TestCase):
    def test_public_kickoff_prior_is_conservative_and_separate_from_labels(self):
        arrays = toy()
        config = dict(CONFIG, step=1, history_samples=0, kickoff_prior_guard_frames=5)
        game = {'BAR-ATH': {'kickoff': {'minute':0,'second':0,'frame':20}}}
        with patch('possession.champion.half_video_blocks', return_value=[(1,0,29)]), patch('possession.champion.GAMES', game):
            labels, _, _ = decode_factorized(arrays, config, np.full(30,.1), np.tile([2.,-2.],(30,1)))
        self.assertTrue((labels[:15] == 'D').all())
        self.assertEqual(labels[-1], 'H')

    def test_uncertain_owner_does_not_reduce_total_live_support(self):
        arrays = toy()
        config = dict(CONFIG, step=1, history_samples=2)
        with patch('possession.champion.half_video_blocks', return_value=[(1, 0, 9), (2, 20, 29)]):
            labels, posterior, histories = decode_factorized(arrays, config, np.full(30, .4), conditional_team=np.zeros((30, 2)))
        self.assertTrue((labels[:10] != 'D').all())
        self.assertTrue((labels[10:20] == 'X').all())
        np.testing.assert_allclose(posterior[:10].sum(1), 1., atol=1e-6)

    def test_team_swap_preserves_dead_states_and_swaps_live_states(self):
        arrays = toy()
        config = dict(CONFIG, step=1, history_samples=0)
        first = np.tile([2., -2.], (30, 1))
        p_dead = np.r_[np.full(10, .1), np.full(10, .9), np.full(10, .1)]
        with patch('possession.champion.half_video_blocks', return_value=[(1, 0, 29)]):
            a, pa, _ = decode_factorized(arrays, config, p_dead, first)
            b, pb, _ = decode_factorized(arrays, config, p_dead, first[:, ::-1])
        self.assertTrue((a[10:20] == 'D').all())
        np.testing.assert_array_equal(np.where(a == 'H', 'A', np.where(a == 'A', 'H', a)), b)
        np.testing.assert_allclose(pa[:, [1, 0, 2]], pb, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
