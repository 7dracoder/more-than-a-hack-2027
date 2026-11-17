import unittest
import numpy as np

from scripts.research.spatial_memory.train_partial_had_memory import batch, selection_result, VIEWS


class PartialHADStudyTest(unittest.TestCase):
    def test_native_or_weak_regression_cannot_win_through_proxy_gain(self):
        baseline = dict(scores={v:dict(spell_f1_mean=.6) for v in VIEWS},
            weak_scores={'all_known':{'weak_ha_accuracy':.8}, 'ball_missing':{'weak_ha_accuracy':.75}})
        scores = {v:dict(spell_f1_mean=.9) for v in VIEWS}
        weak = {'all_known':{'weak_ha_accuracy':.8}, 'ball_missing':{'weak_ha_accuracy':.75}}
        self.assertTrue(selection_result(scores, weak, baseline)['eligible'])
        scores['native']['spell_f1_mean'] = .599
        self.assertFalse(selection_result(scores, weak, baseline)['eligible'])
        scores['native']['spell_f1_mean'] = .6
        weak['ball_missing']['weak_ha_accuracy'] = .739
        self.assertFalse(selection_result(scores, weak, baseline)['eligible'])

    def test_partial_unknowns_and_dead_stay_separate_from_weak_domain(self):
        config = dict(batch_domain_counts=[2, 1, 1], batch_size=4, sequence_length=8,
            strong_native_view_probability=.5)
        state = dict(mean=np.zeros(792), scale=np.ones(792))
        def part(width, value, labels, view=None):
            h = dict(x=np.full((8, width), value, np.float32), y=np.asarray(labels),
                w=(np.asarray(labels) != -100).astype(np.float32))
            if view is not None:
                h['view'] = view
            return h
        strong = [part(1088, 1, [0] * 8, v) for v in VIEWS]
        weak = [part(792, 2, [0, 1, -100, 0, 1, -100, 0, 1])]
        partial = [part(792, 3, [2, 2, -100, 2, -100, 2, 2, -100])]
        x, y, w = batch(np.random.default_rng(12), config, strong, weak, partial, state, np.arange(792))
        self.assertEqual(x.shape, (4, 792, 8))
        np.testing.assert_array_equal(x[:, 0, 0], [1, 1, 2, 3])
        self.assertTrue(set(y[2]).issubset({0, 1, -100}))
        np.testing.assert_array_equal(y[3], partial[0]['y'])
        self.assertTrue((w[y == -100] == 0).all())


if __name__ == '__main__':
    unittest.main()
