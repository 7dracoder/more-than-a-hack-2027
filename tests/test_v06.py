"""v0.6 invariants: reproduction of v0.5 when switched off, gating semantics, symmetry."""
import unittest

import numpy as np

from possession import v06
from possession.decode import infer
from possession.champion import fit_classifier, classifier_probability

CONFIG = {'classifier_iterations': 10, 'classifier_learning_rate': .1, 'classifier_max_leaf_nodes': 7,
          'classifier_l2': 1., 'classifier_min_samples_leaf': 5, 'classifier_training_stride': 5, 'seed': 1}


def toy(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.array(['H', 'A', 'D', 'H', 'D', 'A']), n // 6)
    signal = np.where(labels == 'H', 1., np.where(labels == 'A', -1., 0.))
    dead = (labels == 'D').astype(float)
    matrix = np.column_stack([signal + rng.normal(0, .8, n), dead + rng.normal(0, .5, n), rng.normal(size=n)])
    names = ['anti', 'sym', 'noise']
    swap = lambda m, _n: m * np.array([-1., 1., 1.])
    return matrix, names, labels, swap


class GatedViterbiTests(unittest.TestCase):
    def test_constant_costs_match_reference_decoder(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            scores = rng.normal(size=(200, 2)) * rng.uniform(.2, 3)
            path, _, _ = infer(scores, np.array([[0., -6.], [-6., 0.]]), samples=0)
            gated = v06.gated_viterbi(scores, np.full(200, 6.), np.full(200, 6.))
            np.testing.assert_array_equal(path, gated)

    def test_cheaper_switch_only_where_gated(self):
        scores = np.zeros((40, 2))
        scores[:20, 0] = .3
        scores[20:, 1] = .3            # weak away evidence for 20 bins: 6.0 > 20 * 0.3
        high = np.full(40, 6.)
        self.assertTrue((v06.gated_viterbi(scores, high, high) == 0).all())
        to_away = high.copy(); to_away[20] = 3.
        path = v06.gated_viterbi(scores, high, to_away)
        self.assertTrue((path[:20] == 0).all() and (path[20:] == 1).all())

    def test_gates_are_directional_and_dilated(self):
        contacts = np.zeros(50, np.int8)
        contacts[22] = 1
        home, away = v06.contact_gates(contacts, 0, 50, 5, dilate=1)
        np.testing.assert_array_equal(np.flatnonzero(home), [3, 4, 5])
        self.assertFalse(away.any())


class FittingTests(unittest.TestCase):
    def test_offset_zero_reproduces_champion_classifier(self):
        matrix, names, labels, swap = toy()
        mask = np.ones(len(labels), bool)
        old, _ = fit_classifier(matrix, names, labels, mask, CONFIG, swap=swap)
        new = v06.fit_three_class(matrix, names, labels, mask, CONFIG, swap, offset=0)
        np.testing.assert_allclose(classifier_probability(old, matrix, names, swap=swap),
                                   v06.three_class_probability(new, matrix, names, swap))

    def test_live_team_model_is_symmetric_and_ignores_dead_labels(self):
        matrix, names, labels, swap = toy()
        mask = np.ones(len(labels), bool)
        model = v06.fit_live_team(matrix, names, labels, mask, CONFIG, swap)
        p = v06.live_team_probability(model, matrix, names, swap)
        np.testing.assert_allclose(p.sum(1), 1.)
        np.testing.assert_allclose(v06.live_team_probability(model, swap(matrix, names), names, swap), p[:, ::-1])
        relabelled = labels.copy()
        relabelled[labels == 'D'] = 'X'   # removing dead labels must not change the live model
        other = v06.fit_live_team(matrix, names, np.where(labels == 'D', 'D', labels), mask & (labels != 'D'),
                                  CONFIG, swap)
        np.testing.assert_allclose(v06.live_team_probability(other, matrix, names, swap), p)

    def test_offsets_use_disjoint_rows(self):
        labels = np.repeat(np.array(['H', 'A', 'D']), 50)
        mask = np.ones(150, bool)
        seen = [set(v06._rows(labels, mask, 5, k)[0]) for k in v06.BAG_OFFSETS]
        self.assertEqual(sum(len(s) for s in seen), 150)
        self.assertEqual(len(set().union(*seen)), 150)


class ExclusionTests(unittest.TestCase):
    def test_heldout_labels_cannot_change_either_model(self):
        matrix, names, labels, swap = toy()
        mask = np.ones(len(labels), bool)
        mask[500:700] = False
        changed = labels.copy()
        changed[~mask] = 'X'
        for offset in v06.BAG_OFFSETS:
            for fit, predict in ((v06.fit_three_class, v06.three_class_probability),
                                 (v06.fit_live_team, v06.live_team_probability)):
                first = fit(matrix, names, labels, mask, CONFIG, swap, offset)
                second = fit(matrix, names, changed, mask, CONFIG, swap, offset)
                np.testing.assert_array_equal(predict(first, matrix, names, swap),
                                              predict(second, matrix, names, swap))

    def test_gating_preserves_dead_and_unknown_masks(self):
        from unittest.mock import patch
        from possession.champion import decode_factorized
        rng = np.random.default_rng(42)
        config = dict(step=5, team_switch_cost=6., stoppage_switch_cost=3.,
                      history_samples=0, seed=1, reset_owner_at_stoppage=True)
        arrays = {'ball': np.zeros((300, 2))}
        dead = np.r_[np.full(100, .01), np.full(80, .99), np.full(120, .01)]
        team = rng.normal(size=(300, 2))
        contacts = rng.choice([-1, 0, 1], size=300)
        halves = [(1, 10, 289)]
        with patch('possession.v06.half_video_blocks', return_value=halves), patch('possession.champion.half_video_blocks', return_value=halves):
            reference = decode_factorized(arrays, config, dead, team)[0]
            plain = v06.decode(arrays, config, dead, team)
            gated = v06.decode(arrays, config, dead, team, contacts)
        np.testing.assert_array_equal(reference, plain)
        for state in ('D', 'X'):
            np.testing.assert_array_equal(reference == state, gated == state)


if __name__ == '__main__':
    unittest.main()
