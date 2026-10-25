import unittest
import numpy as np
from evaluation import possession_metrics as pm
from possession.transition_repair import (Edit, PHYSICAL, propose, edit_features, choose_edits,
                                         apply_edits, fast_spell_score, supervision)
from possession.external import swap_features
from scripts.train_repair_references import validate_plan


CONFIG = dict(max_island_frames=25, max_shift_frames=12, peak_separation=8,
              minimum_alternative_probability=.15, insert_lengths=[3, 6, 12, 25])


class TransitionRepairTests(unittest.TestCase):
    def test_fast_metric_matches_official_with_stoppages_unknown_and_abstentions(self):
        rng = np.random.default_rng(13)
        for _ in range(50):
            truth = np.repeat(rng.integers(0, 4, 25), rng.integers(1, 20, 25)).astype(np.int8)
            pred = rng.integers(0, 4, len(truth)).astype(np.int8)
            actual, _ = fast_spell_score(truth, pred)
            graded = pm.valid_runs(truth)
            expected = pm.spell_scores(truth, pm.fill_abstentions(pred, graded), graded)['mean_f1']
            self.assertEqual(actual, expected)

    def test_candidate_oracle_recovers_a_missed_short_possession(self):
        truth = np.zeros(100, np.int8); truth[45:51] = 1
        base = np.zeros(100, np.int8)
        p = np.tile([.9, .09, .01, .01], (100, 1)).astype(np.float32)
        p[48, :3] = [.2, .79, .01]
        edits = propose(p, base, CONFIG)
        y = supervision(truth, base, edits)
        self.assertGreater(y.max(), 0)
        self.assertTrue(any(e.kind == 'insert' and e.start == 45 and e.stop == 51 for e in edits))

    def test_shifts_and_deletions_are_explicit_and_preserve_dead_unknown(self):
        labels = np.array([2]*5+[0]*20+[1]*4+[0]*20+[3]*5, np.int8)
        p = np.tile([.49, .49, .02, .1], (len(labels), 1))
        edits = propose(p, labels, CONFIG)
        self.assertTrue(any(e.kind == 'delete' for e in edits))
        self.assertTrue(any(e.kind == 'shift' for e in edits))
        for e in edits:
            out = apply_edits(labels, [e])
            np.testing.assert_array_equal(out[labels >= 2], labels[labels >= 2])
        with self.assertRaises(ValueError):
            apply_edits(labels, [Edit(0, 5, 0, 'delete')])

    def test_edit_gain_uses_exact_official_spell_change(self):
        truth = np.array([0]*30+[1]*10+[0]*30)
        base = np.zeros(70, np.int8)
        edit = Edit(30, 40, 1, 'insert')
        before, n = fast_spell_score(truth, base)
        after, _ = fast_spell_score(truth, apply_edits(base, [edit]))
        self.assertAlmostEqual(float(supervision(truth, base, [edit])[0]), (after-before)*n/2., places=6)
        truth[32] = 3
        self.assertTrue(np.isnan(supervision(truth, base, [edit])[0]))

    def test_interval_optimizer_beats_greedy_and_never_overlaps(self):
        edits = [Edit(2, 30, 1, 'insert'), Edit(2, 10, 1, 'insert'), Edit(12, 30, 1, 'insert')]
        chosen = choose_edits(edits, [3., 2., 2.])
        self.assertEqual(chosen, sorted(edits[1:]))
        self.assertEqual(choose_edits(edits, [3., 2., 2.], 4.), [])

    def test_team_swap_preserves_proposals_features_and_selected_changes(self):
        rng = np.random.default_rng(10)
        labels = np.array([0]*30+[1]*5+[0]*35)
        p = rng.random((len(labels), 4)).astype(np.float32)
        p[:, :3] /= p[:, :3].sum(1, keepdims=True)
        x = rng.random((len(labels), len(PHYSICAL))).astype(np.float32)
        names = list(PHYSICAL)
        edits = propose(p, labels, CONFIG)
        flipped = propose(p[:, [1, 0, 2, 3]], 1-labels, CONFIG)
        self.assertEqual(flipped, sorted(Edit(e.start, e.stop, 1-e.target, e.kind) for e in edits))
        np.testing.assert_array_equal(edit_features(p, labels, x, names, edits),
            edit_features(p[:, [1, 0, 2, 3]], 1-labels, swap_features(x, names), names, flipped))

    def test_crossfit_plan_requires_exact_disjoint_training_pair_coverage(self):
        plan = dict(training_matches=list('abcdef'), excluded_pairs=[list('ab'), list('cd'), list('ef')])
        validate_plan(plan)
        plan['excluded_pairs'][2] = list('ea')
        with self.assertRaises(ValueError):
            validate_plan(plan)
        plan = dict(training_matches=['J03WPY', *list('abcde')],
                    excluded_pairs=[['J03WPY', 'a'], list('bc'), list('de')])
        with self.assertRaises(ValueError):
            validate_plan(plan)


if __name__ == '__main__':
    unittest.main()
