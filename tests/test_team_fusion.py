"""Invariants for the single predeclared team-only contacts experiment."""
import unittest
from unittest.mock import patch
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

from possession.champion import decode_factorized
from possession.team_fusion import team_only_inputs, decode_team_only
from test_possession import CONFIG
from scripts.run_v04_experiments import Setup, export_candidate


class TeamFusionTests(unittest.TestCase):
    def test_candidate_export_accepts_relative_output_directory(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder)
            relative = root.relative_to(Path.cwd()) / 'experiment'
            fake = SimpleNamespace(output=relative, valid=np.ones(199319, dtype=bool),
                fingerprint='test', provenance={}, last_prediction_audit={},
                predict=lambda group, mask: (np.full(199319, 'X'), 78))
            with patch('scripts.run_v04_experiments.RESULTS_MD', root / 'log.md'):
                export_candidate(fake, 'contacts_team_only')
            self.assertTrue((relative / 'contacts_team_only/candidate/CoreOrigin.csv').exists())
            self.assertIn('contacts_team_only', (root / 'log.md').read_text())

    def test_both_fitted_heads_ignore_excluded_labels(self):
        random = np.random.default_rng(7)
        matrix = random.normal(size=(60, 4))
        labels = np.resize(np.array(['H', 'A', 'D']), 60)
        mask = np.arange(60) >= 30
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs/champion_v031.json').read_text())
        config.update(classifier_iterations=5, classifier_min_samples_leaf=2,
                      classifier_training_stride=1, kickoff_prior_guard_frames=None, history_samples=0)

        def run(truth):
            setup = Setup.__new__(Setup)
            setup.config, setup.truth = config, truth
            setup.arrays = {'ball': np.zeros((60, 2))}
            setup.probability_cache = {}
            def design(group):
                width = 3 if group == 'baseline' else 4
                values = matrix[:, :width]
                def exchange(x, names):
                    swapped = x.copy(); swapped[:, 0] *= -1
                    return swapped
                return values, ['x'+str(i) for i in range(width)], exchange
            setup.matrix = design
            with patch('possession.champion.half_video_blocks', return_value=[(1, 0, 59)]):
                prediction = setup.predict('contacts_team_only', mask)[0]
            self.assertTrue(setup.last_prediction_audit['identical_training_mask'])
            for group in ('baseline', 'contacts'):
                self.assertEqual(setup.probability(group, mask)[2]['n_training_frames'], 30)
            return prediction, [value[0] for value in setup.probability_cache.values()]

        first, probabilities = run(labels.copy())
        labels[:30] = 'X'
        second, changed_probabilities = run(labels)
        np.testing.assert_array_equal(first, second)
        for a, b in zip(probabilities, changed_probabilities):
            np.testing.assert_array_equal(a, b)

    def test_head_sources_are_separate(self):
        baseline = np.array([[.7, .2, .1], [.1, .1, .8]])
        contacts = np.array([[.06, .24, .7], [.36, .04, .6]])
        dead, team = team_only_inputs(baseline, contacts)
        np.testing.assert_array_equal(dead, baseline[:, 2])
        np.testing.assert_allclose(np.exp(team), [[.2, .8], [.9, .1]])
        changed = contacts.copy()
        changed[:, :2] *= 2
        changed[:, 2] = 1 - changed[:, :2].sum(1)
        np.testing.assert_allclose(team_only_inputs(baseline, changed)[1], team)
        swapped = baseline[:, [1, 0, 2]]
        np.testing.assert_array_equal(team_only_inputs(swapped, contacts)[0], dead)

    def test_dead_masks_posteriors_and_sampled_dead_histories_are_unchanged(self):
        baseline = np.tile([.2, .7, .1], (60, 1))
        baseline[20:29] = [.01, .01, .98]
        contacts = np.tile([.95, .01, .04], (60, 1))
        contacts[40:] = [.01, .95, .04]
        arrays = {'ball': np.zeros((60, 2))}
        config = dict(CONFIG, step=1, history_samples=3, reset_owner_at_stoppage=True,
                      kickoff_prior_guard_frames=None, stoppage_switch_cost=1., team_switch_cost=1.)
        p = baseline.clip(1e-6, 1.)
        with patch('possession.champion.half_video_blocks', return_value=[(1, 0, 29), (2, 35, 59)]):
            original = decode_factorized(arrays, config, p[:, 2], np.log(p[:, :2] / p[:, :2].sum(1)[:, None]))
            hybrid = decode_team_only(arrays, config, baseline, contacts)
        np.testing.assert_array_equal(original[0] == 'D', hybrid[0] == 'D')
        np.testing.assert_array_equal(original[0] == 'X', hybrid[0] == 'X')
        np.testing.assert_array_equal(original[1][:, 2], hybrid[1][:, 2])
        np.testing.assert_array_equal(original[2] == 2, hybrid[2] == 2)
        self.assertTrue((original[0] != hybrid[0]).any())

    def test_tiny_baseline_live_mass_does_not_flatten_contact_team_odds(self):
        dead, team = team_only_inputs(np.array([[0., 0., 1.]]), np.array([[.9, .1, 0.]]))
        np.testing.assert_allclose(np.exp(team), [[.9, .1]])
        self.assertEqual(dead[0], 1.)

    def test_team_swap_keeps_dead_evidence_and_exchanges_team_odds(self):
        baseline = np.array([[.2, .5, .3]])
        contacts = np.array([[.6, .1, .3]])
        dead, team = team_only_inputs(baseline, contacts)
        other_dead, other_team = team_only_inputs(baseline[:, [1, 0, 2]], contacts[:, [1, 0, 2]])
        np.testing.assert_array_equal(dead, other_dead)
        np.testing.assert_allclose(team[:, ::-1], other_team)

    def test_bad_probability_inputs_fail(self):
        good = np.array([[.2, .3, .5]])
        for bad in (np.zeros((2, 3)), np.array([[np.nan, .3, .5]]), np.array([[-.1, .6, .5]])):
            with self.assertRaises(ValueError):
                team_only_inputs(good, bad)


if __name__ == '__main__':
    unittest.main()
