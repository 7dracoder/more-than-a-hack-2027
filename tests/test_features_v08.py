"""Physical/contact features retain the data-isolation and symmetry contracts."""
import unittest
import numpy as np
from possession.features_v08 import build_features, contact_context, velocity
from possession.external import swap_features
from scripts.prepare_external_v08 import corrupt_tracking


class ContactFeatureTests(unittest.TestCase):
    def test_frozen_reference_stays_on_original_grid_for_finer_inputs(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from scripts.run_local import reference_score
        rng=np.random.default_rng(19)
        names=['home_a','away_a','ball_seen']
        x=rng.normal(size=(60,3)).astype(np.float32)
        classifier=HistGradientBoostingClassifier(max_iter=3,min_samples_leaf=2,early_stopping=False).fit(x,np.tile([0,1,2],20))
        truth=np.repeat(np.tile(['H','A','D'],20),2)
        coarse=[dict(x=x,truth=truth)]
        fine=[dict(x=np.repeat(x,2,axis=0),truth=truth)]
        a=reference_score(dict(sample_step=2),classifier,coarse,names,names)
        b=reference_score(dict(sample_step=1,baseline_sample_step=2),classifier,fine,names,names)
        self.assertEqual(a,b)

    def test_fractional_focal_loss_has_finite_confident_and_ignored_gradients(self):
        import torch
        from possession.temporal import frame_loss
        logits = torch.tensor([[[100., 0., 0.], [-100., 0., 0.], [-100., 0., 0.]]], requires_grad=True)
        labels = torch.tensor([[0, -100, 2]])
        loss = frame_loss(logits, labels, .5)
        self.assertEqual(float(loss[0,1].detach()), 0.)
        self.assertTrue(torch.isfinite(loss).all())
        loss.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad[:,:,1] == 0).all())

    def test_development_only_split_cannot_claim_an_independent_test(self):
        from scripts.run_local import validate_splits, evaluate_test
        from pathlib import Path
        config = dict(train_matches=['OTHER1','OTHER2'], validation_matches=['OTHER3'], test_matches=[], development_only=True)
        validate_splits(config)
        with self.assertRaisesRegex(ValueError, 'no independent test'):
            evaluate_test(config, Path('unused'), 'cpu')
        with self.assertRaises(ValueError):
            validate_splits(dict(config, development_only=False))
        with self.assertRaises(ValueError):
            validate_splits(dict(config, test_matches=['OTHER4']))

    def test_classifier_adapter_preserves_probability_and_team_symmetry(self):
        import torch
        from sklearn.ensemble import HistGradientBoostingClassifier
        from possession.temporal import ClassifierAdapter, symmetric_probability
        rng = np.random.default_rng(5)
        x = rng.normal(size=(60, 3)).astype(np.float32)
        classifier = HistGradientBoostingClassifier(max_iter=3, min_samples_leaf=2, early_stopping=False).fit(x, np.tile([0,1,2], 20))
        names = ['home_a', 'away_a', 'ball_seen']
        p = symmetric_probability(ClassifierAdapter(classifier), x, names, np.zeros(3), np.ones(3), 'cpu')
        expected = (classifier.predict_proba(x)+classifier.predict_proba(swap_features(x, names))[:, [1,0,2]])/2
        np.testing.assert_allclose(p, expected, atol=1e-6)
        q = symmetric_probability(ClassifierAdapter(classifier), swap_features(x, names), names, np.zeros(3), np.ones(3), 'cpu')
        np.testing.assert_allclose(p, q[:, [1,0,2]], atol=1e-6)
        from scripts.run_local import reference_probability
        extended = np.column_stack([x,np.ones(len(x))])
        projected = reference_probability(classifier,extended,names+['new_evidence'],names)
        np.testing.assert_allclose(projected,expected,atol=1e-6)
        with self.assertRaisesRegex(ValueError,'not a subset'):
            reference_probability(classifier,extended,names+['new_evidence'],['missing'])

    def test_hybrid_uses_full_sequence_and_produces_finite_logits(self):
        import torch
        from possession.temporal import make_model
        torch.set_num_threads(2)
        model = make_model(7, dict(architecture='hybrid', hidden_channels=8, dropout=0.))
        model.eval()
        with torch.inference_mode():
            result = model(torch.zeros(2, 7, 60))
        self.assertEqual(tuple(result.shape), (2, 3, 60))
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(model.full_sequence)

    def test_future_contact_does_not_wrap_or_invent_missing_observations(self):
        distance = np.full((100, 2), np.inf)
        distance[20] = [0.5, 4.]
        distance[70] = [4., 0.5]
        columns = contact_context(distance, 1.)
        self.assertEqual(columns['home_future_owner'][0], 1)
        self.assertEqual(columns['home_past_owner'][0], 0)
        self.assertEqual(columns['away_future_owner'][80], 0)
        self.assertEqual(columns['away_past_owner'][80], 1)
        empty = contact_context(np.full((100, 2), np.inf), 1.)
        self.assertTrue((empty['home_future_owner'] == 0).all())
        self.assertTrue((empty['away_past_owner'] == 0).all())

    def test_velocity_does_not_bridge_missing_coordinates(self):
        x = np.column_stack([np.arange(30)/25., np.zeros(30)]).astype(np.float32)
        x[10] = np.nan
        result = velocity(x)
        np.testing.assert_allclose(result[3:7, 0], 1., atol=1e-6)
        self.assertTrue(np.isnan(result[8:13]).all())

    def test_features_are_team_and_slot_symmetric_at_finer_resolution(self):
        rng = np.random.default_rng(2028)
        players = rng.normal(size=(100, 8, 2)).cumsum(0)*.02+[50, 34]
        ball = rng.normal(size=(100, 2)).cumsum(0)*.05+[50, 34]
        teams = np.repeat([0, 1], 4)
        a, names = build_features(players, teams, ball, (105, 68), step=2)
        b, other = build_features(players, 1-teams, ball, (105, 68), step=2)
        self.assertEqual(a.shape[0], 50)
        self.assertEqual(names, other)
        np.testing.assert_allclose(swap_features(a, names), b, atol=1e-6)
        order = rng.permutation(8)
        c, _ = build_features(players[:, order], teams[order], ball, (105, 68), step=2)
        np.testing.assert_allclose(a, c, atol=1e-6)

    def test_corruption_is_reproducible_and_preserves_source_coordinates(self):
        players = np.ones((2000, 8, 2), np.float32)
        ball = np.ones((2000, 2), np.float32)
        a, b = corrupt_tracking(players, ball, 5)
        c, d = corrupt_tracking(players, ball, 5)
        np.testing.assert_array_equal(a, c); np.testing.assert_array_equal(b, d)
        self.assertTrue((players == 1).all()); self.assertTrue((ball == 1).all())
        self.assertTrue(np.isnan(a).any()); self.assertTrue(np.isnan(b).any())

    def test_missing_ball_creates_no_contact_evidence(self):
        x, names = build_features(np.ones((100, 8, 2)), np.repeat([0, 1], 4), np.full((100, 2), np.nan), (105, 68))
        self.assertTrue(np.isfinite(x).all())
        for name in names:
            if '_owner_' in name and not name.endswith('_known'):
                self.assertTrue((x[:, names.index(name)] == 0).all())


if __name__ == '__main__':
    unittest.main()
