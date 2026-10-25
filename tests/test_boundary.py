import unittest
import itertools
import numpy as np
import torch
from possession.boundary import BoundaryHybrid, boundary_loss


class BoundaryTests(unittest.TestCase):
    def test_dynamic_viterbi_agrees_with_exhaustive_path_objective(self):
        from possession.temporal import viterbi_labels
        rng = np.random.default_rng(8)
        p = rng.dirichlet([1,1,1],size=6)
        q = np.array([.1,.3,.8,.05,.5,.9])
        def objective(path,bias):
            result = sum(np.log(p[t,label]) for t,label in enumerate(path))
            for t in range(1,len(path)):
                if path[t] != path[t-1]:
                    result -= 1.5 if 2 in (path[t-1],path[t]) else 2.
                    result += .7*(np.log(q[t]/(1-q[t]))+bias)
            return result
        for bias in (0.,2.):
            exact = max(itertools.product(range(3),repeat=6),key=lambda path:objective(path,bias))
            actual = viterbi_labels(np.column_stack([p,q]),2.,1.5,.7,bias)
            np.testing.assert_array_equal(actual,np.array(['H','A','D'])[list(exact)])
        np.testing.assert_array_equal(viterbi_labels(p,2.,1.5),viterbi_labels(np.column_stack([p,q]),2.,1.5,0.))

    def test_state_and_change_evidence_are_both_team_symmetric(self):
        from possession.temporal import symmetric_probability
        from possession.external import swap_features
        torch.set_num_threads(2); torch.manual_seed(8)
        model = BoundaryHybrid(3,8,0.)
        names = ['home_a','away_a','ball_seen']
        x = np.random.default_rng(8).normal(size=(60,3)).astype(np.float32)
        a = symmetric_probability(model,x,names,np.zeros(3),np.ones(3),'cpu')
        b = symmetric_probability(model,swap_features(x,names),names,np.zeros(3),np.ones(3),'cpu')
        self.assertEqual(a.shape,(60,4))
        np.testing.assert_allclose(a,b[:,[1,0,2,3]],atol=1e-6)
        np.testing.assert_allclose(a[:,:3].sum(1),1.,atol=1e-6)

    def test_unknown_labels_and_crop_edges_are_not_transition_targets(self):
        logits = torch.zeros(1,6,requires_grad=True)
        labels = torch.tensor([[0,0,-100,1,1,2]])
        loss = boundary_loss(logits,labels,10.)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad[0,0]),0.)
        self.assertEqual(float(logits.grad[0,2]),0.)
        self.assertEqual(float(logits.grad[0,3]),0.)
        self.assertGreater(float(logits.grad[0,1]),0.)
        self.assertLess(float(logits.grad[0,5]),0.)

    def test_calibration_removes_positive_class_weight_from_odds(self):
        model = BoundaryHybrid(3,8,0.,10.)
        value = torch.tensor([np.log(10.)],dtype=torch.float32)
        self.assertAlmostEqual(float(model.calibrated_boundary(value)[0]),.5,places=6)

    def test_both_heads_share_context_and_remain_finite(self):
        torch.set_num_threads(2)
        model = BoundaryHybrid(3,8,0.)
        model.eval()
        values = torch.zeros(2,3,30)
        logits,boundary = model.forward_with_boundary(values)
        self.assertEqual(tuple(logits.shape),(2,3,30))
        self.assertEqual(tuple(boundary.shape),(2,30))
        self.assertTrue(torch.isfinite(logits).all())
        torch.testing.assert_close(logits,model(values))
