import unittest
import numpy as np
import torch
from possession.temporal import make_model,symmetric_probability


class BoundaryTCNTests(unittest.TestCase):
    def test_bounded_overlap_matches_whole_sequence_for_both_heads(self):
        torch.set_num_threads(2);torch.manual_seed(28)
        model=make_model(3,dict(architecture='boundary_tcn',hidden_channels=8,dilations=[1,2,4],dropout=0.))
        self.assertFalse(model.full_sequence)
        self.assertEqual(model.receptive_radius,7)
        names=['home_a','away_a','ball_seen']
        x=np.random.default_rng(28).normal(size=(151,3)).astype(np.float32)
        whole=symmetric_probability(model,x,names,np.zeros(3),np.ones(3),'cpu',chunk=151)
        chunked=symmetric_probability(model,x,names,np.zeros(3),np.ones(3),'cpu',chunk=17)
        self.assertEqual(whole.shape,(151,4))
        np.testing.assert_allclose(chunked,whole,atol=1e-6)
