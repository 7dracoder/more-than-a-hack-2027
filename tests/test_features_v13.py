import unittest
import numpy as np
from possession.features_v13 import extra_features
from possession.external import swap_features


class FineMotionFeaturesTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(201)
        self.players=np.cumsum(rng.normal(size=(101,8,2))*.015,axis=0).astype(np.float32)+20
        self.ball=np.cumsum(rng.normal(size=(101,2))*.035,axis=0).astype(np.float32)+20
        self.teams=np.array([0]*4+[1]*4)

    def test_team_swap_and_player_permutation(self):
        x,names=extra_features(self.players,self.teams,self.ball)
        swapped,other=extra_features(self.players,1-self.teams,self.ball)
        self.assertEqual(names,other)
        np.testing.assert_allclose(swapped,swap_features(x,names),atol=1e-6)
        order=np.array([4,1,7,3,0,6,2,5])
        permuted,_=extra_features(self.players[:,order],self.teams[order],self.ball)
        np.testing.assert_allclose(x,permuted,atol=1e-6)

    def test_gaps_are_finite_and_time_grid_is_exact(self):
        self.ball[40:50]=np.nan
        x,names=extra_features(self.players,self.teams,self.ball)
        coarse,other=extra_features(self.players,self.teams,self.ball,step=2)
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_array_equal(x[::2],coarse)
        self.assertEqual(names,other)
        np.testing.assert_array_equal(x[44],np.zeros(x.shape[1]))

    def test_coincident_players_do_not_break_slot_invariance(self):
        self.players[:,1]=self.players[:,0]
        x,_=extra_features(self.players,self.teams,self.ball)
        order=np.array([1,0,2,3,4,5,6,7])
        y,_=extra_features(self.players[:,order],self.teams[order],self.ball)
        np.testing.assert_array_equal(x,y)
