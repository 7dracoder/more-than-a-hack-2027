import unittest
import numpy as np
from possession.features_v25 import extra_features,build_features
from possession.features_v13 import build_features as base_features
from possession.external import swap_features


class PlayerTrajectoryFeaturesTests(unittest.TestCase):
    def fixture(self):
        t=np.arange(40,dtype=np.float32)
        ball=np.stack([.12*t,np.sin(t/20)],axis=1)
        initial=np.array([[1.,1.],[3.,-2.],[8.,2.],[-2.,1.],[-5.,-1.],[-9.,3.]],np.float32)
        players=initial[None,:,:]+np.stack([.02*t,.01*t],axis=1)[:,None,:]
        return players,np.array([0,0,0,1,1,1]),ball

    def test_trajectory_follows_centre_player_when_distance_rank_changes(self):
        t=np.arange(25,dtype=np.float32);players=np.zeros((25,6,2),np.float32)
        players[:,:,0]=np.array([1.,2.,10.,-10.,-11.,-12.])[None,:]
        players[:,0,0]+=.01*t;players[:,1,0]-=.1*t
        ball=np.zeros((25,2),np.float32);x,names=extra_features(players,np.array([0,0,0,1,1,1]),ball)
        self.assertAlmostEqual(float(x[8,names.index('home_tube1_o8_distance')]),1.16/10,places=6)
        self.assertAlmostEqual(float(x[8,names.index('home_tube1_o0_distance')]),1.08/10,places=6)

    def test_rotation_reflection_and_translation_invariance(self):
        players,teams,ball=self.fixture();expected,names=extra_features(players,teams,ball)
        angle=.71;rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]],np.float32)
        for matrix in (rotation,rotation@np.diag([1.,-1.]).astype(np.float32)):
            actual,other=extra_features(players@matrix+np.array([12.,-7.]),teams,ball@matrix+np.array([12.,-7.]))
            self.assertEqual(names,other);np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)

    def test_permutation_and_team_swap(self):
        players,teams,ball=self.fixture();x,names=extra_features(players,teams,ball)
        order=[5,2,0,4,1,3];permuted,_=extra_features(players[:,order],teams[order],ball)
        np.testing.assert_allclose(x,permuted,atol=1e-7)
        swapped,_=extra_features(players,1-teams,ball)
        np.testing.assert_allclose(swapped,swap_features(x,names),atol=1e-7)

    def test_equal_distance_ties_are_permutation_invariant(self):
        players,teams,ball=self.fixture();players[:,:,1]=0.;players[:,0,0]=ball[:,0]+1.;players[:,1,0]=ball[:,0]-1.
        x,names=extra_features(players,teams,ball);order=[1,0,2,5,4,3]
        actual,_=extra_features(players[:,order],teams[order],ball)
        np.testing.assert_allclose(actual,x,rtol=1e-6,atol=1e-7)

    def test_gaps_and_impossible_player_jumps_are_not_bridged(self):
        players,teams,ball=self.fixture();players[:,0]=ball+np.array([.5,0.]);players[11,0]=np.nan
        x,names=extra_features(players,teams,ball)
        self.assertEqual(float(x[8,names.index('home_tube1_o8_known')]),0.)
        players,teams,ball=self.fixture();players[:,0]=ball+np.array([.5,0.]);players[11:13,0,0]+=20.
        x,names=extra_features(players,teams,ball)
        self.assertEqual(float(x[8,names.index('home_tube1_o8_known')]),0.)

    def test_empty_observations_are_finite_and_sampling_is_exact(self):
        players,teams,ball=self.fixture();full,names=extra_features(players,teams,ball)
        small,other=extra_features(players,teams,ball,step=2)
        self.assertEqual(len(names),180);self.assertEqual(names,other)
        np.testing.assert_array_equal(small,full[::2])
        missing,_=extra_features(np.full_like(players,np.nan),teams,np.full_like(ball,np.nan))
        np.testing.assert_array_equal(missing,np.zeros_like(missing))

    def test_original_feature_prefix_is_exact(self):
        players,teams,ball=self.fixture();pitch=(105.,68.)
        old,names=base_features(players,teams,ball,pitch,step=1)
        new,new_names=build_features(players,teams,ball,pitch)
        self.assertEqual(new_names[:len(names)],names);np.testing.assert_array_equal(new[:,:old.shape[1]],old)
