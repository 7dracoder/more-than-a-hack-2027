import unittest
import numpy as np
from possession.features_v19 import extra_features,build_features
from possession.features_v13 import build_features as previous_features
from possession.external import swap_features


class ContactGeometryTests(unittest.TestCase):
    def test_expansion_preserves_previous_normalization_schema_prefix(self):
        players,teams,ball=self.fixture()
        old,names=previous_features(players,teams,ball,(105.,68.),step=1)
        new,expanded=build_features(players,teams,ball,(105.,68.),step=1)
        self.assertEqual(new.shape[1],836);self.assertEqual(len(expanded),len(set(expanded)))
        self.assertEqual(expanded[:len(names)],names)
        np.testing.assert_array_equal(new[:,:old.shape[1]],old)

    def fixture(self):
        time=np.arange(101)[:,None,None]/25.
        positions=np.array([[0.,0.],[1.,2.],[2.,1.],[3.,3.],[4.,2.],[2.,4.]])
        velocities=np.array([[2.,.5],[-1.,2.],[2.,-1.],[-2.,.5],[1.,1.],[-1.,-1.]])
        players=positions[None]+time*velocities[None]
        ball=np.column_stack([np.arange(101)/25.*5.-1.,np.sin(np.arange(101)/15.)])
        return players,np.array([0,0,0,1,1,1]),ball

    def test_between_frame_contact_is_visible_without_sampled_contact(self):
        players=np.zeros((3,2,2),np.float32);players[:,1]=[5.,5.]
        ball=np.array([[-1.,0.],[1.,0.],[3.,0.]],np.float32)
        x,names=extra_features(players,np.array([0,1]),ball)
        self.assertEqual(x[0,names.index('home_segment_future_distance')],0.)
        self.assertEqual(x[0,names.index('home_segment_future_distance_known')],1.)
        self.assertEqual(x[1,names.index('home_segment_past_distance')],0.)
        self.assertEqual(x[-1,names.index('home_segment_future_distance_known')],0.)
        self.assertEqual(np.linalg.norm(ball[0]-players[0,0]),1.)

    def test_team_swap_and_slot_permutation(self):
        players,teams,ball=self.fixture();x,names=extra_features(players,teams,ball)
        swapped,_=extra_features(players,1-teams,ball)
        np.testing.assert_allclose(swapped,swap_features(x,names),atol=1e-6)
        order=[5,0,3,2,1,4];permuted,_=extra_features(players[:,order],teams[order],ball)
        np.testing.assert_allclose(x,permuted,atol=1e-6)

    def test_rigid_motion_preserves_geometry(self):
        players,teams,ball=self.fixture();x,_=extra_features(players,teams,ball)
        rotation=np.array([[0.,-1.],[1.,0.]])
        moved,_=extra_features(players@rotation+50.,teams,ball@rotation+50.)
        np.testing.assert_allclose(x,moved,atol=1e-6)

    def test_missing_gap_is_not_bridged_and_time_grid_is_exact(self):
        players,teams,ball=self.fixture();ball[40:50]=np.nan
        x,names=extra_features(players,teams,ball);coarse,other=extra_features(players,teams,ball,step=2)
        self.assertTrue(np.isfinite(x).all());self.assertEqual(x.shape[1],56)
        np.testing.assert_array_equal(x[::2],coarse);self.assertEqual(names,other)
        np.testing.assert_array_equal(x[44],np.zeros(56))
        self.assertEqual(x[39,names.index('home_segment_future_distance_known')],0.)
