import unittest
import numpy as np
from possession.features_v08 import build_features as previous_features
from possession.features_v09 import build_features,shift
from possession.external import swap_features


class PhysicalTouchTests(unittest.TestCase):
    def test_prefix_matches_frozen_previous_feature_bank(self):
        rng = np.random.default_rng(9)
        players = rng.normal(size=(80,8,2)).cumsum(0)*.01+[50,34]
        ball = rng.normal(size=(80,2)).cumsum(0)*.04+[50,34]
        teams = np.repeat([0,1],4)
        old,names = previous_features(players,teams,ball,(105,68))
        new,extra = build_features(players,teams,ball,(105,68))
        self.assertEqual(extra[:len(names)],names)
        np.testing.assert_array_equal(old,new[:,:len(names)])
        full,full_names=build_features(players,teams,ball,(105,68),step=1)
        self.assertEqual(full_names,extra)
        np.testing.assert_array_equal(full[::2],new)

    def test_team_swap_and_slot_reordering_preserve_physical_evidence(self):
        rng = np.random.default_rng(19)
        players = rng.normal(size=(80,8,2)).cumsum(0)*.01+[50,34]
        ball = rng.normal(size=(80,2)).cumsum(0)*.04+[50,34]
        teams = np.repeat([0,1],4)
        x,names = build_features(players,teams,ball,(105,68))
        y,_ = build_features(players,1-teams,ball,(105,68))
        np.testing.assert_allclose(swap_features(x,names),y,atol=1e-6)
        order = rng.permutation(8)
        z,_ = build_features(players[:,order],teams[order],ball,(105,68))
        np.testing.assert_allclose(x,z,atol=1e-6)

    def test_missing_ball_has_no_invented_turn_or_relative_motion(self):
        x,names = build_features(np.ones((80,8,2)),np.repeat([0,1],4),np.full((80,2),np.nan),(105,68))
        self.assertTrue(np.isfinite(x).all())
        for key in ['ball_acceleration_known','ball_turn2_known','home_relative_motion1_known','away_projected_distance0.2_known']:
            self.assertTrue((x[:,names.index(key)]==0).all())

    def test_shift_never_wraps_opposite_end_of_half(self):
        x = np.arange(10,dtype=np.float32)
        self.assertTrue(np.isnan(shift(x,2)[:2]).all())
        self.assertTrue(np.isnan(shift(x,-2)[-2:]).all())
        np.testing.assert_array_equal(shift(x,2)[2:],x[:-2])
