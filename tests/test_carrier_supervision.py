import unittest
import numpy as np
from possession.carrier_supervision import carrier_targets


class CarrierSupervisionTests(unittest.TestCase):
    def fixture(self):
        players=np.array([[[.2,0],[.8,0],[.5,0],[1.,0]]],np.float32)
        return players,np.array([0,0,1,1]),np.zeros((1,2),np.float32)

    def test_carrier_rank_is_a_label_and_not_always_the_nearest_player(self):
        p,t,b=self.fixture()
        np.testing.assert_array_equal(carrier_targets(p,t,b,np.array([1]),np.array(['H'])),[1])
        np.testing.assert_array_equal(carrier_targets(p,t,b,np.array([2]),np.array(['A'])),[0])

    def test_unknown_wrong_team_missing_and_tied_candidates_stay_unknown(self):
        p,t,b=self.fixture()
        for carrier,label in ((-1,'H'),(2,'H'),(1,'X')):
            np.testing.assert_array_equal(carrier_targets(p,t,b,np.array([carrier]),np.array([label])),[-100])
        p[0,0]=p[0,1]
        np.testing.assert_array_equal(carrier_targets(p,t,b,np.array([1]),np.array(['H'])),[-100])
        b[:]=np.nan
        np.testing.assert_array_equal(carrier_targets(p,t,b,np.array([1]),np.array(['H'])),[-100])

    def test_physical_slot_permutation_does_not_change_the_target_rank(self):
        p,t,b=self.fixture();permutation=[3,1,0,2]
        a=carrier_targets(p,t,b,np.array([1]),np.array(['H']))
        c=carrier_targets(p[:,permutation],t[permutation],b,np.array([permutation.index(1)]),np.array(['H']))
        np.testing.assert_array_equal(a,c)
