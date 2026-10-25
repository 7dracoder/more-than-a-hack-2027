import unittest
import numpy as np
from possession.fast_decode import path,path_reference


class FastDecoderTests(unittest.TestCase):
    def test_random_sequences_and_transition_odds_match_reference_exactly(self):
        rng=np.random.default_rng(419)
        for length in (1,2,3,11,1025,10001):
            emission=rng.normal(size=(length,3))
            for team,dead in ((0.,0.),(0.,2.),(8.,12.)):
                transition=np.array([[0.,-team,-dead],[-team,0.,-dead],[-dead,-dead,0.]])
                for odds in (np.zeros(length),rng.normal(size=length)):
                    np.testing.assert_array_equal(path(emission,transition,odds),path_reference(emission,transition,odds))

    def test_ties_keep_lowest_source_and_final_state(self):
        for epsilon in (0.,1e-15):
            emission=np.zeros((101,3)); emission[::3,1]=epsilon
            transition=np.zeros((3,3)); odds=np.zeros(101)
            np.testing.assert_array_equal(path(emission,transition,odds),path_reference(emission,transition,odds))
