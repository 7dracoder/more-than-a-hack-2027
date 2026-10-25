import copy
import unittest
import numpy as np
from possession.sampling import short_spell_anchors,sample_crop_start


class ShortSpellSamplingTests(unittest.TestCase):
    def test_only_complete_known_team_spells_supply_anchors(self):
        labels=np.array([0,0,1,1,2,0,0,0,-100,1,1,0,0,2])
        np.testing.assert_array_equal(short_spell_anchors(labels,2),[2,11])
        np.testing.assert_array_equal(short_spell_anchors(np.zeros(20,int),20),[])

    def test_disabled_sampling_preserves_uniform_crop_and_rng_exactly(self):
        a=np.random.default_rng(91);b=np.random.default_rng(91)
        actual=[sample_crop_start(a,1000,100,np.array([500]),0.) for _ in range(100)]
        expected=[int(b.integers(901)) for _ in range(100)]
        self.assertEqual(actual,expected)
        self.assertEqual(a.bit_generator.state,b.bit_generator.state)

    def test_anchor_position_varies_within_crop_and_resume_is_exact(self):
        rng=np.random.default_rng(92)
        starts=[sample_crop_start(rng,1000,100,np.array([500]),1.) for _ in range(100)]
        self.assertTrue(all(a<=500<a+100 for a in starts))
        self.assertGreater(len(set(starts)),20)
        state=copy.deepcopy(rng.bit_generator.state)
        expected=[sample_crop_start(rng,1000,100,np.array([20,500,990]),.5) for _ in range(100)]
        other=np.random.default_rng();other.bit_generator.state=state
        actual=[sample_crop_start(other,1000,100,np.array([20,500,990]),.5) for _ in range(100)]
        self.assertEqual(expected,actual)

    def test_no_anchors_falls_back_to_uniform_and_bounds_are_enforced(self):
        rng=np.random.default_rng(93)
        self.assertEqual(sample_crop_start(rng,100,100,np.array([]),1.),0)
        with self.assertRaises(ValueError):sample_crop_start(rng,10,11,np.array([]),1.)
        with self.assertRaises(ValueError):sample_crop_start(rng,10,4,np.array([10]),1.)
