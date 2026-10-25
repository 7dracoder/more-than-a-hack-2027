import unittest
import numpy as np
from possession.view_sampling import sample_training_view,validate_view_sampling


class ViewSamplingTests(unittest.TestCase):
    def test_default_reproduces_original_values_and_rng_stream(self):
        a=np.random.default_rng(19);b=np.random.default_rng(19)
        self.assertEqual([sample_training_view(a,24,12) for _ in range(50)],[int(b.integers(24)) for _ in range(50)])
        self.assertEqual(a.random(),b.random())

    def test_native_only_and_corrupt_only_never_select_opposite_view(self):
        rng=np.random.default_rng(23)
        native=[sample_training_view(rng,24,12,1.) for _ in range(200)]
        corrupt=[sample_training_view(rng,24,12,0.) for _ in range(200)]
        self.assertEqual(set(native),set(range(12)))
        self.assertEqual(set(corrupt),set(range(12,24)))

    def test_requested_frequency_and_each_half_remain_represented(self):
        rng=np.random.default_rng(29)
        indices=np.array([sample_training_view(rng,24,12,.9) for _ in range(20000)])
        self.assertLess(abs(float((indices<12).mean())-.9),.015)
        self.assertEqual(set(indices.tolist()),set(range(24)))
        counts=np.bincount(indices[indices<12],minlength=12)
        self.assertLess(float(counts.max()/counts.min()),1.2)

    def test_invalid_probability_or_unpaired_banks_are_rejected(self):
        for probability in (-1.,2.,float('nan'),float('inf'),'native'):
            with self.assertRaises(ValueError):validate_view_sampling(24,12,probability)
        for total,native in ((12,12),(24,0),(23,12)):
            with self.assertRaises(ValueError):validate_view_sampling(total,native,.9)

    def test_saved_rng_resumes_identical_view_and_crop_draws(self):
        a=np.random.default_rng(31)
        for _ in range(10):sample_training_view(a,24,12,.9)
        b=np.random.default_rng();b.bit_generator.state=a.bit_generator.state
        for _ in range(20):
            self.assertEqual(sample_training_view(a,24,12,.9),sample_training_view(b,24,12,.9))
            self.assertEqual(a.integers(50000),b.integers(50000))
