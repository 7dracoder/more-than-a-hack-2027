"""Invariants for the v0.5 out-of-play hold and contact snapping."""
import unittest
from unittest.mock import patch

import numpy as np

from possession import structure


def ball_track(n=200):
    ball = np.full((n, 3), np.nan)
    ball[:, 0] = .5
    ball[:, 1] = .5
    return ball


class OutOfPlayTests(unittest.TestCase):
    def test_hold_resets_at_halftime(self):
        ball = ball_track()
        ball[90:95, 1] = 1.05
        ball[95:110, :2] = np.nan
        with patch('possession.structure.half_video_blocks', return_value=[(1, 0, 99), (2, 100, 199)]):
            hard, soft = structure.out_of_play_hold(ball)
        self.assertTrue(soft[95:100].all())
        self.assertFalse(hard[100:].any() or soft[100:].any())

    def test_single_out_observation_hold_is_capped(self):
        ball = ball_track()
        ball[50, 1] = 1.05
        ball[51:, :2] = np.nan
        with patch('possession.structure.HOLD_CAP_FRAMES', 20):
            hard, soft = structure.out_of_play_hold(ball)
        self.assertEqual(int((hard | soft).sum()), 20)
        self.assertFalse(soft[70:].any())

    def setUp(self):
        self.blocks = patch('possession.structure.half_video_blocks', return_value=[(1, 0, 199)])
        self.blocks.start()

    def tearDown(self):
        self.blocks.stop()

    def test_missing_ball_never_starts_a_hold(self):
        ball = np.full((200, 3), np.nan)
        hard, soft = structure.out_of_play_hold(ball)
        self.assertFalse(hard.any() or soft.any())

    def test_hold_lasts_until_ball_inside_and_kicked(self):
        ball = ball_track()
        ball[50:60, 1] = 1.05            # 3.4 m beyond the touchline
        ball[60:100, :2] = np.nan        # not observed
        ball[100:120, 1] = .5            # placed inside, still
        ball[120:, 0] = .5 + np.arange(80) * .003   # kicked: ~7.9 m/s
        hard, soft = structure.out_of_play_hold(ball)
        self.assertTrue(hard[50:60].all())
        self.assertTrue(soft[60:120].all())
        self.assertFalse(hard[120:].any() or soft[121:].any())

    def test_within_margin_is_not_out(self):
        ball = ball_track()
        ball[50:60, 1] = 1. + .5 / 68    # 0.5 m beyond, inside the 1 m tolerance
        hard, soft = structure.out_of_play_hold(ball)
        self.assertFalse(hard.any() or soft.any())

    def test_prior_only_raises_dead_probability(self):
        p = np.linspace(.01, .99, 200)
        hard = np.zeros(200, bool); soft = np.zeros(200, bool)
        hard[:10] = True; soft[10:50] = True
        q = structure.apply_dead_prior(p, hard, soft)
        self.assertTrue((q >= p - 1e-12).all())
        np.testing.assert_allclose(q[50:], p[50:].clip(1e-6, 1 - 1e-6))
        self.assertTrue((q[:10] >= structure.HARD_DEAD_PROBABILITY).all())


class SnapTests(unittest.TestCase):
    def test_neighbouring_moves_cannot_cross_and_erase_short_possession(self):
        labels = np.array(list('HHHHHAAAAAHHHHH'))
        contacts = np.zeros(15, np.int8)
        contacts[9] = -1; contacts[6] = 1
        out, moved = structure.snap_transitions(labels, contacts, window=5)
        self.assertEqual(''.join(out), 'HHHHHHHHHAHHHHH')
        self.assertEqual(moved, 1)

    def test_preserves_ordered_spells_dead_and_ungraded_frames_on_short_runs(self):
        rng = np.random.default_rng(23)
        runs = lambda x: x[np.r_[True, x[1:] != x[:-1]]].tolist()
        for _ in range(1000):
            labels = np.repeat(rng.choice(list('HADX'), 20), rng.integers(1, 10, 20))
            contacts = rng.choice([-1, 0, 1], len(labels)).astype(np.int8)
            out, _ = structure.snap_transitions(labels, contacts)
            self.assertEqual(runs(out), runs(labels))
            np.testing.assert_array_equal(out == 'D', labels == 'D')
            np.testing.assert_array_equal(out == 'X', labels == 'X')
            exchange = lambda x: np.where(x == 'H', 'A', np.where(x == 'A', 'H', x))
            exchanged, _ = structure.snap_transitions(exchange(labels), -contacts)
            np.testing.assert_array_equal(exchanged, exchange(out))

    def test_moves_change_to_new_team_contact(self):
        labels = np.array(list('HHHHHHHHHHAAAAAAAAAA'))
        contacts = np.zeros(20, np.int8); contacts[13] = -1
        out, moved = structure.snap_transitions(labels, contacts, window=5)
        self.assertEqual(''.join(out), 'HHHHHHHHHHHHHAAAAAAA')
        self.assertEqual(moved, 1)

    def test_never_crosses_dead_ball_or_other_change(self):
        labels = np.array(list('HHHHDDAAAHHAAHHH'))
        contacts = np.zeros(16, np.int8)
        contacts[2] = -1    # away contact before the stoppage: must not pull A across D
        contacts[11] = 1    # home contact on the previous change: moving there would delete the A spell
        out, moved = structure.snap_transitions(labels, contacts, window=5)
        np.testing.assert_array_equal(out, labels)
        self.assertEqual(moved, 0)

    def test_preserves_counts_of_spells(self):
        rng = np.random.default_rng(0)
        labels = np.repeat(rng.choice(list('HA'), 30), rng.integers(10, 40, 30))
        contacts = rng.choice([-1, 0, 0, 0, 1], len(labels)).astype(np.int8)
        out, _ = structure.snap_transitions(labels, contacts)
        count = lambda x: int((x[1:] != x[:-1]).sum())
        self.assertEqual(count(out), count(labels))
        self.assertEqual(set(np.flatnonzero(out == 'D')), set(np.flatnonzero(labels == 'D')))


if __name__ == '__main__':
    unittest.main()
