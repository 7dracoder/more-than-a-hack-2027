"""The published anchor must be structural: bounded, idempotent and surgical."""
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from possession.anchors import (GRADED_BLOCKS, SECOND_HALF_KICKOFF,  # noqa: E402
                                apply_published_dead, published_dead_intervals)


class AnchorTests(unittest.TestCase):
    def test_interval_is_inside_the_graded_second_half_and_stops_before_the_anchor(self):
        (start, stop), = published_dead_intervals()
        self.assertEqual(start, GRADED_BLOCKS[1][0])
        self.assertLess(stop, SECOND_HALF_KICKOFF)
        self.assertLessEqual(stop, GRADED_BLOCKS[1][1])
        self.assertGreater(stop, start)

    def test_every_frame_in_the_interval_becomes_dead_and_nothing_else_moves(self):
        labels = np.array(['H', 'A', 'D'] * 70000, dtype='<U1')
        out, changed = apply_published_dead(labels)
        (start, stop), = published_dead_intervals()
        self.assertTrue((out[start:stop] == 'D').all())
        self.assertTrue(np.array_equal(out[:start], labels[:start]))
        self.assertTrue(np.array_equal(out[stop:], labels[stop:]))
        self.assertEqual(changed, int((labels[start:stop] != 'D').sum()))

    def test_second_application_is_a_no_op(self):
        labels = np.array(['H'] * 199319, dtype='<U1')
        once, _ = apply_published_dead(labels)
        twice, changed = apply_published_dead(once)
        self.assertEqual(changed, 0)
        self.assertTrue(np.array_equal(once, twice))

    def test_anchor_is_not_silently_moved_by_a_guard(self):
        (start, stop), = published_dead_intervals(guard_seconds=0.)
        self.assertEqual(stop, SECOND_HALF_KICKOFF)
        with self.assertRaises(ValueError):
            published_dead_intervals(guard_seconds=1e6)

    def test_labels_stay_inside_the_submission_alphabet(self):
        labels = np.array(['A'] * 199319, dtype='<U1')
        out, _ = apply_published_dead(labels)
        self.assertEqual(set(np.unique(out)), {'A', 'D'})


if __name__ == '__main__':
    unittest.main()
