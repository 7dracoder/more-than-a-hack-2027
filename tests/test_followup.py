"""Tests for leakage controls, checkpoint recovery and query selection isolation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from possession.validation import grouped_split, query_benchmark
from possession.quality import jump_flags
from possession.stoppage import scores
from possession.calibrate import fit
from scripts.run_followup import Checkpoints, ROOT
from test_possession import toy, CONFIG


class SplitTests(unittest.TestCase):
    def test_no_group_is_split_and_context_is_embargoed(self):
        labels = np.array(list('XXHHHAAADDDHHHHHAAADDDXXX'))
        train, test, provenance = grouped_split(labels, target=10, embargo=4, context=2)
        edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
        for a, b in zip(edges[:-1], edges[1:]):
            self.assertIn(int(train[a:b].sum()), (0, b-a))
            self.assertIn(int(test[a:b].sum()), (0, b-a))
            self.assertFalse(train[a:b].any() and test[a:b].any())
        self.assertGreaterEqual(np.flatnonzero(test)[0] - np.flatnonzero(train)[-1] - 1, 4)
        self.assertFalse(train[labels == 'X'].any() or test[labels == 'X'].any())
        self.assertEqual(provenance['train_frames'], int(train.sum()))

    def test_invalid_context_rejected(self):
        with self.assertRaises(ValueError):
            grouped_split(np.array(list('HHADDAAHH')), 4, embargo=2, context=2)

    def test_masked_fitting_ignores_excluded_labels(self):
        a = toy()
        labels = np.resize(np.array(['H', 'A', 'D']), 30)
        mask = np.zeros(30, dtype=bool)
        mask[:12] = True
        altered = labels.copy()
        altered[~mask] = 'X'
        with patch('possession.calibrate.half_video_blocks', return_value=[(1, 0, 29)]):
            first = fit(a, CONFIG, labels, 0, 30, train_mask=mask)
            second = fit(a, CONFIG, altered, 0, 30, train_mask=mask)
        self.assertEqual(first, second)
        self.assertEqual(first['n_train_frames'], 12)


class CheckpointTests(unittest.TestCase):
    def temporary(self):
        directory = ROOT / 'data/test-tmp'
        directory.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=directory)

    def test_corruption_and_source_changes_trigger_recompute(self):
        with self.temporary() as temporary:
            calls = []
            def action(directory):
                calls.append(1)
                (directory / 'result.txt').write_text('verified')
            checkpoint = Checkpoints(temporary, 'source-v1', True)
            checkpoint.run('stage', action)
            checkpoint.run('stage', action)
            self.assertEqual(len(calls), 1)
            (Path(temporary) / 'stage/result.txt').write_text('corrupted')
            checkpoint.run('stage', action)
            self.assertEqual(len(calls), 2)
            Checkpoints(temporary, 'source-v2', True).run('stage', action)
            self.assertEqual(len(calls), 3)
            marker = Path(temporary) / 'stage/checkpoint.json'
            marker.write_text('{truncated')
            (Path(temporary) / 'stage/checkpoint.partial.json').write_text('interrupted write')
            Checkpoints(temporary, 'source-v2', True).run('stage', action)
            self.assertEqual(len(calls), 4)
            Checkpoints(temporary, 'source-v2', True).run('stage', action)
            self.assertEqual(len(calls), 4)

    def test_failed_stage_has_no_completion_marker(self):
        with self.temporary() as temporary:
            def failure(directory):
                (directory / 'partial.txt').write_text('partial')
                raise RuntimeError('interrupted')
            with self.assertRaises(RuntimeError):
                Checkpoints(temporary, 'v1', True).run('stage', failure)
            self.assertFalse((Path(temporary) / 'stage/checkpoint.json').exists())


class EvidenceTests(unittest.TestCase):
    def test_query_selection_cannot_see_evaluation_labels(self):
        n = 400
        arrays = {'ball': np.full((n, 2), .5)}
        posterior = np.tile([.4, .35, .25], (n, 1))
        posterior[100:180] = [.5, .2, .3]
        cfg = dict(CONFIG, query_budget=3, step=2, query_spacing_seconds=2)
        truth = np.full(n, 'H')
        changed = np.full(n, 'D')
        first = query_benchmark(posterior, np.zeros(n), arrays, cfg, truth, truth, [(1, 0, n-1)])
        second = query_benchmark(posterior, np.zeros(n), arrays, cfg, changed, truth, [(1, 0, n-1)])
        for policy in first['policies']:
            self.assertEqual(first['policies'][policy]['queries'], second['policies'][policy]['queries'])
            self.assertEqual(first['policies'][policy]['queries_count'], 3)

    def test_missing_ball_and_unknown_motion_do_not_create_dead_evidence(self):
        arrays = toy()
        arrays['ball'][:] = np.nan
        arrays['nearest'][:] = np.nan
        arrays['median_speed'][:] = np.nan
        arrays['stationary'][:] = np.nan
        with patch('possession.stoppage.half_video_blocks', return_value=[(1, 0, 29)]):
            output = scores(arrays, CONFIG)
        self.assertTrue(np.isfinite(output).all())
        self.assertTrue((output.argmax(1) != 2).all())

    def test_jump_flags_require_contiguous_observations(self):
        previous = np.array([[.1, .1], [np.nan, np.nan], [.1, .1]])
        current = np.array([[.1, .1], [.9, .9], [.9, .9]])
        flags, _ = jump_flags(previous, current, 3.)
        np.testing.assert_array_equal(flags, [False, False, True])


if __name__ == '__main__':
    unittest.main()
