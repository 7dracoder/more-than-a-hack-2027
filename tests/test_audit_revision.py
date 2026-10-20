"""Regression coverage for the audit findings, including the generated viewer."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from possession.champion import owner_with_restarts, fixed_time_folds
from possession.data import load_cached_arrays, sha256
from possession.metric_features import design
from possession.model import choose_queries
from possession.review import build_review
from test_possession import CONFIG, toy


class RevisionTests(unittest.TestCase):
    def test_fixed_time_folds_partition_window_and_embargo_training(self):
        valid=np.zeros(100,dtype=bool);valid[10:90]=True
        coverage=np.zeros(100,dtype=int)
        for train,test,info in fixed_time_folds(valid,4,3):
            coverage+=test
            self.assertEqual(int(test.sum()),20)
            self.assertFalse(train[max(0,info['test_first']-3):info['test_stop']+3].any())
        np.testing.assert_array_equal(coverage,valid.astype(int))

    def test_dead_ball_evidence_cannot_change_live_owner(self):
        states = np.r_[np.zeros(10), np.ones(10), np.zeros(10)]
        scores = np.r_[np.tile([3., -3.], (10, 1)), np.tile([100., -100.], (10, 1)), np.tile([-.1, .1], (10, 1))]
        a, pa = owner_with_restarts(scores, states, 6.)
        scores[10:20] *= -1
        b, pb = owner_with_restarts(scores, states, 6.)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(pa, pb)
        self.assertTrue((a[:10] == 0).all())
        self.assertTrue((a[20:] == 1).all())
        np.testing.assert_allclose(pa[10:20], .5)

    def test_restarted_owner_samples_are_reproducible_and_all_dead_is_valid(self):
        scores = np.zeros((20, 2)); states = np.r_[np.zeros(5), np.ones(5), np.zeros(10)]
        a = owner_with_restarts(scores, states, 2., seed=9, sample=True)[0]
        b = owner_with_restarts(scores, states, 2., seed=9, sample=True)[0]
        np.testing.assert_array_equal(a, b)
        _, p = owner_with_restarts(scores, np.ones(20), 2.)
        np.testing.assert_allclose(p, .5)

    def test_missing_motion_is_distinguishable_from_zero_motion(self):
        a = toy()
        a.update(player_speed_quantiles=np.zeros((30, 3)), running_fraction=np.zeros(30), team_speed=np.zeros((30, 2)),
                 phase_support=np.zeros(30), phase_conflict=np.zeros(30), restart_owner=np.zeros(30), restart_types=np.zeros((30, 5)))
        b = {k:v.copy() for k,v in a.items()}
        for key in ('player_speed_quantiles', 'stationary', 'running_fraction', 'team_speed'):
            b[key][:] = np.nan
        config = dict(CONFIG, feature_windows_frames=[1], include_observation_masks=True)
        with patch('possession.metric_features.half_video_blocks', return_value=[(1, 0, 29)]):
            x, names = design(a, config); y, _ = design(b, config)
        np.testing.assert_allclose(x[:, names.index('speed50_1')], y[:, names.index('speed50_1')])
        self.assertTrue((x[:, names.index('speed_observed_1')] == 1).all())
        self.assertTrue((y[:, names.index('speed_observed_1')] == 0).all())

    def test_cache_rejects_corruption_changed_code_and_bad_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory)/'cache.npz'; meta=Path(directory)/'cache.json'
            np.savez(cache, x=np.arange(3)); fp={'source':'version1'}
            meta.write_text(json.dumps({'fingerprint':fp, 'cache_sha256':sha256(cache)}))
            self.assertIsNotNone(load_cached_arrays(cache, meta, fp))
            self.assertIsNone(load_cached_arrays(cache, meta, {'source':'version2'}))
            cache.write_bytes(b'broken')
            self.assertIsNone(load_cached_arrays(cache, meta, fp))
            meta.write_text('{interrupted')
            self.assertIsNone(load_cached_arrays(cache, meta, fp))

    def test_queries_follow_exported_transitions_not_marginal_argmax(self):
        posterior=np.tile([.6,.3,.1],(200,1)); arrays={'ball':np.zeros((200,2))}
        config=dict(CONFIG, step=1, query_budget=1)
        labels=np.array(['H']*100+['A']*100)
        query=choose_queries(posterior,np.zeros(200),arrays,config,blocks=[(1,0,199)],decoded_labels=labels)[0]
        self.assertTrue(75 <= query['frame'] <= 125)

    def test_review_contains_every_frame_and_handles_browser_storage_and_navigation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'tactical').mkdir()
            (root/'tactical/BAR-ATH_tactical_atd.txt').write_text(''.join(f'{i+1}:.1,.2,1;.8,.7,2:.5,.4,0\n' for i in range(30)))
            q={'frame':10,'half':1,'score':1.,'entropy':.2,'source_disagreement':.3,'ball_missing':False}
            labels=np.array(['H','A','D']*10); posterior=np.tile([.4,.3,.3],(30,1))
            with patch('possession.review.half_video_blocks',return_value=[(1,0,29)]), patch('possession.review.slot_teams',return_value=np.array([0,1])):
                build_review(root,root/'review',[q,dict(q,frame=20)],labels,posterior,encode=False,experiment_id='test-candidate')
            payload=json.loads((root/'review/review_data.json').read_text())
            self.assertEqual(payload['queries'][0]['frame_labels'],labels.tolist())
            self.assertEqual(len(payload['queries'][0]['frame_support']),30)
            node=shutil.which('node')
            if node:
                runner=Path(__file__).with_name('review_ui_checks.js')
                subprocess.run([node,str(runner),str(root/'review/index.html')],check=True,capture_output=True,text=True)


if __name__ == '__main__':
    unittest.main()
