"""Meaningful mathematical/invariance tests for the new research implementation."""

import itertools
import unittest
from unittest.mock import patch

import numpy as np

from possession.data import frame_features
from possession.decode import infer
from possession.model import evidence, predict, choose_queries


class DecoderTests(unittest.TestCase):
    def test_exact_against_exhaustive_paths(self):
        scores = np.array([[.2, -.5, .1], [.3, -.1, .5], [-.7, .8, .1], [.2, .4, -.2]])
        trans = np.array([[0, -.9, -.5], [-.9, 0, -.5], [-.5, -.5, 0]])
        paths = np.array(list(itertools.product(range(3), repeat=4)))
        weight = np.array([scores[np.arange(4), p].sum() + trans[p[:-1], p[1:]].sum() for p in paths])
        distribution = np.exp(weight - weight.max())
        distribution /= distribution.sum()
        expected = np.array([[distribution[paths[:, t] == k].sum() for k in range(3)] for t in range(4)])
        path, posterior, sampled = infer(scores, trans, samples=20, seed=4)
        np.testing.assert_array_equal(path, paths[np.argmax(weight)])
        np.testing.assert_allclose(posterior, expected, atol=1e-12)
        np.testing.assert_array_equal(sampled, infer(scores, trans, samples=20, seed=4)[2])

    def test_invalid_evidence_rejected(self):
        with self.assertRaises(ValueError):
            infer(np.array([[np.nan, 0., 1.]]), np.zeros((3, 3)))


class FeatureTests(unittest.TestCase):
    def test_track_slot_permutation(self):
        players = np.array([[.1,.2,1.], [.2,.3,2.], [.8,.4,3.], [np.nan,np.nan,np.nan]])
        teams = np.array([0,1,0,1])
        permutation = [3,1,0,2]
        first = frame_features(players, np.array([.3,.2,0]), teams, np.array([105.3,68.]))
        second = frame_features(players[permutation], np.array([.3,.2,0]), teams[permutation], np.array([105.3,68.]))
        for a,b in zip(first,second):
            np.testing.assert_allclose(a,b)


def toy():
    n = 30
    return {"ball": np.full((n,2), .4), "nearest": np.tile([.5,5.], (n,1)),
            "counts": np.full((n,2), 11), "median_speed": np.full(n,2.),
            "stationary": np.full(n,.2), "camera_center": np.full((n,2),.5),
            "camera_valid": np.ones(n,dtype=bool), "tag": np.full(n,.8),
            "restart": np.zeros(n,dtype=bool)}


CONFIG = {"step":2,"geometry_weight":2.4,"tag_weight":1.8,"camera_weight":.12,
          "history_samples":2,"seed":7,"team_switch_cost":6.,"stoppage_switch_cost":4.,
          "query_budget":0,"query_spacing_seconds":1}


class ModelTests(unittest.TestCase):
    def test_calibration_cannot_see_labels_outside_training_block(self):
        from possession.calibrate import fit
        a=toy()
        labels=np.resize(np.array(['H','A','D']),30)
        changed=labels.copy(); changed[20:]='X'
        with patch('possession.calibrate.half_video_blocks',return_value=[(1,0,9),(2,20,29)]):
            first=fit(a,CONFIG,labels,0,9)
            second=fit(a,CONFIG,changed,0,9)
        self.assertEqual(first,second)

    def test_team_swap_symmetry(self):
        a = toy()
        b = {k:v.copy() for k,v in a.items()}
        b["nearest"] = b["nearest"][:,::-1]
        b["tag"] *= -1
        sa,_ = evidence(a,0,30,CONFIG)
        sb,_ = evidence(b,0,30,CONFIG)
        np.testing.assert_allclose(sa[:,[1,0,2]],sb,atol=1e-12)

    def test_halftime_reset_and_ungraded_output(self):
        first = toy()
        second = {k:v.copy() for k,v in first.items()}
        second["tag"][:10] *= -1
        second["nearest"][:10] = second["nearest"][:10,::-1]
        with patch('possession.model.half_video_blocks',return_value=[(1,0,9),(2,20,29)]):
            a,pa,ha,_ = predict(first,CONFIG)
            b,pb,hb,_ = predict(second,CONFIG)
        np.testing.assert_array_equal(a[20:],b[20:])
        np.testing.assert_allclose(pa[20:],pb[20:])
        np.testing.assert_array_equal(ha[:,20:],hb[:,20:])
        self.assertTrue((a[10:20]=='X').all())

    def test_zero_query_budget(self):
        a=toy()
        with patch('possession.model.half_video_blocks',return_value=[(1,0,9),(2,20,29)]):
            self.assertEqual(choose_queries(np.full((30,3),1/3),np.zeros(30),a,CONFIG),[])

    def test_missing_ball_is_not_automatically_dead(self):
        a=toy(); a["ball"][:]=np.nan; a["nearest"][:]=np.nan
        scores,_=evidence(a,0,30,CONFIG)
        self.assertTrue((np.argmax(scores,axis=1)!=2).all())


if __name__=='__main__':
    unittest.main()
