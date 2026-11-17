import numpy as np

from scripts.research.video_ball.fit_external_motion import live_speeds
from scripts.research.video_ball.fuse_detections import coordinate_hypotheses


def test_consistent_unique_candidates_fill_only_missing_observations():
    original=np.array([[0.,0.],[np.nan,np.nan],[np.nan,np.nan],[.3,0.]],dtype=np.float32)
    fused,r=coordinate_hypotheses(original,np.array([0,1,2]),np.array([.99,.9,.9]),
                                 np.array([[50,50],[.1,0],[.2,0]]),.5,10.,[(0,4)])
    np.testing.assert_allclose(fused,[[0,0],[.1,0],[.2,0],[.3,0]],atol=1e-7)
    assert r['coordinate_hypothesis_frames']==2
    np.testing.assert_array_equal(fused[[0,3]],original[[0,3]])


def test_ambiguity_unreachable_and_unsupported_points_are_left_missing():
    original=np.array([[0.,0.],[np.nan,np.nan],[np.nan,np.nan],[0.,0.]],dtype=np.float32)
    frames=np.array([1,1,2]);confidence=np.array([.9,.9,.9])
    fused,r=coordinate_hypotheses(original,frames,confidence,np.array([[.1,0],[.2,0],[50,50]]),.5,10.,[(0,4)])
    np.testing.assert_array_equal(np.isfinite(fused),np.isfinite(original))
    assert r['coordinate_hypothesis_frames']==0


def test_adjacency_cannot_cross_half_boundaries():
    original=np.array([[0.,0.],[np.nan,np.nan],[np.nan,np.nan],[.3,0.]],dtype=np.float32)
    fused,r=coordinate_hypotheses(original,np.array([1,2]),np.array([.9,.9]),
                                 np.array([[.04,0],[.26,0]]),.5,2.,[(0,2),(2,4)])
    # Each hypothesis is supported by its own half's observed anchor only.
    assert r['coordinate_hypothesis_frames']==2
    assert np.isfinite(fused).all()
    empty=np.full((4,2),np.nan,np.float32)
    _,r=coordinate_hypotheses(empty,np.array([1,2]),np.array([.9,.9]),
                             np.zeros((2,2)),.5,2.,[(0,2),(2,4)])
    assert r['coordinate_hypothesis_frames']==0


def test_motion_fit_excludes_dead_and_unknown_label_transitions():
    ball=np.array([[0.,0.],[1.,0.],[100.,0.],[101.,0.]])
    np.testing.assert_array_equal(live_speeds(ball,np.array(['H','A','D','X'])),[25.])
