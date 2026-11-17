import numpy as np

from scripts.research.video_ball.calibrate_external_confidence import detection_targets


def test_highest_confidence_unique_ground_truth_match_only():
    boxes=np.array([[0,0,10,10],[0,0,10,10],[20,20,30,30],[50,50,60,60]])
    confidence=np.array([.6,.9,.8,.7])
    truth=[[0,0,10,10],[20,20,10,10]]
    np.testing.assert_array_equal(detection_targets(boxes,confidence,truth),[0,1,1,0])


def test_negative_images_and_empty_outputs_remain_valid():
    np.testing.assert_array_equal(detection_targets(np.array([[0,0,10,10]]),np.array([.7]),[]),[0])
    assert detection_targets(np.empty((0,4)),np.array([]),[[0,0,10,10]]).size==0
