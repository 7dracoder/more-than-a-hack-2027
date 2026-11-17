import numpy as np
import pytest

from scripts.research.video_ball.check_detections import coordinate_checks


def test_zero_detection_frames_and_multiple_candidates_are_counted_correctly():
    ball = np.array([[0., 0.], [np.nan, np.nan], [2., 2.], [3., 3.]])
    frames = np.arange(4)
    result = coordinate_checks(frames, frames, np.array([0, 0, 1, 2]),
                               np.array([.9, .8, .7, .6]),
                               np.array([[.5, 0], [20, 20], [10, 10], [4, 2]]), ball)
    assert result['known_ball_selected_frames'] == 3
    assert result['known_ball_frames_with_candidates'] == 2
    assert result['known_ball_frames_with_candidate_within_1m'] == 1
    assert result['known_ball_frames_with_candidate_within_3m'] == 2
    assert result['missing_ball_frames_with_projected_candidates'] == 1
    assert result['possession_score'] is None
    assert not result['parameter_or_candidate_selection_performed']


def test_no_detections_is_a_valid_complete_result():
    frames = np.arange(3)
    result = coordinate_checks(frames, frames, np.array([], dtype=int),
                               np.array([]), np.empty((0, 2)), np.zeros((3, 2)))
    assert result['raw_detections'] == 0
    assert result['known_ball_nearest_candidate_error_m_quantiles'] is None


def test_incomplete_timeline_outside_frames_and_invalid_confidence_fail():
    frames = np.arange(3)
    ball = np.zeros((3, 2))
    for completed, detected, confidence in [(frames[:2], [0], [.5]),
                                           (frames, [3], [.5]),
                                           (frames, [0], [np.nan])]:
        with pytest.raises(ValueError):
            coordinate_checks(frames, completed, np.array(detected), np.array(confidence),
                              np.zeros((1, 2)), ball)
