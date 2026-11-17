"""Timeline, geometry and resume checks; no video or possession labels required."""
from types import SimpleNamespace

import numpy as np
import pytest

from games import GAMES, half_video_blocks
from scripts.research.video_ball.detect import (
    frame_list, pixel_to_pitch, resume_checkpoint, save_checkpoint,
)
from scripts.research.video_ball.tracking import bracketed_gaps, linear_gap, project_points


def args(mode='clips', clips=40, clip_length=50):
    return SimpleNamespace(frames=mode, clips=clips, clip_length=clip_length)


def test_projection_round_trip_and_invalid_homography():
    h = np.array([[1200, 40, 10], [20, 800, 15], [.1, .2, 1.]])
    pitch = np.array([105.3, 68.])
    xy = np.array([[40., 20.], [80., 60.]])
    uv = project_points(np.stack([h, h]), xy, pitch)
    recovered = np.array([pixel_to_pitch(h, u, v, pitch) for u, v in uv])
    np.testing.assert_allclose(recovered, xy)
    assert np.isnan(pixel_to_pitch(np.zeros((3, 3)), 1, 1, pitch)).all()
    assert np.isnan(pixel_to_pitch(np.full((3, 3), np.nan), 1, 1, pitch)).all()


def test_clips_never_cross_halves_or_holdout_blocks():
    n = GAMES['BAR-ATH']['n_video_frames']
    for mode in ('clips', 'holdout'):
        frames = frame_list(args(mode, clips=300), n)
        assert len(frames) and np.all(np.diff(frames) > 0)
        allowed = np.zeros(n, bool)
        for _, first, last in half_video_blocks('BAR-ATH'):
            allowed[first:last + 1] = True
        assert allowed[frames].all()
        if mode == 'holdout':
            assert ((frames // 15000) % 5 == 4).all()
    with pytest.raises(ValueError):
        frame_list(args(), 164488)
    with pytest.raises(ValueError):
        frame_list(args(clip_length=100000), n)


def test_missing_frames_use_supplied_aligned_tracking_only():
    n = GAMES['BAR-ATH']['n_video_frames']
    ball = np.zeros((n, 2))
    ball[[0, 11185, 190197, n - 1]] = np.nan
    np.testing.assert_array_equal(frame_list(args('missing'), n, ball), [11185, 190197])
    with pytest.raises(ValueError):
        frame_list(args('missing'), n)


def test_gap_anchors_never_cross_halves():
    n = GAMES['BAR-ATH']['n_video_frames']
    ball = np.zeros((n, 2))
    ball[11185:11190] = np.nan
    ball[12000:12003] = np.nan
    ball[90650:110810] = np.nan
    assert bracketed_gaps(ball) == [(1, 12000, 12003)]
    ball[11999], ball[12003] = [0, 0], [4, 8]
    np.testing.assert_allclose(linear_gap(ball, 12000, 12003), [[1, 2], [2, 4], [3, 6]])


def test_resume_preserves_zero_detection_frames_and_rejects_changes(tmp_path):
    out = tmp_path / 'detections.npz'
    selected = np.array([10, 11, 14])
    protocol = {'weights_sha256': 'frozen', 'settings': {'device': 'cpu'}}
    rows = [(10, .9, 30, 40, 3, 4)]
    save_checkpoint(out, selected, [10, 11], rows, protocol)
    completed, loaded = resume_checkpoint(out, selected, protocol)
    assert completed == [10, 11] and loaded == [list(rows[0])]
    with pytest.raises(ValueError):
        resume_checkpoint(out, selected, {'weights_sha256': 'changed'})
    with out.open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError):
        resume_checkpoint(out, selected, protocol)


def test_resume_rejects_a_nonprefix_completed_timeline(tmp_path):
    out = tmp_path / 'detections.npz'
    selected = np.array([10, 11, 14])
    save_checkpoint(out, selected, [10, 14], [], {})
    with pytest.raises(ValueError):
        resume_checkpoint(out, selected, {})
