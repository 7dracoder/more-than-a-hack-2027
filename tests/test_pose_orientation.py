import numpy as np

from possession.pose_orientation import angular_change, orientation


def joints():
    return {name:dict(xyz=xyz, p90_mae_cm=1.) for name, xyz in {
        'nose':[.2, 0, 1.8], 'lEar':[0, .08, 1.8], 'rEar':[0, -.08, 1.8],
        'lShoulder':[0, .2, 1.5], 'rShoulder':[0, -.2, 1.5]}.items()}


def test_head_and_unsigned_body_alignment_are_not_gaze():
    result = orientation(joints())
    assert result['head_known'] and result['body_axis_known']
    np.testing.assert_allclose(result['head_unit'], [1, 0])
    assert result['unsigned_head_body_angle_degrees'] == 0
    assert not result['gaze_known']


def test_missing_uncertain_and_degenerate_head_geometry_stays_unknown():
    assert not orientation(None)['head_known']
    bad = joints(); bad['nose']['p90_mae_cm'] = 30
    assert not orientation(bad)['head_known']
    bad = joints(); bad['nose']['xyz'] = [0, 0, 1.8]
    assert not orientation(bad)['head_known']
    bad = joints(); bad.pop('lEar')
    assert not orientation(bad)['head_known']


def test_head_turn_can_differ_from_body_axis_without_guessing_peripheral_vision():
    value = joints(); value['nose']['xyz'] = [0, .2, 1.8]
    result = orientation(value)
    assert result['unsigned_head_body_angle_degrees'] == 90
    assert not result['gaze_known']
    np.testing.assert_allclose(np.degrees(angular_change(np.radians(179), np.radians(-179))), 2.)
