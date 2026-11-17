"""Uncertainty-gated orientation proxies from external 3D pose joints.

Head direction is not eye gaze, attention, peripheral vision or intention.
Shoulders identify an unsigned body axis, not its forward/backward direction.
No velocity-based replacement is supplied for an unavailable head observation.
"""
import math

import numpy as np


def joint(joints, name):
    value = joints.get(name) if isinstance(joints, dict) else None
    if not isinstance(value, dict):
        return None
    point = np.asarray(value.get('xyz', []), float)
    error = value.get('p90_mae_cm')
    if point.shape != (3,) or not np.isfinite(point).all() or error is None:
        return None
    error = float(error) / 100.
    if not np.isfinite(error) or error < 0:
        return None
    return point[:2], error


def orientation(joints, max_error_to_length=.5):
    """Use nose relative to midpoint of ears; fail closed for uncertain geometry.

    The summed predicted errors provide an engineering reliability gate, not a
    calibrated confidence interval for the resulting angle.
    """
    if not 0 < max_error_to_length < 1:
        raise ValueError('Need a finite relative-uncertainty gate inside (0,1)')
    result = dict(head_known=False, head_unit=None, head_yaw_radians=None,
                  head_error_to_length=None, body_axis_known=False,
                  unsigned_head_body_angle_degrees=None, gaze_known=False)
    nose, left, right = (joint(joints, name) for name in ('nose', 'lEar', 'rEar'))
    if nose is not None and left is not None and right is not None:
        vector = nose[0] - (left[0] + right[0]) / 2.
        length = float(np.linalg.norm(vector))
        error = nose[1] + (left[1] + right[1]) / 2.
        ratio = error / length if length > 1e-8 else math.inf
        if np.isfinite(ratio):
            result['head_error_to_length'] = ratio
        if length > 1e-8 and ratio < max_error_to_length:
            unit = vector / length
            result.update(head_known=True, head_unit=unit.tolist(),
                          head_yaw_radians=float(np.arctan2(unit[1], unit[0])))
    left, right = (joint(joints, name) for name in ('lShoulder', 'rShoulder'))
    if left is not None and right is not None:
        axis = right[0] - left[0]
        length = float(np.linalg.norm(axis))
        if length > 1e-8 and (left[1] + right[1]) / length < max_error_to_length:
            result['body_axis_known'] = True
            normal = np.array([-axis[1], axis[0]]) / length
            if result['head_known']:
                cosine = abs(float(np.dot(result['head_unit'], normal)))
                result['unsigned_head_body_angle_degrees'] = float(np.degrees(np.arccos(np.clip(cosine, 0., 1.))))
    return result


def angular_change(previous, current):
    """Smallest signed angle; callers must enforce person/time continuity."""
    return float(np.arctan2(np.sin(current - previous), np.cos(current - previous)))
