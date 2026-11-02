"""Label-free dead-ball rules from ball position alone (v0.9 candidate).

Rule OOB: the ball observed more than ``margin`` metres outside the pitch for at
least ``min_frames`` consecutive frames is dead. Laws of the Game 9.1: the ball
is out of play once it has wholly crossed a touch or goal line.

Constants were fixed on the six Sportec Open DFL training matches only
(Bassek et al. 2025, doi:10.1038/s41597-025-04505-y, CC BY 4.0), by frame
precision of BallStatus=0 inside the rule's runs: margin 1.0 m, 13 frames
(0.5 s) -> 99.40 percent of 54,760 flagged frames are dead, 78/80 runs >= 90
percent dead. No BAR-ATH label (public window or otherwise) chose anything.
"""
import numpy as np

MARGIN_M = 1.0
MIN_FRAMES = 13


def outside_distance(ball_xy_m, pitch):
    b = np.asarray(ball_xy_m, dtype=np.float64)
    out = np.maximum.reduce([-b[:, 0], b[:, 0] - pitch[0], -b[:, 1], b[:, 1] - pitch[1]])
    return np.where(np.isfinite(out), out, -np.inf)


def runs_of(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def oob_dead_mask(ball_xy_m, pitch, margin=MARGIN_M, min_frames=MIN_FRAMES):
    out = outside_distance(ball_xy_m, pitch) > margin
    mask = np.zeros(len(out), bool)
    for start, stop in runs_of(out):
        if stop - start >= min_frames:
            mask[start:stop] = True
    return mask


def apply_oob(labels, ball_xy_m, pitch, graded=None, margin=MARGIN_M, min_frames=MIN_FRAMES):
    """Return (new_labels, changed_mask). Only H/A labels become D; nothing else moves."""
    labels = np.asarray(labels).astype('<U1').copy()
    mask = oob_dead_mask(ball_xy_m, pitch, margin, min_frames) & np.isin(labels, ['H', 'A'])
    if graded is not None:
        mask &= np.asarray(graded, bool)
    labels[mask] = 'D'
    return labels, mask
