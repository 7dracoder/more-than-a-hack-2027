import numpy as np
from possession.dead_ball_rules import apply_oob, oob_dead_mask

PITCH = (105.0, 68.0)


def track(n=100):
    b = np.full((n, 2), 50.0)
    b[:, 1] = 30.0
    return b


def test_short_excursion_ignored():
    b = track(); b[10:20, 1] = 70.0   # 10 frames < 13
    assert not oob_dead_mask(b, PITCH).any()


def test_long_excursion_dead_and_only_alive_labels_change():
    b = track(); b[10:30, 1] = 70.0
    labels = np.array(['H'] * 100); labels[25:28] = 'X'
    new, changed = apply_oob(labels, b, PITCH)
    assert changed.sum() == 17 and (new[10:25] == 'D').all() and (new[25:28] == 'X').all()
    assert (new[:10] == 'H').all() and (new[30:] == 'H').all()


def test_margin_and_missing_ball():
    b = track(); b[10:40, 0] = 105.5; b[50:80] = np.nan
    assert not oob_dead_mask(b, PITCH).any()


def test_graded_mask_respected():
    b = track(); b[10:30, 0] = -3.0
    graded = np.zeros(100, bool); graded[:20] = True
    _, changed = apply_oob(np.array(['A'] * 100), b, PITCH, graded)
    assert changed.sum() == 10
