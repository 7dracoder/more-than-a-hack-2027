import numpy as np
import pytest

from possession.spatial_memory import build_features, passing_options, symmetric_features


def example(n=240):
    rng = np.random.default_rng(90)
    players = rng.uniform([10, 10], [90, 60], (n, 8, 2)).astype(np.float32)
    ball = rng.uniform([20, 20], [80, 50], (n, 2)).astype(np.float32)
    ball[30:65] = np.nan
    return players, np.repeat([0, 1], 4), ball


def test_memory_is_slot_permutation_invariant_and_team_equivariant():
    players, teams, ball = example()
    x, names = build_features(players, teams, ball, (105., 68.))
    order = np.array([7, 2, 0, 4, 3, 6, 1, 5])
    permuted, pn = build_features(players[:, order], teams[order], ball, (105., 68.))
    swapped, sn = build_features(players, 1 - teams, ball, (105., 68.))
    assert names == pn == sn
    np.testing.assert_allclose(x, permuted, rtol=1e-5, atol=2e-6)
    np.testing.assert_allclose(symmetric_features(x, names), swapped, rtol=1e-5, atol=2e-6)


def test_reflection_and_translation_do_not_invent_facing_direction():
    players, teams, ball = example()
    x, names = build_features(players, teams, ball, (105., 68.))
    p, b = players.copy(), ball.copy()
    p[..., 0] = 105 - p[..., 0]
    b[..., 0] = 105 - b[..., 0]
    changed, _ = build_features(p, teams, b, (105., 68.))
    np.testing.assert_allclose(x, changed, rtol=2e-4, atol=2e-5)
    translated, _ = build_features(players + [5, -3], teams, ball + [5, -3], (105., 68.))
    np.testing.assert_allclose(x, translated, rtol=2e-4, atol=2e-5)
    assert not any('gaze' in name or 'head' in name for name in names)


def test_memory_starts_empty_at_half_boundary_and_retains_ball_missingness():
    players, teams, ball = example()
    ball[:10] = np.nan
    original = ball.copy()
    x, names = build_features(players, teams, ball, (105., 68.))
    assert x[0, names.index('past_anchor_known')] == 0
    assert x[0, names.index('home_memory8_centroid_displacement_known')] == 0
    assert x[0, names.index('home_memory1_near_origin_change_known')] == 0
    assert not x[30:65, names.index('ball_seen')].any()
    np.testing.assert_array_equal(ball, original)
    assert np.isfinite(x).all()


def test_no_tracking_and_missing_team_keep_fixed_finite_schema():
    players, teams, ball = example(12)
    full, names = build_features(players, teams, ball, (105., 68.))
    absent, an = build_features(np.full_like(players, np.nan), teams,
                                np.full_like(ball, np.nan), (105., 68.))
    one, on = build_features(players[:, :4], teams[:4], ball, (105., 68.))
    assert names == an == on and full.shape == absent.shape == one.shape
    assert np.isfinite(absent).all() and np.isfinite(one).all()


def test_receivers_behind_carrier_are_not_excluded_by_an_assumed_visual_cone():
    players = np.array([[[50., 34.], [35., 34.], [65., 34.], [60., 60.]]])
    result = passing_options(players, np.array([0, 0, 0, 1]), np.array([[50., 34.]]))
    assert result['home_current_options'][0] > 0
    assert result['home_current_receiver_distance'][0] == 1.5


def test_bad_input_shapes_and_invalid_fps_are_rejected():
    players, teams, ball = example(12)
    with pytest.raises(ValueError):
        build_features(players, teams, ball[:-1], (105., 68.))
    with pytest.raises(ValueError):
        build_features(players, teams, ball, (105., 68.), fps=0)
