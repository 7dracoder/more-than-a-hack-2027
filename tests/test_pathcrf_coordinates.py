import inspect
import numpy as np

from possession.pathcrf_coordinates import player_features, windows, source_teams


def players():
    rng = np.random.default_rng(403)
    p = np.broadcast_to(rng.uniform([10., 8.], [95., 60.], (22, 2)), (600, 22, 2)).copy().astype(np.float32)
    p[:, 0, 0] = 3.; p[:, 11, 0] = 102.
    p[:, :, 1] += np.arange(600)[:, None] * .01
    return p, np.repeat([0, 1], 11)


def test_adapter_has_no_ball_or_label_input_and_abstains_on_incomplete_roster():
    assert list(inspect.signature(player_features).parameters) == ['players', 'pitch', 'fps']
    p, teams = players()
    original = p.copy()
    p[20:500, 3] = np.nan
    assert not list(windows(player_features(p, (105., 68.)), teams))
    assert np.isfinite(player_features(original, (105., 68.))).all()


def test_team_mapping_follows_field_side_and_outside_is_unknown():
    p, teams = players()
    _, x, left = next(windows(player_features(p, (105., 68.)), teams))
    assert x.shape == (50, 26, 8) and left == 0
    np.testing.assert_array_equal(source_teams([0, 10, 11, 21, 22, 25], left), [0, 0, 1, 1, -1, -1])
    p[:, :, 0] = 105. - p[:, :, 0]
    _, _, reflected_left = next(windows(player_features(p, (105., 68.)), teams))
    assert reflected_left == 1
    np.testing.assert_array_equal(source_teams([0, 11, 22], reflected_left), [1, 0, -1])


def test_velocity_uses_metres_per_second_and_does_not_bridge_gaps():
    p, _ = players()
    x = player_features(p, (105., 68.))
    np.testing.assert_allclose(x[30:400, :, 3], .25, atol=1e-4)
    p[200:300, 5] = np.nan
    x = player_features(p, (105., 68.))
    assert np.isnan(x[200:300, 5]).all()
    assert np.isfinite(x[320:400, 5]).all()
