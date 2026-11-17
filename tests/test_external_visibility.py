import numpy as np
import pytest

from possession.external_visibility import summarize, fit_profile, degrade, remove_stationary_cue


def sample():
    rng = np.random.default_rng(802)
    p = rng.uniform([0, 0], [100, 60], (240, 8, 2)).astype(np.float32)
    b = np.column_stack([np.arange(240) * .05, np.ones(240) * 30]).astype(np.float32)
    d = b.copy(); d[30:60] = np.nan; d[110:160] = np.nan
    dp = p.copy(); dp[50:100, :2] = np.nan
    return p, b, dp, d, np.repeat([0, 1], 4)


def profile():
    p, b, dp, d, teams = sample()
    return fit_profile([summarize(p, b, dp, d, teams, (105, 68), 10)], ['external-training-match'])


def test_visibility_fit_uses_seconds_and_keeps_target_labels_out_of_api():
    p, b, dp, d, teams = sample()
    summary = summarize(p, b, dp, d, teams, (105, 68), 10)
    assert summary['gap_seconds'] == [3., 5.]
    result = fit_profile([summary], ['external-training-match'])
    assert result['missing_fraction'] == 80 / 240
    assert not result['provided_match_used'] and not result['possession_labels_used']
    assert result['natural_gap_count'] == 2


def test_external_degradation_is_deterministic_and_preserves_original_arrays():
    p, b, _, _, teams = sample()
    original_p, original_b = p.copy(), b.copy()
    a, c = degrade(p, teams, b, (105, 68), profile(), 41, mode='none')
    e, f = degrade(p, teams, b, (105, 68), profile(), 41, mode='none')
    np.testing.assert_array_equal(a, e); np.testing.assert_array_equal(c, f)
    np.testing.assert_array_equal(p, original_p); np.testing.assert_array_equal(b, original_b)
    assert np.isnan(c).any()
    known = np.isfinite(c).all(1)
    np.testing.assert_array_equal(c[known], b[known])


def test_target_derived_profiles_are_refused_and_existing_absences_are_not_filled():
    p, b, _, _, teams = sample()
    bad = profile(); bad['provided_match_used'] = True
    with pytest.raises(ValueError, match='external observations'):
        degrade(p, teams, b, (105, 68), bad, 1)
    b[:20] = np.nan
    _, result = degrade(p, teams, b, (105, 68), profile(), 1, mode='none')
    assert np.isnan(result[:20]).all()


def test_stationary_cue_can_be_removed_without_any_dead_ball_labels():
    p, b, _, _, _ = sample()
    b[10:40] = b[10]
    hidden = remove_stationary_cue(p, b, 'hide', 5)
    assert np.isnan(hidden[10:40]).all()
    np.testing.assert_array_equal(hidden[:10], b[:10])
    assert np.isfinite(remove_stationary_cue(p, b, 'carry', 5)[10:40]).all()
