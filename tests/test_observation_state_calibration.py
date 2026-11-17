import numpy as np
import pytest
from possession.observation_state_calibration import calibrate_dead_odds


def test_offset_changes_only_dead_odds_and_preserves_ha_odds():
    p = np.array([[.2,.3,.5,.1],[.1,.7,.2,.9],[0,0,1,.3]], np.float64)
    result = calibrate_dead_odds(p, [1,0,1], .7)
    np.testing.assert_array_equal(result[:,3], p[:,3])
    np.testing.assert_array_equal(result[1], p[1])
    np.testing.assert_array_equal(result[2], p[2])
    np.testing.assert_allclose(result[0,0]/result[0,1], p[0,0]/p[0,1])
    np.testing.assert_allclose(result[0,2]/result[0,:2].sum(), np.exp(.7)*p[0,2]/p[0,:2].sum())
    np.testing.assert_allclose(result[:,:3].sum(1), 1)


def test_zero_offset_and_team_swap_are_exact():
    p = np.array([[.2,.3,.5,.1],[.1,.7,.2,.9]], np.float32)
    np.testing.assert_array_equal(calibrate_dead_odds(p,[1,1],0), p)
    swapped = p[:,[1,0,2,3]]
    np.testing.assert_array_equal(calibrate_dead_odds(swapped,[1,1],-.5),
        calibrate_dead_odds(p,[1,1],-.5)[:,[1,0,2,3]])


def test_invalid_gate_or_distribution_is_rejected():
    p = np.array([[.2,.3,.5,.1]], np.float32)
    for gate,offset in [([.5],0),([np.nan],1),([1],np.inf),([1],21)]:
        with pytest.raises(ValueError):
            calibrate_dead_odds(p,gate,offset)
    with pytest.raises(ValueError):
        calibrate_dead_odds(p*2,[1],0)
