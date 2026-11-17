import numpy as np
from possession.observation_boundary import mix_boundaries


def inputs():
    p = np.tile(np.array([.3,.4,.3,.2], np.float32), (6,1))
    q = np.tile(np.array([.4,.3,.3,.36], np.float32), (3,1))
    memory = np.zeros((3,3), np.float32)
    return p,q,memory,['ball_seen','home_current_visible','away_current_visible']


def test_zero_strength_or_complete_observations_preserve_original():
    p,q,m,n = inputs()
    np.testing.assert_array_equal(mix_boundaries(p,q,m,n,0), p)
    m[:] = 1
    np.testing.assert_array_equal(mix_boundaries(p,q,m,n,1), p)


def test_two_frame_hazards_recover_expert_event_probability():
    p,q,m,n = inputs()
    q[1,3],q[2,3] = .51,.75
    result = mix_boundaries(p,q,m,n,1)
    for j in [1,2]:
        a,b = (j-1)*2+1,j*2+1
        np.testing.assert_allclose(1-np.prod(1-result[a:b,3]), q[j,3], atol=1e-7)
    np.testing.assert_array_equal(result[:, :3], p[:, :3])
    np.testing.assert_array_equal(result[[0,5],3], p[[0,5],3])


def test_bad_clock_or_transition_probability_rejected():
    p,q,m,n = inputs()
    for invalid in [np.nan, -1, 2]:
        q[1,3] = invalid
        try:
            mix_boundaries(p,q,m,n,1)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid expert boundary accepted')
