import numpy as np
import pytest
from possession.visibility_curriculum import choose_crop,weak_visibility_weights


def test_focused_crop_contains_known_missing_center_and_stays_in_half():
    for center in [0,50,99]:
        first,focused = choose_crop(np.random.default_rng(1),np.random.default_rng(2),100,20,np.array([center]),1.)
        assert focused and 0<=first<=80 and first<=center<first+20


def test_focusing_does_not_perturb_control_random_stream_and_empty_falls_back():
    a,b = np.random.default_rng(12),np.random.default_rng(12)
    f,g = np.random.default_rng(3),np.random.default_rng(3)
    for _ in range(20):
        choose_crop(a,f,100,20,np.array([50]),0.)
        choose_crop(b,g,100,20,np.array([50]),1.)
        assert a.integers(10000)==b.integers(10000)
    first,focused = choose_crop(a,f,100,20,np.array([],int),1.)
    assert not focused and 0<=first<=80


def test_unknown_weights_remain_zero_and_visible_weights_unchanged():
    w = weak_visibility_weights(np.array([1,1,0,0],np.float32),[0,1,0,1],4.)
    np.testing.assert_array_equal(w,[4,1,0,0])
    with pytest.raises(ValueError):
        weak_visibility_weights([1],[.5],4.)
