import numpy as np
import pytest
from scripts.research.spatial_memory.train_geometry_extension import added_normalization,full_weak


def test_normalization_balances_domains_and_symmetrizes_teams():
    names = ([f'base_{i}' for i in range(664)] + [f'home_new{i}' for i in range(148)] +
        [f'away_new{i}' for i in range(148)] + [f'memory_{i}' for i in range(128)])
    strong = []
    for view,(home,away) in zip(['native','masked','mixed'],[(1,3),(2,6),(4,8)]):
        x = np.zeros((2,1088),np.float32)
        x[:,664:812],x[:,812:960] = home,away
        strong.append(dict(x=x,view=view))
    extra = np.empty((2,296),np.float32)
    extra[:,:148],extra[:,148:] = 10,20
    mean,scale = added_normalization(strong,[dict(extra=extra)],names,[.25,.125,.125,.5])
    np.testing.assert_array_equal(mean,np.full(296,9.25,np.float32))
    np.testing.assert_allclose(scale,np.sqrt(133.75-9.25**2),rtol=1e-6)
    np.testing.assert_array_equal(mean[:148],mean[148:])
    np.testing.assert_array_equal(scale[:148],scale[148:])


def test_expanded_weak_rows_preserve_old_columns_and_reject_wrong_clock():
    raw = np.arange(3*792,dtype=np.float32).reshape(3,792)
    extra = np.arange(3*296,dtype=np.float32).reshape(3,296)+10000
    expanded = full_weak(raw,extra)
    np.testing.assert_array_equal(expanded[:,:664],raw[:,:664])
    np.testing.assert_array_equal(expanded[:,960:],raw[:,664:])
    np.testing.assert_array_equal(expanded[:,664:960],extra)
    with pytest.raises(ValueError):
        full_weak(raw,extra[:-1])
