import numpy as np
import pytest
from possession.feature_permutation import team_swap_indices
from possession.external import swap_features
from possession.temporal import normalized


def test_duplicate_common_fields_keep_their_own_occurrence():
    names = ['ball_seen','home_x','away_x','ball_seen']
    mapping = team_swap_indices(names)
    np.testing.assert_array_equal(mapping,[0,2,1,3])
    x = np.array([[0,10,20,5]])
    np.testing.assert_array_equal(x[:,mapping][:,mapping],x)


def test_raw_swap_before_normalization_matches_inference_with_unequal_statistics():
    names = ['ball_seen','home_x','away_x','ball_seen']
    raw = np.array([[0,2,4,0],[1,3,5,1]],np.float32)
    mean = np.array([.8,2,3,.7],np.float32)
    scale = np.array([.4,2,3,.44],np.float32)
    correct = normalized(raw[:,team_swap_indices(names)],mean,scale)
    inference = normalized(swap_features(raw,names),mean,scale)
    np.testing.assert_array_equal(correct,inference)
    historical = swap_features(normalized(raw,mean,scale),names)
    assert not np.array_equal(historical,correct)


def test_unpaired_team_occurrences_rejected():
    with pytest.raises(ValueError):
        team_swap_indices(['home_x','home_x','away_x'])
