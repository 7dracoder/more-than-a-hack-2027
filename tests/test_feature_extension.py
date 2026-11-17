import numpy as np
import pytest
import torch
from possession.feature_extension import extend_state,feature_positions
from possession.temporal import normalized


def test_zero_extension_preserves_projection_and_existing_statistics():
    torch.manual_seed(1)
    state = dict(names=['home_x','away_x','memory'],mean=np.array([1,2,3],np.float32),
        scale=np.array([2,3,4],np.float32),model={'project.weight':torch.randn(5,3,1),'project.bias':torch.randn(5)})
    names = ['home_x','new_a','away_x','new_b','memory']
    result = extend_state(state,names,[10,20],[2,5])
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(32,5)).astype(np.float32)
    old_x = torch.from_numpy(normalized(raw[:,[0,2,4]],state['mean'],state['scale']).T[None])
    new_x = torch.from_numpy(normalized(raw,result['mean'],result['scale']).T[None])
    old_y = torch.nn.functional.conv1d(old_x,state['model']['project.weight'],state['model']['project.bias'])
    new_y = torch.nn.functional.conv1d(new_x,result['model']['project.weight'],result['model']['project.bias'])
    torch.testing.assert_close(old_y,new_y,rtol=1e-6,atol=1e-6)
    np.testing.assert_array_equal(result['mean'][[0,2,4]],state['mean'])
    np.testing.assert_array_equal(result['scale'][[0,2,4]],state['scale'])
    assert torch.count_nonzero(result['model']['project.weight'][:,[1,3]]) == 0
    assert state['names'] == ['home_x','away_x','memory']
    assert state['model']['project.weight'].shape == (5,3,1)


def test_dropped_or_duplicate_feature_is_rejected():
    state = dict(names=['old'],mean=np.zeros(1),scale=np.ones(1),model={'project.weight':torch.ones(2,1,1)})
    for names in [['new'],['old','old','new']]:
        with pytest.raises(ValueError):
            extend_state(state,names,[0],[1])


def test_existing_duplicate_names_keep_distinct_weights_and_normalization():
    state = dict(names=['seen','other','seen'],mean=np.array([1,2,4],np.float32),
        scale=np.array([3,5,7],np.float32),model={'project.weight':torch.tensor([[[2.],[3.],[5.]]])})
    names = ['seen','added','other','seen']
    result = extend_state(state,names,[0],[1])
    positions = feature_positions(state['names'],names)
    np.testing.assert_array_equal(positions,[0,2,3])
    np.testing.assert_array_equal(result['mean'][positions],state['mean'])
    np.testing.assert_array_equal(result['scale'][positions],state['scale'])
    torch.testing.assert_close(result['model']['project.weight'][:,positions],state['model']['project.weight'])
    assert result['mean'][0] != result['mean'][3]
