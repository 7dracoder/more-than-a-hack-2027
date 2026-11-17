import numpy as np
import pytest
import torch
from possession.input_alias_repair import repair_state
from possession.temporal import normalized


def state():
    torch.manual_seed(21)
    return dict(names=['ball_seen','ball_seen_known','other','ball_seen','ball_seen_known'],
        mean=np.array([.8,1,2,.7,1],np.float32),scale=np.array([.4,.01,3,.44,.01],np.float32),
        model={'project.weight':torch.randn(4,5,1),'project.bias':torch.randn(4),'other.weight':torch.randn(3,4)})


def test_affine_edits_match_intended_first_layer_interpolation():
    s = state()
    raw = np.array([[0,1,4,0,1],[1,1,7,1,1]],np.float32)
    x = torch.from_numpy(normalized(raw,s['mean'],s['scale']).T[None])
    aliased = x.clone();aliased[:,0] = x[:,3];aliased[:,1] = x[:,4]
    original = torch.nn.functional.conv1d(x,s['model']['project.weight'],s['model']['project.bias'])
    swapped = torch.nn.functional.conv1d(aliased,s['model']['project.weight'],s['model']['project.bias'])
    for strength in [0.,.5,1.]:
        patched,receipt = repair_state(s,strength)
        y = torch.nn.functional.conv1d(x,patched['model']['project.weight'],patched['model']['project.bias'])
        torch.testing.assert_close(y,(1-strength)*original+strength*swapped,rtol=1e-6,atol=1e-6)
        np.testing.assert_array_equal(patched['mean'],s['mean'])
        np.testing.assert_array_equal(patched['scale'],s['scale'])
        torch.testing.assert_close(patched['model']['other.weight'],s['model']['other.weight'],rtol=0,atol=0)
        assert receipt['changes'][1]['alpha']==1 and receipt['changes'][1]['beta']==0


def test_zero_strength_is_exact_and_original_is_not_mutated():
    s = state()
    patched,receipt = repair_state(s,0.)
    assert receipt['changed_weights']==receipt['changed_biases']==0
    for k,v in s['model'].items():
        torch.testing.assert_close(v,patched['model'][k],rtol=0,atol=0)
    snapshot = s['model']['project.weight'].clone()
    repair_state(s,1.)
    torch.testing.assert_close(s['model']['project.weight'],snapshot,rtol=0,atol=0)


def test_invalid_strength_or_clipped_domain_is_rejected():
    for strength in [-1,2,np.nan]:
        with pytest.raises(ValueError):
            repair_state(state(),strength)
    s = state();s['scale'][0]=.001
    with pytest.raises(ValueError):
        repair_state(s,.5)
