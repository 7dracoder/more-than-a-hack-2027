import numpy as np
import torch
from possession.receiver_context import marginal_context as original
from possession.receiver_context_v2 import marginal_context


def test_never_observed_padding_does_not_change_context_or_erase_entropy():
    torch.manual_seed(4)
    w={'candidate.0.weight':torch.randn(8,11)*.1,'candidate.0.bias':torch.zeros(8),
        'candidate.2.weight':torch.randn(8,8)*.1,'candidate.2.bias':torch.zeros(8),
        'candidate.4.weight':torch.randn(1,8)*.1,'candidate.4.bias':torch.zeros(1),'unavailable':torch.zeros(1)}
    h=np.zeros((51,4,2),np.float32);h[:,:,0]=np.arange(4)*5
    teams=np.array([0,0,1,1]);ball=np.zeros((51,2),np.float32)
    expected=marginal_context(h,teams,ball,w)
    assert 0<expected[5]<=1
    np.testing.assert_allclose(expected,original(h,teams,ball,w),atol=1e-6)
    padded=np.concatenate([h,np.full((51,7,2),np.nan,np.float32)],axis=1)
    more_teams=np.r_[teams,[0,1,0,1,0,1,0]]
    actual=marginal_context(padded,more_teams,ball,w)
    np.testing.assert_allclose(actual,expected,atol=1e-6)
    assert abs(original(padded,more_teams,ball,w)[5]-expected[5])>.01


def test_recently_observed_candidates_still_count_when_currently_hidden():
    torch.manual_seed(4)
    w={'candidate.0.weight':torch.randn(8,11)*.1,'candidate.0.bias':torch.zeros(8),
        'candidate.2.weight':torch.randn(8,8)*.1,'candidate.2.bias':torch.zeros(8),
        'candidate.4.weight':torch.randn(1,8)*.1,'candidate.4.bias':torch.zeros(1),'unavailable':torch.zeros(1)}
    h=np.zeros((51,3,2),np.float32);h[:,1,0]=10;h[:,2,0]=20;h[30:,2]=np.nan
    ball=np.zeros((51,2),np.float32)
    np.testing.assert_allclose(marginal_context(h,[0,0,1],ball,w),original(h,[0,0,1],ball,w),atol=1e-6)
