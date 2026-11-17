import numpy as np
import torch
from possession.receiver_geometry import receiver_features
from possession.receiver_context import actor_geometry,marginal_context


def weights():
    torch.manual_seed(1)
    return {'candidate.0.weight':torch.randn(64,11)*.1,'candidate.0.bias':torch.zeros(64),
        'candidate.2.weight':torch.randn(64,64)*.1,'candidate.2.bias':torch.zeros(64),
        'candidate.4.weight':torch.randn(1,64)*.1,'candidate.4.bias':torch.zeros(1),'unavailable':torch.zeros(1)}


def test_vectorized_context_matches_receiver_training_features():
    h=np.random.default_rng(5).normal(size=(51,5,2));h[35:,2]=np.nan;h[:,4]=np.nan;teams=np.array([0,1,0,1,1])
    x,mask,actors,_,_=actor_geometry(h,teams)
    for actor in np.flatnonzero(actors):
        expected,valid=receiver_features(h,teams,actor)
        np.testing.assert_allclose(x[actor],expected,atol=1e-6)
        np.testing.assert_array_equal(mask[actor],valid)


def test_marginalization_preserves_probability_mass_and_team_exchange():
    h=np.zeros((51,4,2));h[:,:,0]=np.arange(4)*5;teams=np.array([0,0,1,1]);ball=np.zeros((51,2));w=weights()
    x=marginal_context(h,teams,ball,w);swapped=marginal_context(h,1-teams,ball,w)
    np.testing.assert_allclose(x[[1,0,3,2,4,5,6,7,8,9]],swapped,atol=1e-6)
    np.testing.assert_allclose(x[:2].sum(),1,atol=1e-6);np.testing.assert_allclose(x[2:5].sum(),1,atol=1e-6)
    order=np.array([3,1,0,2]);permuted=marginal_context(h[:,order],teams[order],ball,w)
    np.testing.assert_allclose(x,permuted,atol=1e-6)


def test_no_ball_or_players_has_unknown_context_and_never_fills_from_future():
    h=np.zeros((51,3,2));ball=np.full((51,2),np.nan);w=weights()
    assert not marginal_context(h,[0,1,0],ball,w).any()
    ball[-1]=0;h[:]=np.nan
    assert not marginal_context(h,[0,1,0],ball,w).any()
