import numpy as np
from possession.receiver_geometry import receiver_features


def test_missing_history_retains_last_position_without_future_fill():
    h=np.zeros((51,3,2));h[:,1,0]=10;h[26:,1]=np.nan;h[:,2]=np.nan
    x,mask=receiver_features(h,[0,0,1],0)
    np.testing.assert_array_equal(mask,[False,True,False])
    assert x[1,4]==0 and x[1,5]==.5
    np.testing.assert_allclose(x[1,0],10/105)


def test_candidate_permutation_preserves_features_and_actor_reference():
    h=np.random.default_rng(2).normal(size=(51,4,2));teams=np.array([0,1,0,1])
    x,mask=receiver_features(h,teams,0)
    order=np.array([2,0,3,1]);y,m=receiver_features(h[:,order],teams[order],1)
    np.testing.assert_array_equal(y,x[order]);np.testing.assert_array_equal(m,mask[order])


def test_future_coordinates_are_not_part_of_window():
    h=np.zeros((70,3,2));h[:51,1,0]=10
    a=receiver_features(h[:51],[0,0,1],0)
    h[51:]=1e6
    b=receiver_features(h[:51],[0,0,1],0)
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
