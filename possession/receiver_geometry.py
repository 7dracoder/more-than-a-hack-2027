"""Conditional receiver features from a strictly past-and-present coordinate window."""
import numpy as np

NAMES=['relative_x','relative_y','distance','same_team','seen_now','age_seconds_over_two',
    'velocity_x_over_15','velocity_y_over_15','motion_known','actor_motion_alignment','actor_motion_known']


def receiver_features(history,teams,actor,pitch=(105.,68.),fps=25.):
    h=np.asarray(history);teams=np.asarray(teams)
    if h.ndim!=3 or h.shape[2]!=2 or h.shape[1]!=len(teams) or not 0<=actor<len(teams) or fps<=0:
        raise ValueError('Invalid causal coordinate window')
    if not len(h) or len(h)>round(2*fps)+1 or not np.isfinite(h[-1,actor]).all():
        raise ValueError('Require present actor and at most two seconds of history')
    current=len(h)-1;seen=np.isfinite(h).all(-1);k=len(teams)
    last=np.full(k,-1,int);velocity=np.zeros((k,2));motion=np.zeros(k,bool)
    for j in range(k):
        ticks=np.flatnonzero(seen[:,j])
        if not len(ticks):continue
        last[j]=ticks[-1];before=ticks[ticks<=last[j]-max(1,round(.2*fps))]
        if len(before):
            v=(h[last[j],j]-h[before[-1],j])*fps/(last[j]-before[-1])
            if np.linalg.norm(v)<=15.:velocity[j]=v;motion[j]=True
    valid=last>=0;valid[actor]=False
    x=np.zeros((k,len(NAMES)),np.float32);origin=h[-1,actor];diagonal=np.linalg.norm(pitch)
    for j in np.flatnonzero(valid):
        relative=h[last[j],j]-origin;distance=np.linalg.norm(relative)
        av=velocity[actor];speed=np.linalg.norm(av)
        alignment=float(relative@av/(distance*speed)) if distance>0 and speed>0 else 0.
        x[j]=[relative[0]/pitch[0],relative[1]/pitch[1],distance/diagonal,teams[j]==teams[actor],
            seen[-1,j],(current-last[j])/fps/2,velocity[j,0]/15,velocity[j,1]/15,motion[j],alignment,motion[actor]]
    return x,valid
