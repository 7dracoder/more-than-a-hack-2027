"""Coordinate-only contact proxies between frames and relative acceleration."""
import numpy as np
from possession.features_v08 import velocity
from possession.features_v13 import build_features as base_features


def segment_distances(relative):
    """Closest linear approach within each adjacent, fully observed frame pair."""
    past=np.full(relative.shape[:-1],np.nan,dtype=np.float32)
    future=past.copy()
    if len(relative)<2:return past,future
    start=relative[:-1];delta=relative[1:]-start
    known=np.isfinite(start).all(-1)&np.isfinite(relative[1:]).all(-1)
    square=(delta*delta).sum(-1)
    fraction=np.divide(-(start*delta).sum(-1),square,out=np.zeros_like(square),where=known&(square>1e-12)).clip(0,1)
    closest=np.linalg.norm(start+fraction[:,:,None]*delta,axis=-1)
    closest=np.where(known,closest,np.nan)
    past[1:]=closest;future[:-1]=closest
    return past,future


def extra_features(players,teams,ball,fps=25.,step=1):
    players,teams,ball=np.asarray(players),np.asarray(teams),np.asarray(ball)
    relative=players-ball[:,None,:]
    distance=np.linalg.norm(relative,axis=-1)
    pv=velocity(players,fps,span=8,max_speed=15.)
    bv=velocity(ball,fps,span=8)
    pa=velocity(pv,fps,span=8,max_speed=80.)
    ba=velocity(bv,fps,span=8,max_speed=500.)
    speed=np.linalg.norm(pv,axis=-1)
    heading=np.divide(pv,speed[:,:,None],out=np.full_like(pv,np.nan),where=(speed>.5)[:,:,None])
    # A geometric forward probe, not an observed foot location or contact label.
    probe=relative+.5*heading
    body_past,body_future=segment_distances(relative)
    probe_past,probe_future=segment_distances(probe)
    columns={}
    for side,prefix in ((0,'home'),(1,'away')):
        choose=teams==side
        def minimum(values):
            if not choose.any():return np.full(len(ball),np.nan)
            result=np.min(np.where(np.isfinite(values[:,choose]),values[:,choose],np.inf),axis=1)
            return np.where(np.isfinite(result),result,np.nan)
        for kind,values in (('segment_past',body_past),('segment_future',body_future),
                            ('probe_past',probe_past),('probe_future',probe_future)):
            closest=minimum(values)
            columns[f'{prefix}_{kind}_distance']=closest.clip(0,30)/10.
            columns[f'{prefix}_{kind}_proximity']=np.exp(-closest/.75)
        nearest=minimum(distance)
        selected=choose[None,:]&np.isfinite(distance)&(distance<=nearest[:,None]+1e-7)
        def average(values):
            known=selected&np.isfinite(values);count=known.sum(1)
            return np.divide(np.where(known,values,0.).sum(1),count,out=np.full(len(ball),np.nan),where=count>0)
        longitudinal=-(relative*heading).sum(-1)
        lateral=np.sqrt(np.maximum(distance**2-longitudinal**2,0.))
        columns[f'{prefix}_nearest_forward']=average(longitudinal).clip(-20,20)/10.
        columns[f'{prefix}_nearest_lateral']=average(lateral).clip(0,30)/10.
        columns[f'{prefix}_probe_nearest_distance']=minimum(np.linalg.norm(probe,axis=-1)).clip(0,30)/10.
        pn=np.linalg.norm(pa,axis=-1);bn=np.linalg.norm(ba,axis=-1)
        columns[f'{prefix}_nearest_acceleration']=average(pn).clip(0,80)/20.
        columns[f'{prefix}_nearest_relative_acceleration']=average(np.linalg.norm(pa-ba[:,None,:],axis=-1)).clip(0,300)/100.
        denominator=pn*bn[:,None]
        cosine=np.divide((pa*ba[:,None,:]).sum(-1),denominator,out=np.full_like(denominator,np.nan),where=denominator>.1)
        columns[f'{prefix}_nearest_acceleration_alignment']=average(cosine.clip(-1,1))
    names=list(columns);values=np.stack([columns[n][::step] for n in names],axis=1)
    known=np.isfinite(values)
    return np.concatenate([np.where(known,values,0.),known.astype(np.float32)],axis=1).astype(np.float32),names+[n+'_known' for n in names]


def build_features(players,teams,ball,pitch,fps=25.,step=1):
    base,names=base_features(players,teams,ball,pitch,fps=fps,step=step)
    extra,extra_names=extra_features(players,teams,ball,fps=fps,step=step)
    return np.concatenate([base,extra],axis=1),names+extra_names
