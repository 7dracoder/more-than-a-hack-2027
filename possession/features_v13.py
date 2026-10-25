"""Fine-scale coordinate motion among the nearest few players, without labels."""
import numpy as np
from possession.features_v08 import velocity
from possession.features_v09 import build_features as base_features,shift


def extra_features(players,teams,ball,fps=25.,step=1):
    players,teams,ball=np.asarray(players),np.asarray(teams),np.asarray(ball)
    distances=np.linalg.norm(players-ball[:,None,:],axis=-1)
    columns={}
    for span in (2,8):
        bv=velocity(ball,fps,span=span)
        pv=velocity(players,fps,span=span,max_speed=15.)
        bs=np.linalg.norm(bv,axis=-1)
        acceleration=velocity(bv,fps,span=2,max_speed=500.)
        columns[f'ball_fine_speed{span}']=bs.clip(0,60)/20.
        columns[f'ball_fine_impulse{span}']=np.linalg.norm(acceleration,axis=-1).clip(0,500)/100.
        before,after=shift(bv,1),shift(bv,-1)
        denominator=np.linalg.norm(before,axis=-1)*np.linalg.norm(after,axis=-1)
        cosine=np.divide((before*after).sum(-1),denominator,out=np.full(len(ball),np.nan),where=denominator>.1)
        columns[f'ball_fine_turn{span}']=1.-np.clip(cosine,-1,1)
        for side,prefix in ((0,'home'),(1,'away')):
            d=distances[:,teams==side]
            xy=players[:,teams==side]
            speed=pv[:,teams==side]
            delta=xy-ball[:,None,:]
            direction=delta/np.maximum(d[:,:,None],.1)
            relative=speed-bv[:,None,:]
            relative_speed=np.linalg.norm(relative,axis=-1)
            player_speed=np.linalg.norm(speed,axis=-1)
            values=dict(relative=relative_speed.clip(0,60)/20.,
                        closing=np.sum(relative*direction,axis=-1).clip(-40,40)/20.,
                        speed=player_speed.clip(0,15)/10.)
            denominator=player_speed*bs[:,None]
            values['alignment']=np.divide((speed*bv[:,None,:]).sum(-1),denominator,
                out=np.full_like(denominator,np.nan),where=denominator>.1)
            for rank in (1,2,3):
                if d.shape[1]:
                    # Include all geometric ties, so slot ordering cannot choose a player.
                    threshold=np.partition(np.where(np.isfinite(d),d,np.inf),min(rank,d.shape[1])-1,axis=1)[:,min(rank,d.shape[1])-1]
                    selected=np.isfinite(d)&(d<=threshold[:,None])
                else:
                    selected=np.zeros(d.shape,bool)
                for key,value in values.items():
                    known=selected&np.isfinite(value)
                    count=known.sum(1)
                    average=np.divide(np.where(known,value,0.).sum(1),count,
                                      out=np.full(len(ball),np.nan),where=count>0)
                    columns[f'{prefix}_fine_{key}{span}_top{rank}']=average
            known=np.isfinite(d)&np.isfinite(relative_speed)
            mass=((d<1.5)&(relative_speed<3.)&known).sum(1)/2.
            columns[f'{prefix}_fine_control_mass{span}']=np.where(known.any(1),mass,np.nan)
    names=list(columns)
    values=np.stack([columns[n][::step] for n in names],axis=1)
    known=np.isfinite(values)
    return np.concatenate([np.where(known,values,0.),known.astype(np.float32)],axis=1).astype(np.float32),names+[n+'_known' for n in names]


def build_features(players,teams,ball,pitch,fps=25.,step=1):
    base,names=base_features(players,teams,ball,pitch,fps=fps,step=step)
    extra,extra_names=extra_features(players,teams,ball,fps=fps,step=step)
    return np.concatenate([base,extra],axis=1),names+extra_names
