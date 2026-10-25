"""Coordinate-only trajectories tied to the player selected at the centre frame."""
import numpy as np
from possession.features_v13 import build_features as base_features

OFFSETS=(-8,-4,0,4,8)
COMPONENTS=('radial','lateral','distance','ball_radial','ball_lateral','known')


def extra_features(players,teams,ball,fps=25.,step=1):
    players=np.asarray(players,dtype=np.float32);ball=np.asarray(ball,dtype=np.float32);teams=np.asarray(teams)
    if players.ndim!=3 or players.shape[-1]!=2 or ball.shape!=(len(players),2) or teams.shape!=(players.shape[1],) or not len(players):
        raise ValueError('Expected aligned player/ball coordinates and team membership')
    if not np.isfinite(fps) or fps<=0 or not isinstance(step,int) or step<1:
        raise ValueError('Invalid trajectory time grid')
    n,p=players.shape[:2];tick=np.arange(n)
    known=np.isfinite(players).all(-1)&np.isfinite(ball).all(-1)[:,None]
    with np.errstate(invalid='ignore',over='ignore'):
        relative=players-ball[:,None,:];distance=np.linalg.norm(relative,axis=-1)
        bad_step=np.zeros((n,p),bool)
        if n>1:
            player_jump=np.linalg.norm(players[1:]-players[:-1],axis=-1)*fps>15.
            ball_jump=np.linalg.norm(ball[1:]-ball[:-1],axis=-1)*fps>80.
            bad_step[1:]=player_jump|ball_jump[:,None]
    point_bad=np.concatenate([np.zeros((1,p),np.int32),np.cumsum(~known,axis=0,dtype=np.int32)])
    step_bad=np.concatenate([np.zeros((1,p),np.int32),np.cumsum(bad_step,axis=0,dtype=np.int32)])
    def point(offset):
        end=tick+offset;inside=(end>=0)&(end<n);safe=end.clip(0,n-1)
        low=np.minimum(tick,safe);high=np.maximum(tick,safe)
        continuous=inside[:,None]&((point_bad[high+1]-point_bad[low])==0)&((step_bad[high+1]-step_bad[low+1])==0)
        return relative[safe],ball[safe]-ball,continuous
    axis=np.zeros_like(relative);axis[...,0]=1.
    directed=known&(distance>1e-8)
    np.divide(relative,distance[...,None],out=axis,where=directed[...,None])
    need=known&~directed
    # A coincident centre can use an observed neighbouring direction. The
    # fallback is irrelevant when every relative/displacement vector is zero.
    for offset in (-8,-4,4,8):
        if not need.any():break
        rel,displacement,continuous=point(offset)
        for candidate in (rel,np.broadcast_to(displacement[:,None,:],relative.shape)):
            norm=np.linalg.norm(candidate,axis=-1);use=need&continuous&(norm>1e-8)
            np.divide(candidate,norm[...,None],out=axis,where=use[...,None]);need&=~use
    columns=[];names=[]
    for side,prefix in ((0,'home'),(1,'away')):
        eligible=known&(teams==side)[None,:]
        ordered=np.sort(np.where(eligible,distance,np.inf),axis=1)
        selections=[]
        for rank in range(3):
            closest=ordered[:,rank] if rank<p else np.full(n,np.inf)
            difference=np.full_like(distance,np.inf)
            np.subtract(distance,closest[:,None],out=difference,where=np.isfinite(distance)&np.isfinite(closest[:,None]))
            selections.append(eligible&(np.abs(difference)<=1e-5))
        for rank,selected in enumerate(selections,1):
            for offset in OFFSETS:
                rel,displacement,continuous=point(offset)
                valid=selected&continuous;count=valid.sum(1)
                with np.errstate(invalid='ignore',over='ignore'):
                    radial=(rel*axis).sum(-1)
                    lateral=np.abs(rel[...,0]*axis[...,1]-rel[...,1]*axis[...,0])
                    d=np.linalg.norm(rel,axis=-1)
                    ball_radial=(displacement[:,None,:]*axis).sum(-1)
                    ball_lateral=np.abs(displacement[:,None,0]*axis[...,1]-displacement[:,None,1]*axis[...,0])
                values=(radial.clip(-30,30)/10.,lateral.clip(0,30)/10.,d.clip(0,30)/10.,
                        ball_radial.clip(-50,50)/10.,ball_lateral.clip(0,50)/10.)
                for component,value in zip(COMPONENTS[:-1],values):
                    mean=np.divide(np.where(valid,value,0.).sum(1),count,out=np.zeros(n,np.float32),where=count>0)
                    columns.append(mean[::step]);names.append(f'{prefix}_tube{rank}_o{offset}_{component}')
                columns.append((count>0).astype(np.float32)[::step]);names.append(f'{prefix}_tube{rank}_o{offset}_known')
    result=np.stack(columns,axis=1).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('Nonfinite trajectory feature')
    return result,names


def build_features(players,teams,ball,pitch,fps=25.,step=1):
    base,names=base_features(players,teams,ball,pitch,fps=fps,step=step)
    extra,extra_names=extra_features(players,teams,ball,fps=fps,step=step)
    return np.concatenate([base,extra],axis=1),names+extra_names
