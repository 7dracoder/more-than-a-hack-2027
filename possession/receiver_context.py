"""Passer-marginalized receiver geometry without event/possession input labels."""
import numpy as np
import torch
import torch.nn.functional as f

NAMES=['home_actor_mass','away_actor_mass','home_receiver_mass','away_receiver_mass',
    'receiver_unavailable_mass','receiver_entropy','receiver_mean_dx','receiver_mean_dy',
    'receiver_anchor_age','receiver_context_known']


def actor_geometry(history,teams,pitch=(105.,68.),fps=25.):
    h=np.asarray(history);teams=np.asarray(teams);n,k,_=h.shape
    if not 0<n<=round(2*fps)+1 or h.shape[-1]!=2 or len(teams)!=k:raise ValueError('Invalid causal player window')
    seen=np.isfinite(h).all(-1);last=np.where(seen,np.arange(n)[:,None],-1).max(0);present=last>=0
    pos=h[np.maximum(last,0),np.arange(k)].copy();pos[~present]=0
    before=np.where(seen&(np.arange(n)[:,None]<=last[None,:]-max(1,round(.2*fps))),np.arange(n)[:,None],-1).max(0)
    velocity=np.zeros((k,2),np.float32);motion=before>=0
    ids=np.flatnonzero(motion)
    velocity[ids]=(pos[ids]-h[before[ids],ids])*fps/(last[ids]-before[ids])[:,None]
    motion &= np.linalg.norm(velocity,axis=1)<=15.;velocity[~motion]=0
    actors=seen[-1];origin=np.where(actors[:,None],h[-1],0)
    relative=pos[None,:,:]-origin[:,None,:];distance=np.linalg.norm(relative,axis=-1)
    valid=actors[:,None]&present[None,:]&~np.eye(k,dtype=bool)
    speed=np.linalg.norm(velocity,axis=1);denominator=distance*speed[:,None]
    alignment=np.divide((relative*velocity[:,None,:]).sum(-1),denominator,out=np.zeros((k,k)),where=denominator>0)
    x=np.stack([relative[:,:,0]/pitch[0],relative[:,:,1]/pitch[1],distance/np.linalg.norm(pitch),
        np.broadcast_to(teams[:,None]==teams[None,:],(k,k)),np.broadcast_to(seen[-1],(k,k)),
        np.broadcast_to((n-1-last)/fps/2,(k,k)),np.broadcast_to(velocity[:,0]/15,(k,k)),
        np.broadcast_to(velocity[:,1]/15,(k,k)),np.broadcast_to(motion,(k,k)),alignment,
        np.broadcast_to(motion[:,None],(k,k))],axis=-1).astype(np.float32)
    x[~valid]=0
    return x,valid,actors,pos,present


def marginal_context(history,teams,ball_history,weights,pitch=(105.,68.),fps=25.):
    """One causal snapshot. Ball window ends now, at most eight seconds long."""
    ball=np.asarray(ball_history);seen=np.isfinite(ball).all(-1);output=np.zeros(len(NAMES),np.float32)
    if len(ball)>round(8*fps)+1:raise ValueError('Ball history exceeds fixed causal horizon')
    if not seen.any():return output
    x,mask,actors,positions,present=actor_geometry(history,teams,pitch,fps)
    if not actors.any():return output
    tick=np.flatnonzero(seen)[-1];anchor=ball[tick];age=(len(ball)-1-tick)/fps
    # Fixed coordinate-only actor prior, with no actor event ID or possession label.
    distance=np.linalg.norm(np.asarray(history)[-1]-anchor,axis=1)
    logits=np.where(actors,-np.nan_to_num(distance,nan=1e6)/2.,-1e9)
    prior=np.exp(logits-logits.max());prior/=prior.sum()
    with torch.no_grad():
        h=torch.as_tensor(x)
        for layer in ('0','2'):h=f.gelu(f.linear(h,weights['candidate.'+layer+'.weight'],weights['candidate.'+layer+'.bias']))
        scores=f.linear(h,weights['candidate.4.weight'],weights['candidate.4.bias']).squeeze(-1)
        scores=scores.masked_fill(~torch.as_tensor(mask),-1e9)
        p=torch.softmax(torch.cat([scores,weights['unavailable'].expand(len(x),1)],dim=1),dim=1).numpy()
    receiver=prior@p;teams=np.asarray(teams)
    entropy=-(receiver*np.log(np.maximum(receiver,1e-12))).sum()/np.log(len(receiver))
    delta=(positions-anchor)/np.asarray(pitch);mean=(receiver[:-1,None]*delta).sum(0)
    output[:]=[prior[teams==0].sum(),prior[teams==1].sum(),receiver[:-1][teams==0].sum(),
        receiver[:-1][teams==1].sum(),receiver[-1],entropy,mean[0],mean[1],age/8,1.]
    assert np.isfinite(output).all()
    return output
