"""Provider identity is a weak training target, never a coordinate-model input."""
import numpy as np


def carrier_targets(players,teams,ball,carrier_index,labels):
    players=np.asarray(players,np.float32);teams=np.asarray(teams);ball=np.asarray(ball,np.float32)
    carrier_index=np.asarray(carrier_index);labels=np.asarray(labels)
    n,p=players.shape[:2]
    if players.shape!=(n,p,2) or ball.shape!=(n,2) or teams.shape!=(p,) or carrier_index.shape!=(n,) or labels.shape!=(n,) or p<1:
        raise ValueError('Carrier labels and coordinate timeline differ')
    known=np.isfinite(players).all(-1)&np.isfinite(ball).all(-1)[:,None]
    with np.errstate(invalid='ignore'):distance=np.linalg.norm(players-ball[:,None,:],axis=-1)
    result=np.full(n,-100,np.int64);frames=np.arange(n);safe=np.clip(carrier_index,0,p-1)
    for side,label in ((0,'H'),(1,'A')):
        eligible=known&(teams==side)[None,:]
        ordered=np.sort(np.where(eligible,distance,np.inf),axis=1)
        valid=(carrier_index>=0)&(carrier_index<p)&(labels==label)&(teams[safe]==side)&eligible[frames,safe]
        for rank in range(min(3,p)):
            nearest=ordered[:,rank];difference=np.full_like(distance,np.inf)
            np.subtract(distance,nearest[:,None],out=difference,where=np.isfinite(distance)&np.isfinite(nearest[:,None]))
            selected=eligible&(np.abs(difference)<=1e-5)
            # A tied feature slot averages identities. It cannot be assigned
            # one carrier identity without inventing a target.
            use=valid&selected[frames,safe]&(selected.sum(1)==1)
            result[use]=rank
    return result
