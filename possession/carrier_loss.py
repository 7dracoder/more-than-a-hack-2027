"""Weak carrier supervision for the shared, within-team attention scorer."""
import torch
import torch.nn.functional as F


def carrier_attention_loss(scores,features,labels,targets,rank_weights=None):
    batch,frames=labels.shape
    if scores.shape!=(batch,2,3,frames) or targets.shape!=labels.shape or features.shape[0]!=batch or features.shape[-1]!=frames or features.shape[1]<180:
        raise ValueError('Carrier attention tensors do not align')
    known=features[:,:180].reshape(batch,2,3,5,6,frames)[:,:,:,2,5]>.5
    side=labels.clamp(0,1)[:,None,None,:].expand(-1,1,3,-1)
    selected=torch.gather(scores,1,side)[:,0]
    visible=torch.gather(known,1,side)[:,0]
    valid=(labels>=0)&(labels<2)&(targets>=0)&(targets<3)
    target_visible=torch.gather(visible,1,targets.clamp(0,2)[:,None,:])[:,0]
    valid&=target_visible
    safe=torch.where(valid,targets,torch.full_like(targets,-100))
    loss=F.cross_entropy(selected.masked_fill(~visible,-1e4),safe,weight=rank_weights,reduction='none',ignore_index=-100)
    denominator=valid.sum() if rank_weights is None else (rank_weights[targets.clamp(0,2)]*valid).sum()
    return loss.sum()/denominator.clamp_min(1)
