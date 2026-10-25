"""A differentiable overlap objective for complete external training spells."""
import torch
from torch.nn import functional as F


def spell_overlap_loss(logits,labels,max_context=125):
    """Average soft IoU over uncensored H/A runs inside each training crop.

    Context errors on another true run of the same class are not charged to
    this run. Crop edges and unknown-label boundaries never fabricate spells.
    The one-hot matrix product avoids duplicate-index gradient atomics.
    """
    batch,_,length=logits.shape
    if labels.shape!=(batch,length) or max_context<1:
        raise ValueError('Invalid spell-loss dimensions or context')
    edges=torch.ones_like(labels,dtype=torch.bool)
    edges[:,1:]=labels[:,1:]!=labels[:,:-1]
    starts=torch.nonzero(edges.reshape(-1),as_tuple=False).flatten()
    ends=torch.cat([starts[1:],starts.new_tensor([labels.numel()])])
    flat=labels.reshape(-1)
    classes=flat[starts]
    row_start=(starts//length)*length
    before=flat[(starts-1).clamp_min(0)]
    after=flat[ends.clamp_max(labels.numel()-1)]
    complete=(starts>row_start)&(ends<row_start+length)&(before>=0)&(after>=0)
    keep=complete&((classes==0)|(classes==1))
    starts,ends,classes=starts[keep],ends[keep],classes[keep]
    if starts.numel()==0:
        return logits.sum()*0.
    rows=starts//length
    start=starts%length
    end=ends-rows*length
    radius=(end-start).clamp(min=1,max=max_context)
    ticks=torch.arange(length,device=logits.device)[None,:]
    inside=(ticks>=start[:,None])&(ticks<end[:,None])
    context=(ticks>=(start-radius).clamp_min(0)[:,None])&(ticks<(end+radius).clamp_max(length)[:,None])
    truth=labels[rows]
    outside=context&~inside&(truth>=0)&(truth!=classes[:,None])
    probabilities=torch.softmax(logits,dim=1).reshape(batch*3,length)
    selectors=F.one_hot(rows*3+classes,num_classes=batch*3).to(probabilities.dtype)
    predicted=selectors@probabilities
    intersection=(predicted*inside).sum(1)
    union=(end-start).to(predicted.dtype)+(predicted*outside).sum(1)
    return (1-intersection/(union+1e-6)).mean()
