"""Exact three-state chain likelihood using a balanced log-semiring reduction."""
import math
import torch


def log_chain_product(matrices):
    """Multiply time-ordered 3x3 transfers in O(log T) sequential operations."""
    if matrices.ndim!=4 or matrices.shape[-2:]!=(3,3) or not matrices.shape[1]:
        raise ValueError('Expected batch/time/3/3 transfer matrices')
    level=matrices
    while level.shape[1]>1:
        pairs=level.shape[1]//2
        left,right=level[:,:2*pairs:2],level[:,1:2*pairs:2]
        product=torch.logsumexp(left[:,:,:,:,None]+right[:,:,None,:,:],dim=-2)
        level=torch.cat([product,level[:,-1:]],dim=1) if level.shape[1]%2 else product
    return level[:,0]


def structured_state_loss(logits,boundary_logits,labels,team_cost=0.,dead_cost=2.):
    """Joint unary/transition NLL; unknown targets reset the chain without supervision.

    The boundary logit is a transition potential, so use positive_weight=1 at
    inference rather than weighted-BCE calibration for a model trained this way.
    Official scoring is separate and unchanged. Double precision avoids loss
    cancellation on long, confidently predicted training crops.
    """
    if labels.ndim!=2 or logits.ndim!=3 or logits.shape[1]!=3 or logits.shape[0]!=labels.shape[0] or logits.shape[2]!=labels.shape[1] or boundary_logits.shape!=labels.shape or not labels.shape[1]:
        raise ValueError('Expected matching batch/3/time unaries and batch/time targets/potentials')
    if not all(math.isfinite(v) and v>=0 for v in (team_cost,dead_cost)):
        raise ValueError('Structured transition costs must be finite and nonnegative')
    known=(labels>=0)&(labels<3)
    unary=torch.log_softmax(logits.double(),dim=1).transpose(1,2)
    unary=torch.where(known[:,:,None],unary,0.)
    safe=torch.where(known,labels,0).long()
    targets=torch.nn.functional.one_hot(safe,3).to(unary.dtype)*known[:,:,None]
    true_score=(unary*targets).sum((1,2))
    if labels.shape[1]>1:
        pair=known[:,1:]&known[:,:-1]
        cost=torch.tensor([[0.,team_cost,dead_cost],[team_cost,0.,dead_cost],[dead_cost,dead_cost,0.]],device=logits.device,dtype=unary.dtype)
        off=1.-torch.eye(3,device=logits.device,dtype=unary.dtype)
        potential=boundary_logits[:,1:].double()[:,:,None,None]*off-cost
        potential=torch.where(pair[:,:,None,None],potential,0.)
        transfers=potential+unary[:,1:,None,:]
        product=log_chain_product(transfers)
        partition=torch.logsumexp(unary[:,0,:,None]+product,dim=(1,2))
        true_transition=(potential*targets[:,:-1,:,None]*targets[:,1:,None,:]).sum((1,2,3))
        true_score=true_score+true_transition
    else:
        partition=torch.logsumexp(unary[:,0],dim=1)
    # Every unknown frame is an independent free state, contributing log(3).
    nll=partition-true_score-(~known).sum(1).to(unary.dtype)*math.log(3.)
    nll=torch.where(known.any(1),nll,0.)
    return nll.sum()/known.sum().clamp_min(1)
