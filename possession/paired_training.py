"""Fit the same team-symmetric probability mean used at inference."""
import math
import torch
from torch.nn import functional as F


def symmetric_outputs(state,swapped_state,boundary,swapped_boundary,positive_weight):
    """Return normalized log state probabilities and prior-adjusted edge logits.

    The second state prediction is in the swapped team's coordinate system.
    Both passes retain gradients; neither prediction acts as a target.
    """
    if state.shape!=swapped_state.shape or state.ndim!=3 or state.shape[1]!=3:
        raise ValueError('Expected matching batch x H/A/D x time state outputs')
    if boundary.shape!=swapped_boundary.shape or boundary.shape!=(state.shape[0],state.shape[2]):
        raise ValueError('Expected matching batch x time boundary outputs')
    if not math.isfinite(positive_weight) or positive_weight<=0:
        raise ValueError('Boundary positive weight must be finite and positive')
    aligned=swapped_state[:,[1,0,2]]
    logits=torch.logaddexp(F.log_softmax(state,dim=1),F.log_softmax(aligned,dim=1))-math.log(2.)
    correction=math.log(positive_weight)
    a=boundary-correction;b=swapped_boundary-correction
    log_yes=torch.logaddexp(F.logsigmoid(a),F.logsigmoid(b))-math.log(2.)
    log_no=torch.logaddexp(F.logsigmoid(-a),F.logsigmoid(-b))-math.log(2.)
    return logits,log_yes-log_no+correction
