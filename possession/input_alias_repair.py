"""Reversible affine projection edits for duplicated availability normalizers.

The old training permutation can alias a normalized base input to its memory
copy; inference swaps raw inputs before normalization. These edits reproduce a
specified interpolation of the two first-layer preactivations. They do not
guarantee a gain or equal a mixture of complete nonlinear-network predictions.
"""
import copy
import numpy as np
import torch


def repair_state(state,strength):
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Repair strength must lie in [0,1]')
    names = state['names']
    mean,scale = np.asarray(state['mean']),np.asarray(state['scale'])
    if (mean.shape != (len(names),) or scale.shape != mean.shape or
            not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any()):
        raise ValueError('Invalid frozen normalization')
    original = state['model']['project.weight']
    bias = state['model']['project.bias']
    if original.ndim != 3 or original.shape[1:] != (len(names),1) or bias.shape != (original.shape[0],):
        raise ValueError('Expected one-point projection with output bias')
    result = copy.deepcopy(state)
    changes = []
    for name in ['ball_seen','ball_seen_known']:
        indices = [i for i,n in enumerate(names) if n==name]
        if len(indices) != 2:
            raise ValueError('Expected the existing base/memory availability pair')
        target,alias = indices
        # For identical raw r: z_alias = (scale_target/scale_alias)*z_target
        #                       + (mean_target-mean_alias)/scale_alias.
        # Availability is binary, so verify neither normalizer clips its domain.
        values = np.array([0.,1.]) if name=='ball_seen' else np.array([1.])
        if any((np.abs((values-mean[i])/scale[i])>10).any() for i in indices):
            raise ValueError('Affine repair is invalid across normalization clipping')
        ratio = float(scale[target])/float(scale[alias])
        shift = (float(mean[target])-float(mean[alias]))/float(scale[alias])
        alpha,beta = 1-strength+strength*ratio,strength*shift
        column = original[:,target,0]
        result['model']['project.weight'][:,target,0] = column*alpha
        result['model']['project.bias'] += column*beta
        changes.append(dict(name=name,target=target,alias=alias,alpha=alpha,beta=beta))
    changed_weights = int((result['model']['project.weight'] != original).sum().item())
    changed_biases = int((result['model']['project.bias'] != bias).sum().item())
    return result,dict(strength=strength,changes=changes,changed_weights=changed_weights,changed_biases=changed_biases)
