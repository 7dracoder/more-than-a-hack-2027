"""Insert zero-weight input features without refitting existing normalization."""
import copy
from collections import Counter,defaultdict
import numpy as np
import torch


def feature_positions(old,names):
    old_counts,new_counts = Counter(old),Counter(names)
    if (not set(old) < set(names) or any(new_counts[n] != count for n,count in old_counts.items()) or
            any(count != 1 for n,count in new_counts.items() if n not in old_counts)):
        raise ValueError('Extension must retain every original feature occurrence and add unique columns')
    available = defaultdict(list)
    for i,name in enumerate(names):
        available[name].append(i)
    used = Counter()
    positions = []
    for name in old:
        positions.append(available[name][used[name]])
        used[name] += 1
    return np.asarray(positions)


def extend_state(state, names, added_mean, added_scale):
    old = state['names']
    positions = feature_positions(old,names)
    added = np.array([i for i,name in enumerate(names) if name not in set(old)])
    added_mean,added_scale = np.asarray(added_mean),np.asarray(added_scale)
    if (added_mean.shape != added.shape or added_scale.shape != added.shape or
            not np.isfinite(added_mean).all() or not np.isfinite(added_scale).all() or (added_scale <= 0).any()):
        raise ValueError('New normalization must match added features')
    original = state['model']['project.weight']
    if original.ndim != 3 or original.shape[1:] != (len(old),1):
        raise ValueError('Expected a one-point input projection')
    result = copy.deepcopy(state)
    weight = original.new_zeros((original.shape[0],len(names),1))
    weight[:,positions,:] = original
    result['model']['project.weight'] = weight
    result['names'] = list(names)
    mean,scale = np.empty(len(names),np.float32),np.empty(len(names),np.float32)
    mean[positions],scale[positions] = state['mean'],state['scale']
    mean[added],scale[added] = added_mean,added_scale
    result['mean'],result['scale'] = mean,scale
    return result
