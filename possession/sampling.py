"""Training-only crop sampling around complete short possession spells."""
import numpy as np


def short_spell_anchors(labels,max_frames,minimum_frames=1):
    labels=np.asarray(labels)
    if labels.ndim!=1 or not 1<=minimum_frames<=max_frames:
        raise ValueError('Short-spell sampling needs one label timeline and valid length bounds')
    cuts=np.r_[0,np.flatnonzero(labels[1:]!=labels[:-1])+1,len(labels)]
    anchors=[]
    for a,b in zip(cuts[:-1],cuts[1:]):
        if (a>0 and b<len(labels) and labels[a] in (0,1) and
                labels[a-1] in (0,1,2) and labels[b] in (0,1,2) and minimum_frames<=b-a<=max_frames):
            anchors.append((a+b-1)//2)
    return np.asarray(anchors,np.int64)


def sample_crop_start(rng,frames,length,anchors,probability=0.):
    anchors=np.asarray(anchors)
    if not 0<length<=frames or not np.isfinite(probability) or not 0<=probability<=1:
        raise ValueError('Invalid crop size or short-spell sampling probability')
    if anchors.ndim!=1 or (len(anchors) and ((anchors<0).any() or (anchors>=frames).any())):
        raise ValueError('Training anchors must belong to the sampled half')
    # The disabled/default case consumes exactly the original RNG draw.
    if not probability or not len(anchors) or rng.random()>=probability:
        return int(rng.integers(frames-length+1))
    anchor=int(anchors[int(rng.integers(len(anchors)))])
    low=length//8;high=max(low+1,min(length,7*length//8+1))
    offset=int(rng.integers(low,high))
    return int(np.clip(anchor-offset,0,frames-length))
