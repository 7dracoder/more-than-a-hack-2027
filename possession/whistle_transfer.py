"""Audio-only transfer features and explicitly uncertain external point targets."""
import numpy as np


def temporal_features(embedding,offsets=(-1,0,1)):
    x=np.asarray(embedding);indices=np.arange(len(x))
    if x.ndim!=2 or not len(x):raise ValueError('Need one nonempty recording')
    return np.concatenate([x[np.clip(indices+offset,0,len(x)-1)] for offset in offsets],axis=1)


def point_targets(times,truth,positive_radius=.32,negative_margin=2.):
    times=np.asarray(times);truth=np.sort(np.asarray(truth))
    if not len(truth) or not 0<=positive_radius<negative_margin:raise ValueError('Invalid point target policy')
    index=np.searchsorted(truth,times)
    distance=np.minimum(np.abs(times-truth[np.clip(index,0,len(truth)-1)]),np.abs(times-truth[np.clip(index-1,0,len(truth)-1)]))
    return np.where(distance<=positive_radius,1,np.where(distance>=negative_margin,0,-100)).astype(np.int64)


def frame_probabilities(native,frame_count):
    # The frozen CNN's first native bin is centred at 0.08 s and spans frames
    # -8..23; subsequent bins advance by 32 100-Hz frames.
    index=(np.arange(frame_count)+8)//32
    if len(index) and index[-1]>=len(native):raise ValueError('Native scores do not cover the recording')
    return np.asarray(native)[index]
