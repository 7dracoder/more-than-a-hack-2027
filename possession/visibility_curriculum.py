"""External missing-ball crop sampling without changing unknown-target masks."""
import numpy as np


def choose_crop(rng,focus_rng,n,length,eligible,probability):
    eligible = np.asarray(eligible)
    if (not isinstance(length,int) or not isinstance(n,int) or not 0<length<=n or
            eligible.ndim!=1 or not np.isfinite(probability) or not 0<=probability<=1 or
            (len(eligible) and (not np.issubdtype(eligible.dtype,np.integer) or eligible.min()<0 or eligible.max()>=n))):
        raise ValueError('Invalid crop clock, eligible centers or focus probability')
    # Always consume the control stream's uniform draw. Focusing uses a separate
    # stream so source-half choices, strong crops and team swaps remain paired.
    first = int(rng.integers(0,n-length+1))
    focus = bool(focus_rng.random()<probability and len(eligible))
    if focus:
        center = int(focus_rng.choice(eligible))
        first = int(np.clip(center-length//2,0,n-length))
    return first,focus


def weak_visibility_weights(weights,ball_seen,multiplier):
    weights,seen = np.asarray(weights),np.asarray(ball_seen)
    if (weights.shape!=seen.shape or not np.isfinite(weights).all() or (weights<0).any() or
            not np.isin(seen,[0.,1.]).all() or not np.isfinite(multiplier) or multiplier<1):
        raise ValueError('Invalid weak weights, availability mask or multiplier')
    return (weights*np.where(seen<.5,multiplier,1.)).astype(np.float32)
