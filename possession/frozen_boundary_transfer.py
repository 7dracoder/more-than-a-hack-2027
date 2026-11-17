"""External boundary labels and analytic correction of balanced training odds."""
import numpy as np


def transition_targets(labels,step=2):
    labels=np.asarray(labels)
    if labels.ndim!=1 or not len(labels) or not isinstance(step,int) or step<1 or set(np.unique(labels))-set('HADX'):
        raise ValueError('Need a nonempty single-half label sequence and positive integer step')
    labels=labels[::step];target=np.full(len(labels),-100,np.int8)
    valid=np.isin(labels[1:],['H','A','D'])&np.isin(labels[:-1],['H','A','D'])
    target[1:]=np.where(valid,labels[1:]!=labels[:-1],-100)
    return target


def balance_weights(target,weight):
    target,weight=np.asarray(target),np.asarray(weight,float)
    if target.shape!=weight.shape or target.ndim!=1 or not np.isin(target,[0,1]).all() or not np.isfinite(weight).all() or (weight<=0).any():
        raise ValueError('Need selected binary targets and positive sampling weights')
    counts=np.bincount(target.astype(int),weights=weight,minlength=2)
    if (counts<=0).any():raise ValueError('Both boundary classes are required')
    factors=counts.sum()/(2*counts)
    return weight*factors[target.astype(int)],float(np.log(factors[1]/factors[0]))


def calibrated_probability(logits,log_weight_odds):
    from scipy.special import expit
    return expit(np.asarray(logits)-log_weight_odds)
