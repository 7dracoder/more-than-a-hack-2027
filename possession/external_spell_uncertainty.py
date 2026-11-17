"""Paired external spell resampling, matching the organiser's bootstrap rule."""
import numpy as np
from evaluation import possession_metrics as pm


def spell_information(truth_parts, prediction_parts):
    if len(truth_parts) != len(prediction_parts) or not truth_parts:
        raise ValueError('Need corresponding nonempty half lists')
    if any(len(t) != len(p) for t,p in zip(truth_parts,prediction_parts)):
        raise ValueError('Truth and prediction clocks differ')
    truth = np.concatenate([x for y in truth_parts for x in (y,np.array(['X']))])
    predicted = np.concatenate([x for y in prediction_parts for x in (y,np.array(['X']))])
    t,p = pm.label_codes(truth),pm.label_codes(predicted)
    runs = pm.valid_runs(t)
    p = pm.fill_abstentions(p,runs)
    gt_spells,pred_spells = pm.possession_spells(t,runs),pm.possession_spells(p,runs)
    overlaps = pm.spell_overlaps(gt_spells,pred_spells)
    flags = np.vstack([pm.matched_gt_flags(overlaps,len(gt_spells),threshold) for threshold in pm.SPELL_IOUS])
    return flags,len(pred_spells),gt_spells


def paired_draws(flags, predicted_counts, n_draws=1000, seed=0):
    flags,counts = np.asarray(flags),np.asarray(predicted_counts)
    if (flags.ndim != 4 or flags.shape[1] != 2 or flags.shape[2] != len(pm.SPELL_IOUS) or
            not flags.shape[0] or not flags.shape[-1] or counts.shape != flags.shape[:2] or
            flags.dtype != np.bool_ or (counts < 0).any() or
            not np.equal(counts,np.round(counts)).all() or not isinstance(n_draws,int) or n_draws < 1):
        raise ValueError('Need aligned view/model/IoU/GT-spell flags and prediction counts')
    n_gt = flags.shape[-1]
    rng = np.random.RandomState(seed)
    picks = np.stack([rng.randint(0,n_gt,n_gt) for _ in range(n_draws)])
    tp = np.take(flags,picks,axis=-1).sum(-1)
    fp = np.maximum(0,counts[:,:,None,None]-tp)
    fn = n_gt-tp
    f1 = np.divide(2.*tp,2*tp+fp+fn,out=np.zeros(tp.shape,np.float64),where=(2*tp+fp+fn)>0)
    # One set of possession draws is shared across models AND correlated views.
    view_scores = f1.mean(2)
    overall = view_scores.mean(0)
    return view_scores,overall[1]-overall[0]
