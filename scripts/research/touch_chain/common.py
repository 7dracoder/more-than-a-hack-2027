import sys, numpy as np
sys.path.insert(0,'/workspace/mtah/repo')
from evaluation import possession_metrics as pm
TRAIN=['J03WMX','J03WN1','J03WOH','J03WOY','J03WQQ','J03WR9']; DEV='J03WPY'
def load(m):
    z=np.load(f'/workspace/mtah/cache/{m}.npz')
    return [dict(pl=z[f'pl{i}'],te=z[f'te{i}'],b=z[f'b{i}'],lab=z[f'lab{i}']) for i in (0,1)]
def ev(truths,preds):
    t=[];p=[]
    for tr,pr in zip(truths,preds): t+=[tr,np.array(['X'])]; p+=[pr,np.array(['X'])]
    r=pm.evaluate(pm.label_codes(np.concatenate(t)),pm.label_codes(np.concatenate(p)))
    return r['spell_f1_mean'], r['macro_f1'], r['n_gt_spells'], r['n_pred_spells']
def geom(h):
    pl,te,b=h['pl'],h['te'],h['b']
    d=np.linalg.norm(pl[...,:2]-b[:,None,:2],axis=-1)  # frames,slots
    d=np.where(np.isfinite(d),d,np.inf)
    dh=d[:,te==0].min(1); da=d[:,te==1].min(1)
    return d,dh,da
def runs(a):
    out=[];s=0
    for i in range(1,len(a)+1):
        if i==len(a) or a[i]!=a[s]: out.append((s,i,a[s])); s=i
    return out
