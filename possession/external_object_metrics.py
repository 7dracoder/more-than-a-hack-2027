"""Single-class IoU-0.5 matching and 101-point interpolated detection AP."""
import numpy as np


def matches(boxes,confidence,truth):
    boxes=np.asarray(boxes,float).reshape(-1,4);truth=np.asarray(truth,float).reshape(-1,4);confidence=np.asarray(confidence,float)
    if confidence.shape!=(len(boxes),) or not np.isfinite(boxes).all() or not np.isfinite(truth).all() or not np.isfinite(confidence).all():raise ValueError('Invalid boxes or confidence')
    if ((confidence<0)|(confidence>1)).any() or (boxes[:,2:]<boxes[:,:2]).any() or (truth[:,2:]<=truth[:,:2]).any():raise ValueError('Invalid geometry or score')
    predicted=np.zeros(len(boxes),bool);assigned=np.full(len(truth),-1,int)
    if not len(boxes) or not len(truth):return predicted,assigned
    intersection=np.maximum(0,np.minimum(boxes[:,None,2:],truth[None,:,2:])-np.maximum(boxes[:,None,:2],truth[None,:,:2])).prod(2)
    union=(boxes[:,2:]-boxes[:,:2]).prod(1)[:,None]+(truth[:,2:]-truth[:,:2]).prod(1)[None,:]-intersection
    iou=intersection/np.maximum(union,1e-12)
    for i in np.argsort(-confidence,kind='stable'):
        available=np.where(assigned<0,iou[i],-1.);j=int(available.argmax())
        if available[j]>=.5:predicted[i]=True;assigned[j]=i
    return predicted,assigned


def summary(images):
    scores=[];flags=[];n_gt=0;n_small=0;small_matched=0
    for row in images:
        matched,assigned=matches(row['boxes'],row['confidence'],row['truth'])
        truth=np.asarray(row['truth']).reshape(-1,4);small=(truth[:,2:]-truth[:,:2]).prod(1)<32**2
        scores.extend(row['confidence']);flags.extend(matched);n_gt+=len(truth);n_small+=int(small.sum());small_matched+=int(((assigned>=0)&small).sum())
    order=np.argsort(-np.asarray(scores),kind='stable');flags=np.asarray(flags,bool)[order]
    tp=np.cumsum(flags);fp=np.cumsum(~flags);recall=tp/max(1,n_gt);precision=tp/np.maximum(1,tp+fp)
    ap=float(np.mean([precision[recall>=r].max(initial=0.) for r in np.linspace(0,1,101)])) if n_gt else 0.
    count=int(flags.sum());false_positive=len(flags)-count;false_negative=n_gt-count
    return dict(images=len(images),ground_truth=n_gt,predictions=len(flags),tp=count,fp=false_positive,fn=false_negative,
        precision=count/max(1,count+false_positive),recall=count/max(1,n_gt),f1=2*count/max(1,2*count+false_positive+false_negative),ap50=ap,
        small_ground_truth=n_small,small_matched=small_matched,small_recall=small_matched/max(1,n_small))
