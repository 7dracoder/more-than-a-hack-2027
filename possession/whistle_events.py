"""External audio pilot's fixed event extraction and one-to-one point scoring."""
import numpy as np


def extract_events(score,threshold,fps=100.,merge_gap=.1,refractory=.35):
    score=np.asarray(score);mask=score>=threshold
    edges=np.diff(np.r_[False,mask,False].astype(np.int8));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    intervals=[]
    for start,end in zip(starts,ends):
        if intervals and (start-intervals[-1][1])/fps<=merge_gap:
            intervals[-1][1]=int(end)
        else:intervals.append([int(start),int(end)])
    peaks=[a+int(np.argmax(score[a:b])) for a,b in intervals]
    selected=[]
    for index in sorted(peaks,key=lambda i:(-score[i],i)):
        if all(abs(index-other)/fps>=refractory for other in selected):selected.append(index)
    return np.array(sorted(selected),dtype=float)/fps


def event_score(truth,predicted,tolerance=.5):
    truth=np.sort(np.asarray(truth,float));predicted=np.sort(np.asarray(predicted,float))
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all() or tolerance<0:
        raise ValueError('Invalid event timestamps or tolerance')
    i=j=matched=0
    while i<len(truth) and j<len(predicted):
        if predicted[j]<truth[i]-tolerance:j+=1
        elif predicted[j]>truth[i]+tolerance:i+=1
        else:matched+=1;i+=1;j+=1
    fp=len(predicted)-matched;fn=len(truth)-matched
    return dict(tp=matched,fp=fp,fn=fn,precision=matched/max(1,matched+fp),recall=matched/max(1,matched+fn),f1=2*matched/max(1,2*matched+fp+fn))
