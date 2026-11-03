import sys,numpy as np
sys.path.insert(0,'/workspace/mtah'); sys.path.insert(0,'/workspace/mtah/repo')
from common import load,TRAIN,DEV
from scipy.ndimage import minimum_filter1d
def team_dist(h):
    pl,te,b=h['pl'],h['te'],h['b']; n=len(b); bx=b[:,:2].astype(np.float64); ok=np.isfinite(bx).all(1); idx=np.arange(n)
    for j in range(2): bx[:,j]=np.interp(idx,idx[ok],bx[ok,j])
    d=np.linalg.norm(pl[...,:2]-bx[:,None,:].astype(np.float32),axis=-1); d=np.where(np.isfinite(d),d,99.)
    return d,bx
def candidates(dT,thr=2.0,w=2):
    lm=(dT==minimum_filter1d(dT,2*w+1,mode='nearest'))&(dT<thr)
    return np.where(lm)[0]
if __name__=='__main__':
    for thr in (1.5,2.0,2.5):
      tot=hit=0; offs=[]; ncand=0
      for m in TRAIN:
        for h in load(m):
            d,bx=team_dist(h); te=h['te']; y=h['lab']
            D=[d[:,te==0].min(1),d[:,te==1].min(1)]
            C=[candidates(D[0],thr),candidates(D[1],thr)]; ncand+=len(C[0])+len(C[1])
            ch=np.where((y[1:]!=y[:-1])&np.isin(y[1:],['H','A'])&np.isin(y[:-1],['H','A']))[0]+1
            for c in ch:
                T=0 if y[c]=='H' else 1; tot+=1
                k=C[T]; j=np.searchsorted(k,c); best=min([abs(k[i]-c) for i in (j-1,j) if 0<=i<len(k)]+[999])
                sgn=[k[i]-c for i in (j-1,j) if 0<=i<len(k)]
                if best<=5: hit+=1; offs.append(min(sgn,key=abs))
      offs=np.array(offs); print(thr,'changes',tot,'recall<=5f',hit/tot,'cands',ncand,'offset pct',np.percentile(offs,[5,25,50,75,95]))
