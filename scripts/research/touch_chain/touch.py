import sys,numpy as np
sys.path.insert(0,'/workspace/mtah'); sys.path.insert(0,'/workspace/mtah/repo')
from common import load,TRAIN,DEV
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from cand import team_dist
FPS=25
def cum_mean(a,lo,hi):
    c=np.concatenate([[0],np.cumsum(a)]); n=len(a); lo=np.clip(lo,0,n); hi=np.clip(hi,0,n)
    return (c[hi]-c[lo])/np.maximum(hi-lo,1)
def build(h,pitch,use_z=True,thr=2.0):
    d,bx=team_dist(h); te=h['te']; n=len(bx); idx=np.arange(n)
    pl=np.nan_to_num(h['pl'][...,:2])
    v=uniform_filter1d(np.gradient(bx,axis=0)*FPS,3,axis=0); sp=np.linalg.norm(v,axis=1)
    acc=np.linalg.norm(np.gradient(v,axis=0)*FPS,axis=1)
    z=np.nan_to_num(h['b'][:,2]) if use_z else np.zeros(n)
    miss=~np.isfinite(h['b'][:,0])
    D=[np.sort(d[:,te==k],1) for k in (0,1)]
    dm=[D[0][:,0],D[1][:,0]]
    J=[np.where(te==k)[0][np.argmin(d[:,te==k],1)] for k in (0,1)]
    pv=np.gradient(pl,axis=0)*FPS
    L,W=pitch; out=np.maximum.reduce([-bx[:,0],bx[:,0]-L,-bx[:,1],bx[:,1]-W])
    hx=np.nanmean(h['pl'][:250,te==0,0]); hl=1. if hx<L/2 else -1.
    lm=[(dm[k]==minimum_filter1d(dm[k],5,mode='nearest'))&(dm[k]<thr) for k in (0,1)]
    C=[np.where(lm[k])[0] for k in (0,1)]
    diff=dm[0]-dm[1]  # <0 home nearer
    nearH=(diff<0).astype(float)
    rows=[];meta=[]
    for T in (0,1):
        O=1-T; sg=1 if T==0 else -1
        cT=C[T]; cO=C[O]
        for c in cT:
            j=J[T][c]
            pre=v[max(c-4,0)]; post=v[min(c+4,n-1)]
            ang=(pre*post).sum()/(np.linalg.norm(pre)*np.linalg.norm(post)+1e-6)
            k=np.searchsorted(cT,c); prevT=c-cT[k-1] if k>0 else 999; nextT=cT[k+1]-c if k+1<len(cT) else 999
            kk=np.searchsorted(cO,c); prevO=c-cO[kk-1] if kk>0 else 999; nextO=cO[kk]-c if kk<len(cO) else 999
            if kk<len(cO) and cO[kk]==c: nextO=0
            dprevO=dm[O][cO[kk-1]] if kk>0 else 9; dnextO=dm[O][cO[kk]] if kk<len(cO) else 9
            f=[dm[T][c],dm[O][c],D[T][c,1],D[O][c,1],(d[c,te==T]<1).sum(),(d[c,te==O]<1).sum(),(d[c,te==T]<3).sum(),(d[c,te==O]<3).sum(),
               sp[c],np.linalg.norm(pre),np.linalg.norm(post),np.linalg.norm(post-pre),ang,acc[max(c-3,0):c+4].max(),
               np.linalg.norm(v[c]-pv[c,j]),np.linalg.norm(pv[c,j]),z[c],z[max(c-3,0):c+4].max(),miss[max(c-12,0):c+13].mean(),
               out[c],(bx[c,0]-L/2)*hl*sg,abs(bx[c,1]-W/2),
               min(prevT,999),min(nextT,999),min(prevO,999),min(nextO,999),dprevO,dnextO,
               dm[O][max(c-12,0):c+13].min(), dm[T][max(c-12,0):c+13].mean()]
            for lo,hi in ((-50,-25),(-25,-5),(-5,0),(1,6),(6,25),(25,50),(50,100)):
                a=cum_mean(diff*sg,np.array([c+lo]),np.array([c+hi]))[0]
                b=cum_mean(nearH if T==0 else 1-nearH,np.array([c+lo]),np.array([c+hi]))[0]
                e=cum_mean(dm[O],np.array([c+lo]),np.array([c+hi]))[0]
                g=cum_mean(dm[T],np.array([c+lo]),np.array([c+hi]))[0]
                s=cum_mean(sp,np.array([c+lo]),np.array([c+hi]))[0]
                f+= [a,b,e,g,s]
            rows.append(f); meta.append((c,T))
    X=np.array(rows,np.float32); M=np.array(meta,np.int64)
    o=np.argsort(M[:,0]*2+M[:,1],kind='stable'); return X[o],M[o]
def targets(M,y,off=4):
    t=np.minimum(M[:,0]+off,len(y)-1); lab=y[t]
    T=np.where(M[:,1]==0,'H','A')
    return (lab==T).astype(int), np.isin(lab,['H','A'])
if __name__=='__main__':
    import os; os.makedirs('/workspace/mtah/tc/tf',exist_ok=True)
    for m in TRAIN+[DEV]:
        pitch=np.load(f'/workspace/mtah/cache/{m}.npz')['pitch']
        for i,h in enumerate(load(m)):
            X,M=build(h,pitch); yy,valid=targets(M,h['lab'])
            np.savez(f'/workspace/mtah/tc/tf/{m}_{i}.npz',X=X,M=M,y=yy,valid=valid)
            print(m,i,X.shape,yy[valid].mean().round(3),flush=True)
