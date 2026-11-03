"""Touch-chain features: tracking-only (player xy, team, ball xy[,z]). Labels never enter."""
import sys, numpy as np
from scipy.ndimage import minimum_filter1d, maximum_filter1d, uniform_filter1d
sys.path.insert(0,'/workspace/mtah'); sys.path.insert(0,'/workspace/mtah/repo')
from common import load, TRAIN, DEV
FPS=25
def _shift(a,k):
    o=np.empty_like(a); 
    if k>0: o[k:]=a[:-k]; o[:k]=a[:1]
    elif k<0: o[:k]=a[-k:]; o[k:]=a[-1:]
    else: o[:]=a
    return o
def roll_min(a,w,side):  # side -1 past window [t-w,t], +1 future [t,t+w], 0 centered
    m=minimum_filter1d(a,w+1 if side else 2*w+1,mode='nearest',origin=(w//2 if side<0 else -(w//2) if side>0 else 0) if side else 0)
    return m
def build(h,pitch,use_z=False):
    pl,te,b=h['pl'],h['te'],h['b']
    n=len(b); bx=b[:,:2].astype(np.float64)
    ok=np.isfinite(bx).all(1)
    # interpolate ball gaps
    idx=np.arange(n)
    for j in range(2):
        bx[:,j]=np.interp(idx,idx[ok],bx[ok,j])
    v=np.gradient(bx,axis=0)*FPS
    vs=uniform_filter1d(v,5,axis=0)
    sp=np.linalg.norm(vs,axis=1)
    acc=np.linalg.norm(np.gradient(vs,axis=0)*FPS,axis=1)
    vp=_shift(vs,3); vf=_shift(vs,-3)
    cosang=(vp*vf).sum(1)/(np.linalg.norm(vp,axis=1)*np.linalg.norm(vf,axis=1)+1e-6)
    dv=np.linalg.norm(vf-vp,axis=1)
    P=pl[...,:2]
    d=np.linalg.norm(P-bx[:,None,:].astype(np.float32),axis=-1); d=np.where(np.isfinite(d),d,99.)
    H=te==0; A=te==1
    dH=np.sort(d[:,H],1)[:,:3]; dA=np.sort(d[:,A],1)[:,:3]
    dh,da=dH[:,0],dA[:,0]
    # player speeds near ball
    pv=np.gradient(np.nan_to_num(P),axis=0)*FPS
    jh=np.argmin(np.where(H[None],d,99),1); ja=np.argmin(np.where(A[None],d,99),1)
    # relative velocity of ball vs nearest player
    rvh=np.linalg.norm(vs-pv[idx,jh],axis=1); rva=np.linalg.norm(vs-pv[idx,ja],axis=1)
    cntH1=(d[:,H]<1.0).sum(1); cntA1=(d[:,A]<1.0).sum(1)
    cntH3=(d[:,H]<3.0).sum(1); cntA3=(d[:,A]<3.0).sum(1)
    L,W=pitch
    out=np.maximum.reduce([-bx[:,0],bx[:,0]-L,-bx[:,1],bx[:,1]-W])
    # attack direction: home mean x in first frames
    hx=np.nanmean(P[:250,H,0]); homeleft=1. if hx<L/2 else -1.
    xr=(bx[:,0]-L/2)*homeleft; yr=np.abs(bx[:,1]-W/2)
    # stationary run length of ball
    still=(sp<0.5).astype(np.float64)
    stillc=np.zeros(n); 
    for t in range(1,n): stillc[t]=stillc[t-1]+1 if still[t] else 0
    stillf=np.zeros(n)
    for t in range(n-2,-1,-1): stillf[t]=stillf[t+1]+1 if still[t] else 0
    # out-of-bounds recency
    oob=(out>0.5).astype(np.float64)
    def since(mask):
        s=np.full(n,9999.); c=9999.
        for t in range(n):
            c=0 if mask[t] else c+1; s[t]=c
        return np.minimum(s,2000)
    def until(mask): return since(mask[::-1])[::-1]
    sinceoob=since(oob>0); untiloob=until(oob>0)
    missing=(~ok).astype(np.float64)
    # touch proxies: close & ball kink
    th=(dh<1.2); ta=(da<1.2)
    sinceH=since(th); sinceA=since(ta); untilH=until(th); untilA=until(ta)
    # player density / team centroid
    cxH=np.nanmean(P[:,H,0],1); cxA=np.nanmean(P[:,A,0],1)
    spreadH=np.nanstd(P[:,H,0],1); spreadA=np.nanstd(P[:,A,0],1)
    # number of visible players (stoppages often have bunching)
    meanplsp=np.nanmean(np.linalg.norm(pv,axis=-1),1)
    base=dict(dh=dh,da=da,dh2=dH[:,1],da2=dA[:,1],ddiff=dh-da,sp=sp,acc=acc,cos=cosang,dv=dv,rvh=rvh,rva=rva,
        cH1=cntH1,cA1=cntA1,cH3=cntH3,cA3=cntA3,out=out,xr=xr,yr=yr,stillc=np.minimum(stillc,2000),stillf=np.minimum(stillf,2000),
        sinceoob=sinceoob,untiloob=untiloob,miss=missing,sinceH=sinceH,sinceA=sinceA,untilH=untilH,untilA=untilA,
        cxH=(cxH-L/2)*homeleft,cxA=(cxA-L/2)*homeleft,sprH=spreadH,sprA=spreadA,plsp=meanplsp)
    if use_z:
        z=np.nan_to_num(b[:,2]); base['z']=z
    F=dict(base)
    for w in (5,12,25,50,100):
        for k in ('dh','da','ddiff','sp','dv'):
            a=base[k]
            F[f'{k}_m{w}']=uniform_filter1d(a,2*w+1,mode='nearest')
            # past/future means
            c=np.cumsum(np.r_[0,a]); 
            lo=np.clip(idx-w,0,n); F[f'{k}_p{w}']=(c[idx+1]-c[lo])/(idx+1-lo)
            hi=np.clip(idx+w+1,0,n); F[f'{k}_f{w}']=(c[hi]-c[idx])/(hi-idx)
        F[f'minh_m{w}']=minimum_filter1d(dh,2*w+1,mode='nearest'); F[f'mina_m{w}']=minimum_filter1d(da,2*w+1,mode='nearest')
        F[f'maxsp_m{w}']=maximum_filter1d(sp,2*w+1,mode='nearest')
        F[f'maxout_m{w}']=maximum_filter1d(out,2*w+1,mode='nearest')
        F[f'plsp_m{w}']=uniform_filter1d(meanplsp,2*w+1,mode='nearest')
    names=sorted(F); X=np.stack([F[k] for k in names],1).astype(np.float32)
    return X,names
if __name__=='__main__':
    import os
    os.makedirs('/workspace/mtah/tc/feat',exist_ok=True)
    for m in TRAIN+[DEV]:
        pitch=np.load(f'/workspace/mtah/cache/{m}.npz')['pitch']
        for i,h in enumerate(load(m)):
            X,names=build(h,pitch)
            np.savez(f'/workspace/mtah/tc/feat/{m}_{i}.npz',X=X,y=h['lab'],names=np.array(names))
            print(m,i,X.shape,flush=True)
