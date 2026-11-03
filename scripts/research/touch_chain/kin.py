import sys,numpy as np
sys.path.insert(0,'/workspace/mtah'); sys.path.insert(0,'/workspace/mtah/repo')
from common import load,TRAIN,DEV
def interp_ball(b):
    n=len(b); idx=np.arange(n); out=b[:,:3].astype(np.float64).copy()
    for j in range(3):
        ok=np.isfinite(out[:,j]); out[:,j]=np.interp(idx,idx[ok],out[ok,j]) if ok.any() else 0
    return out
def linfit(P,lo,hi):
    t=np.arange(lo,hi,dtype=float); Y=P[lo:hi]; tm=t.mean(); A=((t-tm)[:,None]*(Y-Y.mean(0))).sum(0)/(((t-tm)**2).sum()+1e-9)
    res=Y-(Y.mean(0)+A*(t-tm)[:,None]); return A*25, np.sqrt((res**2).sum(1).mean())
def kin(b,cands,W=(4,6,10)):
    P=interp_ball(b); n=len(P); F=[]
    for c in cands:
        f=[]
        for w in W:
            lo=max(c-w,0); hi=min(c+w+1,n)
            if c-lo<2 or hi-c<3: f+=[0]*9; continue
            vpre,rpre=linfit(P[:,:2],lo,c+1); vpost,rpost=linfit(P[:,:2],c,hi); vall,rall=linfit(P[:,:2],lo,hi)
            sp0=np.linalg.norm(vpre); sp1=np.linalg.norm(vpost)
            cos=(vpre*vpost).sum()/(sp0*sp1+1e-6)
            zpre,_=linfit(P[:,2:3],lo,c+1); zpost,_=linfit(P[:,2:3],c,hi)
            f+=[sp0,sp1,np.linalg.norm(vpost-vpre),cos,rall,rall-0.5*(rpre+rpost),sp1/(sp0+0.5),float(zpost[0]-zpre[0]),P[c,2]]
        # frame of max kink in +-4
        F.append(f)
    return np.array(F,np.float32)
if __name__=='__main__':
    for m in TRAIN+[DEV]:
        for i,h in enumerate(load(m)):
            z=dict(np.load(f'/workspace/mtah/tc/tf/{m}_{i}.npz'))
            K=kin(h['b'],z['M'][:,0]); z['K']=K
            np.savez(f'/workspace/mtah/tc/tf/{m}_{i}.npz',**z); print(m,i,K.shape,flush=True)
