import sys,numpy as np
sys.path.insert(0,'/workspace/mtah'); sys.path.insert(0,'/workspace/mtah/repo')
from common import load,TRAIN,DEV
from kin import interp_ball
OFF=np.arange(-12,13,2)
def seqf(h,M):
    pl=h['pl'][...,:2]; te=h['te']; P=interp_ball(h['b']); n=len(P)
    sp=np.r_[0,np.linalg.norm(np.diff(P[:,:2],axis=0),axis=1)*25]
    F=[]
    for c,T in M:
        t=np.clip(c+OFF,0,n-1)
        dT=np.linalg.norm(pl[c][te==T]-P[c,:2],axis=1); dT=np.where(np.isfinite(dT),dT,99); j=np.where(te==T)[0][np.argmin(dT)]
        dO=np.linalg.norm(pl[c][te!=T]-P[c,:2],axis=1); dO=np.where(np.isfinite(dO),dO,99); k=np.where(te!=T)[0][np.argmin(dO)]
        dj=np.linalg.norm(pl[t,j]-P[t,:2],axis=1); dk=np.linalg.norm(pl[t,k]-P[t,:2],axis=1)
        # all-team nearest over time
        dTt=np.nanmin(np.linalg.norm(pl[t][:,te==T]-P[t,None,:2],axis=-1),1); dOt=np.nanmin(np.linalg.norm(pl[t][:,te!=T]-P[t,None,:2],axis=-1),1)
        F.append(np.r_[np.nan_to_num(dj,nan=99),np.nan_to_num(dk,nan=99),dTt-dOt,sp[t],P[t,2]])
    return np.array(F,np.float32)
if __name__=='__main__':
    for m in TRAIN+[DEV]:
        for i,h in enumerate(load(m)):
            z=dict(np.load(f'/workspace/mtah/tc/tf/{m}_{i}.npz')); z['S']=seqf(h,z['M'])
            np.savez(f'/workspace/mtah/tc/tf/{m}_{i}.npz',**z); print(m,i,z['S'].shape,flush=True)
