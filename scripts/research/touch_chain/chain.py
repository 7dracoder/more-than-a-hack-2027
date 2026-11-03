import sys,numpy as np, lightgbm as lgb
sys.path.insert(0,'/workspace/mtah'); sys.path.insert(0,'/workspace/mtah/repo')
from common import TRAIN,DEV,ev,runs,load
def ldt(m,i):
    z=np.load(f'/workspace/mtah/tc/tf/{m}_{i}.npz'); return z['X'],z['M'],z['y'],z['valid']
PRM=dict(n_estimators=400,learning_rate=0.03,num_leaves=31,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,min_child_samples=30,verbose=-1,n_jobs=8)
def fit(ms):
    X=[];Y=[]
    for m in ms:
        for i in (0,1):
            x,M,y,v=ldt(m,i); X.append(x[v]); Y.append(y[v])
    return lgb.LGBMClassifier(**PRM).fit(np.concatenate(X),np.concatenate(Y))
def chain_decode(M,p,alive,lam=2.0,n=None,eps=1e-4):
    """Candidate-level 2-state Viterbi per alive segment; render label from each candidate frame."""
    out=np.full(n,'D','<U1')
    lp=np.log(np.clip(p,eps,1-eps)); lq=np.log(np.clip(1-p,eps,1-eps))
    for s,e,v in runs(alive):
        if not v: continue
        k=np.where((M[:,0]>=s)&(M[:,0]<e))[0]
        if len(k)==0: out[s:e]='X'; continue
        T=M[k,1]
        # emissions: state 0=H,1=A
        em=np.stack([np.where(T==0,lp[k],lq[k]),np.where(T==1,lp[k],lq[k])],1)
        S=em[0].copy(); bp=np.zeros((len(k),2),np.int8)
        for j in range(1,len(k)):
            NS=np.empty(2)
            for st in (0,1):
                a=S[st]; b=S[1-st]-lam
                bp[j,st]=st if a>=b else 1-st; NS[st]=max(a,b)+em[j,st]
            S=NS
        path=np.empty(len(k),np.int8); path[-1]=np.argmax(S)
        for j in range(len(k)-1,0,-1): path[j-1]=bp[j,path[j]]
        lab=np.array(['H','A'])[path]
        fr=M[k,0]
        out[s:e]=lab[0]
        for j in range(1,len(k)):
            if path[j]!=path[j-1]: out[fr[j]:e]=lab[j]
    return out
