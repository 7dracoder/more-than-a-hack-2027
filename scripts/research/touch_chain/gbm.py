import sys,numpy as np, lightgbm as lgb
sys.path.insert(0,'/workspace/mtah'); sys.path.insert(0,'/workspace/mtah/repo')
from common import TRAIN,DEV,ev
from dec import decode
def ld(m,i):
    z=np.load(f'/workspace/mtah/tc/feat/{m}_{i}.npz'); return z['X'],z['y']
cls={'H':0,'A':1,'D':2}
Xs=[];ys=[]
for m in TRAIN:
    for i in (0,1):
        X,y=ld(m,i); k=(y!='X'); sel=np.zeros(len(y),bool); sel[::3]=True
        ch=np.where(y[1:]!=y[:-1])[0]
        for c in ch: sel[max(0,c-25):c+25]=True
        sel&=k; Xs.append(X[sel]); ys.append(np.array([cls[v] for v in y[sel]]))
X=np.concatenate(Xs); y=np.concatenate(ys); print(X.shape,flush=True)
mdl=lgb.LGBMClassifier(n_estimators=600,learning_rate=0.05,num_leaves=63,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,min_child_samples=50,verbose=-1,n_jobs=8)
mdl.fit(X,y)
mdl.booster_.save_model('/workspace/mtah/tc/gbm1.txt')
tr=[];P=[]
for i in (0,1):
    Xd,yd=ld(DEV,i); P.append(mdl.predict_proba(Xd)); tr.append(yd)
np.savez('/workspace/mtah/tc/gbm1_dev.npz',p0=P[0],p1=P[1])
L=np.array(['H','A','D'])
print('argmax',ev(tr,[L[p.argmax(1)] for p in P]))
for team in (1,2,4,6,8):
    for stop in (1,2,4):
        print(team,stop,ev(tr,[decode(p,team,stop) for p in P]),flush=True)
