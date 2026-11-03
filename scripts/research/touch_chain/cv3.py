from chain import *
import pickle
from sklearn.metrics import roc_auc_score
def ldk(m,i):
    z=np.load(f'/workspace/mtah/tc/tf/{m}_{i}.npz'); return np.hstack([z['X'],z['K'],z['S']]),z['M'],z['y'],z['valid']
def fitX(data):
    X=np.concatenate([d[0][d[3]] for d in data]); Y=np.concatenate([d[2][d[3]] for d in data])
    return lgb.LGBMClassifier(**PRM).fit(X,Y)
def ctx(X,M,p,K=4):
    n=len(M); F=[p]
    for o in list(range(-K,0))+list(range(1,K+1)):
        j=np.clip(np.arange(n)+o,0,n-1); valid=(np.arange(n)+o>=0)&(np.arange(n)+o<n)
        same=(M[j,1]==M[:,1]).astype(float)
        F+= [np.where(valid,p[j],0.5), np.where(valid,same,0.5), np.where(valid,np.abs(M[j,0]-M[:,0]),999)]
    return np.hstack([X,np.stack(F,1)])
D={(m,i):ldk(m,i) for m in TRAIN+[DEV] for i in (0,1)}
# stage1 OOF
P1={}
for m in TRAIN:
    mdl=fitX([D[(x,i)] for x in TRAIN if x!=m for i in (0,1)])
    for i in (0,1): P1[(m,i)]=mdl.predict_proba(D[(m,i)][0])[:,1]
mdl1=fitX([D[(x,i)] for x in TRAIN for i in (0,1)])
for i in (0,1): P1[(DEV,i)]=mdl1.predict_proba(D[(DEV,i)][0])[:,1]
print('stage1 AUC train', np.mean([roc_auc_score(D[k][2][D[k][3]],P1[k][D[k][3]]) for k in P1 if k[0]!=DEV]).round(4),
      'dev',np.mean([roc_auc_score(D[k][2][D[k][3]],P1[k][D[k][3]]) for k in P1 if k[0]==DEV]).round(4),flush=True)
# stage2 OOF (nested: stage1 OOF features already)
D2={k:(ctx(D[k][0],D[k][1],P1[k]),D[k][1],D[k][2],D[k][3]) for k in D}
P2={}
for m in TRAIN:
    mdl=fitX([D2[(x,i)] for x in TRAIN if x!=m for i in (0,1)])
    for i in (0,1): P2[(m,i)]=mdl.predict_proba(D2[(m,i)][0])[:,1]
mdl2=fitX([D2[(x,i)] for x in TRAIN for i in (0,1)])
for i in (0,1): P2[(DEV,i)]=mdl2.predict_proba(D2[(DEV,i)][0])[:,1]
print('stage2 AUC train', np.mean([roc_auc_score(D[k][2][D[k][3]],P2[k][D[k][3]]) for k in P2 if k[0]!=DEV]).round(4),
      'dev',np.mean([roc_auc_score(D[k][2][D[k][3]],P2[k][D[k][3]]) for k in P2 if k[0]==DEV]).round(4),flush=True)
pickle.dump(dict(P1=P1,P2=P2),open('/workspace/mtah/tc/p12s.pkl','wb'))
H={m:load(m) for m in TRAIN+[DEV]}
for nm,PP in (('s1',P1),('s2',P2)):
    for lam in (0.25,0.5,1,2):
        sc=[]
        for m in TRAIN:
            tr=[h['lab'] for h in H[m]]
            sc.append(ev(tr,[chain_decode(D[(m,i)][1],PP[(m,i)],alive=(tr[i]!='D'),lam=lam,n=len(tr[i])) for i in (0,1)])[0])
        tr=[h['lab'] for h in H[DEV]]
        dv=ev(tr,[chain_decode(D[(DEV,i)][1],PP[(DEV,i)],alive=(tr[i]!='D'),lam=lam,n=len(tr[i])) for i in (0,1)])
        print(nm,'lam',lam,'trainOOF',np.mean(sc).round(4),'dev(truthD)',np.round(dv[0],4),dv[3],flush=True)
