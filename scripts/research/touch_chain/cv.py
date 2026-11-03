from chain import *
from r48 import vdec, P48, TR
import itertools
# OOF on train
OOF={}
for m in TRAIN:
    mdl=fit([x for x in TRAIN if x!=m])
    for i in (0,1):
        x,M,y,v=ldt(m,i); OOF[(m,i)]=(M,mdl.predict_proba(x)[:,1])
    print('fold',m,flush=True)
import pickle; pickle.dump(OOF,open('/workspace/mtah/tc/oof.pkl','wb'))
H={m:load(m) for m in TRAIN}
for lam in (0.5,1,2,3,5):
    sc=[]
    for m in TRAIN:
        tr=[h['lab'] for h in H[m]]
        pr=[chain_decode(*OOF[(m,i)],alive=(tr[i]!='D'),lam=lam,n=len(tr[i])) for i in (0,1)]
        sc.append(ev(tr,pr)[0])
    print('train-OOF truthD lam',lam,np.round(sc,4),np.mean(sc).round(4),flush=True)
