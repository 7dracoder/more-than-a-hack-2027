from chain import *
import pickle
from sklearn.metrics import roc_auc_score
OOF=pickle.load(open('/workspace/mtah/tc/oof.pkl','rb'))
H={m:load(m) for m in TRAIN}
sc=[];sc2=[];aucs=[]
for m in TRAIN:
    tr=[h['lab'] for h in H[m]]; pr=[];pr2=[]
    for i in (0,1):
        x,M,y,v=ldt(m,i); M2,p=OOF[(m,i)]
        aucs.append(roc_auc_score(y[v],p[v]))
        pr.append(chain_decode(M,np.where(v,y*0.98+0.01,0.5),alive=(tr[i]!='D'),lam=0.5,n=len(tr[i])))
    sc.append(ev(tr,pr)[0])
print('AUC',np.round(aucs,3)); print('oracle cand labels',np.round(sc,4),np.mean(sc))
