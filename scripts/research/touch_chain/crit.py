from chain import *
import pickle
from sklearn.metrics import roc_auc_score
pp=pickle.load(open('/workspace/mtah/tc/p12s.pkl','rb'))
for grp in (TRAIN,[DEV]):
    Y=[];P=[]
    for m in grp:
        H=load(m)
        for i in (0,1):
            z=np.load(f'/workspace/mtah/tc/tf/{m}_{i}.npz'); M,y,v=z['M'],z['y'],z['valid']; lab=H[i]['lab']
            prev=lab[np.maximum(M[:,0]-3,0)]; T=np.where(M[:,1]==0,'H','A')
            c=v&np.isin(prev,['H','A'])&(prev!=T)
            Y.append(y[c]);P.append(pp['P2'][(m,i)][c])
    Y=np.concatenate(Y);P=np.concatenate(P)
    print(grp[0],'critical n',len(Y),'pos rate',Y.mean().round(3),'AUC',round(roc_auc_score(Y,P),3),'acc@.5',((P>.5)==Y).mean().round(3))
