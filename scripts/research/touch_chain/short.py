from chain import *
from r48 import vdec,P48,TR
import pickle
pp=pickle.load(open('/workspace/mtah/tc/p12.pkl','rb'))
def spells(lab):
    return [(s,e,l) for s,e,l in runs(lab) if l in 'HA']
def iou(a,b): 
    i=max(0,min(a[1],b[1])-max(a[0],b[0])); return i/(max(a[1],b[1])-min(a[0],b[0]))
def short_inserts(lab,M,p,maxlen=50):
    """sandwiched short spells in chain output, with min p of candidates starting them"""
    out=[]; r=runs(lab)
    for k in range(1,len(r)-1):
        s,e,l=r[k]
        if l in 'HA' and e-s<maxlen and r[k-1][2]==r[k+1][2] and r[k-1][2] in 'HA' and r[k-1][2]!=l:
            T=0 if l=='H' else 1; j=np.where((M[:,0]>=s-2)&(M[:,0]<e)&(M[:,1]==T))[0]
            out.append((s,e,l,p[j].max() if len(j) else 0))
    return out
def prec(m,i,lab,Mi,p,lam=0.5):
    ch=chain_decode(Mi,p,alive=(lab!='D'),lam=lam,n=len(lab))
    ins=short_inserts(ch,Mi,p); ts=spells(lab); res=[]
    for s,e,l,pm in ins:
        ok=any(iou((s,e),(a,b))>=0.5 and l==ll for a,b,ll in ts if b>s-100 and a<e+100)
        res.append((pm,ok))
    return res
H={m:load(m) for m in TRAIN}
R=[]
for m in TRAIN:
    for i in (0,1):
        R+=prec(m,i,H[m][i]['lab'],np.load(f'/workspace/mtah/tc/tf/{m}_{i}.npz')['M'],pp['P2'][(m,i)])
R=np.array(R)
for t in (0.5,0.7,0.8,0.9,0.95):
    s=R[:,0]>=t; print('train thr',t,'n',s.sum(),'prec',R[s,1].mean().round(3))
