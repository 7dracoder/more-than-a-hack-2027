from chain import *
from r48 import vdec,P48,TR,path,L
import pickle
pp=pickle.load(open('/workspace/mtah/tc/p12.pkl','rb'))
def chain_post(M,p,alive,lam,n,eps=1e-4):
    """forward-backward over candidates in alive segments; frame q(H)"""
    q=np.full(n,0.5)
    lp=np.log(np.clip(p,eps,1-eps)); lq=np.log(np.clip(1-p,eps,1-eps))
    T=np.log(np.array([[1,np.exp(-lam)],[np.exp(-lam),1]])/(1+np.exp(-lam)))
    for s,e,v in runs(alive):
        if not v: continue
        k=np.where((M[:,0]>=s)&(M[:,0]<e))[0]
        if len(k)==0: continue
        Tm=M[k,1]; em=np.stack([np.where(Tm==0,lp[k],lq[k]),np.where(Tm==1,lp[k],lq[k])],1)
        K=len(k); a=np.zeros((K,2)); b=np.zeros((K,2)); a[0]=em[0]
        for j in range(1,K): a[j]=np.logaddexp(a[j-1][:,None]+T,0).max(0)*0+np.logaddexp.reduce(a[j-1][:,None]+T,axis=0)+em[j]
        for j in range(K-2,-1,-1): b[j]=np.logaddexp.reduce(T+em[j+1][None]+b[j+1][None],axis=1)
        g=a+b; g=np.exp(g-g.max(1,keepdims=True)); g/=g.sum(1,keepdims=True)
        fr=M[k,0]; q[s:fr[0]]=g[0,0]
        for j in range(K):
            q[fr[j]:(fr[j+1] if j+1<K else e)]=g[j,0]
    return q
M=[np.load(f'/workspace/mtah/tc/tf/{DEV}_{i}.npz')['M'] for i in (0,1)]
base=[vdec(p) for p in P48]
print('run48',ev(TR,base))
res=[]
for lam in (0.5,1.0):
    Q=[chain_post(M[i],pp['P2'][(DEV,i)],alive=(base[i]!='D'),lam=lam,n=len(TR[i])) for i in (0,1)]
    for a in (0.1,0.2,0.35,0.5,0.75,1.0):
        out=[]
        for i in (0,1):
            p=P48[i].copy().astype(float); q=np.clip(Q[i],1e-3,1-1e-3)
            ha=p[:,0]+p[:,1]; h=p[:,0]**1*q**a; aa=p[:,1]*(1-q)**a; s=h+aa
            p[:,0]=h/s*ha; p[:,1]=aa/s*ha
            out.append(vdec(p))
        r0=ev([TR[0]],[out[0]]); r1=ev([TR[1]],[out[1]]); r=ev(TR,out)
        print(lam,a,'full',np.round(r[0],4),r[3],'h0',np.round(r0[0],4),'h1',np.round(r1[0],4),flush=True)
print('base halves',ev([TR[0]],[base[0]])[0],ev([TR[1]],[base[1]])[0])
