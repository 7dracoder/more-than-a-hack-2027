import numpy as np, numba
@numba.njit(cache=True)
def viterbi(logp, C):
    n=logp.shape[0]; S=np.zeros(3); bp=np.zeros((n,3),np.int8)
    for s in range(3): S[s]=logp[0,s]
    for t in range(1,n):
        NS=np.empty(3)
        for s in range(3):
            best=-1e18; arg=0
            for r in range(3):
                v=S[r]-C[r,s]
                if v>best: best=v; arg=r
            NS[s]=best+logp[t,s]; bp[t,s]=arg
        S=NS
    out=np.empty(n,np.int8); out[n-1]=np.argmax(S)
    for t in range(n-1,0,-1): out[t-1]=bp[t,out[t]]
    return out
L=np.array(['H','A','D'])
def decode(p,team=2.,stop=1.,eps=1e-6):
    C=np.array([[0,team,stop],[team,0,stop],[stop,stop,0]],np.float64)
    return L[viterbi(np.log(np.clip(p,eps,1)).astype(np.float64),C)]
