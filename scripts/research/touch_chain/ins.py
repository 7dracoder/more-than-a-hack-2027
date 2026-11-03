from short import *
base=[vdec(p) for p in P48]
print('run48',ev(TR,base))
for t in (0.8,0.9,0.95,0.98):
    out=[]; n=0
    for i in (0,1):
        Mi=np.load(f'/workspace/mtah/tc/tf/{DEV}_{i}.npz')['M']; p=pp['P2'][(DEV,i)]
        ch=chain_decode(Mi,p,alive=(base[i]!='D'),lam=0.5,n=len(base[i]))
        b=base[i].copy()
        for s,e,l,pm in short_inserts(ch,Mi,p):
            if pm>=t and (b[s:e]!=l).all() and (b[s:e]!='D').all(): b[s:e]=l; n+=1
        out.append(b)
    print(t,'inserted',n,ev(TR,out))
