from short import *
from cand import team_dist
base=[vdec(p) for p in P48]
HD=load(DEV)
stats=[]
for i in (0,1):
    d,bx=team_dist(HD[i]); te=HD[i]['te']; dm=[d[:,te==0].min(1),d[:,te==1].min(1)]
    z=np.nan_to_num(HD[i]['b'][:,2]); miss=~np.isfinite(HD[i]['b'][:,0])
    ps=spells(base[i]); 
    for s,e,l in spells(TR[i]):
        ok=any(iou((s,e),(a,b))>=0.5 and l==ll for a,b,ll in ps if b>s-500 and a<e+500)
        T=0 if l=='H' else 1
        stats.append((ok,e-s,dm[T][max(s-5,0):s+6].min(),z[max(s-5,0):s+6].max(),miss[max(s-5,0):s+6].mean(), (base[i][s:e]==l).mean()))
S=np.array(stats)
for ok in (1,0):
    x=S[S[:,0]==ok]; print('matched' if ok else 'missed',len(x),'len med',np.median(x[:,1]),'mindist pct',np.percentile(x[:,2],[25,50,75,90]).round(2),'z>1.5 frac',(x[:,3]>1.5).mean().round(2),'ballmiss',x[:,4].mean().round(3),'frac frames right',np.median(x[:,5]).round(2))
x=S[S[:,0]==0]; print('missed len buckets', np.histogram(x[:,1],[0,25,50,125,250,1e6])[0], 'frac right buckets',np.histogram(x[:,5],[0,.1,.5,.9,1.01])[0])
