import sys,numpy as np
sys.path.insert(0,'/workspace/mtah'); sys.path.insert(0,'/workspace/mtah/repo')
from possession.fast_decode import path
from common import ev, runs
L=np.array(['H','A','D'])
def vdec(p,team=0.,dead=2.,cw=1.,cb=0.):
    p=np.asarray(p,float); em=np.log(np.maximum(p[:,:3],1e-9))
    tr=np.array([[0,-team,-dead],[-team,0,-dead],[-dead,-dead,0.]])
    if p.shape[1]==4 and cw:
        c=p[:,3].clip(1e-6,1-1e-6); od=(np.log(c)-np.log1p(-c)+cb)*cw
    else: od=np.zeros(len(p))
    return L[path(em,tr,od)]
z=np.load('/workspace/mtah/pt.npz'); P48=[z['p0'],z['p1']]; TR=[z['truth0'],z['truth1']]
