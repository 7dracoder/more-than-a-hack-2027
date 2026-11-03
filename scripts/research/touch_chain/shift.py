from r48 import *
base=[vdec(p) for p in P48]
for k in (-6,-3,-1,0,1,3,6):
    out=[np.roll(b,k) for b in base]; print(k,np.round(ev(TR,out)[0],4))
