import sys, numpy as np, xml.etree.ElementTree as ET, glob
sys.path.insert(0,'/workspace/mtah/repo')
from possession.external import external_labels
m=sys.argv[1]; folder='/workspace/mtah/dfl/'
meta=glob.glob(folder+'*matchinformation*'+m+'.xml')[0]; trk=glob.glob(folder+'*positions*'+m+'.xml')[0]
root=ET.parse(meta).getroot(); g=next(root.iter('General'))
sides={g.get('HomeTeamId'):0,g.get('GuestTeamId'):1}
env=next(root.iter('Environment')); pitch=(float(env.get('PitchX')),float(env.get('PitchY')))
P={}
for _,it in ET.iterparse(trk,events=('end',)):
    if it.tag!='FrameSet': continue
    team,person,per=it.get('TeamId',''),it.get('PersonId',''),it.get('GameSection')
    isb=team.lower()=='ball' or person.lower()=='ball'
    if per in ('firstHalf','secondHalf') and (team in sides or isb):
        rows=list(it.iter('Frame'))
        ids=np.fromiter((int(x.get('N')) for x in rows),np.int64)
        if isb:
            xyz=np.array([(float(x.get('X','nan')),float(x.get('Y','nan')),float(x.get('Z','nan')),float(x.get('S','nan'))) for x in rows],np.float32)
            lab=external_labels([x.get('BallStatus','') for x in rows],[x.get('BallPossession','') for x in rows])
            P.setdefault(per,{})['ball']=(ids,xyz,lab)
        else:
            xy=np.array([(float(x.get('X','nan')),float(x.get('Y','nan')),float(x.get('S','nan'))) for x in rows],np.float32)
            P.setdefault(per,{})[person]=(ids,xy,sides[team])
    it.clear()
out={}
for i,per in enumerate(('firstHalf','secondHalf')):
    o=P[per]; bids,bxyz,lab=o.pop('ball')
    f0=bids.min(); n=bids.max()-f0+1
    pl=np.full((n,len(o),3),np.nan,np.float32); te=np.zeros(len(o),np.int8)
    for s,(k,(ids,xy,sd)) in enumerate(sorted(o.items())):
        j=ids-f0; ok=(j>=0)&(j<n); pl[j[ok],s]=xy[ok]; te[s]=sd
    b=np.full((n,4),np.nan,np.float32); L=np.full(n,'X','<U1'); b[bids-f0]=bxyz; L[bids-f0]=lab
    pl[...,:2]+=np.array(pitch,np.float32)/2; b[:,:2]+=np.array(pitch,np.float32)/2
    out.update({f'pl{i}':pl,f'te{i}':te,f'b{i}':b,f'lab{i}':L})
np.savez_compressed(f'/workspace/mtah/cache/{m}.npz',pitch=np.array(pitch),**out)
print(m,[out[f'pl{i}'].shape for i in (0,1)])
