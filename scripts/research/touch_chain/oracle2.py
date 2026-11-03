from r48 import *
base=[vdec(p) for p in P48]; print('run48',ev(TR,base))
a=[];b=[]
for bb,t in zip(base,TR):
    x=bb.copy(); x[t=='D']='D'; m=(x=='D')&(t!='D'); 
    # fill wrongly-dead frames with nearest alive pred? use truth team there (pure dead oracle needs team) -> use previous alive pred
    x[m]='X'; a.append(x)
    y=t.copy(); alive=(bb!='D'); y[(bb=='D')]='D'; mm=(bb!='D')&(t=='D'); y[mm]='X'; b.append(y)
print('truth dead-mask + run48 teams',ev(TR,a))
print('run48 dead-mask + truth teams',ev(TR,b))
