"""Optional compiled evaluation of the unchanged three-state MAP recurrence."""
import numpy as np


def path_reference(emission, transition, odds):
    back=np.zeros(emission.shape,np.int8)
    best=emission[0].copy()
    off_diagonal=1-np.eye(3)
    for i in range(1,len(emission)):
        # Keep addition grouping identical to the original implementation.
        candidate=best[:,None]+(transition+off_diagonal*odds[i])
        back[i]=candidate.argmax(0)
        best=candidate[back[i],np.arange(3)]+emission[i]
        best-=best.max()
    path=np.empty(len(emission),np.int8)
    path[-1]=best.argmax()
    for i in range(len(path)-2,-1,-1):
        path[i]=back[i+1,path[i+1]]
    return path


def _path_scalar(emission,transition,odds):
    back=np.zeros(emission.shape,np.int8)
    best=emission[0].copy()
    next_best=np.empty(3,np.float64)
    for i in range(1,len(emission)):
        for destination in range(3):
            value=best[0]+(transition[0,destination]+(odds[i] if destination!=0 else 0.))
            parent=0
            for source in range(1,3):
                candidate=best[source]+(transition[source,destination]+(odds[i] if source!=destination else 0.))
                if candidate>value:
                    value=candidate
                    parent=source
            back[i,destination]=parent
            next_best[destination]=value+emission[i,destination]
        maximum=max(next_best[0],next_best[1],next_best[2])
        for destination in range(3):
            best[destination]=next_best[destination]-maximum
    path=np.empty(len(emission),np.int8)
    last=0
    for state in range(1,3):
        if best[state]>best[last]:
            last=state
    path[-1]=last
    for i in range(len(path)-2,-1,-1):
        path[i]=back[i+1,path[i+1]]
    return path


try:
    from numba import njit
except ImportError:
    path=path_reference
    BACKEND='numpy'
else:
    path=njit(cache=True,fastmath=False)(_path_scalar)
    BACKEND='numba'
