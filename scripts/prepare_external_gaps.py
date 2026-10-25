"""Create external training views with label-independent missing-ball intervals.

Only external coordinates are altered. Official external labels stay separate and
unchanged. The mask never examines possession or in-play state.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from possession.external import build_features,read_sportec
from possession.data import sha256


def missing_intervals(n,seed,fps=25.):
    rng=np.random.default_rng(seed)
    missing=np.zeros(n,bool); start=0; visible=True
    while start<n:
        mean=9. if visible else 7.
        length=max(5,round(rng.exponential(mean)*fps))
        if not visible: missing[start:start+length]=True
        start+=length; visible=not visible
    return missing


def prepare(match,step):
    folder=ROOT/'data/external/sportec'; cache=ROOT/'data/external/features-v07-gaps'
    cache.mkdir(parents=True,exist_ok=True)
    metadata=next(folder.glob('*matchinformation*'+match+'.xml'))
    tracking=next(folder.glob('*positions*'+match+'.xml'))
    provenance=dict(source_sha256=sha256(ROOT/'possession/external.py'),
                    augmentation_source_sha256=sha256(Path(__file__)),
                    tracking_sha256=sha256(tracking),metadata_sha256=sha256(metadata),
                    step=step,mask='Alternating exponential visible/missing runs, 9s/7s means, min 5 frames; seed from match/half only')
    path=cache/(match+'.npz'); record_path=path.with_suffix('.json')
    if path.exists() and record_path.exists():
        record=json.loads(record_path.read_text())
        if record['provenance']==provenance and record['sha256']==sha256(path): return
    output={}; records=[]
    for i,half in enumerate(read_sportec(metadata,tracking)):
        seed=int(hashlib.sha256((match+'/'+str(i)+'/2027').encode()).hexdigest()[:8],16)
        missing=missing_intervals(len(half.frames),seed)
        ball=half.ball.copy(); ball[missing]=np.nan
        x,names=build_features(half.players,half.teams,ball,half.pitch,step=step)
        output.update({f'x{i}':x,f'y{i}':half.labels[::step],f'truth{i}':half.labels,f'frames{i}':half.frames})
        records.append(dict(period=half.period,seed=seed,artificially_missing_fraction=float(missing.mean())))
    output['names']=np.array(names)
    part=path.with_suffix('.partial.npz'); np.savez_compressed(part,**output); part.replace(path)
    record_path.write_text(json.dumps(dict(provenance=provenance,sha256=sha256(path),halves=records),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matches',nargs='+',default=['J03WMX','J03WN1','J03WOH','J03WOY'])
    args=p.parse_args()
    for match in args.matches:
        t=time.monotonic(); print('Preparing missing-ball external view '+match,flush=True)
        prepare(match,5)
        print(json.dumps(dict(match=match,seconds=round(time.monotonic()-t,2))),flush=True)
