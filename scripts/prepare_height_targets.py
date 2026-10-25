"""Prepare auxiliary Z targets for the six external training matches only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from possession.data import sha256
from possession.height import read_ball_height,align_height


def prepare(match):
    if 'BAR' in match.upper() or 'ATH' in match.upper():
        raise ValueError('Auxiliary targets are restricted to other external matches')
    folder=ROOT/'data/external/sportec'
    tracking=next(folder.glob('*positions*'+match+'.xml'))
    features=ROOT/'data/external/features-v10'/(match+'.npz')
    out=ROOT/'data/external/height-targets'; out.mkdir(parents=True,exist_ok=True)
    path=out/(match+'.npz')
    source=dict(tracking_sha256=sha256(tracking),reader_source_sha256=sha256(ROOT/'possession/height.py'),
                preparation_source_sha256=sha256(Path(__file__)))
    if path.exists() and path.with_suffix('.json').exists():
        previous=json.loads(path.with_suffix('.json').read_text())
        if all(previous.get(k)==v for k,v in source.items()) and previous.get('sha256')==sha256(path):
            return
    periods=read_ball_height(tracking)
    arrays={}; digests={}; stats=[]
    with np.load(features,allow_pickle=False) as x:
        for i,period in enumerate(('firstHalf','secondHalf')):
            frames=x[f'frames{i}']; ids,z=periods[period]
            target=align_height(frames,ids,z)
            arrays.update({f'z{i}':target,f'frames{i}':frames})
            digests[str(i)]=hashlib.sha256(frames.tobytes()).hexdigest()
            finite=target[np.isfinite(target)]
            stats.append(dict(period=period,frames=len(frames),known=len(finite),minimum=float(finite.min()),maximum=float(finite.max()),mean=float(finite.mean())))
    part=path.with_suffix('.partial.npz'); np.savez_compressed(part,**arrays); part.replace(path)
    source.update(sha256=sha256(path),frame_sha256=digests,halves=stats,
                  scope='External training supervision only; never an input feature')
    path.with_suffix('.json').write_text(json.dumps(source,indent=2))
    print(json.dumps(dict(match=match,halves=stats)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/local_v10_fullrate.json')
    args=p.parse_args(); config=json.loads(args.config.read_text())
    if set(config['train_matches'])&set(config['validation_matches']):
        raise ValueError('Training/development overlap')
    for match in config['train_matches']:
        started=time.monotonic(); print('Preparing external height target '+match,flush=True)
        prepare(match); print('Seconds '+str(round(time.monotonic()-started,2)),flush=True)
