"""Store the same physical evidence at all 25 frames/second, without ZIP compression."""
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
from possession.external import read_sportec
from possession.features_v09 import build_features
from scripts.prepare_external_v08 import corrupt_tracking


def prepare(match,corrupt):
    folder=ROOT/'data/external/sportec'
    cache=ROOT/('data/external/features-v10-corrupt' if corrupt else 'data/external/features-v10')
    cache.mkdir(parents=True,exist_ok=True)
    path=cache/(match+'.npz')
    metadata=next(folder.glob('*matchinformation*'+match+'.xml'))
    tracking=next(folder.glob('*positions*'+match+'.xml'))
    provenance=dict(source_sha256=sha256(ROOT/'possession/external.py'),
             feature_source_sha256=sha256(ROOT/'possession/features_v08.py'),
             preparation_source_sha256=sha256(ROOT/'scripts/prepare_external_v08.py'),
             v09_feature_source_sha256=sha256(ROOT/'possession/features_v09.py'),
             v10_preparation_source_sha256=sha256(Path(__file__)),
             gap_mask_source_sha256=sha256(ROOT/'scripts/prepare_external_gaps.py'),
             tracking_sha256=sha256(tracking),metadata_sha256=sha256(metadata),step=1,augmentation=corrupt)
    if path.exists() and path.with_suffix('.json').exists():
        old=json.loads(path.with_suffix('.json').read_text())
        if old['provenance']==provenance and old['sha256']==sha256(path):
            return
    started=time.monotonic(); halves=read_sportec(metadata,tracking); parse_seconds=time.monotonic()-started
    arrays,records={},[]
    for i,half in enumerate(halves):
        seed=int(hashlib.sha256((match+'/'+str(i)+'/v08').encode()).hexdigest()[:8],16)
        players,ball=corrupt_tracking(half.players,half.ball,seed) if corrupt else (half.players,half.ball)
        started=time.monotonic(); x,names=build_features(players,half.teams,ball,half.pitch,step=1)
        records.append(dict(period=half.period,frames=len(half.frames),feature_seconds=time.monotonic()-started,seed=seed))
        arrays.update({f'x{i}':x,f'y{i}':half.labels,f'truth{i}':half.labels,f'frames{i}':half.frames})
    arrays['names']=np.array(names)
    started=time.monotonic(); partial=path.with_suffix('.partial.npz')
    np.savez(partial,**arrays); partial.replace(path)
    record=dict(provenance=provenance,sha256=sha256(path),halves=records,parse_seconds=parse_seconds,
                write_seconds=time.monotonic()-started,bytes=path.stat().st_size,compression='none')
    path.with_suffix('.json').write_text(json.dumps(record,indent=2))
    print(json.dumps({k:record[k] for k in ('parse_seconds','write_seconds','bytes','halves')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches',nargs='+',default=['J03WMX','J03WN1','J03WOH','J03WOY','J03WQQ','J03WR9','J03WPY'])
    parser.add_argument('--corrupt',action='store_true')
    args=parser.parse_args()
    for match in args.matches:
        print('Preparing full-rate '+match+(' corrupt' if args.corrupt else ' native'),flush=True)
        prepare(match,args.corrupt)
