"""Prepare physical touch evidence with the same targets/corruptions as v0.8."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from possession.data import sha256
from possession.external import read_sportec
from possession.features_v09 import build_features
from scripts.prepare_external_v08 import corrupt_tracking


def prepare(match, augmented):
    folder = ROOT/'data/external/sportec'
    cache = ROOT/('data/external/features-v09-corrupt' if augmented else 'data/external/features-v09')
    cache.mkdir(parents=True,exist_ok=True)
    path = cache/(match+'.npz')
    metadata = next(folder.glob('*matchinformation*'+match+'.xml'))
    tracking = next(folder.glob('*positions*'+match+'.xml'))
    provenance = dict(source_sha256=sha256(ROOT/'possession/external.py'),
               feature_source_sha256=sha256(ROOT/'possession/features_v08.py'),
               v09_feature_source_sha256=sha256(ROOT/'possession/features_v09.py'),
               v09_preparation_source_sha256=sha256(Path(__file__)),
               preparation_source_sha256=sha256(ROOT/'scripts/prepare_external_v08.py'),
               gap_mask_source_sha256=sha256(ROOT/'scripts/prepare_external_gaps.py'),
               tracking_sha256=sha256(tracking),metadata_sha256=sha256(metadata),step=2,augmentation=augmented)
    if path.exists() and path.with_suffix('.json').exists():
        old = json.loads(path.with_suffix('.json').read_text())
        if old['provenance']==provenance and old['sha256']==sha256(path):
            return
    arrays,records = {},[]
    for i,half in enumerate(read_sportec(metadata,tracking)):
        # Identical corruption seeds make the previous bank a strict prefix.
        seed = int(hashlib.sha256((match+'/'+str(i)+'/v08').encode()).hexdigest()[:8],16)
        players,ball = corrupt_tracking(half.players,half.ball,seed) if augmented else (half.players,half.ball)
        x,names = build_features(players,half.teams,ball,half.pitch,step=2)
        arrays.update({f'x{i}':x,f'y{i}':half.labels[::2],f'truth{i}':half.labels,f'frames{i}':half.frames})
        records.append(dict(period=half.period,frames=len(half.frames),seed=seed))
    arrays['names'] = np.array(names)
    partial = path.with_suffix('.partial.npz'); np.savez_compressed(partial,**arrays); partial.replace(path)
    path.with_suffix('.json').write_text(json.dumps(dict(provenance=provenance,sha256=sha256(path),halves=records),indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches',nargs='+',default=['J03WMX','J03WN1','J03WOH','J03WOY','J03WQQ','J03WR9','J03WPY'])
    parser.add_argument('--corrupt',action='store_true')
    args = parser.parse_args()
    for match in args.matches:
        started = time.monotonic(); print('Preparing v09 '+match+(' corrupt' if args.corrupt else ' native'),flush=True)
        prepare(match,args.corrupt)
        print(json.dumps(dict(match=match,seconds=round(time.monotonic()-started,2))),flush=True)
