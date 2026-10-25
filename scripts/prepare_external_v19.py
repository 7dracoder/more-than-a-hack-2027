"""Append contact geometry to frozen v13 caches without altering their prefix."""
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
from possession.features_v19 import extra_features
from scripts.prepare_external_v08 import corrupt_tracking


def prepare(match,corrupt):
    suffix='-corrupt' if corrupt else ''
    base=ROOT/('data/external/features-v13'+suffix)/(match+'.npz')
    base_meta=json.loads(base.with_suffix('.json').read_text())
    if base_meta['sha256']!=sha256(base):raise ValueError('Base cache checksum differs')
    source_map=dict(source_sha256='possession/external.py',feature_source_sha256='possession/features_v08.py',
        preparation_source_sha256='scripts/prepare_external_v08.py',v09_feature_source_sha256='possession/features_v09.py',
        v10_preparation_source_sha256='scripts/prepare_external_v10.py',gap_mask_source_sha256='scripts/prepare_external_gaps.py',
        v13_feature_source_sha256='possession/features_v13.py',v13_preparation_source_sha256='scripts/prepare_external_v13.py')
    if any(base_meta['provenance'][key]!=sha256(ROOT/path) for key,path in source_map.items()):
        raise ValueError('Base feature sources differ')
    folder=ROOT/'data/external/sportec'
    metadata=next(folder.glob('*matchinformation*'+match+'.xml'));tracking=next(folder.glob('*positions*'+match+'.xml'))
    if sha256(tracking)!=base_meta['provenance']['tracking_sha256'] or sha256(metadata)!=base_meta['provenance']['metadata_sha256']:
        raise ValueError('Raw and cached tracking differ')
    provenance=dict(base_meta['provenance'],v19_base_cache_sha256=sha256(base),
        v19_feature_source_sha256=sha256(ROOT/'possession/features_v19.py'),v19_preparation_source_sha256=sha256(Path(__file__)))
    path=ROOT/('data/external/features-v19'+suffix)/(match+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.with_suffix('.json').exists():
        old=json.loads(path.with_suffix('.json').read_text())
        if old['provenance']==provenance and old['sha256']==sha256(path):return
    started=time.monotonic();halves=read_sportec(metadata,tracking);arrays={};records=[]
    with np.load(base,allow_pickle=False) as z:
        base_names=z['names'].tolist()
        for i,half in enumerate(halves):
            np.testing.assert_array_equal(z[f'frames{i}'],half.frames);np.testing.assert_array_equal(z[f'truth{i}'],half.labels)
            seed=int(hashlib.sha256((match+'/'+str(i)+'/v08').encode()).hexdigest()[:8],16)
            players,ball=corrupt_tracking(half.players,half.ball,seed) if corrupt else (half.players,half.ball)
            extra,names=extra_features(players,half.teams,ball);prefix=z[f'x{i}']
            arrays[f'x{i}']=np.concatenate([prefix,extra],axis=1)
            arrays.update({f'y{i}':z[f'y{i}'],f'truth{i}':half.labels,f'frames{i}':half.frames})
            records.append(dict(period=half.period,frames=len(half.frames),prefix_features=prefix.shape[1],extra_features=extra.shape[1],seed=seed))
            del prefix,extra,players,ball
    names=base_names+names
    if len(names)!=len(set(names)):raise ValueError('New contact features duplicate an existing name')
    arrays['names']=np.array(names)
    partial=path.with_suffix('.partial.npz');np.savez(partial,**arrays);partial.replace(path)
    record=dict(provenance=provenance,sha256=sha256(path),halves=records,seconds=time.monotonic()-started,compression='none')
    path.with_suffix('.json').write_text(json.dumps(record,indent=2));print(json.dumps(records),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches',nargs='+',default=['J03WMX','J03WN1','J03WOH','J03WOY','J03WQQ','J03WR9','J03WPY'])
    parser.add_argument('--both',action='store_true');args=parser.parse_args()
    for match in args.matches:
        for corrupt in ([False,True] if args.both else [False]):
            print('Appending contact geometry '+match+(' corrupt' if corrupt else ' native'),flush=True);prepare(match,corrupt)
