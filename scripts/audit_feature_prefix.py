"""Verify an appended feature cache against its parent using bounded memory."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from possession.data import sha256


def header(stream):
    version=np.lib.format.read_magic(stream)
    if version==(1,0):return np.lib.format.read_array_header_1_0(stream)
    if version==(2,0):return np.lib.format.read_array_header_2_0(stream)
    raise ValueError('Unsupported array header')


def audit(base,current,version='v19'):
    previous=sha256(base);digest=sha256(current)
    meta=json.loads(current.with_suffix('.json').read_text())
    if meta['sha256']!=digest or meta['provenance'][version+'_base_cache_sha256']!=previous:
        raise ValueError('Cache lineage checksum differs')
    records=[]
    with np.load(base,allow_pickle=False) as old,np.load(current,allow_pickle=False) as new:
        names=old['names'];extended=new['names']
        np.testing.assert_array_equal(names,extended[:len(names)])
        if len(set(extended.tolist()))!=len(extended):raise ValueError('Duplicate feature names')
        for key in old.files:
            if key!='names' and not key.startswith('x'):np.testing.assert_array_equal(old[key],new[key])
    with zipfile.ZipFile(base) as old,zipfile.ZipFile(current) as new:
        for name in old.namelist():
            if not name.startswith('x') or not name.endswith('.npy'):continue
            with old.open(name) as a,new.open(name) as b:
                shape_a,order_a,dtype_a=header(a);shape_b,order_b,dtype_b=header(b)
                if order_a or order_b or dtype_a!=np.dtype('float32') or dtype_b!=dtype_a or shape_a[0]!=shape_b[0] or shape_b[1]!=len(extended):
                    raise ValueError('Unexpected feature layout')
                for start in range(0,shape_a[0],4096):
                    size=min(4096,shape_a[0]-start)
                    x=np.frombuffer(a.read(size*shape_a[1]*4),dtype=dtype_a).reshape(size,shape_a[1])
                    y=np.frombuffer(b.read(size*shape_b[1]*4),dtype=dtype_b).reshape(size,shape_b[1])
                    np.testing.assert_array_equal(x,y[:,:shape_a[1]])
                    if not np.isfinite(y).all():raise ValueError('Nonfinite feature')
                if a.read(1) or b.read(1):raise ValueError('Unexpected trailing array bytes')
                records.append(dict(array=name,rows=shape_a[0],prefix=shape_a[1],features=shape_b[1]))
    return dict(path=str(current),sha256=digest,parent_sha256=previous,exact_prefix=True,labels_and_frames_equal=True,finite=True,arrays=records)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',choices=['v19','v25'],default='v19')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();records=[]
    if args.out is None:args.out=ROOT/('data/local-v08/'+('contact_geometry' if args.version=='v19' else 'player_trajectory')+'_cache_audit.json')
    for suffix in ('','-corrupt'):
        for current in sorted((ROOT/('data/external/features-'+args.version+suffix)).glob('*.npz')):
            record=audit(ROOT/('data/external/features-v13'+suffix)/current.name,current,args.version)
            records.append(record);print(json.dumps(record),flush=True)
    if len(records)!=14:raise ValueError('Expected seven native and seven corrupted caches')
    args.out.write_text(json.dumps(dict(records=records,audit_source_sha256=sha256(Path(__file__))),indent=2))
