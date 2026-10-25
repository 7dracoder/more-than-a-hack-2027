"""Memoize deterministic coordinate features, guarding every raw/source/value checksum."""
import hashlib,json
from pathlib import Path
import numpy as np
from possession.data import sha256


def values_sha(matrix):return hashlib.sha256(memoryview(np.ascontiguousarray(matrix))).hexdigest()


def cached_block(root,key,builder,rows):
    rows=int(rows)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest();path=root/(digest+'.npy');sidecar=path.with_suffix('.json')
    reused=path.exists() or sidecar.exists()
    if reused:
        record=json.loads(sidecar.read_text())
        if record['key']!=key or record['file_sha256']!=sha256(path):raise ValueError('Inference feature cache provenance changed')
        matrix=np.load(path,mmap_mode='r',allow_pickle=False);names=record['names']
    else:
        generated,names=builder();generated=np.asarray(generated)
        if generated.dtype!=np.float32 or generated.shape!=(rows,len(names)) or len(names)!=len(set(names)) or not np.isfinite(generated).all():
            raise ValueError('Invalid generated coordinate features')
        data_sha=values_sha(generated);partial=path.with_suffix('.partial.npy');np.save(partial,generated,allow_pickle=False);partial.replace(path)
        matrix=np.load(path,mmap_mode='r',allow_pickle=False)
        if values_sha(matrix)!=data_sha:raise ValueError('Stored inference feature values differ')
        record=dict(key=key,names=names,file_sha256=sha256(path),values_sha256=data_sha,rows=rows,dtype='float32')
        sidecar.write_text(json.dumps(record,indent=2))
    if matrix.dtype!=np.float32 or matrix.shape!=(rows,len(names)) or values_sha(matrix)!=record['values_sha256']:
        raise ValueError('Inference feature cache values changed')
    return matrix,names,dict(key_sha256=digest,reused=reused,file_sha256=record['file_sha256'],
        values_sha256=record['values_sha256'],source_and_raw_input_checksums=key)
