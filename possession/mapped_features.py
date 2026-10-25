"""Lossless, checksum-verified float32 feature maps for memory-constrained local fitting."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
from possession.data import sha256


def open_verified(path,key):
    metadata=json.loads(path.with_suffix('.json').read_text())
    if metadata['key']!=key or sha256(path)!=metadata['sha256']:raise ValueError('Mapped feature cache changed')
    return open_map(path,metadata['sha256'])


def open_map(path,digest):
    matrix=np.load(path,mmap_mode='r',allow_pickle=False)
    if matrix.ndim!=2 or matrix.dtype!=np.float32 or not matrix.flags.c_contiguous:raise ValueError('Expected a C-order float32 feature matrix')
    matrix._verified_file_sha256=digest
    return matrix


def commit(partial,path,key,digest):
    partial.replace(path)
    path.with_suffix('.json').write_text(json.dumps(dict(key=key,sha256=digest,lossless_float32=True),indent=2))
    return open_map(path,digest)


def mapped_member(archive,member,root,archive_sha256=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);archive=Path(archive)
    if '/' in member or '\\' in member or not member.endswith('.npy'):raise ValueError('Expected a plain array member')
    digest=sha256(archive) if archive_sha256 is None else archive_sha256
    key=dict(archive_sha256=digest,member=member,format='exact-npy-member-v1')
    name=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest();path=root/(name+'.npy')
    if path.exists() or path.with_suffix('.json').exists():return open_verified(path,key)
    partial=path.with_suffix('.partial');actual=hashlib.sha256()
    with zipfile.ZipFile(archive) as source,source.open(member) as stream,partial.open('wb') as output:
        while block:=stream.read(1_048_576):output.write(block);actual.update(block)
    return commit(partial,path,key,actual.hexdigest())


def mapped_normalized(matrix,mean,scale,root,chunk=4096):
    from possession.temporal import normalized
    from possession import temporal
    if not isinstance(matrix,np.memmap) or matrix.flags.writeable or chunk<1:
        raise ValueError('Normalization mapping requires a verified read-only feature map')
    raw_digest=getattr(matrix,'_verified_file_sha256',None)
    if raw_digest is None:raise ValueError('Raw feature map has no verified lineage')
    mean=np.asarray(mean);scale=np.asarray(scale)
    if mean.shape!=(matrix.shape[1],) or scale.shape!=mean.shape:raise ValueError('Normalization schema differs')
    key=dict(raw_sha256=raw_digest,mean_sha256=hashlib.sha256(mean.tobytes()).hexdigest(),
        scale_sha256=hashlib.sha256(scale.tobytes()).hexdigest(),normalizer_source_sha256=sha256(Path(temporal.__file__)),
        mapper_source_sha256=sha256(Path(__file__)),format='pointwise-float32-normalization-v1')
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    name=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest();path=root/(name+'.npy')
    if path.exists() or path.with_suffix('.json').exists():return open_verified(path,key)
    partial=path.with_suffix('.partial.npy')
    destination=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float32,shape=matrix.shape)
    for first in range(0,len(matrix),chunk):destination[first:first+chunk]=normalized(matrix[first:first+chunk],mean,scale)
    destination.flush();del destination
    return commit(partial,path,key,sha256(partial))
