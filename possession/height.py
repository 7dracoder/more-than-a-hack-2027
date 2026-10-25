"""External ball-height supervision, kept separate from every model input feature."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from possession.data import sha256


def read_ball_height(path):
    periods={}; ball=False; period=None
    for event,item in ET.iterparse(path,events=('start','end')):
        if event=='start' and item.tag=='FrameSet':
            ball=item.get('TeamId','').lower()=='ball' or item.get('PersonId','').lower()=='ball'
            period=item.get('GameSection')
        elif event=='end' and item.tag=='Frame':
            if ball and period in ('firstHalf','secondHalf'):
                periods.setdefault(period,[]).append((int(item.get('N')),float(item.get('Z','nan'))))
            item.clear()
        elif event=='end' and item.tag=='FrameSet':
            item.clear(); ball=False; period=None
    return {key:(np.array([r[0] for r in rows],np.int64),np.array([r[1] for r in rows],np.float32)) for key,rows in periods.items()}


def align_height(frames,ids,values):
    if len(ids)!=len(np.unique(ids)):
        raise ValueError('Duplicate external height frame IDs')
    indices=np.searchsorted(frames,ids)
    valid=indices<len(frames)
    valid[valid]&=frames[indices[valid]]==ids[valid]
    result=np.full(len(frames),np.nan,np.float32)
    result[indices[valid]]=values[valid]
    return result


def attach_height_targets(halves,directory,feature_dir,step,root):
    loaded={}
    for match in dict.fromkeys(h['match'] for h in halves):
        path=Path(directory)/(match+'.npz')
        metadata=json.loads(path.with_suffix('.json').read_text())
        if metadata['sha256']!=sha256(path) or metadata['reader_source_sha256']!=sha256(Path(__file__)) or metadata['preparation_source_sha256']!=sha256(Path(root)/'scripts/prepare_height_targets.py'):
            raise ValueError('Stale/corrupted external height targets')
        feature_path=Path(feature_dir)/(match+'.npz')
        feature_meta=json.loads(feature_path.with_suffix('.json').read_text())
        if feature_meta['provenance']['tracking_sha256']!=metadata['tracking_sha256']:
            raise ValueError('Height targets came from a different tracking source')
        values=[]
        with np.load(path,allow_pickle=False) as z,np.load(feature_path,allow_pickle=False) as x:
            for i in range(2):
                frames=x[f'frames{i}']
                if hashlib.sha256(frames.tobytes()).hexdigest()!=metadata['frame_sha256'][str(i)]:
                    raise ValueError('Height targets and feature timeline differ')
                np.testing.assert_array_equal(frames,z[f'frames{i}'])
                # Targets only. Neither these values nor their statistics enter x.
                values.append(np.log1p(np.clip(z[f'z{i}'],0,50)).astype(np.float32)[::step])
        loaded[match]=values
    for half in halves:
        target=loaded[half['match']][half['half']]
        if len(target)!=len(half['y']) or len(target)!=len(half['x']):
            raise ValueError('Height supervision and sampled features differ in length')
        half['height_target']=target


def height_regression_loss(prediction,target):
    import torch
    if prediction.shape!=target.shape:
        raise ValueError('Height regression shapes differ')
    known=torch.isfinite(target)
    safe=torch.nan_to_num(target,nan=0.,posinf=0.,neginf=0.)
    values=torch.nn.functional.smooth_l1_loss(prediction,safe,reduction='none')
    return (values*known).sum()/known.sum().clamp_min(1)
