"""Load only pinned external development feature/target/parent triples."""
import json
from pathlib import Path
import numpy as np
from possession.data import sha256
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.train_real_broadcast_memory import STRONG,TRANSFER,VIEWS

ROOT = Path(__file__).resolve().parents[3]


def load(out,reference):
    split = reference['plan']['config']
    assert reference['complete'] and not reference['provided_match_used'] and not reference['public_labels_used']
    specs = []
    for match in split['strong_development_matches']:
        for half in (0,1):
            for view in VIEWS:
                stem = STRONG / 'cache' / f'{match}_{half}_{view}'
                specs.append(dict(domain='strong',view=view,x_path=stem.with_suffix('.npy'),
                    target=stem.with_name(stem.name+'_truth.npy'),parent_path=stem.with_name(stem.name+'_parent.npy')))
    for match in split['weak_development_matches']:
        for half in (1,2):
            stem = f'{match}_{half}_detected'
            specs.append(dict(domain='weak',x_path=TRANSFER / 'features' / (stem+'.npy'),
                target=TRANSFER / 'features' / (stem+'_weak.npz'),parent_path=TRANSFER / 'predictions' / (stem+'.npz')))
    allowed = [h[k] for h in specs for k in ('x_path','target','parent_path')]+[h['x_path'].with_suffix('.json') for h in specs]
    denied = restrict_external(out,allowed)
    hashes = {}
    for path in allowed:
        key = str(path.relative_to(ROOT))
        digest = sha256(path)
        assert digest == reference['plan']['caches'][key],key
        hashes[key] = digest
    strong,weak = [],[]
    for h in specs:
        metadata = json.loads(h['x_path'].with_suffix('.json').read_text())
        h['x'] = np.load(h['x_path'],mmap_mode='r')
        if h['domain'] == 'weak':
            h['x'] = h['x'][::2]
        h['names'],h['memory_names'] = metadata['names'],metadata['names'][960:]
        assert len(h['names']) == 1088
        if h['domain'] == 'strong':
            h['truth'] = np.load(h['target'])
            h['parent'] = np.load(h['parent_path'])
            strong.append(h)
        else:
            with np.load(h['target']) as data:
                h.update(truth=data['truth'],ticks=data['ticks'],seen=data['detected_ball_seen'])
            with np.load(h['parent_path']) as data:
                h['parent'] = data['parent']
            weak.append(h)
    return strong,weak,hashes,denied
