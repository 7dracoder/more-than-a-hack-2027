"""Independently check weak cache bytes, physical masks and raw-frame target alignment."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from possession.data import sha256


def main():
    raw_root=ROOT/'data/external/skillcorner-raw'
    inventory=json.loads((raw_root/'inventory.json').read_text());records=[]
    for match in inventory['matches']:
        directory=raw_root/match;source=json.loads((directory/'source.json').read_text())
        if sha256(directory/'tracking.jsonl')!=source['tracking_sha256'] or sha256(directory/'metadata.json')!=source['metadata_sha256']:
            raise ValueError('Pinned raw source changed')
        metadata=json.loads((directory/'metadata.json').read_text())
        bounds={p['period']:(p['start_frame'],p['end_frame']) for p in metadata['match_periods'] if p['period'] in (1,2)}
        labels={period:np.full(last-first+1,'X',dtype='<U1') for period,(first,last) in bounds.items()}
        with (directory/'tracking.jsonl').open() as stream:
            for line in stream:
                row=json.loads(line);period=row.get('period')
                if period not in bounds:continue
                group=(row.get('possession') or {}).get('group')
                if group not in (None,'home team','away team'):raise ValueError('Unknown provider group')
                labels[period][row['frame']-bounds[period][0]]={None:'X','home team':'H','away team':'A'}[group]
        for view in ('native','detected'):
            path=ROOT/('data/external/skillcorner-trajectories-'+view)/(match+'.npz')
            record=json.loads(path.with_suffix('.json').read_text())
            if record['source']!=source or record['sha256']!=sha256(path):raise ValueError('Weak cache checksum differs')
            if any(sha256(ROOT/p)!=digest for p,digest in record['producers'].items()):raise ValueError('Weak cache producer changed')
            with np.load(path,allow_pickle=False) as cache:
                names=cache['names'].tolist()
                if len(names)!=180 or len(set(names))!=180:raise ValueError('Unexpected trajectory schema')
                masks=[i for i,name in enumerate(names) if name.endswith('_known')]
                if len(masks)!=30:raise ValueError('Expected thirty trajectory observation masks')
                for i,period in enumerate((1,2)):
                    x=cache[f'x{i}'];y=cache[f'y{i}'];tick=cache[f'source_tick{i}']
                    length=(len(labels[period])-1)*5//2+1
                    expected_tick=np.arange(length,dtype=np.int64)*2//5
                    np.testing.assert_array_equal(tick,expected_tick)
                    np.testing.assert_array_equal(y,labels[period][expected_tick])
                    if x.shape!=(length,180) or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('Invalid physical features')
                    if not np.isin(x[:,masks],[0.,1.]).all() or 'D' in y:raise ValueError('Invalid mask or inferred D target')
            records.append(dict(match=match,view=view,sha256=record['sha256'],raw_frame_targets_exact=True,finite=True,binary_masks=True,no_inferred_D=True))
            print(json.dumps(records[-1]),flush=True)
    if len(records)!=2*len(inventory['matches']):raise ValueError('Incomplete weak-source audit')
    out=ROOT/'data/local-v08/skillcorner_pretraining_cache_audit.json'
    partial=out.with_suffix('.partial.json')
    partial.write_text(json.dumps(dict(commit=inventory['commit'],records=records,audit_source_sha256=sha256(Path(__file__)),used_for_model_training=False),indent=2));partial.replace(out)


if __name__=='__main__':main()
