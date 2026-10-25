"""Attach separate rank-of-carrier weak targets to frozen coordinate-only caches."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from possession.data import sha256
from possession.skillcorner import roster,load_match,resample_coordinates
from possession.carrier_supervision import carrier_targets


def main():
    raw_root=ROOT/'data/external/skillcorner-raw';inventory=json.loads((raw_root/'inventory.json').read_text())
    producers={p:sha256(ROOT/p) for p in ('possession/carrier_supervision.py','scripts/prepare_carrier_supervision.py','possession/skillcorner.py')}
    for match in inventory['matches']:
        directory=raw_root/match;source=json.loads((directory/'source.json').read_text())
        if sha256(directory/'tracking.jsonl')!=source['tracking_sha256'] or sha256(directory/'metadata.json')!=source['metadata_sha256']:raise ValueError('Pinned source changed')
        metadata=json.loads((directory/'metadata.json').read_bytes());ids,teams=roster(metadata);index={identity:i for i,identity in enumerate(ids)}
        halves,_=load_match(metadata,directory/'tracking.jsonl')
        carriers={period:np.full(len(half['labels']),-1,np.int64) for period,half in halves.items()}
        with (directory/'tracking.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);period=row.get('period')
                if period not in halves:continue
                carriers[period][row['frame']-halves[period]['first']]=index.get((row.get('possession') or {}).get('player_id'),-1)
        for view in ('native','detected'):
            features=ROOT/('data/external/skillcorner-trajectories-'+view)/(match+'.npz')
            feature_sha=sha256(features)
            if feature_sha!=json.loads(features.with_suffix('.json').read_text())['sha256']:raise ValueError('Weak feature cache changed')
            out=ROOT/('data/external/skillcorner-carrier-'+view);out.mkdir(parents=True,exist_ok=True)
            path=out/(match+'.npz');sidecar=path.with_suffix('.json')
            if path.exists() or sidecar.exists():
                record=json.loads(sidecar.read_text())
                if record['source']!=source or record['producers']!=producers or record['feature_sha256']!=feature_sha or sha256(path)!=record['sha256']:raise ValueError('Existing carrier cache differs')
                continue
            arrays={};counts=[]
            with np.load(features,allow_pickle=False) as cached:
                for i,period in enumerate((1,2)):
                    players,ball=halves[period][view]
                    players,low=resample_coordinates(players,speed_limit=15.);ball,ball_low=resample_coordinates(ball,speed_limit=80.)
                    np.testing.assert_array_equal(low,ball_low);np.testing.assert_array_equal(low,cached[f'source_tick{i}'])
                    labels=halves[period]['labels'][low];np.testing.assert_array_equal(labels,cached[f'y{i}'])
                    target=carrier_targets(players,teams,ball,carriers[period][low],labels)
                    arrays[f'carrier{i}']=target;counts.append(int((target>=0).sum()))
            partial=path.with_suffix('.partial.npz');np.savez_compressed(partial,**arrays);partial.replace(path)
            sidecar.write_text(json.dumps(dict(source=source,producers=producers,feature_sha256=feature_sha,sha256=sha256(path),
                label_kind='provider carrier rank weak target; no identity codes enter features',known_per_half=counts),indent=2))
            print(json.dumps(dict(match=match,view=view,known_per_half=counts)),flush=True)


if __name__=='__main__':main()
