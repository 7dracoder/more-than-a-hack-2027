"""Prepare coordinate-only player trajectories and separate provider H/A weak targets."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from possession.data import sha256
from possession.skillcorner import load_match,resample_coordinates
from possession.features_v25 import extra_features


def prepare(match):
    directory=ROOT/'data/external/skillcorner-raw'/match
    source=json.loads((directory/'source.json').read_text())
    if sha256(directory/'tracking.jsonl')!=source['tracking_sha256'] or sha256(directory/'metadata.json')!=source['metadata_sha256']:
        raise ValueError('Raw SkillCorner source changed')
    producers={path:sha256(ROOT/path) for path in ('possession/skillcorner.py','possession/features_v25.py','scripts/prepare_skillcorner_pretraining.py')}
    halves,teams=load_match(json.loads((directory/'metadata.json').read_text()),directory/'tracking.jsonl')
    for view in ('native','detected'):
        out=ROOT/('data/external/skillcorner-trajectories-'+view);out.mkdir(parents=True,exist_ok=True)
        path=out/(match+'.npz');sidecar=path.with_suffix('.json')
        if path.exists() or sidecar.exists():
            record=json.loads(sidecar.read_text())
            if record['source']!=source or record['producers']!=producers or record['sha256']!=sha256(path):raise ValueError('Existing weak cache differs')
            print(json.dumps(dict(match=match,view=view,reused=True)),flush=True);continue
        arrays={};records=[]
        for i,period in enumerate((1,2)):
            half=halves[period];players,ball=half[view]
            players,low=resample_coordinates(players,speed_limit=15.)
            ball,ball_low=resample_coordinates(ball,speed_limit=80.)
            np.testing.assert_array_equal(low,ball_low)
            # Feature extraction has no label or provider-possession argument.
            features,names=extra_features(players,teams,ball,fps=25.)
            labels=half['labels'][low].copy()
            if set(np.unique(labels))-set('HAX'):raise ValueError('Weak labels must never infer D')
            arrays.update({f'x{i}':features,f'y{i}':labels,f'source_tick{i}':low})
            records.append(dict(period=period,source_first=half['first'],source_last=half['last'],frames=len(features),
                labels={label:int((labels==label).sum()) for label in 'HAX'},source_frames_missing=int((~half['seen']).sum())))
        arrays['names']=np.array(names)
        partial=path.with_suffix('.partial.npz');np.savez_compressed(partial,**arrays);partial.replace(path)
        record=dict(source=source,producers=producers,sha256=sha256(path),feature_count=180,source_fps=10,target_fps=25,
            physical_units=True,view=view,periods=records,
            label_policy='Provider H/A only, null/unknown stays X. No dead-ball labels inferred. Not independent human ground truth.',trained=False)
        sidecar.write_text(json.dumps(record,indent=2));print(json.dumps(dict(match=match,view=view,periods=records)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--matches',nargs='*');parser.add_argument('--all',action='store_true')
    args=parser.parse_args()
    root=ROOT/'data/external/skillcorner-raw'
    matches=sorted(p.name for p in root.iterdir() if p.is_dir() and (p/'source.json').exists()) if args.all else args.matches
    if not matches:parser.error('No downloaded matches selected')
    for match in matches:
        if not match.isdigit():raise ValueError('Expected a numeric public match identifier')
        started=time.monotonic();prepare(match);print(json.dumps(dict(match=match,seconds=round(time.monotonic()-started,2))),flush=True)


if __name__=='__main__':main()
