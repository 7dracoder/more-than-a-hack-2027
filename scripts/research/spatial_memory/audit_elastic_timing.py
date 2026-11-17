"""TRAIN-only label-neighborhood diagnostic; never fit a timestamp correction."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from possession.data import sha256
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump
from scripts.research.spatial_memory.audit_elastic_training import team_label,INTENTIONAL
from scripts.research.spatial_memory.prepare_elastic_training import TRAIN,COMMIT


def neighborhood(truth,index,label):
    if truth[index]==label:
        return 'exact'
    for radius in (1,5,25):
        if label in truth[max(0,index-radius):min(len(truth),index+radius+1)]:
            return f'agrees_within_{radius}_frames'
    return 'no_agreement_within_25_frames'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'results.json').exists():
        raise ValueError('Preserve completed diagnostic')
    source=ROOT/'data/external/elastic-benchmark-e245224aad74'
    paths=[source/'source.json',ROOT/'data/research/elastic-training-audit-20261114/results.json']
    for match in TRAIN:
        paths += [source/f'benchmark/gt/{match}.parquet']
        paths += [ROOT/f'data/external/features-v25/{match}.{suffix}' for suffix in ('npz','json')]
    denied=restrict_external(args.out,paths)
    import pandas as pd
    receipt=json.loads((source/'source.json').read_text())
    assert receipt['commit']==COMMIT and receipt['matches']==TRAIN and not receipt['downloaded_development_annotations']
    ledger={str(p.relative_to(ROOT)):sha256(p) for p in paths}
    sources=[Path(__file__),ROOT/'scripts/research/spatial_memory/audit_elastic_training.py',
        ROOT/'scripts/research/spatial_memory/prepare_elastic_training.py']
    source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in sources}
    plan=dict(input_sha256=ledger,source_sha256=source_hashes,matches=TRAIN,radii_frames=[1,5,25],fps=25,
        policy='Describe neighborhoods and source-type disagreement only. No lag selection, label overwrite, model fitting or candidate selection.',
        provided_match_used=False,public_labels_used=False)
    dump(args.out/'plan.json',plan)
    results=[]
    for match in TRAIN:
        path=ROOT/f'data/external/features-v25/{match}.npz'
        assert sha256(path)==json.loads(path.with_suffix('.json').read_text())['sha256']
        assert sha256(source/f'benchmark/gt/{match}.parquet')==receipt['files'][f'benchmark/gt/{match}.parquet']['sha256']
        with np.load(path) as z:
            clocks={i+1:(z[f'frames{i}'],z[f'truth{i}']) for i in (0,1)}
        events=pd.read_parquet(source/f'benchmark/gt/{match}.parquet')
        points=[];teams=defaultdict(set)
        for row in events.itertuples():
            if row.spadl_type in INTENTIONAL:
                points.append(('intentional_actor',row.spadl_type,int(row.period_id),int(row.frame_id),team_label(row.player_id)))
            if pd.notna(row.receive_frame_id) and team_label(row.receiver_id):
                assert row.receive_frame_id==int(row.receive_frame_id)
                points.append(('player_reception',row.spadl_type,int(row.period_id),int(row.receive_frame_id),team_label(row.receiver_id)))
        for scope,kind,period,tick,label in points:
            if label: teams[period,tick].add(label)
        scopes=defaultdict(Counter);kinds=defaultdict(Counter)
        for scope,kind,period,tick,label in points:
            if not label: continue
            frames,truth=clocks[period];index=tick-int(frames[0])
            assert 0<=index<len(truth) and np.all(np.diff(frames)==1)
            outcome=neighborhood(truth,index,label)
            scopes[scope][outcome]+=1;kinds[scope+'/'+kind][outcome]+=1
            if len(teams[period,tick])>1:
                scopes[scope]['conflicting_frame_points']+=1
        results.append(dict(match=match,scopes={k:dict(v) for k,v in scopes.items()},
            by_action_type={k:dict(v) for k,v in kinds.items()}))
    for name,digest in {**ledger,**source_hashes}.items(): assert sha256(ROOT/name)==digest
    result=dict(complete=True,rows=results,plan_sha256=sha256(args.out/'plan.json'),training_only=True,
        selected_timestamp_offset=None,changed_labels=False,model_fitted=False,provided_match_used=False,
        public_labels_used=False,unexpected_competition_access=denied,competition_target_verified=False)
    dump(args.out/'results.json',result)
    print(json.dumps(dict(complete=True,scopes=[dict(match=r['match'],scopes=r['scopes']) for r in results])))


if __name__=='__main__': main()
