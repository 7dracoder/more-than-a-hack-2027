"""Prepare conditional receiver choices exclusively from two external TRAIN matches."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from possession.data import sha256
from possession.receiver_geometry import receiver_features,NAMES
from scripts.research.spatial_memory.audit_elastic_identity import canonical_players
from scripts.research.spatial_memory.prepare_elastic_training import TRAIN,COMMIT
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump

PLAN=dict(matches=TRAIN,event_types=['pass','cross'],fps=25,history_frames=51,
    maximum_flight_seconds=10.,seed=20351124,views=['native','history_masked'],
    masked_candidate_probability=.35,masked_recent_frames=25,
    task='Actual receiving player conditional on a known passer at release, not intent, gaze or possession labels.',
    unavailable_target='Extra class K if annotated receiver has no observation in the causal two-second history.',
    features=NAMES,provided_match_used=False,public_labels_used=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'preparation.json').exists():raise ValueError('Preserve completed preparation')
    base=ROOT/'data/external/elastic-benchmark-e245224aad74';specs=[];paths=[base/'source.json']
    for match in TRAIN:
        meta=next((ROOT/'data/external/sportec').glob('*matchinformation*'+match+'.xml'))
        raw=next((ROOT/'data/external/sportec').glob('*positions_raw*'+match+'.xml'))
        events=base/f'benchmark/gt/{match}.parquet';clock=ROOT/f'data/external/features-v25/{match}.npz'
        specs.append((match,meta,raw,events,clock));paths += [meta,raw,events,clock,clock.with_suffix('.json')]
    denied=restrict_external(args.out,paths)
    import pandas as pd
    receipt=json.loads((base/'source.json').read_text());assert receipt['commit']==COMMIT and receipt['matches']==TRAIN
    ledger={str(p.relative_to(ROOT)):sha256(p) for p in paths}
    sources=[Path(__file__),ROOT/'possession/receiver_geometry.py',ROOT/'scripts/research/spatial_memory/audit_elastic_identity.py']
    source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in sources}
    plan=dict(config=PLAN,inputs=ledger,sources=source_hashes)
    dump(args.out/'plan.json',plan);summaries=[];files={}
    for match,meta,raw,event,clock in specs:
        assert sha256(clock)==json.loads(clock.with_suffix('.json').read_text())['sha256']
        assert sha256(event)==receipt['files'][f'benchmark/gt/{match}.parquet']['sha256']
        mapping=canonical_players(ET.parse(meta).getroot());people=sorted(mapping.values());slots={p:i for i,p in enumerate(people)}
        canonical_slot={key:slots[value] for key,value in mapping.items()}
        teams=np.empty(len(people),np.int8)
        for key,slot in canonical_slot.items():teams[slot]=0 if key.startswith('home_') else 1
        with np.load(clock) as z:frames={i+1:z[f'frames{i}'] for i in (0,1)}
        arrays={period:np.full((len(ticks),len(people),2),np.nan,np.float32) for period,ticks in frames.items()}
        for _,item in ET.iterparse(raw,events=('end',)):
            if item.tag!='FrameSet':continue
            period={'firstHalf':1,'secondHalf':2}.get(item.get('GameSection'));person=item.get('PersonId')
            if period in arrays and person in slots:
                ticks=frames[period];slot=slots[person]
                for point in item.iter('Frame'):
                    index=int(point.get('N'))-int(ticks[0])
                    if 0<=index<len(ticks):arrays[period][index,slot]=[float(point.get('X','nan')),float(point.get('Y','nan'))]
            item.clear()
        events=pd.read_parquet(event);eligible=[];reasons=Counter();groups=defaultdict(set)
        for row in events.itertuples():
            if row.spadl_type not in PLAN['event_types']:continue
            if row.player_id not in canonical_slot or row.receiver_id not in canonical_slot or pd.isna(row.receive_frame_id):
                reasons['nonplayer_or_missing_target']+=1;continue
            if row.player_id==row.receiver_id:reasons['self_receiver']+=1;continue
            duration=float(row.receive_frame_id)-int(row.frame_id)
            if not 0<duration<=PLAN['maximum_flight_seconds']*25:reasons['invalid_flight_duration']+=1;continue
            period=int(row.period_id);release=int(row.frame_id);receive=int(row.receive_frame_id)
            if period not in frames or not frames[period][0]<=release<receive<=frames[period][-1]:
                reasons['outside_half']+=1;continue
            key=(period,release,str(row.player_id));groups[key].add(str(row.receiver_id))
            eligible.append((key,str(row.receiver_id)))
        x,mask,y,periods,ticks,views,actors,receivers=[],[],[],[],[],[],[],[]
        used=set();rng=np.random.default_rng(PLAN['seed']+TRAIN.index(match))
        for key,receiver in eligible:
            if len(groups[key])>1:reasons['ambiguous_receiver_at_release']+=1;continue
            if key in used:reasons['duplicate_release']+=1;continue
            used.add(key);period,tick,actor=key;index=tick-int(frames[period][0]);a=canonical_slot[actor];r=canonical_slot[receiver]
            history=arrays[period][max(0,index-50):index+1]
            if not np.isfinite(history[-1,a]).all():reasons['actor_missing_at_release']+=1;continue
            for view in PLAN['views']:
                observed=history.copy()
                if view=='history_masked':
                    hide=rng.random(len(people))<PLAN['masked_candidate_probability'];hide[a]=False
                    observed[-PLAN['masked_recent_frames']:,hide,:]=np.nan
                features,valid=receiver_features(observed,teams,a)
                x.append(features);mask.append(valid);y.append(r if valid[r] else len(people))
                periods.append(period);ticks.append(tick);views.append(view);actors.append(actor);receivers.append(receiver)
        values=dict(x=np.asarray(x),mask=np.asarray(mask),target=np.asarray(y),period=np.asarray(periods),
            frame=np.asarray(ticks),view=np.asarray(views),actor=np.asarray(actors),receiver=np.asarray(receivers),canonical_ids=np.asarray(sorted(canonical_slot,key=canonical_slot.get)))
        assert values['x'].shape==(len(y),len(people),len(NAMES)) and np.isfinite(values['x']).all()
        assert all(target==len(people) or valid[target] for target,valid in zip(y,mask))
        path=args.out/(match+'.npz');np.savez_compressed(path,**values);files[path.name]=sha256(path)
        summaries.append(dict(match=match,examples=len(y),unique_releases=len(y)//2,candidates=len(people),exclusions=dict(reasons),
            unavailable_target_by_view={view:int(((values['view']==view)&(values['target']==len(people))).sum()) for view in PLAN['views']}))
        print(json.dumps(summaries[-1]),flush=True)
    for name,digest in {**ledger,**source_hashes}.items():assert sha256(ROOT/name)==digest
    dump(args.out/'preparation.json',dict(complete=True,plan_sha256=sha256(args.out/'plan.json'),summaries=summaries,files=files,
        provided_match_used=False,public_labels_used=False,training_matches_only=True,model_fitted=False,
        possession_labels_changed=False,unexpected_competition_access=denied,competition_target_verified=False))


if __name__=='__main__':main()
