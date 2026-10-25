"""Check all carrier-target masks and sampled identities directly against pinned raw frames."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from possession.data import sha256


def point(value,detected):
    if value is None or (detected and value.get('is_detected') is not True):return np.full(2,np.nan,np.float32)
    return np.array([np.nan if value.get(k) is None else value[k] for k in ('x','y')],np.float32)


def interpolate(a,b,fraction,limit):
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.linalg.norm(b-a)*10>limit:return np.full(2,np.nan,np.float32)
    return (a.astype(np.float64)+(b-a).astype(np.float64)*fraction).astype(np.float32)


def main():
    raw_root=ROOT/'data/external/skillcorner-raw';inventory=json.loads((raw_root/'inventory.json').read_text());records=[]
    for match in inventory['matches']:
        directory=raw_root/match;source=json.loads((directory/'source.json').read_text());metadata=json.loads((directory/'metadata.json').read_bytes())
        if sha256(directory/'tracking.jsonl')!=source['tracking_sha256']:raise ValueError('Raw carrier source changed')
        team_for={p['id']:int(p['team_id']==metadata['away_team']['id']) for p in metadata['players']};ids=list(team_for)
        periods={p['period']:p for p in metadata['match_periods'] if p['period'] in (1,2)}
        requests=[];required=set();pending=[]
        for view in ('native','detected'):
            path=ROOT/('data/external/skillcorner-carrier-'+view)/(match+'.npz');record=json.loads(path.with_suffix('.json').read_text())
            features=ROOT/('data/external/skillcorner-trajectories-'+view)/(match+'.npz')
            if record['source']!=source or sha256(path)!=record['sha256'] or sha256(features)!=record['feature_sha256']:raise ValueError('Carrier cache lineage differs')
            if any(sha256(ROOT/p)!=digest for p,digest in record['producers'].items()):raise ValueError('Carrier producer changed')
            count=0
            with np.load(path,allow_pickle=False) as cache,np.load(features,allow_pickle=False) as original:
                for i,period in enumerate((1,2)):
                    target=cache[f'carrier{i}'];x=original[f'x{i}'];y=original[f'y{i}']
                    if target.shape!=(len(y),) or not np.isin(target,[-100,0,1,2]).all():raise ValueError('Invalid carrier rank')
                    known=np.flatnonzero(target>=0);count+=len(known)
                    if not np.isin(y[known],['H','A']).all():raise ValueError('Carrier target on an unknown group')
                    side=(y[known]=='A').astype(np.int64);columns=side*90+target[known]*30+17
                    if not (x[known,columns]==1.).all():raise ValueError('Labelled carrier has no coordinate observation')
                    sample=known[np.linspace(0,len(known)-1,min(20,len(known)),dtype=int)] if len(known) else []
                    for tick in sample:
                        low=int(periods[period]['start_frame'])+int(tick)*2//5;remainder=int(tick)*2%5;high=low+int(remainder>0)
                        required.update((low,high))
                        rank=int(target[tick]);side=int(y[tick]=='A')
                        requests.append((view,low,high,remainder/5.,rank,side,float(x[tick,side*90+rank*30+14])*10))
            pending.append(dict(match=match,view=view,sha256=record['sha256'],known=count,all_targets_have_finite_candidates=True))
        rows={}
        with (directory/'tracking.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line)
                if row['frame'] in required:rows[row['frame']]=row
        checked={'native':0,'detected':0}
        for view,low,high,fraction,rank,side,feature_distance in requests:
            a,b=rows[low],rows[high];possession=a.get('possession') or {};carrier=possession.get('player_id')
            if possession.get('group')!=('home team' if side==0 else 'away team') or team_for.get(carrier)!=side:raise ValueError('Raw carrier team differs')
            detected=view=='detected';left={p['player_id']:p for p in a['player_data']};right={p['player_id']:p for p in b['player_data']}
            ball=interpolate(point(a.get('ball_data'),detected),point(b.get('ball_data'),detected),fraction,80.)
            players=np.array([interpolate(point(left.get(identity),detected),point(right.get(identity),detected),fraction,15.) for identity in ids])
            with np.errstate(invalid='ignore'):distances=np.linalg.norm(players-ball,axis=-1)
            eligible=np.array([team_for[identity]==side for identity in ids])&np.isfinite(distances)
            ordered=np.sort(np.where(eligible,distances,np.inf));index=ids.index(carrier)
            if not eligible[index] or abs(float(distances[index])-float(ordered[rank]))>1e-5 or int((eligible&(np.abs(distances-ordered[rank])<=1e-5)).sum())!=1:
                raise ValueError('Carrier rank disagrees with raw player identity/coordinates')
            if not np.isclose(feature_distance,np.clip(distances[index],0,30),atol=1e-4,rtol=1e-6):raise ValueError('Carrier distance differs from coordinate feature')
            checked[view]+=1
        for record in pending:
            record['sampled_raw_identity_checks']=checked[record['view']];records.append(record);print(json.dumps(record),flush=True)
    if len(records)!=40 or not sum(r['known'] for r in records):raise ValueError('Incomplete carrier audit')
    out=ROOT/'data/local-v08/skillcorner_carrier_cache_audit.json'
    partial=out.with_suffix('.partial.json')
    partial.write_text(json.dumps(dict(commit=inventory['commit'],records=records,audit_source_sha256=sha256(Path(__file__)),
        target_kind='Provider carrier identity weak labels, not independent human ground truth'),indent=2));partial.replace(out)


if __name__=='__main__':main()
