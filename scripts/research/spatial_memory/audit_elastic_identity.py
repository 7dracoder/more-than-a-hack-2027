"""Verify external receiver identities from metadata, never nearest-player guesses."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from possession.data import sha256
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump
from scripts.research.spatial_memory.prepare_elastic_training import TRAIN,COMMIT
from scripts.research.spatial_memory.audit_elastic_training import team_label,INTENTIONAL


def canonical_players(root):
    general=next(root.iter('General'))
    sides={general.get('HomeTeamId'):'home',general.get('GuestTeamId'):'away'}
    mapping={}
    for team in root.iter('Team'):
        players=team.find('Players')
        if players is None: continue
        side=sides[team.get('TeamId')]
        assert side==('away' if team.get('Role')=='guest' else 'home')
        for player in players.findall('Player'):
            key=f"{side}_{int(player.get('ShirtNumber'))}"
            if key in mapping: raise ValueError('Ambiguous shirt number in team metadata')
            mapping[key]=player.get('PersonId')
    assert len(set(mapping.values()))==len(mapping)
    return mapping


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'results.json').exists(): raise ValueError('Preserve completed audit')
    source=ROOT/'data/external/elastic-benchmark-e245224aad74'
    paths=[source/'source.json',source/'mapping-source/sportec_data.py',source/'mapping-source/match_data.py']
    specs=[]
    for match in TRAIN:
        meta=next((ROOT/'data/external/sportec').glob('*matchinformation*'+match+'.xml'))
        tracking=next((ROOT/'data/external/sportec').glob('*positions_raw*'+match+'.xml'))
        event=source/f'benchmark/gt/{match}.parquet';paths += [meta,tracking,event];specs.append((match,meta,tracking,event))
    denied=restrict_external(args.out,paths)
    import pandas as pd
    receipt=json.loads((source/'source.json').read_text());assert receipt['commit']==COMMIT and receipt['matches']==TRAIN
    ledger={str(p.relative_to(ROOT)):sha256(p) for p in paths}
    plan=dict(commit=COMMIT,matches=TRAIN,input_sha256=ledger,source_sha256=sha256(Path(__file__)),
        mapping='Source load_lineup_data: team role plus integer ShirtNumber -> PersonId; verify against General home/guest IDs.',
        policy='Coordinate-consistency diagnostic only; no distance-based identity selection, threshold fitting, labels or model changes.',provided_match_used=False,public_labels_used=False)
    dump(args.out/'plan.json',plan);results=[]
    for match,meta,tracking,event in specs:
        assert sha256(event)==receipt['files'][f'benchmark/gt/{match}.parquet']['sha256']
        mapping=canonical_players(ET.parse(meta).getroot());events=pd.read_parquet(event);points=[]
        for row in events.itertuples():
            if row.spadl_type in INTENTIONAL: points.append(('actor',int(row.period_id),int(row.frame_id),str(row.player_id)))
            if pd.notna(row.receive_frame_id) and team_label(row.receiver_id):
                points.append(('receiver',int(row.period_id),int(row.receive_frame_id),str(row.receiver_id)))
        assert all(identifier in mapping for _,_,_,identifier in points)
        needed={(period,tick,mapping[identifier]) for _,period,tick,identifier in points}
        ticks={(period,tick) for _,period,tick,_ in points};observed={};balls={};current=None
        for kind,item in ET.iterparse(tracking,events=('start','end')):
            if kind=='start' and item.tag=='FrameSet':
                section=item.get('GameSection');period={'firstHalf':1,'secondHalf':2}.get(section)
                person=item.get('PersonId','');ball=item.get('TeamId','').lower()=='ball' or person.lower()=='ball'
                current=(period,person,ball)
            elif kind=='end' and item.tag=='Frame' and current:
                period,person,ball=current;tick=int(item.get('N'))
                if (ball and (period,tick) in ticks) or (period,tick,person) in needed:
                    xy=[float(item.get('X','nan')),float(item.get('Y','nan'))]
                    if ball: balls[period,tick]=xy
                    else: observed[period,tick,person]=xy
                item.clear()
            elif kind=='end' and item.tag=='FrameSet': current=None;item.clear()
        scopes={}
        for scope in ('actor','receiver'):
            counts=Counter();distances=[]
            for kind,period,tick,identifier in points:
                if kind!=scope:continue
                counts['points']+=1
                xy=np.asarray(observed.get((period,tick,mapping[identifier]),[np.nan,np.nan]))
                ball=np.asarray(balls.get((period,tick),[np.nan,np.nan]))
                if np.isfinite(xy).all():counts['player_coordinates_present']+=1
                if np.isfinite(xy).all() and np.isfinite(ball).all(): distances.append(float(np.linalg.norm(xy-ball)))
            scopes[scope]=dict(counts=counts,finite_distances=len(distances),median_metres=float(np.median(distances)),p90_metres=float(np.percentile(distances,90)),within_metres={str(r):sum(d<=r for d in distances) for r in (1,2,5)})
        results.append(dict(match=match,unique_metadata_players=len(mapping),all_anchor_ids_resolved=True,scopes=scopes))
        print(json.dumps(results[-1]),flush=True)
    for name,digest in ledger.items():assert sha256(ROOT/name)==digest
    dump(args.out/'results.json',dict(complete=True,rows=results,plan_sha256=sha256(args.out/'plan.json'),training_only=True,provided_match_used=False,public_labels_used=False,unexpected_competition_access=denied,model_changed=False,competition_target_verified=False))


if __name__=='__main__':main()
