"""Compare external TRAIN event anchors to provider labels without changing either."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from possession.data import sha256
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump
from scripts.research.spatial_memory.prepare_elastic_training import COMMIT,TRAIN,FILES

INTENTIONAL = {'pass','cross','throw_in','freekick_short','freekick_crossed',
    'goalkick','corner_crossed','corner_short','shot','shot_penalty','shot_freekick',
    'keeper_pick_up','keeper_claim','ball_recovery'}


def team_label(identifier):
    match = re.fullmatch(r'(home|away)_\d+',str(identifier))
    return {'home':'H','away':'A'}[match[1]] if match else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'results.json').exists():
        raise ValueError('Preserve completed audit')
    source = ROOT/'data/external/elastic-benchmark-e245224aad74'
    allowed = [source/'source.json']+[source/name for name in FILES]
    for match in TRAIN:
        allowed += [ROOT/f'data/external/features-v25/{match}.{suffix}' for suffix in ('npz','json')]
    denied = restrict_external(args.out,allowed)
    resolved = {p.resolve() for p in allowed}

    def exact_external(event,values):
        if event!='open' or not values or not isinstance(values[0],(str,bytes,os.PathLike)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        if path.is_relative_to((ROOT/'data').resolve()) and not path.is_relative_to(args.out) and path not in resolved:
            denied.append(str(path))
            raise PermissionError('This audit reads only the two pinned TRAIN fixtures')

    sys.addaudithook(exact_external)
    import pandas as pd
    receipt = json.loads((source/'source.json').read_text())
    assert receipt['complete'] and receipt['commit']==COMMIT and set(receipt['files'])==set(FILES)
    assert receipt['matches']==TRAIN and not receipt['downloaded_development_annotations']
    assert sha256(ROOT/'scripts/research/spatial_memory/prepare_elastic_training.py')==receipt['source_sha256']
    for name,record in receipt['files'].items():
        assert sha256(source/name)==record['sha256']
    ledger = {str(p.relative_to(ROOT)):sha256(p) for p in allowed}
    plan = dict(commit=COMMIT,matches=TRAIN,intentional_types=sorted(INTENTIONAL),
        input_sha256=ledger,source_sha256=sha256(Path(__file__)),
        team_mapping='Only full home_<integer> or away_<integer> identifiers yield H/A. Nonplayer receivers are excluded.',
        policy='External TRAIN diagnostic only. Do not overwrite provider truth or generate training targets from this comparison.',
        provided_match_used=False,public_labels_used=False)
    dump(args.out/'plan.json',plan)
    rows = []
    for match in TRAIN:
        path = ROOT/f'data/external/features-v25/{match}.npz'
        metadata = json.loads(path.with_suffix('.json').read_text())
        assert sha256(path)==metadata['sha256']
        with np.load(path) as data:
            clocks = {i+1:(data[f'frames{i}'],data[f'truth{i}']) for i in (0,1)}
        for frames,truth in clocks.values():
            assert len(frames)==len(truth) and np.all(np.diff(frames)==1)
        events = pd.read_parquet(source/f'benchmark/gt/{match}.parquet')
        counts = {scope:Counter() for scope in ('actors','intentional_actors','player_receptions')}
        nonplayer_receivers,conflicts = Counter(),{}
        for row in events.itertuples():
            period = int(row.period_id)
            assert period in clocks
            points = [('actors',row.frame_id,row.player_id)]
            if row.spadl_type in INTENTIONAL:
                points.append(('intentional_actors',row.frame_id,row.player_id))
            if pd.notna(row.receive_frame_id):
                if team_label(row.receiver_id):
                    points.append(('player_receptions',row.receive_frame_id,row.receiver_id))
                else:
                    nonplayer_receivers[str(row.receiver_id)]+=1
            for scope,tick,identifier in points:
                label = team_label(identifier)
                if label is None:
                    counts[scope]['unmapped_team']+=1
                    continue
                assert np.isfinite(tick) and tick==int(tick)
                frames,truth = clocks[period]
                index = int(tick)-int(frames[0])
                if not 0<=index<len(truth):
                    counts[scope]['outside_tracking']+=1
                    continue
                known = str(truth[index])
                counts[scope]['mapped_points']+=1
                counts[scope]['provider_'+known]+=1
                counts[scope]['team_agrees']+=int(known==label)
                if scope!='actors':
                    conflicts.setdefault((period,int(tick)),set()).add(label)
        rows.append(dict(match=match,events=len(events),receive_timestamp_fields=int(events.receive_frame_id.notna().sum()),
            nonplayer_receive_timestamps=dict(nonplayer_receivers),scopes={k:dict(v) for k,v in counts.items()},
            conflicting_control_teams_at_same_frame=sum(len(v)>1 for v in conflicts.values())))
    for name,digest in ledger.items():
        assert sha256(ROOT/name)==digest
    result = dict(complete=True,rows=rows,plan_sha256=sha256(args.out/'plan.json'),
        training_matches_only=True,development_annotations_downloaded_or_read=False,
        changed_provider_truth=False,used_to_fit_or_select_current_model=False,
        provided_match_used=False,public_labels_used=False,unexpected_competition_access=denied,
        independent_test=False,competition_target_verified=False)
    dump(args.out/'results.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    main()
