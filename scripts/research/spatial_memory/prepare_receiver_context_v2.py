"""Migrate entropy using causal visibility counts and verify fresh neural snapshots."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from possession.data import sha256
from possession.receiver_context_v2 import marginal_context,NAMES
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump
from scripts.research.spatial_memory.train_natural_visibility import seed_for

CONFIG=dict(stride_frames=10,output_step=2,fps=25,actor_distance_scale_metres=2,
    player_history_seconds=2,ball_history_seconds=8,interpolation='Causal hold; explicit snapshot-age feature.',
    entropy_normalization='Recently observed candidates plus unavailable class; independent of never-observed padding.',names=NAMES+['receiver_snapshot_age'],provided_match_used=False,public_labels_used=False)


def migrate(previous,players):
    seen=np.isfinite(players).all(-1);ticks=np.arange(len(players))
    last=np.maximum.accumulate(np.where(seen,ticks[:,None],-1),axis=0)
    sample=ticks[::10]
    count=((last[sample]>=np.maximum(0,sample-50)[:,None])&(last[sample]>=0)).sum(1)
    factor=np.log(players.shape[1]+1)/np.log(np.maximum(2,count+1))
    output=previous.copy();output[:,5]*=factor[np.arange(0,len(players),2)//10]
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    split=json.loads((ROOT/'configs/real_broadcast_memory_adaptation.json').read_text())
    natural=ROOT/'data/research/natural-visibility-20261114'
    model=ROOT/'data/research/receiver-choice-model-20261114/run-01'
    previous=ROOT/'data/research/receiver-context-20261114/run-03'
    allowed=[natural/'external_visibility.json',previous/'preparation.json',previous/'plan.json']+list(previous.glob('*.npy'))+list(previous.glob('*.json'))+[model/name for name in ('verification.json','results.json','selected.pt')]
    for match in split['strong_train_matches']+split['strong_development_matches']:
        for half in (0,1):
            for view in ('native','masked','mixed'):
                path=natural/'cache'/f'{match}_{half}_{view}.npy';allowed += [path,path.with_suffix('.json')]
    transfer=ROOT/'data/research/broadcast-transfer-20261114'
    for match in split['weak_development_matches']:
        for period in (1,2):
            p=transfer/'features'/f'{match}_{period}_detected.npy';allowed += [p,p.with_suffix('.json')]
    denied=restrict_external(args.out,allowed)
    import torch
    from possession.external import read_sportec
    from possession.external_visibility import degrade
    from possession.skillcorner import load_match,resample_coordinates
    torch.set_num_threads(2)
    verified=json.loads((model/'verification.json').read_text());assert verified['complete'] and verified['functional_forward_exactly_reproduced']
    result=json.loads((model/'results.json').read_text());assert sha256(model/'selected.pt')==result['checkpoint_sha256']
    weights=torch.load(model/'selected.pt',map_location='cpu',weights_only=False)['model']
    profile=json.loads((natural/'external_visibility.json').read_text());assert not profile['provided_match_used'] and not profile['possession_labels_used']
    old_preparation=json.loads((previous/'preparation.json').read_text())
    assert old_preparation['complete'] and not old_preparation['unexpected_competition_access']
    old_plan=json.loads((previous/'plan.json').read_text())
    assert sha256(previous/'plan.json')==old_preparation['plan_sha256']
    for name,digest in old_plan['sources'].items():assert sha256(ROOT/name)==digest
    sources=[Path(__file__),ROOT/'possession/receiver_context_v2.py',ROOT/'possession/receiver_context.py',ROOT/'possession/external.py',ROOT/'scripts/research/spatial_memory/train_natural_visibility.py',ROOT/'possession/receiver_geometry.py',ROOT/'possession/external_visibility.py',ROOT/'possession/skillcorner.py']
    plan=dict(config=CONFIG,split=split,previous_preparation_sha256=sha256(previous/'preparation.json'),receiver_model_sha256=sha256(model/'selected.pt'),profile_sha256=sha256(natural/'external_visibility.json'),sources={str(p.relative_to(ROOT)):sha256(p) for p in sources})
    if (args.out/'plan.json').exists():assert json.loads((args.out/'plan.json').read_text())==plan
    dump(args.out/'plan.json',plan);files={}
    def save(name,p,teams,b,pitch,cache,extra):
        path=args.out/(name+'.npy');meta=path.with_suffix('.json')
        cached=json.loads(cache.with_suffix('.json').read_text());assert sha256(cache)==cached['sha256']
        old=previous/(name+'.npy');old_meta=old.with_suffix('.json')
        assert sha256(old)==old_preparation['files'][old.name] and sha256(old_meta)==old_preparation['files'][old_meta.name]
        assert json.loads(old_meta.read_text())['fingerprint']['coordinate_sources']==extra
        expected=dict(plan=plan,coordinate_sources=extra,baseline_cache_sha256=sha256(cache),previous_feature_sha256=sha256(old))
        if path.exists() and meta.exists():
            record=json.loads(meta.read_text());assert record['fingerprint']==expected and sha256(path)==record['sha256']
        else:
            base=np.load(cache,mmap_mode='r');compare=base if len(base)==len(b[::2]) else base[::2]
            np.testing.assert_array_equal(compare[:,0],np.isfinite(b[::2]).all(1).astype(np.float32))
            original=np.load(old);x=migrate(original,p)
            assert len(x)==len(compare) and np.isfinite(x).all()
            np.testing.assert_array_equal(x[:,[i for i in range(11) if i!=5]],original[:,[i for i in range(11) if i!=5]])
            assert x[:,5].min()>=0 and x[:,5].max()<=1.00001
            for tick in np.unique(np.linspace(0,(len(b)-1)//10,20,dtype=int))*10:
                direct=marginal_context(p[max(0,tick-50):tick+1],teams,b[max(0,tick-200):tick+1],weights,pitch)
                np.testing.assert_allclose(x[tick//2,:10],direct,atol=2e-6,rtol=1e-6)
            np.save(path,x);dump(meta,dict(fingerprint=expected,sha256=sha256(path),rows=len(x),features=CONFIG['names']))
        files[path.name]=sha256(path);files[meta.name]=sha256(meta)
        print('Prepared '+name,flush=True)
    folder=ROOT/'data/external/sportec'
    for match in split['strong_train_matches']+split['strong_development_matches']:
        meta=next(folder.glob('*matchinformation*'+match+'.xml'));raw=next(folder.glob('*positions_raw*'+match+'.xml'))
        extra={str(p.relative_to(ROOT)):sha256(p) for p in [meta,raw]}
        for half,h in enumerate(read_sportec(meta,raw)):
            for view in ('native','masked','mixed'):
                p,b=h.players,h.ball
                if view!='native':p,b=degrade(p,h.teams,b,h.pitch,profile,seed_for(match,half,view),mode='none' if view=='masked' else 'mixed')
                save(f'{match}_{half}_{view}',p,h.teams,b,h.pitch,natural/'cache'/f'{match}_{half}_{view}.npy',extra)
    for match in split['weak_development_matches']:
        folder=ROOT/'data/external/skillcorner-raw'/match
        source=json.loads((folder/'source.json').read_text());meta=folder/'metadata.json';raw=folder/'tracking.jsonl'
        assert sha256(meta)==source['metadata_sha256'] and sha256(raw)==source['tracking_sha256']
        metadata=json.loads(meta.read_text(encoding='utf-8'));pitch=np.array([metadata['pitch_length'],metadata['pitch_width']],np.float32)
        halves,teams=load_match(metadata,raw)
        for period in (1,2):
            p,b=halves[period]['detected'];p,low=resample_coordinates(p+pitch/2,speed_limit=15.);b,blow=resample_coordinates(b+pitch/2,speed_limit=80.)
            np.testing.assert_array_equal(low,blow)
            save(f'{match}_{period}_detected',p,teams,b,pitch,transfer/'features'/f'{match}_{period}_detected.npy',{str(q.relative_to(ROOT)):sha256(q) for q in [meta,raw]})
    for name,digest in plan['sources'].items():assert sha256(ROOT/name)==digest
    dump(args.out/'preparation.json',dict(complete=True,plan_sha256=sha256(args.out/'plan.json'),files=files,provided_match_used=False,public_labels_used=False,external_possession_labels_parsed_by_shared_loader=True,possession_labels_used_as_features=False,unexpected_competition_access=denied))


if __name__=='__main__':main()
