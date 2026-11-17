"""Reproduce receiver scores functionally and quantify a conditional auxiliary gain."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from possession.data import sha256
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();args.run=args.run.resolve()
    result=json.loads((args.run/'results.json').read_text());plan=json.loads((args.run/'plan.json').read_text())
    paths=[ROOT/name for name in plan['inputs']];denied=restrict_external(args.run,paths)
    for name,digest in plan['inputs'].items():assert sha256(ROOT/name)==digest
    assert sha256(ROOT/'scripts/research/spatial_memory/train_receiver_choices.py')==plan['source_sha256']
    assert sha256(args.run/'selected.pt')==result['checkpoint_sha256'] and sha256(args.run/'plan.json')==result['plan_sha256']
    assert result['complete'] and not result['unexpected_competition_access']
    import torch
    import torch.nn.functional as f
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    state=torch.load(args.run/'selected.pt',map_location='cpu',weights_only=False)
    assert state['epoch']==result['selected_epoch'] and state['config']==plan['config']
    data_path=next(p for p in paths if p.name=='J03WN1.npz')
    with np.load(data_path) as data:
        x=data['x'];mask=data['mask'];target=data['target'];view=data['view'];frames=data['frame'];periods=data['period']
    weights=state['model']
    with torch.no_grad():
        h=torch.as_tensor(x)
        for layer in ('0','2'):
            h=f.gelu(f.linear(h,weights['candidate.'+layer+'.weight'],weights['candidate.'+layer+'.bias']))
        scores=f.linear(h,weights['candidate.4.weight'],weights['candidate.4.bias']).squeeze(-1)
        scores=scores.masked_fill(~torch.as_tensor(mask),-1e9)
        logits=torch.cat([scores,weights['unavailable'].expand(len(x),1)],dim=1).numpy()
    with np.load(args.run/'selected_development.npz') as saved:np.testing.assert_array_equal(logits,saved['logits'])
    predictions=np.argsort(-logits,axis=1);learned=(predictions[:,0]==target)
    same=x[:,:,3]>.5;eligible=mask&same;distance=x[:,:,2]
    baseline=np.where(eligible,-distance,-1e9)
    baseline=np.c_[baseline,np.where(eligible.any(1),-1e9,0)]
    reference=np.argmax(baseline,axis=1)==target
    for kind in ('native','history_masked'):
        take=view==kind;expected=result['receiver_development_scores'][kind]
        assert float(learned[take].mean())==expected['top1']
        assert float((predictions[take,:3]==target[take,None]).any(1).mean())==expected['top3']
        # Use the original sort convention for the fixed baseline too.
        ref=np.argsort(-baseline[take],axis=1)[:,0]==target[take]
        assert float(ref.mean())==result['baselines']['nearest_teammate'][kind]['top1']
        reference[take]=ref
    grouped={}
    for i,key in enumerate(zip(periods,frames)):
        grouped.setdefault(tuple(map(int,key)),[]).append(i)
    assert all(len(v)==2 and set(view[v])=={'native','history_masked'} for v in grouped.values())
    differences=np.array([(learned[indices].astype(float)-reference[indices]).mean() for indices in grouped.values()])
    rng=np.random.default_rng(0);draws=np.array([rng.choice(differences,len(differences),replace=True).mean() for _ in range(1000)])
    verification=dict(complete=True,functional_forward_exactly_reproduced=True,scores_reproduced=True,
        unique_release_clusters=len(grouped),observed_receiver_top1_difference=float(differences.mean()),
        paired_release_interval_5_95=np.percentile(draws,[5,95]).tolist(),
        limitation='Conditional on selected epoch, one auxiliary development fixture, two correlated views grouped by release. No epoch-search or match-level uncertainty adjustment; no possession-F1 conclusion.',
        input_hashes_verified=True,source_sha256=sha256(Path(__file__)),provided_match_used=False,public_labels_used=False,
        unexpected_competition_access=denied,integrated_into_possession_model=False,competition_target_verified=False)
    dump(args.run/'verification.json',verification);print(json.dumps(verification))


if __name__=='__main__':main()
