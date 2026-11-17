"""Exploratory receiver-choice auxiliary model; never a possession score."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from possession.data import sha256
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump

CONFIG=dict(seed=20351124,train_match='J03WMX',development_match='J03WN1',epochs=100,
    learning_rate=.001,weight_decay=.001,batch_size=128,hidden=64,
    selection='Highest mean native/history-masked development top-1 accuracy; earlier epoch wins ties.',
    limitation='Both fixtures belong to the existing possession TRAIN pool. This is auxiliary cross-match development, not independent possession evaluation.',
    provided_match_used=False,public_labels_used=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'results.json').exists():raise ValueError('Preserve completed fit')
    paths=[args.data/name for name in ['plan.json','preparation.json','J03WMX.npz','J03WN1.npz']]
    denied=restrict_external(args.out,paths)
    import torch
    torch.set_num_threads(2);torch.manual_seed(CONFIG['seed']);torch.use_deterministic_algorithms(True)
    preparation=json.loads(paths[1].read_text());assert preparation['complete'] and preparation['training_matches_only'] and not preparation['unexpected_competition_access']
    assert sha256(paths[0])==preparation['plan_sha256']
    for path in paths[2:]:assert sha256(path)==preparation['files'][path.name]
    plan=dict(config=CONFIG,inputs={str(p.resolve().relative_to(ROOT)):sha256(p) for p in paths},source_sha256=sha256(Path(__file__)))
    dump(args.out/'plan.json',plan)
    loaded=[]
    for path in paths[2:]:
        with np.load(path) as z:loaded.append({k:z[k] for k in ['x','mask','target','view','frame','period']})
    train,dev=loaded
    class Receiver(torch.nn.Module):
        def __init__(self):
            super().__init__();self.candidate=torch.nn.Sequential(torch.nn.Linear(train['x'].shape[-1],64),torch.nn.GELU(),torch.nn.Linear(64,64),torch.nn.GELU(),torch.nn.Linear(64,1));self.unavailable=torch.nn.Parameter(torch.zeros(1))
        def forward(self,x,mask):
            logits=self.candidate(x).squeeze(-1).masked_fill(~mask,-1e9)
            return torch.cat([logits,self.unavailable.expand(len(x),1)],dim=1)
    model=Receiver();optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    x=torch.tensor(train['x']);mask=torch.tensor(train['mask']);y=torch.tensor(train['target'],dtype=torch.long)
    dx=torch.tensor(dev['x']);dm=torch.tensor(dev['mask']);dy=dev['target'];rng=np.random.default_rng(CONFIG['seed'])
    history=[];best=-1.;best_epoch=0
    def metrics(logits):
        order=np.argsort(-logits,axis=1);result={}
        for view in ['native','history_masked']:
            keep=dev['view']==view
            result[view]=dict(examples=int(keep.sum()),top1=float((order[keep,0]==dy[keep]).mean()),top3=float((order[keep,:3]==dy[keep,None]).any(1).mean()))
        return result
    # Baseline uses only causal release-time geometry and team membership.
    distance=dev['x'][:,:,2];same=dev['x'][:,:,3]>.5
    baselines={}
    for name,eligible in [('nearest_candidate',dev['mask']),('nearest_teammate',dev['mask']&same)]:
        logits=np.full((len(dx),dx.shape[1]+1),-1e9);logits[:,:-1]=np.where(eligible,-distance,-1e9);logits[:,-1]=np.where(eligible.any(1),-1e9,0)
        baselines[name]=metrics(logits)
    for epoch in range(1,CONFIG['epochs']+1):
        model.train();losses=[]
        for start in range(0,len(x),CONFIG['batch_size']):
            if start==0:order=rng.permutation(len(x))
            take=order[start:start+CONFIG['batch_size']];optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(x[take],mask[take]),y[take]);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():logits=model(dx,dm).numpy()
        result=metrics(logits);objective=float(np.mean([v['top1'] for v in result.values()]))
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),scores=result))
        if objective>best:
            best=objective;best_epoch=epoch;torch.save(dict(model=model.state_dict(),config=CONFIG,input_features=train['x'].shape[-1],epoch=epoch),args.out/'selected.pt');np.savez_compressed(args.out/'selected_development.npz',logits=logits,target=dy,view=dev['view'])
    saved=np.load(args.out/'selected_development.npz');selected=metrics(saved['logits']);assert best==float(np.mean([v['top1'] for v in selected.values()]))
    for name,digest in plan['inputs'].items():assert sha256(ROOT/name)==digest
    assert sha256(Path(__file__))==plan['source_sha256']
    dump(args.out/'history.json',history)
    result=dict(complete=True,selected_epoch=best_epoch,receiver_development_scores=selected,baselines=baselines,
        checkpoint_sha256=sha256(args.out/'selected.pt'),plan_sha256=sha256(args.out/'plan.json'),
        provided_match_used=False,public_labels_used=False,unexpected_competition_access=denied,
        integrated_into_possession_model=False,independent_possession_test=False,competition_target_verified=False)
    dump(args.out/'results.json',result);print(json.dumps(result))


if __name__=='__main__':main()
