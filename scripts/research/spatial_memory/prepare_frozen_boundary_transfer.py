"""Freeze whole-half external embeddings; verify their original boundary outputs."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from possession.data import sha256
from possession.temporal import normalized
from possession.external import swap_features
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump,torch_setup
from scripts.research.spatial_memory.train_real_broadcast_memory import STRONG,TRANSFER,VIEWS
from scripts.research.spatial_memory.train_natural_visibility import model_matrix


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'embedding_freeze.json').exists():raise ValueError('Preserve completed preparation')
    config_path=ROOT/'configs/frozen_boundary_transfer.json';cfg=json.loads(config_path.read_text())
    model_dir=ROOT/cfg['initial_model'];selection=json.loads((model_dir/'selection.json').read_text());split=selection['plan']['config']
    assert selection['complete'] and not selection['provided_match_used'] and not selection['public_labels_used']
    boundary=ROOT/'data/research/observation-boundary-20261114';specs=[];dev_index=0
    for match in split['strong_train_matches']+split['strong_development_matches']:
        for half in (0,1):
            for view in VIEWS:
                stem=f'{match}_{half}_{view}';x=STRONG/'cache'/(stem+'.npy')
                h=dict(stem=stem,domain='strong',train=match in split['strong_train_matches'],view=view,x=x,truth=x.with_name(stem+'_truth.npy'))
                if not h['train']:h['reference']=boundary/'probabilities'/f'strong_{dev_index}.npz';dev_index+=1
                specs.append(h)
    for index,(match,half) in enumerate((m,h) for m in split['weak_development_matches'] for h in (1,2)):
        stem=f'{match}_{half}_detected';x=TRANSFER/'features'/(stem+'.npy')
        specs.append(dict(stem=stem,domain='weak',train=False,x=x,truth=x.with_name(stem+'_weak.npz'),reference=boundary/'probabilities'/f'weak_{index}.npz'))
    paths=[boundary/'probability_freeze.json']
    for h in specs:paths += [h[k] for k in ('x','truth','reference') if k in h]+[h['x'].with_suffix('.json')]
    denied=restrict_external(args.out,paths)
    freeze=json.loads((boundary/'probability_freeze.json').read_text())
    assert sha256(model_dir/'selected.pt')==selection['selected_model_sha256']==freeze['model_sha256']
    sources=[Path(__file__),ROOT/'possession/boundary.py',ROOT/'possession/recurrent.py',ROOT/'possession/temporal.py',ROOT/'possession/external.py']
    plan=dict(config=cfg,config_sha256=sha256(config_path),model_sha256=selection['selected_model_sha256'],split=split,
        sources={str(p.relative_to(ROOT)):sha256(p) for p in sources},inputs={str(p.relative_to(ROOT)):sha256(p) for p in paths},
        provided_match_used=False,independent_test=False)
    dump(args.out/'plan.json',plan)
    import torch
    from scripts.run_local import selected_model
    device=torch_setup('cuda');model,state=selected_model(model_dir,device);model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    records=[]
    with torch.inference_mode():
        for h in specs:
            started=time.monotonic();metadata=json.loads(h['x'].with_suffix('.json').read_text())
            assert sha256(h['x'])==metadata['sha256']==selection['plan']['caches'][str(h['x'].relative_to(ROOT))]
            assert sha256(h['truth'])==selection['plan']['caches'][str(h['truth'].relative_to(ROOT))]
            assert metadata['names'][:664]+metadata['names'][960:]==state['names']
            raw=np.load(h['x'],mmap_mode='r');raw=raw if h['domain']=='strong' else raw[::2]
            matrix=model_matrix(raw)
            original=normalized(matrix,state['mean'],state['scale']);swapped=normalized(swap_features(matrix,state['names']),state['mean'],state['scale'])
            batch=torch.as_tensor(np.stack([original.T,swapped.T]),device=device)
            embedding=model.representation(batch)
            q=model.calibrated_boundary(model.boundary(embedding)[:,0]).float().cpu().numpy().mean(0)
            logits=model.head(embedding);probability=torch.softmax(logits,1).float().cpu().numpy();probability=(probability[0]+probability[1,[1,0,2]])/2
            max_difference=None
            if 'reference' in h:
                assert sha256(h['reference'])==freeze['files'][h['reference'].name]
                with np.load(h['reference']) as z:reference=z['expert']
                np.testing.assert_allclose(q,reference[:,3],atol=1e-7,rtol=1e-6)
                np.testing.assert_allclose(probability.T,reference[:,:3],atol=1e-7,rtol=1e-6)
                max_difference=float(max(np.max(np.abs(q-reference[:,3])),np.max(np.abs(probability.T-reference[:,:3]))))
            path=args.out/(h['stem']+'.npy');values=embedding.permute(2,0,1).contiguous().float().cpu().numpy();np.save(path,values)
            record=dict(stem=h['stem'],domain=h['domain'],train=h['train'],view=h.get('view'),shape=list(values.shape),sha256=sha256(path),truth=str(h['truth'].relative_to(ROOT)),truth_sha256=sha256(h['truth']),reference=str(h['reference'].relative_to(ROOT)) if 'reference' in h else None,max_original_output_difference=max_difference,seconds=time.monotonic()-started)
            dump(path.with_suffix('.json'),record);records.append(record)
            print(json.dumps(dict(stem=h['stem'],frames=len(values),max_original_output_difference=max_difference)),flush=True)
            del raw,matrix,original,swapped,batch,embedding,values,logits,probability
    for name,digest in {**plan['sources'],**plan['inputs']}.items():assert sha256(ROOT/name)==digest
    assert sha256(model_dir/'selected.pt')==plan['model_sha256']
    dump(args.out/'embedding_freeze.json',dict(complete=True,records=records,plan_sha256=sha256(args.out/'plan.json'),backbone_frozen=True,original_development_outputs_reproduced=True,provided_match_used=False,unexpected_competition_access=denied))


if __name__=='__main__':main()
