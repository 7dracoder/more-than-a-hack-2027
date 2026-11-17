"""Fit one external full-sequence boundary classifier on a frozen backbone."""
import argparse
import json
from pathlib import Path
import sys
import warnings
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from possession.data import sha256
from possession.frozen_boundary_transfer import transition_targets,balance_weights,calibrated_probability
from possession.observation_boundary import mix_boundaries
from possession.uncertain_decode import decode_labels
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump,score_parts
from scripts.research.spatial_memory.train_real_broadcast_memory import VIEWS
from scripts.research.spatial_memory.train_partial_had_memory import selection_result
from scripts.research.spatial_memory.audit_broadcast_transfer import weak_counts,summarize_counts


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--features',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'selection.json').exists():raise ValueError('Preserve completed selection')
    preparation=json.loads((args.features/'embedding_freeze.json').read_text());feature_plan=json.loads((args.features/'plan.json').read_text());cfg=feature_plan['config']
    assert preparation['complete'] and preparation['backbone_frozen'] and preparation['original_development_outputs_reproduced']
    assert not preparation['unexpected_competition_access']
    assert sha256(args.features/'plan.json')==preparation['plan_sha256']
    paths=[args.features/'embedding_freeze.json',args.features/'plan.json']+[ROOT/p for p in feature_plan['inputs']]
    paths += [args.features/(r['stem']+'.npy') for r in preparation['records']]
    denied=restrict_external(args.out,paths)
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    for name,digest in {**feature_plan['sources'],**feature_plan['inputs']}.items():assert sha256(ROOT/name)==digest
    assert sha256(ROOT/'configs/frozen_boundary_transfer.json')==feature_plan['config_sha256']
    sources=[Path(__file__),ROOT/'possession/frozen_boundary_transfer.py',ROOT/'possession/observation_boundary.py',ROOT/'possession/uncertain_decode.py',ROOT/'scripts/research/spatial_memory/run_external.py',ROOT/'scripts/research/spatial_memory/train_partial_had_memory.py']
    fallback=json.loads((ROOT/cfg['fallback']/'selection.json').read_text())
    plan=dict(config=cfg,preparation_sha256=sha256(args.features/'embedding_freeze.json'),inputs={str(p.resolve().relative_to(ROOT)):sha256(p) for p in paths},sources={str(p.relative_to(ROOT)):sha256(p) for p in sources},fallback_sha256=sha256(ROOT/cfg['fallback']/'selection.json'),provided_match_used=False,independent_test=False)
    dump(args.out/'plan.json',plan)
    xs,ys,ws,inventory=[],[],[],[]
    for r in preparation['records']:
        path=args.features/(r['stem']+'.npy');assert sha256(path)==r['sha256']
        if not r['train']:continue
        assert r['domain']=='strong'
        truth=np.load(ROOT/r['truth']);target=transition_targets(truth)
        embedding=np.load(path,mmap_mode='r');assert len(embedding)==len(target)
        negatives=np.flatnonzero(target==0);chosen_negatives=negatives[::cfg['negative_stride']]
        indices=np.sort(np.r_[np.flatnonzero(target==1),chosen_negatives])
        multiplier=len(negatives)/len(chosen_negatives)
        x=embedding[indices].reshape(-1,embedding.shape[-1]);y=np.repeat(target[indices],2)
        weight=np.repeat(np.where(target[indices]==1,1.,multiplier)*cfg['train_views'][r['view']]/2,2)
        xs.append(x);ys.append(y);ws.append(weight)
        inventory.append(dict(stem=r['stem'],positive=int((target==1).sum()),negative=len(negatives),excluded=int((target<0).sum()),selected_negative=len(chosen_negatives),negative_sampling_multiplier=multiplier,view_weight=cfg['train_views'][r['view']],two_inference_orientations=True))
    x=np.concatenate(xs);y=np.concatenate(ys);sampling=np.concatenate(ws);del xs,ys,ws
    assert set(np.unique(y))=={0,1} and len(inventory)==36
    balanced,shift=balance_weights(y,sampling);scaler=StandardScaler()
    with threadpool_limits(limits=2):x=scaler.fit_transform(x,sample_weight=sampling)
    model=LogisticRegression(C=cfg['C'],max_iter=cfg['max_iter'],random_state=cfg['seed'])
    print(json.dumps(dict(training_rows=len(x),dimensions=x.shape[1],log_weight_odds=shift)),flush=True)
    with threadpool_limits(limits=2),warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning);model.fit(x,y,sample_weight=balanced)
    if any(issubclass(w.category,ConvergenceWarning) for w in caught):raise RuntimeError('Boundary classifier did not converge')
    joblib.dump(dict(model=model,scaler=scaler,log_weight_odds=shift,config=cfg),args.out/'selected.joblib')
    dump(args.out/'training_inventory.json',dict(rows=inventory,total_rows=len(x),positive_rows=int((y==1).sum()),log_weight_odds=shift,normalization_training_only=True,backbone_frozen=True))
    del x,y,sampling,balanced
    parts=[]
    for r in preparation['records']:
        if r['train']:continue
        embedding=np.load(args.features/(r['stem']+'.npy'),mmap_mode='r');shape=embedding.shape
        with threadpool_limits(limits=2):native=calibrated_probability(model.decision_function(scaler.transform(embedding.reshape(-1,shape[-1]))),shift).reshape(shape[:2]).mean(1)
        with np.load(ROOT/r['reference']) as z:
            h={k:z[k] for k in ('probability','expert','gate','memory','memory_names')}
        h.update(domain=r['domain'],view=r['view']);h['expert']=h['expert'].copy();h['expert'][:,3]=native
        if r['domain']=='strong':h['truth']=np.load(ROOT/r['truth'])
        else:
            with np.load(ROOT/r['truth']) as z:h.update(truth=z['truth'],ticks=z['ticks'],seen=z['detected_ball_seen'])
        np.savez_compressed(args.out/(r['stem']+'_boundary.npz'),native_probability=native)
        parts.append(h)
    import torch
    state=torch.load(ROOT/cfg['initial_model']/'selected.pt',map_location='cpu',weights_only=False)
    rows=[]
    for strength in cfg['boundary_strengths']:
        predictions=[]
        for h in parts:
            p=mix_boundaries(h['probability'],h['expert'],h['memory'],h['memory_names'].tolist(),strength)
            predictions.append(decode_labels(p,h['gate'],cfg['continuity_penalty'],*state['decoder_costs']))
        scores={}
        for view in VIEWS:
            indices=[i for i,h in enumerate(parts) if h['domain']=='strong' and h['view']==view]
            scores[view]=score_parts([parts[i]['truth'] for i in indices],[predictions[i] for i in indices])
        weak={'all_known':[],'ball_missing':[]}
        for i,h in enumerate(parts):
            if h['domain']!='weak':continue
            for scope,mask in [('all_known',np.ones(len(h['ticks']),bool)),('ball_missing',~h['seen'])]:weak[scope].append(weak_counts(h['probability'][h['ticks']],predictions[i][h['ticks']],h['truth'],mask))
        weak={scope:summarize_counts(values) for scope,values in weak.items()}
        row=dict(boundary_strength=strength,**selection_result(scores,weak,fallback['selected']))
        if strength==0:
            for view in VIEWS:assert abs(scores[view]['spell_f1_mean']-fallback['selected']['scores'][view]['spell_f1_mean'])<1e-12
            for scope in weak:assert abs(weak[scope]['weak_ha_accuracy']-fallback['selected']['weak_scores'][scope]['weak_ha_accuracy'])<1e-12
        rows.append(row);print(json.dumps(row),flush=True)
    best=max((r for r in rows if r['eligible']),key=lambda r:r['objective'])
    for name,digest in {**plan['inputs'],**plan['sources']}.items():assert sha256(ROOT/name)==digest
    dump(args.out/'selection.json',dict(complete=True,selected=best,candidates=rows,plan=plan,model_sha256=sha256(args.out/'selected.joblib'),iterations=model.n_iter_.tolist(),backbone_frozen=True,provided_match_used=False,public_labels_used=False,independent_test=False,unexpected_competition_access=denied,competition_target_verified=False))


if __name__=='__main__':main()
