"""Train appended receiver-context columns on strong external fixtures only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from possession.data import sha256
from possession.feature_extension import extend_state
from possession.feature_permutation import team_swap_indices
from possession.frozen_projection import configure_added_columns,decay_added_columns,assert_inherited_unchanged
from possession.observation_gate import mix_probabilities,observation_loss
from possession.uncertain_decode import decode_labels
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump,score_parts,torch_setup
from scripts.research.spatial_memory.train_real_broadcast_memory import allowed_inputs,load_data,VIEWS
from scripts.research.spatial_memory.train_partial_had_memory import selection_result
from scripts.research.spatial_memory.audit_broadcast_transfer import weak_counts,summarize_counts


def extension_inputs(study,split):
    result=[study/'plan.json',study/'preparation.json']
    for match in split['strong_train_matches']+split['strong_development_matches']:
        for half in (0,1):
            for view in VIEWS:
                p=study/f'{match}_{half}_{view}.npy';result += [p,p.with_suffix('.json')]
    for match in split['weak_development_matches']:
        for half in (1,2):
            p=study/f'{match}_{half}_detected.npy';result += [p,p.with_suffix('.json')]
    return result


def augment_parts(parts,study,caches):
    for h in parts:
        view=h.get('view','detected');p=study/f"{h['match']}_{h['half']}_{view}.npy"
        receipt=json.loads(p.with_suffix('.json').read_text());assert sha256(p)==receipt['sha256']
        context=np.load(p,mmap_mode='r');assert context.shape==(len(h['x']),11)
        h['x']=np.concatenate([h['x'][:,:664],h['x'][:,960:],context],axis=1)
        for path in [p,p.with_suffix('.json')]:caches[str(path.relative_to(ROOT))]=sha256(path)


def added_normalization(strong,weak,full_names,weights):
    mean=np.zeros(11);second=np.zeros(11)
    for view,weight in zip(VIEWS,weights):
        matrices=[h['x'][:,792:] for h in strong if h['view']==view]
        n=sum(len(x) for x in matrices)
        mean += weight*sum(x.astype(np.float64).sum(0) for x in matrices)/n
        second += weight*sum(np.square(x.astype(np.float64)).sum(0) for x in matrices)/n
    mapping=team_swap_indices(full_names)[792:]-792
    mean=.5*(mean+mean[mapping]);second=.5*(second+second[mapping])
    return mean.astype(np.float32),np.sqrt(np.maximum(second-mean**2,1e-4)).astype(np.float32)


def evaluate(model,state,strong,weak,device):
    from possession.temporal import symmetric_probability
    scores={}
    def prediction(h):
        q=symmetric_probability(model,h['x'],state['names'],state['mean'],state['scale'],device)
        memory=h['x'][:,664:792]
        p=mix_probabilities(h['parent'],q,memory,h['memory_names'],1.)
        gate=np.repeat(observation_loss(memory,h['memory_names']),2)[:len(p)]
        return p,decode_labels(p,gate,8.,*state['decoder_costs'])
    for view in VIEWS:
        current=[h for h in strong if h['view']==view]
        scores[view]=score_parts([h['truth'] for h in current],[prediction(h)[1] for h in current])
    rows={'all_known':[],'ball_missing':[]}
    for h in weak:
        p,labels=prediction(h)
        for scope,mask in [('all_known',np.ones(len(h['ticks']),bool)),('ball_missing',~h['seen'])]:
            rows[scope].append(weak_counts(p[h['ticks']],labels[h['ticks']],h['truth'],mask))
    return scores,{scope:summarize_counts(values) for scope,values in rows.items()}


def batch(rng,config,strong,weak,state,mapping):
    from possession.temporal import normalized
    grouped=[[h for h in strong if h['view']==view] for view in VIEWS]
    bx,by,bw=[],[],[]
    for _ in range(config['batch_size']):
        group=int(rng.choice(3,p=[.5,.25,.25]));h=grouped[group][int(rng.integers(len(grouped[group])))]
        first=int(rng.integers(0,len(h['x'])-config['sequence_length']+1));last=first+config['sequence_length']
        raw=h['x'][first:last];y=h['y'][first:last].copy()
        if rng.random()<.5:raw=raw[:,mapping];y=np.where(y==0,1,np.where(y==1,0,y))
        bx.append(normalized(raw,state['mean'],state['scale']).T);by.append(y);bw.append(h['w'][first:last])
    return np.stack(bx),np.stack(by),np.stack(bw),config['batch_size']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--extension-study',type=Path,default=ROOT / 'data/research/receiver-context-v2-20261114/run-01')
    parser.add_argument('--weak-study',type=Path,default=ROOT / 'data/research/real-broadcast-adaptation-20261114')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    config_path = ROOT / 'configs/receiver_context_v2_adaptation.json'
    config = json.loads(config_path.read_text())
    split = json.loads((ROOT / config['split_config']).read_text())
    assert not config['provided_match_used'] and not config['public_labels_used'] and not config['independent_test_matches']
    assert config['feature_count'] == 803 and config['decoder_penalty'] == 8. and config['dead_offset'] == 0
    assert sum(config['normalization_group_weights']) == 1
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out / 'selection.json').exists():
        raise ValueError('Preserve completed training')
    extensions = extension_inputs(args.extension_study,split)
    denied = restrict_external(args.out,allowed_inputs(split,args.weak_study)+extensions)
    preparation = json.loads((args.extension_study / 'preparation.json').read_text())
    assert preparation['complete'] and not preparation['provided_match_used'] and not preparation['public_labels_used']
    assert not preparation['possession_labels_used_as_features'] and not preparation['unexpected_competition_access']
    prep_plan=json.loads((args.extension_study/'plan.json').read_text())
    assert sha256(args.extension_study/'plan.json')==preparation['plan_sha256'] and prep_plan['split']==split
    for path,digest in prep_plan['sources'].items():assert sha256(ROOT/path)==digest
    device = torch_setup(args.device)
    import torch
    from scripts.run_local import model_from_state,save_torch
    from possession.external import swap_features
    from possession.temporal import frame_loss
    from possession.weak_supervision import conditional_ha_loss
    from possession.boundary import boundary_loss
    initial_dir = ROOT / config['initial_model']
    origin = json.loads((initial_dir / 'selection.json').read_text())
    fallback = json.loads((ROOT / config['continuity_model'] / 'selection.json').read_text())
    assert origin['complete'] and not origin['provided_match_used'] and not origin['public_labels_used']
    assert sha256(initial_dir / 'selected.pt') == origin['selected_model_sha256'] == fallback['selected_model_sha256']
    initial = torch.load(initial_dir / 'selected.pt',map_location='cpu',weights_only=False)
    strong,weak,strong_dev,weak_dev,caches = load_data(split,args.weak_study,initial['names'])
    for path in extensions:
        caches[str(path.relative_to(ROOT))] = sha256(path)
    full_names=initial['names']+prep_plan['config']['names']
    assert len(full_names)==803
    augment_parts(strong+strong_dev+weak_dev,args.extension_study,caches)
    weak=[]  # No weak replay or weak-derived fitting of appended normalization.
    mean,scale = added_normalization(strong,weak,full_names,config['normalization_group_weights'])
    state = extend_state(initial,full_names,mean,scale)
    np.savez(args.out / 'added_normalization.npz',mean=mean,scale=scale)
    sources = ['scripts/research/spatial_memory/train_receiver_context_v2.py','possession/feature_extension.py','possession/feature_permutation.py','possession/frozen_projection.py',
        'scripts/research/spatial_memory/verify_receiver_context_v2.py','possession/receiver_context.py','possession/receiver_context_v2.py',
        'scripts/research/spatial_memory/run_external.py','evaluation/possession_metrics.py',
        'scripts/research/spatial_memory/train_real_broadcast_memory.py',
        'scripts/research/spatial_memory/train_partial_had_memory.py',
        'scripts/research/spatial_memory/external_guard.py','scripts/research/spatial_memory/audit_broadcast_transfer.py',
        'possession/temporal.py','possession/boundary.py','possession/external.py',
        'possession/observation_gate.py','possession/uncertain_decode.py','possession/weak_supervision.py','scripts/run_local.py']
    plan = dict(config=config,split=split,config_sha256=sha256(config_path),
        sources={p:sha256(ROOT / p) for p in sources},caches=caches,
        initial_model_sha256=sha256(initial_dir / 'selected.pt'),initial_protocol_sha256=sha256(initial_dir / 'fixed_protocol.json'),
        initial_selection_sha256=sha256(initial_dir / 'selection.json'),
        fallback_selection_sha256=sha256(ROOT / config['continuity_model'] / 'selection.json'),
        normalization_sha256=sha256(args.out / 'added_normalization.npz'),
        provided_match_used=False,public_labels_used=False,independent_test=False)
    plan_path = args.out / 'plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan
    dump(plan_path,plan)
    fingerprint = hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    torch.manual_seed(config['seed']);rng = np.random.default_rng(config['seed'])
    model = model_from_state(state,device)
    scores,weak_scores = evaluate(model,state,strong_dev,weak_dev,device)
    for view in VIEWS:
        assert abs(scores[view]['spell_f1_mean']-fallback['selected']['scores'][view]['spell_f1_mean']) < 1e-10
    for scope in weak_scores:
        for metric in ('weak_ha_accuracy','conditional_ha_accuracy','conditional_ha_macro_f1'):
            assert abs(weak_scores[scope][metric]-fallback['selected']['weak_scores'][scope][metric]) < 1e-10
    baseline = dict(scores=scores,weak_scores=weak_scores)
    best = dict(epoch=0,fallback=True,**selection_result(scores,weak_scores,baseline))
    dump(args.out / 'initial_reproduction.json',dict(passed=True,baseline=best,zero_extension_verified=True))
    print(json.dumps(dict(initial_reproduced=True,objective=best['objective'],features=len(full_names))),flush=True)
    network_config = dict(state['config'])
    network_config.update(full_geometry_adaptation=config,training_phase='frozen_receiver_context_strong_only',
        epochs=config['epochs'],feature_count=803,train_matches=split['strong_train_matches'])
    state['config'] = network_config
    dump(args.out / 'fixed_protocol.json',dict(config=network_config,fingerprint=fingerprint,preparation=plan,baseline=best))

    def snapshot(epoch,result):
        return dict(model=model.state_dict(),names=state['names'],mean=state['mean'],scale=state['scale'],
            config=network_config,decoder_costs=state['decoder_costs'],epoch=epoch,validation=result,
            fingerprint=fingerprint,warm_start_sha256=plan['initial_model_sha256'],warm_start_epoch=initial['epoch'])

    added_columns = np.arange(792,803)
    trainable_projection = configure_added_columns(model,added_columns)
    assert trainable_projection.shape[0]*len(added_columns) == config['trainable_coefficients'] == 1408
    assert_inherited_unchanged(model.state_dict(),state['model'],added_columns)
    optimizer = torch.optim.AdamW([trainable_projection],lr=config['learning_rate'],weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config['epochs'],eta_min=config['learning_rate']/10.)
    mapping = team_swap_indices(full_names)
    history,start = [],1
    if (args.out / 'latest.pt').exists():
        latest = torch.load(args.out / 'latest.pt',map_location='cpu',weights_only=False)
        assert latest['fingerprint'] == fingerprint and latest['selected_sha256'] == sha256(args.out / 'selected.pt')
        model.load_state_dict(latest['model']);optimizer.load_state_dict(latest['optimizer']);scheduler.load_state_dict(latest['scheduler'])
        assert_inherited_unchanged(model.state_dict(),state['model'],added_columns)
        history,best,start = latest['history'],latest['best'],latest['epoch']+1
        rng.bit_generator.state = latest['rng'];torch.set_rng_state(latest['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(latest['cuda_rng'])
    elif not args.smoke:
        save_torch(args.out / 'selected.pt',snapshot(0,best))
    for epoch in range(start,config['epochs']+1):
        started = time.monotonic();model.train();losses = []
        for _ in range(1 if args.smoke else config['batches_per_epoch']):
            bx,by,bw,ns = batch(rng,config,strong,weak,state,mapping)
            tx,ty,tw = [torch.as_tensor(a,device=device) for a in (bx,by,bw)]
            optimizer.zero_grad(set_to_none=True)
            logits,boundary = model.forward_with_boundary(tx)
            strong_loss = (frame_loss(logits[:ns],ty[:ns])*tw[:ns]).sum()/tw[:ns].sum().clamp_min(1.)
            loss = strong_loss
            loss += config['strong_boundary_auxiliary_weight']*boundary_loss(boundary[:ns],ty[:ns],model.positive_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite receiver-context loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            decay_added_columns(trainable_projection,added_columns,optimizer.param_groups[0]['lr'],config['weight_decay'])
            optimizer.step();losses.append(float(loss.detach()))
        assert_inherited_unchanged(model.state_dict(),state['model'],added_columns)
        if args.smoke:
            dump(args.out / 'gpu_preflight.json',dict(passed=True,loss=losses[-1],trained_weights_saved=False,inherited_tensors_unchanged=True,trainable_coefficients=1408,provided_match_used=False))
            print('Receiver-context GPU preflight passed; updated weights discarded.',flush=True)
            return
        row = dict(epoch=epoch,loss=float(np.mean(losses)),seconds=time.monotonic()-started,inherited_tensors_unchanged=True)
        if epoch == 1 or epoch % config['validation_interval_epochs'] == 0 or epoch == config['epochs']:
            scores,weak_scores = evaluate(model,state,strong_dev,weak_dev,device)
            result = selection_result(scores,weak_scores,baseline)
            row['validation'] = result
            if result['eligible'] and result['objective'] > best['objective']+1e-10:
                best = dict(epoch=epoch,fallback=False,**result)
                save_torch(args.out / 'selected.pt',snapshot(epoch,best))
        history.append(row);scheduler.step()
        save_torch(args.out / 'latest.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            history=history,best=best,epoch=epoch,fingerprint=fingerprint,selected_sha256=sha256(args.out / 'selected.pt'),
            rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None))
        dump(args.out / 'history.json',history)
        print(json.dumps(row),flush=True)
    for path,digest in {**plan['sources'],**plan['caches']}.items():
        assert sha256(ROOT / path) == digest,path
    dump(args.out / 'selection.json',dict(complete=True,selected=dict(kind='uncertain_observation_loss',strength=1.,penalty=8.,
        source_epoch=best['epoch'],**best),plan=plan,selected_model_sha256=sha256(args.out / 'selected.pt'),
        parent_sha256=origin['parent_sha256'],parent_protocol_sha256=origin['parent_protocol_sha256'],initial_reproduced=True,
        provided_match_used=False,public_labels_used=False,unexpected_competition_access=denied,
        independent_test=False,competition_target_achieved=False))


if __name__ == '__main__':
    main()
