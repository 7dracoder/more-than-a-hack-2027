"""Train an external expert with a fixed known-missing-ball curriculum."""
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
from possession.visibility_curriculum import choose_crop,weak_visibility_weights
from possession.observation_gate import mix_probabilities,observation_loss
from possession.uncertain_decode import decode_labels
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump,score_parts,torch_setup
from scripts.research.spatial_memory.train_real_broadcast_memory import allowed_inputs,load_data,VIEWS
from scripts.research.spatial_memory.train_partial_had_memory import selection_result
from scripts.research.spatial_memory.audit_broadcast_transfer import weak_counts,summarize_counts


def extension_inputs(study,split):
    result = [study / 'extension_plan.json',study / 'extension_preparation.json']
    for match in split['weak_train_matches']:
        for half in (1,2):
            path = study / 'weak_extension' / f'{match}_{half}_detected.npy'
            result += [path,path.with_suffix('.json')]
    return result


def full_weak(raw,extra):
    if raw.ndim != 2 or extra.shape != (len(raw),296) or raw.shape[1] != 792:
        raise ValueError('Weak prefix, added geometry and memory clocks differ')
    return np.concatenate([raw[:,:664],extra,raw[:,664:]],axis=1)


def added_normalization(strong,weak,full_names,weights):
    from possession.external import swap_features
    grouped = [[h['x'][:,664:960] for h in strong if h['view'] == view] for view in VIEWS]
    grouped.append([h['extra'] for h in weak])
    mean,second = np.zeros(296),np.zeros(296)
    for weight,matrices in zip(weights,grouped):
        n,total,square = 0,np.zeros(296),np.zeros(296)
        for x in matrices:
            for first in range(0,len(x),16384):
                block = x[first:first+16384].astype(np.float64)
                n += len(block);total += block.sum(0);square += np.square(block).sum(0)
        if n == 0:
            raise ValueError('Empty normalization group')
        mean += weight*total/n;second += weight*square/n
    mapping = swap_features(np.arange(len(full_names)),full_names).astype(int)[664:960]-664
    assert sorted(mapping.tolist()) == list(range(296))
    mean = .5*(mean+mean[mapping]);second = .5*(second+second[mapping])
    return mean.astype(np.float32),np.sqrt(np.maximum(second-mean**2,1e-4)).astype(np.float32)


def evaluate(model,state,strong,weak,device):
    from possession.temporal import symmetric_probability
    strong_scores = {}
    for view in VIEWS:
        truth,predicted = [],[]
        for h in strong:
            if h['view'] != view:
                continue
            q = symmetric_probability(model,h['x'],state['names'],state['mean'],state['scale'],device)
            p = mix_probabilities(h['parent'],q,h['x'][:,960:],h['memory_names'],1.)
            gate = np.repeat(observation_loss(h['x'][:,960:],h['memory_names']),2)[:len(p)]
            truth.append(h['truth']);predicted.append(decode_labels(p,gate,8.,*state['decoder_costs']))
        strong_scores[view] = score_parts(truth,predicted)
    rows = {'all_known':[],'ball_missing':[]}
    for h in weak:
        q = symmetric_probability(model,h['x'],state['names'],state['mean'],state['scale'],device)
        p = mix_probabilities(h['parent'],q,h['x'][:,960:],h['memory_names'],1.)
        gate = np.repeat(observation_loss(h['x'][:,960:],h['memory_names']),2)[:len(p)]
        labels = decode_labels(p,gate,8.,*state['decoder_costs'])
        for scope,mask in [('all_known',np.ones(len(h['ticks']),bool)),('ball_missing',~h['seen'])]:
            rows[scope].append(weak_counts(p[h['ticks']],labels[h['ticks']],h['truth'],mask))
    return strong_scores,{scope:summarize_counts(values) for scope,values in rows.items()}


def batch(rng,focus_rng,config,strong,weak,state,mapping):
    from possession.temporal import normalized
    nweak = round(config['batch_size']*config['weak_batch_fraction'])
    ns = config['batch_size']-nweak
    grouped = [[h for h in strong if h['view'] == view] for view in VIEWS]
    bx,by,bw = [],[],[]
    for i in range(config['batch_size']):
        if i < ns:
            native = config['strong_native_view_probability']
            group = int(rng.choice(3,p=[native,(1-native)/2,(1-native)/2]))
            h = grouped[group][int(rng.integers(len(grouped[group])))]
        else:
            h = weak[int(rng.integers(len(weak)))]
        if i < ns:
            first = int(rng.integers(0,len(h['x'])-config['sequence_length']+1))
        else:
            first,_ = choose_crop(rng,focus_rng,len(h['x']),config['sequence_length'],
                h['missing_centers'],config['weak_missing_crop_probability'])
        last = first+config['sequence_length']
        raw = h['x'][first:last]
        if i >= ns:
            raw = full_weak(raw,h['extra'][first:last])
        y = h['y'][first:last].copy()
        if rng.random() < .5:
            raw = raw[:,mapping];y = np.where(y==0,1,np.where(y==1,0,y))
        x = normalized(raw,state['mean'],state['scale'])
        weight = h['w'][first:last]
        if i >= ns:
            weight = weak_visibility_weights(weight,raw[:,0],config['weak_missing_loss_multiplier'])
        bx.append(x.T);by.append(y);bw.append(weight)
    return np.stack(bx),np.stack(by),np.stack(bw),ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--extension-study',type=Path,default=ROOT / 'data/research/full-geometry-adaptation-20261114')
    parser.add_argument('--weak-study',type=Path,default=ROOT / 'data/research/real-broadcast-adaptation-20261114')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    config_path = ROOT / 'configs/missing_ball_curriculum_adaptation.json'
    config = json.loads(config_path.read_text())
    split = json.loads((ROOT / config['split_config']).read_text())
    assert not config['provided_match_used'] and not config['public_labels_used'] and not config['independent_test_matches']
    assert config['feature_count'] == 1088 and config['decoder_penalty'] == 8. and config['dead_offset'] == 0
    assert sum(config['normalization_group_weights']) == 1
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out / 'selection.json').exists():
        raise ValueError('Preserve completed training')
    extensions = extension_inputs(args.extension_study,split)
    denied = restrict_external(args.out,allowed_inputs(split,args.weak_study)+extensions)
    preparation = json.loads((args.extension_study / 'extension_preparation.json').read_text())
    assert preparation['complete'] and not preparation['provided_match_used'] and not preparation['public_labels_used']
    assert not preparation['validation_matches_used'] and not preparation['unexpected_competition_access']
    for path,digest in preparation['plan']['sources'].items():
        assert sha256(ROOT / path) == digest,path
    assert preparation['plan']['config'] == json.loads((ROOT / config['feature_preparation_config']).read_text()) and preparation['plan']['split'] == split
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
    full_names = preparation['plan']['full_names']
    assert full_names[:664]+full_names[960:] == initial['names']
    for h in weak:
        path = args.extension_study / 'weak_extension' / f"{h['match']}_{h['half']}_detected.npy"
        receipt = json.loads(path.with_suffix('.json').read_text())
        assert receipt['fingerprint']['plan'] == preparation['plan'] and receipt['sha256'] == sha256(path)
        assert receipt['names'] == full_names[664:960]
        h['extra'] = np.load(path,mmap_mode='r')
        assert h['extra'].shape == (len(h['x']),296)
    inventory = []
    for h in weak:
        seen = h['x'][:,0]
        assert np.isin(seen,[0.,1.]).all() and np.array_equal(h['w']>0,h['y']>=0)
        h['missing_centers'] = np.flatnonzero((h['y']>=0)&(seen<.5))
        inventory.append(dict(match=h['match'],half=h['half'],frames=len(seen),
            known=int((h['y']>=0).sum()),known_missing=len(h['missing_centers']),
            known_visible=int(((h['y']>=0)&(seen>=.5)).sum()),
            center_sha256=hashlib.sha256(h['missing_centers'].astype('<i8').tobytes()).hexdigest()))
    dump(args.out / 'training_visibility_inventory.json',dict(rows=inventory,training_only=True,
        provided_match_used=False,public_labels_used=False))
    mean,scale = added_normalization(strong,weak,full_names,config['normalization_group_weights'])
    state = extend_state(initial,full_names,mean,scale)
    np.savez(args.out / 'added_normalization.npz',mean=mean,scale=scale)
    sources = ['scripts/research/spatial_memory/train_missing_ball_curriculum.py','possession/feature_extension.py','possession/feature_permutation.py','possession/visibility_curriculum.py',
        'scripts/research/spatial_memory/train_real_broadcast_memory.py',
        'scripts/research/spatial_memory/train_partial_had_memory.py',
        'scripts/research/spatial_memory/external_guard.py','scripts/research/spatial_memory/audit_broadcast_transfer.py',
        'possession/temporal.py','possession/boundary.py','possession/external.py',
        'possession/observation_gate.py','possession/uncertain_decode.py','possession/weak_supervision.py','scripts/run_local.py']
    plan = dict(config=config,split=split,config_sha256=sha256(config_path),
        training_visibility_inventory_sha256=sha256(args.out / 'training_visibility_inventory.json'),
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
    focus_rng = np.random.default_rng(config['focus_seed'])
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
    network_config.update(full_geometry_adaptation=config,training_phase='missing_ball_curriculum_strong_HAD_weak_HA',
        epochs=config['epochs'],feature_count=1088,train_matches=split['strong_train_matches']+split['weak_train_matches'])
    state['config'] = network_config
    dump(args.out / 'fixed_protocol.json',dict(config=network_config,fingerprint=fingerprint,preparation=plan,baseline=best))

    def snapshot(epoch,result):
        return dict(model=model.state_dict(),names=state['names'],mean=state['mean'],scale=state['scale'],
            config=network_config,decoder_costs=state['decoder_costs'],epoch=epoch,validation=result,
            fingerprint=fingerprint,warm_start_sha256=plan['initial_model_sha256'],warm_start_epoch=initial['epoch'])

    optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config['epochs'],eta_min=config['learning_rate']/10.)
    mapping = team_swap_indices(full_names)
    history,start = [],1
    if (args.out / 'latest.pt').exists():
        latest = torch.load(args.out / 'latest.pt',map_location='cpu',weights_only=False)
        assert latest['fingerprint'] == fingerprint and latest['selected_sha256'] == sha256(args.out / 'selected.pt')
        model.load_state_dict(latest['model']);optimizer.load_state_dict(latest['optimizer']);scheduler.load_state_dict(latest['scheduler'])
        history,best,start = latest['history'],latest['best'],latest['epoch']+1
        rng.bit_generator.state = latest['rng'];torch.set_rng_state(latest['torch_rng'])
        focus_rng.bit_generator.state = latest['focus_rng']
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(latest['cuda_rng'])
    elif not args.smoke:
        save_torch(args.out / 'selected.pt',snapshot(0,best))
    for epoch in range(start,config['epochs']+1):
        started = time.monotonic();model.train();losses = []
        for _ in range(1 if args.smoke else config['batches_per_epoch']):
            bx,by,bw,ns = batch(rng,focus_rng,config,strong,weak,state,mapping)
            tx,ty,tw = [torch.as_tensor(a,device=device) for a in (bx,by,bw)]
            optimizer.zero_grad(set_to_none=True)
            logits,boundary = model.forward_with_boundary(tx)
            strong_loss = (frame_loss(logits[:ns],ty[:ns])*tw[:ns]).sum()/tw[:ns].sum().clamp_min(1.)
            weak_loss = conditional_ha_loss(logits[ns:],ty[ns:],tw[ns:])
            loss = strong_loss+config['weak_loss_weight']*weak_loss
            loss += config['strong_boundary_auxiliary_weight']*boundary_loss(boundary[:ns],ty[:ns],model.positive_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite full geometry loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();losses.append(float(loss.detach()))
        if args.smoke:
            dump(args.out / 'gpu_preflight.json',dict(passed=True,loss=losses[-1],trained_weights_saved=False,provided_match_used=False))
            print('Full geometry GPU preflight passed; updated weights discarded.',flush=True)
            return
        row = dict(epoch=epoch,loss=float(np.mean(losses)),seconds=time.monotonic()-started)
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
            rng=rng.bit_generator.state,focus_rng=focus_rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None))
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
