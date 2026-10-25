"""Train a trajectory encoder on audited public weak labels, with fixed final-epoch selection."""
import argparse,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from possession.data import sha256
from possession.player_trajectory_model import PlayerTrajectoryCorrector
from possession.temporal import training_weights,frame_loss
from possession.boundary import boundary_loss


def save(path,value):path.write_text(json.dumps(value,indent=2))


def read_bank(matches,audit,names=None,weight_power=.5,mmap_root=None,carrier_audit=None):
    bank=[];records=[]
    for view in ('native','detected'):
        for match in matches:
            path=ROOT/('data/external/skillcorner-trajectories-'+view)/(match+'.npz')
            metadata=json.loads(path.with_suffix('.json').read_text())
            digest=sha256(path)
            if audit[(match,view)]!=digest or metadata['sha256']!=digest:raise ValueError('Audited weak cache changed')
            if any(sha256(ROOT/p)!=s for p,s in metadata['producers'].items()):raise ValueError('Weak producer changed')
            carrier_arrays=None
            if carrier_audit is not None:
                carrier_path=ROOT/('data/external/skillcorner-carrier-'+view)/(match+'.npz')
                carrier_meta=json.loads(carrier_path.with_suffix('.json').read_text())
                if sha256(carrier_path)!=carrier_audit[(match,view)] or carrier_meta['feature_sha256']!=digest or carrier_meta['source']!=metadata['source']:
                    raise ValueError('Audited carrier targets changed')
                if any(sha256(ROOT/p)!=s for p,s in carrier_meta['producers'].items()):raise ValueError('Carrier producer changed')
                with np.load(carrier_path,allow_pickle=False) as data:carrier_arrays=[data[f'carrier{i}'] for i in range(2)]
            with np.load(path,allow_pickle=False) as data:
                current=data['names'].tolist()
                if names is None:names=current
                if current!=names:raise ValueError('Weak trajectory schemas differ')
                for i in range(2):
                    if mmap_root is None:x=data[f'x{i}']
                    else:
                        from possession.mapped_features import mapped_member
                        x=mapped_member(path,f'x{i}.npy',mmap_root,digest)
                    labels=data[f'y{i}'].copy()
                    if x.shape[1]!=180 or not np.isfinite(x).all() or set(np.unique(labels))-set('HAX'):raise ValueError('Invalid weak features/labels')
                    # No useful player-contact signal is available if the labelled
                    # team's central trajectories are all unobserved.
                    home=x[:,[17,47,77]].max(1)>.5;away=x[:,[107,137,167]].max(1)>.5
                    labels[((labels=='H')&~home)|((labels=='A')&~away)]='X'
                    y=np.full(len(labels),-100,np.int64);y[labels=='H']=0;y[labels=='A']=1
                    weights=training_weights(labels,weight_power)
                    carrier=np.full(len(labels),-100,np.int64) if carrier_arrays is None else carrier_arrays[i].copy()
                    if carrier.shape!=y.shape:raise ValueError('Carrier target timeline differs')
                    carrier[y<0]=-100
                    bank.append((x,y,weights,carrier))
                    records.append(dict(match=match,view=view,half=i,frames=len(x),supervised=int((y>=0).sum()),
                        carrier_supervised=int((carrier>=0).sum()),sha256=digest))
    return bank,names,records


def validation(model,bank,device,radius):
    model.eval();total=correct=0;loss_sum=0.
    with torch.inference_mode():
        for x,y,_,_ in bank:
            for first in range(0,len(x),8192):
                last=min(len(x),first+8192);left=max(0,first-radius);right=min(len(x),last+radius)
                batch=np.concatenate([x[left:right].T,np.zeros((4,right-left),np.float32)])[None]
                logits=model(torch.as_tensor(batch,device=device))[:,:,first-left:last-left]
                labels=torch.as_tensor(y[first:last][None],device=device);known=labels>=0
                losses=frame_loss(logits,labels)
                total+=int(known.sum());correct+=int(((logits.argmax(1)==labels)&known).sum());loss_sum+=float(losses.sum())
    if not total:raise ValueError('Weak validation has no known targets')
    return dict(weak_provider_HA_cross_entropy=loss_sum/total,weak_provider_HA_accuracy=correct/total,frames=total,
                comparable_to_development_spell_F1=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=ROOT/'configs/skillcorner_encoder_pretrain.json');parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text());out=args.out
    if out.exists():raise ValueError('Preserve existing pretraining evidence; choose a new run')
    if config['label_kind']!='provider_HA_weak' or config['selection_rule']!='fixed_final_epoch' or not config['physical_units']:raise ValueError('Unexpected weak-training protocol')
    train=config['train_matches'];valid=config['validation_matches']
    if not train or not valid or set(train)&set(valid) or any(not match.isdigit() for match in train+valid):raise ValueError('Weak match split must be numeric and disjoint')
    audit_path=ROOT/'data/local-v08/skillcorner_pretraining_cache_audit.json';receipt=json.loads(audit_path.read_text())
    if receipt['audit_source_sha256']!=sha256(ROOT/'scripts/audit_skillcorner_pretraining.py'):raise ValueError('Weak audit source changed')
    audit={(r['match'],r['view']):r['sha256'] for r in receipt['records']}
    if any(not r['raw_frame_targets_exact'] or not r['no_inferred_D'] for r in receipt['records']):raise ValueError('Weak-source audit incomplete')
    if set(train+valid)!={match for match,view in audit}:raise ValueError('Weak cohort differs from audited public source')
    carrier_weight=config.get('carrier_loss_weight',0.);carrier_audit=None;carrier_receipt=None
    if not math.isfinite(carrier_weight) or carrier_weight<0:raise ValueError('Invalid carrier-loss weight')
    if carrier_weight:
        if config['player_pooling']!='attention':raise ValueError('Carrier supervision needs the attention model')
        carrier_path=ROOT/'data/local-v08/skillcorner_carrier_cache_audit.json'
        deadline=time.monotonic()+3600
        while not carrier_path.exists():
            if time.monotonic()>deadline:raise TimeoutError('Carrier-target preparation/audit did not finish')
            time.sleep(2)
        carrier_receipt=json.loads(carrier_path.read_text())
        if carrier_receipt['audit_source_sha256']!=sha256(ROOT/'scripts/audit_carrier_supervision.py'):raise ValueError('Carrier audit source changed')
        carrier_audit={(r['match'],r['view']):r['sha256'] for r in carrier_receipt['records']}
        if set(carrier_audit)!=set(audit) or any(not r['all_targets_have_finite_candidates'] for r in carrier_receipt['records']):raise ValueError('Carrier audit is incomplete')
    torch.set_num_threads(2);torch.manual_seed(config['seed']);torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('Local CUDA is required')
    device=torch.device('cuda');out.mkdir(parents=True)
    sources=['scripts/pretrain_player_encoder.py','possession/player_trajectory_model.py','possession/temporal.py','possession/boundary.py','possession/features_v25.py','possession/skillcorner.py','scripts/prepare_skillcorner_pretraining.py','scripts/audit_skillcorner_pretraining.py']
    if config.get('mmap_features'):sources.append('possession/mapped_features.py')
    if carrier_weight:sources+=['possession/carrier_loss.py','possession/carrier_supervision.py','scripts/prepare_carrier_supervision.py','scripts/audit_carrier_supervision.py']
    record=dict(config=config,sources={p:sha256(ROOT/p) for p in sources},weak_cache_audit_sha256=sha256(audit_path),
        external_caches={match+'-'+view:digest for (match,view),digest in audit.items()},label_kind='provider_HA_weak')
    if carrier_weight:
        record['carrier_cache_audit_sha256']=sha256(carrier_path)
        record['carrier_caches']={match+'-'+view:digest for (match,view),digest in carrier_audit.items()}
    record['fingerprint']=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest();save(out/'fixed_protocol.json',record)
    subprocess.run([sys.executable,'scripts/snapshot_local_run.py','--run',str(out)],cwd=ROOT,check=True)
    mmap_root=ROOT/'data/external/mapped-features' if config.get('mmap_features') else None
    training,names,train_receipt=read_bank(train,audit,weight_power=config['run_weight_power'],mmap_root=mmap_root,carrier_audit=carrier_audit)
    held_out,_,valid_receipt=read_bank(valid,audit,names,config['run_weight_power'],mmap_root,carrier_audit)
    save(out/'weak_supervision.json',dict(training=train_receipt,validation=valid_receipt,targets_read_for_optimization=train,development_spell_score=None))
    model=PlayerTrajectoryCorrector(184,0,4,config['residual_channels'],config['player_channels'],config['residual_dilations'],config['residual_dropout'],config['player_pooling']).to(device)
    with torch.no_grad():model.project.weight[:,-4:].zero_()
    attention_capture={};hook=None
    if carrier_weight:
        from possession.carrier_loss import carrier_attention_loss
        ranks=config.get('carrier_rank_weights',[1.,1.,1.])
        if len(ranks)!=3 or not all(math.isfinite(v) and v>0 for v in ranks):raise ValueError('Carrier rank weights must be three positive finite values')
        rank_weights=torch.as_tensor(ranks,dtype=torch.float32,device=device)
        hook=model.attention.register_forward_hook(lambda module,args,output:attention_capture.update(scores=output))
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config['epochs'],eta_min=1e-5)
    rng=np.random.default_rng(config['seed']);mapping=np.array([names.index('away_'+n[5:] if n.startswith('home_') else 'home_'+n[5:]) for n in names])
    batch_size,length=config['batch_size'],config['sequence_length'];steps=max(16,math.ceil(sum(len(row[0]) for row in training)/(batch_size*length)));history=[]
    for epoch in range(1,config['epochs']+1):
        model.train();losses=[];carrier_losses=[];started=time.monotonic()
        for _ in range(steps):
            xs=[];ys=[];ws=[];cs=[]
            for _ in range(batch_size):
                x,y,w,c=training[int(rng.integers(len(training)))];first=int(rng.integers(max(1,len(x)-length+1)))
                crop=x[first:first+length];labels=y[first:first+length].copy()
                if rng.random()<.5:
                    crop=crop[:,mapping];labels=np.where(labels==0,1,np.where(labels==1,0,labels))
                xs.append(np.concatenate([crop.T,np.zeros((4,len(crop)),np.float32)]));ys.append(labels);ws.append(w[first:first+length]);cs.append(c[first:first+length])
            bx=torch.as_tensor(np.stack(xs),device=device);by=torch.as_tensor(np.stack(ys),device=device);bw=torch.as_tensor(np.stack(ws),device=device)
            optimizer.zero_grad(set_to_none=True);state,edge=model.forward_with_boundary(bx)
            loss=(frame_loss(state,by)*bw).sum()/bw.sum().clamp_min(1.)
            loss=loss+config['boundary_auxiliary_weight']*boundary_loss(edge,by,config['boundary_positive_weight'])
            if carrier_weight:
                scores=attention_capture['scores'].reshape(batch_size,2,3,length)
                carrier_loss=carrier_attention_loss(scores,bx,by,torch.as_tensor(np.stack(cs),device=device),rank_weights)
                loss=loss+carrier_weight*carrier_loss;carrier_losses.append(float(carrier_loss.detach()))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite weak pretraining loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
        entry=dict(epoch=epoch,loss=float(np.mean(losses)),seconds=round(time.monotonic()-started,2),learning_rate=optimizer.param_groups[0]['lr'])
        if carrier_losses:entry['weak_carrier_loss']=float(np.mean(carrier_losses))
        if epoch==1 or epoch%5==0 or epoch==config['epochs']:entry['weak_validation']=validation(model,held_out,device,sum(config['residual_dilations']))
        scheduler.step();history.append(entry);save(out/'history.json',history)
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=config,epoch=epoch,history=history,fingerprint=record['fingerprint'],names=names),out/'latest.pt')
        print(json.dumps(entry),flush=True)
    # Selection is the predeclared final epoch, regardless of weak validation.
    if hook is not None:hook.remove()
    final=dict(model=model.state_dict(),config=config,epoch=config['epochs'],fingerprint=record['fingerprint'],names=names)
    torch.save(final,out/'selected.pt');save(out/'selection.json',dict(epoch=config['epochs'],selection_rule='fixed_final_epoch',checkpoint_sha256=sha256(out/'selected.pt'),weak_validation=history[-1]['weak_validation'],development_spell_score=None,official_competition_score=None))
    save(out/'environment.json',dict(torch=torch.__version__,numpy=np.__version__,device=torch.cuda.get_device_name(),dtype='float32'))


if __name__=='__main__':main()
