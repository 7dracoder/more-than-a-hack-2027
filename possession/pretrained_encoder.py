"""Transfer only an audited weak trajectory representation into a new zero-head corrector."""
import json
from pathlib import Path
import torch
from possession.data import sha256
from scripts.snapshot_local_run import verify_source_archive


def initialize_encoder(model,names,config,record,root,device):
    directory=Path(root)/config['encoder_pretrain_run'];path=directory/'selected.pt'
    digest=sha256(path);selection=json.loads((directory/'selection.json').read_text());fixed=json.loads((directory/'fixed_protocol.json').read_text())
    if digest!=selection['checkpoint_sha256'] or digest!=record['encoder_pretrain_checkpoint_sha256']:
        raise ValueError('Weak encoder checkpoint differs from frozen provenance')
    verify_source_archive(directory,fixed)
    state=torch.load(path,map_location=device,weights_only=False);parent=state['config']
    if state['fingerprint']!=fixed['fingerprint'] or parent!=fixed['config'] or selection['selection_rule']!='fixed_final_epoch' or state['epoch']!=parent['epochs']:
        raise ValueError('Weak encoder must use its completed fixed-final-epoch protocol')
    training=parent['train_matches'];held=parent['validation_matches']
    if parent['label_kind']!='provider_HA_weak' or 'possession/skillcorner.py' not in fixed['sources'] or not training or set(training)&set(held) or any(not match.isdigit() for match in training+held):
        raise ValueError('Weak encoder must use separate public SkillCorner supervision')
    if config.get('encoder_pretrain_training_matches')!=training or set(training)&set(config['validation_matches']):
        raise ValueError('Additional weak supervision differs from the declared cohort')
    if not config.get('trajectory_physical_units') or config.get('residual_kind')!='player_trajectories' or not config.get('residual_teacher_context'):
        raise ValueError('Weak encoder transfer requires matching physical trajectory units and context')
    if list(names[-180:])!=list(state['names']) or model.corrector.teacher_features!=len(names)-180:
        raise ValueError('Weak encoder trajectory schema differs')
    for key in ('residual_channels','player_channels','residual_dilations','player_pooling'):
        if config.get(key)!=parent[key]:raise ValueError('Weak encoder architecture differs: '+key)
    if any(p.requires_grad for p in model.teacher.parameters()):raise ValueError('Weak encoder transfer must preserve a frozen teacher')
    for head in (model.corrector.head[-1],model.corrector.boundary[-1]):
        if int(head.weight.count_nonzero()) or int(head.bias.count_nonzero()):raise ValueError('New correction heads must start at zero')
    destination=model.corrector.state_dict();loaded=[]
    for key,value in state['model'].items():
        if not key.startswith(('encoder.','attention.','project.','blocks.')):continue
        if key not in destination or destination[key].shape!=value.shape:raise ValueError('Weak representation tensor shape differs')
        destination[key]=value;loaded.append(key)
    if not loaded or not any(key.startswith('encoder.') for key in loaded):raise ValueError('No trajectory encoder weights')
    model.corrector.load_state_dict(destination)
    # These four channels were zero throughout weak pretraining. Give them
    # zero initial coefficients rather than transferring untrained weights.
    with torch.no_grad():model.corrector.project.weight[:,-4:].zero_()
    if sha256(path)!=digest:raise ValueError('Weak checkpoint changed while loading')
    return dict(pretrain_run=str(directory),checkpoint_sha256=digest,weak_training_matches=training,
        additional_weak_targets=('provider carrier rank' if parent.get('carrier_loss_weight',0.) else None),
        weak_validation_matches=held,selection_rule='fixed_final_epoch',transferred=loaded,
        prediction_heads_transferred=False,initial_correction_heads_zero=True,teacher_unchanged=True,
        weak_labels_are_independent_ground_truth=False,validation_targets_used_for_gradient=False)
