"""Initialize a new external-only experiment from a verified frozen parent."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from possession.data import sha256
from scripts.snapshot_local_run import verify_source_archive


def load_warm_start(model,names,mean,scale,config,record,root,device):
    wrapper=hasattr(model,'warm_start_model')
    model=getattr(model,'warm_start_model',model)
    directory=Path(root)/config['warm_start_run']
    checkpoint=directory/'selected.pt';digest=sha256(checkpoint)
    selection=json.loads((directory/'selection.json').read_text())
    fixed=json.loads((directory/'fixed_protocol.json').read_text())
    if digest!=selection['checkpoint_sha256'] or digest!=record['warm_start_checkpoint_sha256']:
        raise ValueError('Warm-start checkpoint differs from frozen provenance')
    verify_source_archive(directory,fixed)
    state=torch.load(checkpoint,map_location=device,weights_only=False)
    if state['fingerprint']!=fixed['fingerprint'] or state['config']!=fixed['config'] or 'model' not in state:
        raise ValueError('Warm start requires a completed raw neural checkpoint with matching protocol')
    parent=state['config'];training=parent.get('train_matches',[])
    if config.get('teacher_model_config') is not None and config['teacher_model_config']!=parent:
        raise ValueError('Declared teacher configuration differs from verified parent')
    if not training or 'possession/external.py' not in fixed.get('sources',{}) or any('BAR' in str(m).upper() or 'ATH' in str(m).upper() for m in training):
        raise ValueError('Warm start must use external-only supervision')
    if set(training)&set(config['validation_matches']):
        raise ValueError('Warm parent trained on a currently held-out match')
    if not set(training)<=set(config['train_matches']):
        raise ValueError('Warm parent supervision exceeds the declared training cohort')
    if not set(parent.get('encoder_pretrain_training_matches',[]))<=set(config.get('encoder_pretrain_training_matches',[])):
        raise ValueError('Warm parent weak supervision exceeds the declared cohort')
    old_count=len(state['names'])
    expanded=bool(config.get('allow_feature_prefix_expansion') and len(names)>old_count)
    if list(state['names'])!=list(names[:old_count] if expanded else names):
        raise ValueError('Warm-start feature schema or normalization differs')
    native=record.get('prefix_parent_caches',{}) if expanded else record['external_caches']
    augmented=record.get('prefix_parent_augmentation_caches',{}) if expanded else record.get('augmentation_caches',{})
    for match in training:
        if fixed.get('external_caches',{}).get(match)!=native.get(match):
            raise ValueError('Warm parent and current training caches differ')
        if fixed.get('augmentation_caches',{}).get(match)!=augmented.get(match):
            raise ValueError('Warm parent and current augmentation caches differ')
    weights=state['model']
    sliced=False
    if expanded:
        if not np.allclose(state['mean'],mean[:old_count],rtol=1e-6,atol=1e-7) or not np.allclose(state['scale'],scale[:old_count],rtol=1e-6,atol=1e-7):
            raise ValueError('Warm-start prefix normalization differs')
        input_count=getattr(model,'input_feature_count',None)
        if input_count is None:input_count=getattr(getattr(model,'project',None),'in_channels',None)
        sliced=bool(wrapper and config.get('teacher_feature_count')==old_count and input_count==old_count)
        if not sliced:
            old=weights['project.weight'];new=model.project.weight
            if old.ndim!=new.ndim or old.shape[0]!=new.shape[0] or old.shape[1]!=old_count or old.shape[2:]!=new.shape[2:] or new.shape[1]!=len(names):
                raise ValueError('Warm-start feature expansion needs a compatible input projection')
        # Reuse the parent's training-only prefix statistics, avoiding reduction
        # roundoff from a wider matrix. New feature coefficients start at zero.
        mean[:old_count]=state['mean'];scale[:old_count]=state['scale']
        if not sliced:
            weights=weights.copy()
            with torch.no_grad():
                extended=torch.zeros_like(new);extended[:,:old_count]=old
            weights['project.weight']=extended
    elif not np.array_equal(state['mean'],mean) or not np.array_equal(state['scale'],scale):
        raise ValueError('Warm-start feature schema or normalization differs')
    widen_factor=config.get('warm_start_widen_factor',1)
    if widen_factor!=1:
        from possession.widening import widen_boundary_hybrid
        weights=widen_boundary_hybrid(weights,model,widen_factor)
    model.load_state_dict(weights)
    shift=0.
    if getattr(model,'has_boundary_head',False):
        old=parent.get('boundary_positive_weight',10.);new=model.positive_weight
        if not all(math.isfinite(v) and v>0 for v in (old,new)):
            raise ValueError('Warm-start boundary calibration weights must be positive')
        shift=math.log(new/old)
        if shift:
            with torch.no_grad():model.boundary[-1].bias.add_(shift)
    if sha256(checkpoint)!=digest:
        raise ValueError('Warm-start parent changed while loading')
    return dict(parent_run=str(directory),parent_checkpoint_sha256=digest,parent_epoch=state.get('epoch'),
                parent_training_matches=training,parent_selection_matches=parent.get('validation_matches',[]),
                parent_weak_training_matches=parent.get('encoder_pretrain_training_matches',[]),
                normalization_exact=not expanded,normalization_prefix_exact=True,
                feature_prefix_extension=len(names)-old_count if expanded else 0,new_input_weights_zero_initialized=expanded and not sliced,
                teacher_excludes_new_features=sliced,
                hidden_channel_widen_factor=widen_factor,
                boundary_bias_shift=shift,optimizer_reset=True,
                validation_targets_used_for_gradient=False)
