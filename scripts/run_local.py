"""Train on external matches, select on one external match, infer BAR-ATH locally.

The BAR-ATH public label file is never opened. --test requires a frozen selected
checkpoint and writes a permanent receipt: repeated adaptive test runs are refused.
"""
import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import joblib
import numpy as np
import pandas as pd
import torch
from evaluation import possession_metrics as pm
from possession.champion import fit_classifier, classifier_probability
from possession.data import sha256, slot_teams
from possession.external import build_features, swap_features
from possession.temporal import (make_model, normalized, symmetric_normalization,
                                  symmetric_probability, training_weights, viterbi_labels, frame_loss, make_weight_average)


def save_json(path, value):
    part = path.with_suffix(path.suffix + '.partial')
    part.write_text(json.dumps(value, indent=2), encoding='utf-8')
    part.replace(path)


def save_torch(path, value):
    part = path.with_suffix(path.suffix + '.partial')
    torch.save(value, part)
    part.replace(path)


def validate_splits(config):
    groups = [config[k] for k in ('train_matches', 'validation_matches', 'test_matches')]
    flat = sum(groups, [])
    development_only = config.get('development_only', False)
    if not groups[0] or not groups[1] or (not groups[2] and not development_only) or (groups[2] and development_only) or len(flat) != len(set(flat)) or any('BAR' in m.upper() or 'ATH' in m.upper() for m in flat):
        raise ValueError('Need disjoint nonempty external train/validation/test match ids')


def training_table(halves):
    """Keep half boundaries ungraded when pooling frame classifier examples."""
    features, targets = [], []
    for half in halves:
        if len(half['x']) != len(half['y']):
            raise ValueError('Training feature/target length mismatch')
        features.extend([half['x'], np.zeros((1, half['x'].shape[1]), np.float32)])
        targets.extend([half['y'], np.array(['X'])])
    return np.concatenate(features), np.concatenate(targets)


def load_halves(matches, directory=None,mmap_root=None):
    result, names = [], None
    for match in matches:
        path = (ROOT / 'data/external/features-v07' if directory is None else Path(directory)) / (match + '.npz')
        metadata = json.loads(path.with_suffix('.json').read_text())
        digest=sha256(path)
        if metadata.get('sha256') != digest or metadata['provenance']['source_sha256'] != sha256(ROOT / 'possession/external.py'):
            raise ValueError('Stale/corrupted external cache: rerun scripts/prepare_external.py for ' + match)
        augmentation = metadata['provenance'].get('augmentation_source_sha256')
        if augmentation and augmentation != sha256(ROOT / 'scripts/prepare_external_gaps.py'):
            raise ValueError('Stale missing-ball augmentation cache')
        for field, source in [('feature_source_sha256', 'possession/features_v08.py'),
                              ('preparation_source_sha256', 'scripts/prepare_external_v08.py'),
                              ('gap_mask_source_sha256', 'scripts/prepare_external_gaps.py'),
                              ('v09_feature_source_sha256', 'possession/features_v09.py'),
                              ('v09_preparation_source_sha256', 'scripts/prepare_external_v09.py'),
                              ('v10_preparation_source_sha256', 'scripts/prepare_external_v10.py'),
                              ('v13_feature_source_sha256', 'possession/features_v13.py'),
                              ('v13_preparation_source_sha256', 'scripts/prepare_external_v13.py'),
                              ('v19_feature_source_sha256', 'possession/features_v19.py'),
                              ('v19_preparation_source_sha256', 'scripts/prepare_external_v19.py'),
                              ('v25_feature_source_sha256', 'possession/features_v25.py'),
                              ('v25_preparation_source_sha256', 'scripts/prepare_external_v25.py')]:
            if field in metadata['provenance'] and metadata['provenance'][field] != sha256(ROOT/source):
                raise ValueError('Stale v08 feature/preparation cache: '+match)
        with np.load(path, allow_pickle=False) as stored:
            current = stored['names'].tolist()
            if names is not None and current != names:
                raise ValueError('Feature schemas disagree')
            names = current
            for i in range(2):
                if mmap_root is None:matrix=stored[f'x{i}']
                else:
                    from possession.mapped_features import mapped_member
                    matrix=mapped_member(path,f'x{i}.npy',mmap_root,digest)
                result.append(dict(match=match, half=i, x=matrix, y=stored[f'y{i}'], truth=stored[f'truth{i}']))
    return result, names


def score_probabilities(halves, probabilities, costs, step):
    # Separate halves/matches by X to prevent possessions merging across fixtures.
    truth, predictions = [], []
    for half, probability in zip(halves, probabilities):
        small = viterbi_labels(probability, *costs)
        prediction = np.repeat(small, step)[:len(half['truth'])]
        if len(prediction) != len(half['truth']):
            raise ValueError('Prediction length disagrees')
        truth.extend([half['truth'], np.array(['X'])])
        predictions.extend([prediction, np.array(['X'])])
    result = pm.evaluate(pm.label_codes(np.concatenate(truth)), pm.label_codes(np.concatenate(predictions)))
    return {k: result[k] for k in ('spell_f1_mean', 'macro_f1', 'alive_dead_f1', 'n_gt_spells', 'n_pred_spells')}


def reference_probability(baseline, matrix, names, reference_names):
    """Reuse a frozen feature-bank reference through an explicitly named subset."""
    if reference_names != names:
        if len(set(names)) != len(names) or any(n not in names for n in reference_names):
            raise ValueError('Frozen reference feature bank is not a subset of current features')
        matrix = matrix[:,[names.index(n) for n in reference_names]]
    return classifier_probability(baseline, matrix, reference_names, swap=swap_features)


def reference_score(config, baseline, halves, names, reference_names):
    step=config['sample_step']
    reference_step=config.get('baseline_sample_step',step)
    if reference_step<step or reference_step%step:
        raise ValueError('Reference time grid must be a whole multiple of the input grid')
    stride=reference_step//step
    probability=[reference_probability(baseline,h['x'][::stride],names,reference_names) for h in halves]
    return score_probabilities(halves,probability,(6.,4.),reference_step)


def provenance(config, matches):
    feature_dir = ROOT/config.get('features_dir', 'data/external/features-v07')
    inputs = {m: sha256(feature_dir / (m + '.npz')) for m in matches}
    sources = {p: sha256(ROOT / p) for p in ('possession/external.py', 'possession/temporal.py', 'scripts/run_local.py', 'evaluation/possession_metrics.py')}
    sources['possession/recurrent.py']=sha256(ROOT/'possession/recurrent.py')
    sources['possession/fast_decode.py']=sha256(ROOT/'possession/fast_decode.py')
    sources['possession/learning_rate.py']=sha256(ROOT/'possession/learning_rate.py')
    if config.get('mask_crop_context'):
        sources['possession/training_context.py']=sha256(ROOT/'possession/training_context.py')
    if config.get('feature_version') in ('v08','v09','v13','v19','v25'):
        for source in ('possession/features_v08.py', 'scripts/prepare_external_v08.py'):
            sources[source] = sha256(ROOT/source)
    if config.get('feature_version') in ('v09','v13','v19','v25'):
        for source in ('possession/features_v09.py','scripts/prepare_external_v09.py'):
            sources[source] = sha256(ROOT/source)
    if config.get('full_rate'):
        sources['scripts/prepare_external_v10.py']=sha256(ROOT/'scripts/prepare_external_v10.py')
    if config.get('feature_version') in ('v13','v19','v25'):
        for source in ('possession/features_v13.py','scripts/prepare_external_v13.py'):
            sources[source]=sha256(ROOT/source)
    if config.get('feature_version')=='v19':
        for source in ('possession/features_v19.py','scripts/prepare_external_v19.py'):
            sources[source]=sha256(ROOT/source)
    if config.get('feature_version')=='v25':
        for source in ('possession/features_v25.py','scripts/prepare_external_v25.py'):
            sources[source]=sha256(ROOT/source)
    if config.get('spell_iou_loss_weight',0.):
        sources['possession/spell_loss.py']=sha256(ROOT/'possession/spell_loss.py')
    if config.get('structured_loss_weight',0.):
        sources['possession/structured_loss.py']=sha256(ROOT/'possession/structured_loss.py')
    if config.get('paired_symmetric_training',False):
        sources['possession/paired_training.py']=sha256(ROOT/'possession/paired_training.py')
    if config.get('short_spell_sampling_probability',0.):
        sources['possession/sampling.py']=sha256(ROOT/'possession/sampling.py')
    if config.get('native_view_probability') is not None:
        sources['possession/view_sampling.py']=sha256(ROOT/'possession/view_sampling.py')
    if config.get('unit_feature_names') or config.get('trajectory_physical_units'):
        sources['possession/feature_normalization.py']=sha256(ROOT/'possession/feature_normalization.py')
    if config.get('cache_validation_teacher'):
        sources['possession/teacher_cache.py']=sha256(ROOT/'possession/teacher_cache.py')
    if config.get('mmap_features'):
        sources['possession/mapped_features.py']=sha256(ROOT/'possession/mapped_features.py')
    if config.get('cache_inference_features'):
        sources['possession/inference_cache.py']=sha256(ROOT/'possession/inference_cache.py')
    if config.get('warm_start_run'):
        for source in ('possession/warm_start.py','scripts/snapshot_local_run.py'):
            sources[source]=sha256(ROOT/source)
    if config.get('encoder_pretrain_run'):
        sources['possession/pretrained_encoder.py']=sha256(ROOT/'possession/pretrained_encoder.py')
    if config.get('warm_start_widen_factor',1)!=1:
        sources['possession/widening.py']=sha256(ROOT/'possession/widening.py')
    if config.get('height_targets_dir'):
        for source in ('possession/height.py','scripts/prepare_height_targets.py'):
            sources[source]=sha256(ROOT/source)
    if config.get('architecture') in ('boundary_hybrid','boundary_tcn','height_hybrid','local_attention','recurrent_attention','residual_boundary'):
        sources['possession/boundary.py'] = sha256(ROOT/'possession/boundary.py')
    if config.get('architecture')=='residual_boundary':
        sources['possession/residual.py']=sha256(ROOT/'possession/residual.py')
        if config.get('residual_kind')=='player_trajectories':
            sources['possession/player_trajectory_model.py']=sha256(ROOT/'possession/player_trajectory_model.py')
    if config.get('architecture') in ('local_attention','recurrent_attention'):
        sources['possession/local_attention.py'] = sha256(ROOT/'possession/local_attention.py')
    if config.get('architecture') == 'feature_ensemble':
        sources['possession/ensemble.py'] = sha256(ROOT/'possession/ensemble.py')
        sources['possession/boundary.py'] = sha256(ROOT/'possession/boundary.py')
    record = dict(config=config, external_caches=inputs, sources=sources)
    if config.get('feature_version') in ('v19','v25'):
        parent_field=config['feature_version']+'_base_cache_sha256'
        record['prefix_parent_caches']={m:json.loads((feature_dir/(m+'.json')).read_text())['provenance'][parent_field] for m in matches}
    if config.get('height_targets_dir'):
        record['height_targets']={m:sha256(ROOT/config['height_targets_dir']/(m+'.npz')) for m in config['train_matches']}
    if config.get('frozen_baseline_dir'):
        record['frozen_baseline_sha256'] = sha256(ROOT/config['frozen_baseline_dir']/'baseline.joblib')
    if config.get('augmentation_features_dir'):
        record['augmentation_caches'] = {m: sha256(Path(config['augmentation_features_dir']) / (m+'.npz')) for m in config['train_matches']}
        if config.get('feature_version') in ('v19','v25'):
            parent_field=config['feature_version']+'_base_cache_sha256'
            record['prefix_parent_augmentation_caches']={m:json.loads((ROOT/config['augmentation_features_dir']/(m+'.json')).read_text())['provenance'][parent_field] for m in config['train_matches']}
    if config.get('warm_start_run'):
        parent=ROOT/config['warm_start_run']
        record['warm_start_checkpoint_sha256']=sha256(parent/'selected.pt')
        record['warm_start_protocol_fingerprint']=json.loads((parent/'fixed_protocol.json').read_text())['fingerprint']
    if config.get('encoder_pretrain_run'):
        pretrain=ROOT/config['encoder_pretrain_run']
        record['encoder_pretrain_checkpoint_sha256']=sha256(pretrain/'selected.pt')
        record['encoder_pretrain_protocol_fingerprint']=json.loads((pretrain/'fixed_protocol.json').read_text())['fingerprint']
    record['fingerprint'] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return record


def train(config, out, resume, device):
    if config.get('architecture')=='residual_boundary' and not config.get('warm_start_run'):
        raise ValueError('Residual training requires a verified external warm parent')
    structured_weight=config.get('structured_loss_weight',0.)
    short_probability=config.get('short_spell_sampling_probability',0.)
    if not math.isfinite(structured_weight) or structured_weight<0:
        raise ValueError('Structured loss weight must be finite and nonnegative')
    if not math.isfinite(short_probability) or not 0<=short_probability<=1:
        raise ValueError('Short-spell crop probability must lie between zero and one')
    if structured_weight and config.get('boundary_positive_weight',10.)!=1.:
        raise ValueError('Structured transition potentials require boundary_positive_weight=1')
    structured_costs=config.get('structured_train_costs',[0.,2.])
    if structured_weight and (len(structured_costs)!=2 or not all(math.isfinite(v) and v>=0 for v in structured_costs)):
        raise ValueError('Structured training needs two finite nonnegative transition costs')
    short_seconds=config.get('short_spell_max_seconds',1.)
    if short_probability and (not math.isfinite(short_seconds) or short_seconds<=0):
        raise ValueError('Short-spell duration bound must be finite and positive')
    directory = ROOT/config.get('features_dir', 'data/external/features-v07')
    mmap_root=ROOT/'data/external/mapped-features' if config.get('mmap_features') else None
    training, names = load_halves(config['train_matches'], directory,mmap_root)
    validation, val_names = load_halves(config['validation_matches'], directory,mmap_root)
    if names != val_names:
        raise ValueError('Feature schema mismatch')
    record = provenance(config, config['train_matches'] + config['validation_matches'])
    fixed = out / 'fixed_protocol.json'
    if fixed.exists() and json.loads(fixed.read_text())['fingerprint'] != record['fingerprint']:
        raise ValueError('Source/config/data changed: use a new output directory')
    save_json(fixed, record)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    from possession.fast_decode import BACKEND
    from importlib.metadata import version
    save_json(out / 'environment.json', dict(python=sys.version, executable=sys.executable,
              platform=platform.platform(), numpy=np.__version__, pandas=pd.__version__,
              torch=torch.__version__, cuda_build=torch.version.cuda, device=str(device),
              gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
              deterministic=True,decoder_backend=BACKEND,
              numba=version('numba') if BACKEND=='numba' else None))
    print('Training device: ' + str(device), flush=True)
    # Same small symmetric classifier family as v031, retrained only on external matches.
    # Arrays are already sampled every five frames; avoid applying the stride twice.
    baseline_path = out / 'baseline.joblib'
    if config.get('frozen_baseline_dir'):
        import shutil
        reference = ROOT/config['frozen_baseline_dir']
        if not baseline_path.exists():
            for filename in ('baseline.joblib','baseline_training.json','baseline_validation.json'):
                shutil.copyfile(reference/filename, out/filename)
        if sha256(baseline_path) != record['frozen_baseline_sha256']:
            raise ValueError('Frozen baseline bytes changed')
        save_json(out/'baseline_reference.json', dict(directory=str(reference),
                  sha256=record['frozen_baseline_sha256'], training_matches=config['baseline_train_matches'],
                  note='Frozen external four-match reference; neural training membership is recorded separately.'))
    if not ((resume or config.get('frozen_baseline_dir')) and baseline_path.exists()):
        x, y = training_table(training)
        baseline_config = dict(config, classifier_training_stride=config.get('baseline_training_stride', 1))
        baseline, audit = fit_classifier(x, names, y, y != 'X', baseline_config, swap=swap_features)
        audit.update(supervision='Official external DFL states only', train_matches=config['train_matches'],
                     feature_schema='Coordinates-only v07 feature bank; same bank as temporal model',
                     warning='Small external dataset; domain transfer to BAR-ATH remains unmeasured')
        joblib.dump(baseline, baseline_path)
        save_json(out / 'baseline_training.json', audit)
        del x,y
    baseline = joblib.load(baseline_path)
    reference_names = json.loads((out/'baseline_training.json').read_text())['feature_names']
    baseline_scores = reference_score(config,baseline,validation,names,reference_names)
    save_json(out / 'baseline_validation.json', baseline_scores)
    print('Baseline validation ' + json.dumps(baseline_scores), flush=True)

    native_half_count=len(training)
    if config.get('augmentation_features_dir'):
        augmented, augmented_names = load_halves(config['train_matches'], config['augmentation_features_dir'],mmap_root)
        if augmented_names != names:
            raise ValueError('Augmentation schema differs from native features')
        for native, view in zip(training, augmented):
            if native['match'] != view['match'] or not np.array_equal(native['y'], view['y']):
                raise ValueError('Synthetic missing-ball view altered training targets')
        training += augmented
        del augmented

    native_probability=config.get('native_view_probability')
    if native_probability is not None:
        from possession.view_sampling import validate_view_sampling,sample_training_view
        validate_view_sampling(len(training),native_half_count,native_probability)
        save_json(out/'view_sampling_audit.json',dict(native_probability=native_probability,
            native_halves=native_half_count,augmented_halves=len(training)-native_half_count,
            training_matches=config['train_matches'],normalization_uses_both_training_views=True,
            development_targets_used_for_gradient=False))

    if config.get('height_targets_dir'):
        from possession.height import attach_height_targets
        attach_height_targets(training,ROOT/config['height_targets_dir'],directory,config['sample_step'],ROOT)
    mean, scale = symmetric_normalization([h['x'] for h in training], names)
    if config.get('unit_feature_names'):
        from possession.feature_normalization import preserve_unit_masks
        mean,scale=preserve_unit_masks(mean,scale,names,config['unit_feature_names'])
        save_json(out/'normalization_policy.json',dict(unit_masks=config['unit_feature_names'],statistics_matches=config['train_matches']))
    if config.get('trajectory_physical_units'):
        from possession.feature_normalization import preserve_trajectory_units
        mean,scale=preserve_trajectory_units(mean,scale,names)
        save_json(out/'normalization_policy.json',dict(unit_masks=config.get('unit_feature_names',[]),
            physical_trajectory_suffix=180,statistics_matches=config['train_matches']))
    model = make_model(len(names), config).to(device)
    paired_training=config.get('paired_symmetric_training',False)
    if not isinstance(paired_training,bool):
        raise ValueError('Paired symmetric training must be a boolean')
    if paired_training:
        if not getattr(model,'has_boundary_head',False) or getattr(model,'has_height_head',False):
            raise ValueError('Paired training requires a state/boundary model without height auxiliary targets')
        from possession.paired_training import symmetric_outputs
    if structured_weight:
        if not getattr(model,'has_boundary_head',False):
            raise ValueError('Structured training needs a transition head')
        from possession.structured_loss import structured_state_loss
    if config.get('warm_start_run'):
        from possession.warm_start import load_warm_start
        save_json(out/'warm_start.json',load_warm_start(model,names,mean,scale,config,record,ROOT,device))
    if config.get('encoder_pretrain_run'):
        from possession.pretrained_encoder import initialize_encoder
        save_json(out/'encoder_initialization.json',initialize_encoder(model,names,config,record,ROOT,device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    from possession.learning_rate import make_scheduler
    scheduler=make_scheduler(optimizer,config)
    averaged=make_weight_average(model,config)
    rng = np.random.default_rng(config['seed'])
    best_score, start_epoch, history = -1., 1, []
    latest = out / 'latest.pt'
    if resume and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        if scheduler is not None:
            if state.get('lr_scheduler') is None:
                raise ValueError('Scheduled training resume needs its saved learning-rate state')
            scheduler.load_state_dict(state['lr_scheduler'])
        if averaged is not None:
            if state.get('ema_model') is None:
                raise ValueError('EMA-enabled resume needs its saved averaging state')
            averaged.load_state_dict(state['ema_model'])
        best_score, start_epoch, history = state['best_score'], state['epoch'] + 1, state['history']
        rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(state['torch_rng'].cpu())
        if device.type == 'cuda' and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    tensors = [];sampling_anchors=[];sampling_rows=[]
    if short_probability:
        from possession.sampling import short_spell_anchors,sample_crop_start
        short_frames=round(short_seconds*config['fps']/config['sample_step'])
        if short_frames<1:raise ValueError('Short-spell bound is below the sampled time resolution')
    for half in training:
        labels = pm.label_codes(half['y']).astype(np.int64)
        labels[labels == pm.X] = -100
        if short_probability:
            anchors=short_spell_anchors(labels,short_frames)
            sampling_anchors.append(anchors)
            sampling_rows.append(dict(tensor_index=len(tensors),match=half['match'],half=half['half'],frames=len(labels),anchors=len(anchors)))
        weights = training_weights(half['y'], config.get('run_weight_power', .5))
        if config.get('boundary_weight', 1.) != 1.:
            # Label-derived loss weights are computed on training data only.
            changes = np.flatnonzero((labels[1:] != labels[:-1]) & (labels[1:] >= 0) & (labels[:-1] >= 0))+1
            radius = round(config.get('boundary_seconds', .4)*config['fps']/config['sample_step'])
            near = np.zeros(len(labels), bool)
            for change in changes:
                near[max(0,change-radius):change+radius+1] = True
            weights[near] *= config['boundary_weight']
        # Release one raw half as its normalized replacement is created.
        raw_matrix=half.pop('x')
        if mmap_root is None:matrix=normalized(raw_matrix,mean,scale)
        else:
            from possession.mapped_features import mapped_normalized
            matrix=mapped_normalized(raw_matrix,mean,scale,mmap_root)
        tensors.append((matrix,labels,weights,half.get('height_target')))
        del raw_matrix,matrix
    # Only normalized arrays are needed after this point. Keeping both banks
    # doubles host memory and causes paging on the local 16 GB machine.
    del training,half
    if mmap_root is not None:
        save_json(out/'feature_mapping_receipt.json',dict(lossless_float32=True,training_maps=[str(t[0].filename) for t in tensors],
            feature_values_and_normalization_unchanged=True,validation_maps=[str(h['x'].filename) for h in validation]))
    if short_probability:
        save_json(out/'sampling_audit.json',dict(scope='Training targets only; no development targets used for crop sampling',
                  training_matches=config['train_matches'],probability=short_probability,max_seconds=short_seconds,halves=sampling_rows))
    mapping = torch.as_tensor([names.index('away_' + n[5:] if n.startswith('home_') else 'home_' + n[5:] if n.startswith('away_') else n) for n in names], device=device)
    length, batch_size = config['sequence_length'], config['batch_size']
    crop_context=0
    if config.get('mask_crop_context'):
        if model.full_sequence:
            raise ValueError('Crop context masking requires a model with bounded context')
        crop_context=model.receptive_radius
        if 2*crop_context>=length:
            raise ValueError('Training crop is shorter than its two context margins')
    steps = max(16, math.ceil(sum(len(t[0]) for t in tensors) / (length * batch_size)))
    validation_teacher_cache=[None]*len(validation)
    if config.get('cache_validation_teacher'):
        from possession.teacher_cache import build_teacher_cache
        validation_teacher_cache=[build_teacher_cache(model,h['x'],names,mean,scale,device) for h in validation]
        save_json(out/'teacher_cache_receipt.json',dict(scope='Ephemeral validation inference only; no training targets',
            entries=[dict(input_sha256=c.input_sha256,teacher_sha256=c.teacher_sha256) for c in validation_teacher_cache],
            fresh_audit_and_export_use_full_model=True))
    if config.get('warm_start_run') and not history:
        started=time.monotonic()
        probabilities=[symmetric_probability(model,h['x'],names,mean,scale,device,teacher_cache=cache) for h,cache in zip(validation,validation_teacher_cache)]
        candidates=[dict(weights_kind='warm_initial',costs=cost,score=score_probabilities(validation,probabilities,cost,config['sample_step'])) for cost in config['development_decoder_costs']]
        chosen=max(candidates,key=lambda item:item['score']['spell_f1_mean'])
        best_score=chosen['score']['spell_f1_mean']
        save_torch(out/'selected.pt',dict(model=model.state_dict(),mean=mean,scale=scale,names=names,config=config,
                   decoder_costs=chosen['costs'],validation=chosen['score'],epoch=0,weights_kind='warm_initial',fingerprint=record['fingerprint']))
        entry=dict(epoch=0,loss=None,seconds=round(time.monotonic()-started,2),validation=chosen['score'],
                   decoder_costs=chosen['costs'],weights_kind='warm_initial',decoder_candidates=candidates)
        history.append(entry);save_json(out/'history.json',history);print(json.dumps(entry),flush=True)
    for epoch in range(start_epoch, config['epochs'] + 1):
        started = time.monotonic(); model.train(); losses = []
        for _ in range(steps):
            batch_x, batch_y, batch_w, batch_height = [], [], [], []
            for _ in range(batch_size):
                tensor_index=(sample_training_view(rng,len(tensors),native_half_count,native_probability)
                              if native_probability is not None else int(rng.integers(len(tensors))))
                x, y, w, z = tensors[tensor_index]
                a=(sample_crop_start(rng,len(x),length,sampling_anchors[tensor_index],short_probability)
                   if short_probability else int(rng.integers(max(1,len(x)-length+1))))
                batch_x.append(x[a:a+length].T); batch_y.append(y[a:a+length]); batch_w.append(w[a:a+length])
                if z is not None:
                    batch_height.append(z[a:a+length])
            bx = torch.as_tensor(np.stack(batch_x), device=device)
            by = torch.as_tensor(np.stack(batch_y), device=device)
            bw = torch.as_tensor(np.stack(batch_w), device=device)
            augment = torch.as_tensor(rng.random(batch_size) < .5, device=device)
            bx[augment] = bx[augment][:, mapping]
            by[augment] = torch.where(by[augment] == 0, 1, torch.where(by[augment] == 1, 0, by[augment]))
            if crop_context:
                from possession.training_context import mask_crop_context
                mask_crop_context(by,bw,crop_context)
            optimizer.zero_grad(set_to_none=True)
            auxiliary = height_prediction = None
            if paired_training:
                both_state,both_boundary=model.forward_with_boundary(torch.cat([bx,bx[:,mapping]],dim=0))
                logits,auxiliary=symmetric_outputs(both_state[:batch_size],both_state[batch_size:],
                    both_boundary[:batch_size],both_boundary[batch_size:],model.positive_weight)
            elif getattr(model, 'has_height_head', False):
                if len(batch_height)!=batch_size:
                    raise ValueError('Height auxiliary model needs training targets for every crop')
                logits, auxiliary, height_prediction = model.forward_with_auxiliary(bx)
            elif getattr(model, 'has_boundary_head', False):
                logits, auxiliary = model.forward_with_boundary(bx)
            else:
                logits = model(bx)
            loss_values = frame_loss(logits, by, config.get('focal_gamma', 0.))
            loss = (loss_values * bw).sum() / bw.sum().clamp_min(1.)
            if height_prediction is not None:
                from possession.height import height_regression_loss
                loss=loss+config.get('height_auxiliary_weight',.1)*height_regression_loss(height_prediction,torch.as_tensor(np.stack(batch_height),device=device))
            if auxiliary is not None:
                from possession.boundary import boundary_loss
                soft_radius=round(config.get('boundary_soft_radius_seconds',0.)*config['fps']/config['sample_step'])
                loss = loss+config.get('boundary_auxiliary_weight', .3)*boundary_loss(auxiliary, by, model.positive_weight,soft_radius)
            if structured_weight:
                loss=loss+structured_weight*structured_state_loss(logits,auxiliary,by,*structured_costs)
            if config.get('spell_iou_loss_weight',0.):
                from possession.spell_loss import spell_overlap_loss
                context=max(1,round(config.get('spell_context_seconds',5.)*config['fps']/config['sample_step']))
                loss=loss+config['spell_iou_loss_weight']*spell_overlap_loss(logits,by,context)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss; preserve this run and diagnose before selection')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
            if averaged is not None:
                averaged.update_parameters(model)
            losses.append(float(loss.detach()))
        entry = dict(epoch=epoch, loss=float(np.mean(losses)), seconds=round(time.monotonic()-started, 2),
                     learning_rate=float(optimizer.param_groups[0]['lr']))
        if epoch == 1 or epoch % config['validation_interval_epochs'] == 0 or epoch == config['epochs']:
            variants=[('raw',model)]+([('ema',averaged.module)] if averaged is not None else [])
            candidates=[]
            for kind,candidate_model in variants:
                probabilities = [symmetric_probability(candidate_model, h['x'], names, mean, scale, device,teacher_cache=cache) for h,cache in zip(validation,validation_teacher_cache)]
                for cost in config['development_decoder_costs']:
                    candidates.append(dict(weights_kind=kind,costs=cost,score=score_probabilities(validation,probabilities,cost,config['sample_step'])))
            chosen=max(candidates,key=lambda item:item['score']['spell_f1_mean'])
            costs,score,kind=chosen['costs'],chosen['score'],chosen['weights_kind']
            entry.update(validation=score,decoder_costs=costs,weights_kind=kind,decoder_candidates=candidates)
            if score['spell_f1_mean'] > best_score:
                best_score = score['spell_f1_mean']
                candidate_model=model if kind=='raw' else averaged.module
                save_torch(out / 'selected.pt', dict(model=candidate_model.state_dict(), mean=mean, scale=scale, names=names,
                                config=config, decoder_costs=costs, validation=score,
                                epoch=epoch, weights_kind=kind, fingerprint=record['fingerprint']))
        history.append(entry)
        if scheduler is not None:
            scheduler.step()
        save_torch(latest, dict(model=model.state_dict(), optimizer=optimizer.state_dict(), best_score=best_score,
                        lr_scheduler=scheduler.state_dict() if scheduler is not None else None,
                        epoch=epoch, history=history, numpy_rng=rng.bit_generator.state,
                        ema_model=averaged.state_dict() if averaged is not None else None,
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None))
        save_json(out / 'history.json', history)
        print(json.dumps(entry), flush=True)
        if config.get('stop_when_target_met') and best_score>=config['target_spell_f1']:
            print('Fixed external development target reached; finishing with the selected checkpoint.',flush=True)
            break
    selected = torch.load(out / 'selected.pt', map_location='cpu', weights_only=False)
    save_json(out / 'selection.json', dict(epoch=selected['epoch'], validation=selected['validation'],
              weights_kind=selected.get('weights_kind','raw'),
              decoder_costs=selected['decoder_costs'], checkpoint_sha256=sha256(out / 'selected.pt'),
              baseline_sha256=sha256(baseline_path), feature_names=names,
              target_spell_f1=config.get('target_spell_f1'),
              target_met_on_development=(selected['validation']['spell_f1_mean']>=config['target_spell_f1']) if 'target_spell_f1' in config else None,
              test_opened=False, official_competition_score=None))


def model_from_state(state, device):
    if 'members' in state:
        from possession.ensemble import FeatureEnsemble
        models = [model_from_state(member,device) for member in state['members']]
        model = FeatureEnsemble(models,state['names'],state['members'],state['weights']).to(device)
    elif 'classifier' in state:
        from possession.temporal import ClassifierAdapter
        model = ClassifierAdapter(state['classifier']).to(device)
    else:
        model = make_model(len(state['names']), state['config']).to(device)
        model.load_state_dict(state['model'])
    model.eval()
    return model


def selected_model(out, device):
    state = torch.load(out / 'selected.pt', map_location=device, weights_only=False)
    model = model_from_state(state, device)
    fixed = json.loads((out / 'fixed_protocol.json').read_text())
    if state['fingerprint'] != fixed['fingerprint']:
        raise ValueError('Selected model provenance differs from frozen protocol')
    return model, state


def evaluate_test(config, out, device):
    if config.get('development_only'):
        raise ValueError('Development-only training has no independent test cohort; obtain new external matches.')
    receipt = out / 'test_receipt.json'
    if receipt.exists():
        raise ValueError('Test already opened. Preserve receipt; obtain new matches before adaptive retesting.')
    selection = json.loads((out / 'selection.json').read_text())
    fixed = json.loads((out / 'fixed_protocol.json').read_text())
    if fixed['config'] != config:
        raise ValueError('Evaluation configuration differs from frozen training protocol')
    if sha256(out / 'selected.pt') != selection['checkpoint_sha256'] or sha256(out / 'baseline.joblib') != selection['baseline_sha256']:
        raise ValueError('Selected artifacts changed')
    from possession.evaluation_guard import reserve_test_cohort
    reserve_test_cohort(ROOT / 'data/local-v07/test_registry.json',
                       {m: sha256(ROOT / config.get('features_dir', 'data/external/features-v07') / (m+'.npz')) for m in config['test_matches']},
                       selection['checkpoint_sha256'], out)
    selection['test_opened'] = True
    save_json(out / 'selection.json', selection)
    save_json(receipt, dict(status='opened', selected_sha256=selection['checkpoint_sha256'],
                           test_matches=config['test_matches'], warning='These matches may not be adaptively reused as independent test data.'))
    halves, names = load_halves(config['test_matches'], ROOT/config.get('features_dir', 'data/external/features-v07'))
    model, state = selected_model(out, device)
    if names != state['names']:
        raise ValueError('Test feature schema mismatch')
    baseline = joblib.load(out / 'baseline.joblib')
    reference_names = json.loads((out/'baseline_training.json').read_text())['feature_names']
    tp = [symmetric_probability(model, h['x'], names, state['mean'], state['scale'], device) for h in halves]
    bscore = reference_score(config,baseline,halves,names,reference_names)
    tscore = score_probabilities(halves, tp, state['decoder_costs'], config['sample_step'])
    gain = tscore['spell_f1_mean'] - bscore['spell_f1_mean']
    result = dict(baseline=bscore, temporal=tscore, absolute_spell_gain=gain,
                  target=config['target_absolute_spell_gain'], target_met=gain >= config['target_absolute_spell_gain'],
                  scope='Two unseen external DFL matches. Not comparable to historical BAR-ATH development scores.',
                  official_competition_score=None, selected_sha256=selection['checkpoint_sha256'])
    save_json(out / 'test_results.json', result)
    save_json(receipt, dict(status='completed', **result, test_matches=config['test_matches']))
    print(json.dumps(result, indent=2), flush=True)


def infer_competition(config, out, device):
    from games import GAME, GAMES, asset_paths, half_video_blocks
    paths = asset_paths(data_dir=str(ROOT / 'data'))
    input_sha256={k:sha256(paths[k]) for k in ('atd_meta','atd_raw')}
    n, pitch = GAMES[GAME]['n_video_frames'], GAMES[GAME]['pitch']
    teams = slot_teams(paths['atd_meta'])
    players = np.full((n, len(teams), 2), np.nan, np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    count = 0
    with open(paths['atd_raw']) as stream:
        for i, line in enumerate(stream):
            fields = line.strip().split(':')
            if len(fields) != 3 or int(fields[0]) != i+1 or i >= n:
                raise ValueError('Tactical frame alignment failed')
            row = np.fromstring(fields[1].replace(';', ','), sep=',')
            point = np.fromstring(fields[2], sep=',')
            if row.size != len(teams)*3 or point.size < 2:
                raise ValueError('Malformed tracking row')
            players[i] = row.reshape(-1, 3)[:, :2] * pitch
            ball[i] = point[:2] * pitch
            count += 1
    if count != n:
        raise ValueError('Incomplete tracking')
    if {k:sha256(paths[k]) for k in ('atd_meta','atd_raw')}!=input_sha256:
        raise ValueError('Tactical input changed while parsing')
    model, state = selected_model(out, device)
    if config != state['config']:
        raise ValueError('Inference configuration differs from selected model')
    labels = np.full(n, 'X', dtype='<U1')
    feature_builder = build_features
    if config.get('feature_version') == 'v08':
        from possession.features_v08 import build_features as feature_builder
    elif config.get('feature_version') == 'v09':
        from possession.features_v09 import build_features as feature_builder
    elif config.get('feature_version') == 'v13':
        from possession.features_v13 import build_features as feature_builder
    elif config.get('feature_version') == 'v19':
        from possession.features_v19 import build_features as feature_builder
    elif config.get('feature_version') == 'v25':
        from possession.features_v25 import build_features as feature_builder
    feature_receipts=[];feature_sources=None
    if config.get('cache_inference_features'):
        from possession.inference_cache import cached_block
        import scipy
        source_files=['possession/external.py','possession/data.py','scripts/run_local.py','games.py','possession/inference_cache.py']
        branches={'v08':['v08'],'v09':['v08','v09'],'v13':['v08','v09','v13'],'v19':['v08','v09','v13','v19'],'v25':['v08','v09','v13','v25']}
        source_files+=['possession/features_'+version+'.py' for version in branches.get(config.get('feature_version'),[])]
        feature_sources={p:sha256(ROOT/p) for p in source_files}
    for _, first, last in half_video_blocks(GAME):
        def generate():return feature_builder(players[first:last+1], teams, ball[first:last+1], pitch, step=config['sample_step'])
        if feature_sources is None:x,names=generate()
        else:
            key=dict(raw_inputs=input_sha256,sources=feature_sources,game=str(GAME),first=int(first),last=int(last),
                pitch=[float(v) for v in pitch],sample_step=config['sample_step'],feature_version=config.get('feature_version'),
                numpy=np.__version__,scipy=scipy.__version__)
            x,names,receipt=cached_block(ROOT/'data/inference-features',key,generate,(last-first)//config['sample_step']+1)
            feature_receipts.append(receipt)
        if names != state['names']:
            raise ValueError('Competition feature schema differs from training')
        p = symmetric_probability(model, x, names, state['mean'], state['scale'], device)
        labels[first:last+1] = np.repeat(viterbi_labels(p, *state['decoder_costs']), config['sample_step'])[:last-first+1]
    candidate = out / 'candidate'; candidate.mkdir(exist_ok=True)
    path = candidate / 'CoreOrigin.csv'
    frame = pd.DataFrame(dict(frame=np.arange(n), label=labels))
    failure = pm.validate_submission(frame, n)
    if failure:
        raise ValueError(failure)
    if {k:sha256(paths[k]) for k in ('atd_meta','atd_raw')}!=input_sha256:
        raise ValueError('Tactical input changed during inference')
    frame.to_csv(path, index=False)
    save_json(candidate / 'manifest.json', dict(rows=n, sha256=sha256(path), selected_sha256=sha256(out / 'selected.pt'),
              input_sha256=input_sha256,coordinate_feature_cache=feature_receipts or None,
              device=str(device), public_labels_read=False, training_matches=config['train_matches'],
              selection_matches=config['validation_matches'], official_competition_score=None))
    print('Validated local candidate: ' + str(path), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/local_v07.json')
    parser.add_argument('--out', default='data/local-v07/reproduction')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text()); validate_splits(config)
    if not (args.train or args.test or args.infer):
        parser.error('Select --train, --test and/or --infer')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use the project CUDA environment')
    device = torch.device(args.device); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.train:
        train(config, out, args.resume, device)
    if args.test:
        evaluate_test(config, out, device)
    if args.infer:
        infer_competition(config, out, device)


if __name__ == '__main__':
    main()
