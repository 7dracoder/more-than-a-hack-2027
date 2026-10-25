"""Compact temporal network for external-only possession supervision."""
import numpy as np
import torch
from torch import nn
from possession.recurrent import inference_gru


class TemporalBlock(nn.Module):
    def __init__(self, channels, dilation, dropout=.1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.conv(x)
        residual = self.norm(residual.transpose(1, 2)).transpose(1, 2)
        return x + self.dropout(torch.nn.functional.gelu(residual))


class PossessionTCN(nn.Module):
    def __init__(self, inputs, channels=96, dilations=(1, 2, 4, 8, 16, 32), dropout=.1):
        super().__init__()
        self.receptive_radius = sum(dilations)
        self.full_sequence = False
        self.project = nn.Conv1d(inputs, channels, 1)
        self.blocks = nn.Sequential(*(TemporalBlock(channels, d, dropout) for d in dilations))
        self.head = nn.Conv1d(channels, 3, 1)

    def forward(self, x):
        return self.head(self.blocks(self.project(x)))


class PossessionBiGRU(nn.Module):
    """Bidirectional temporal memory; inference processes a whole half at once."""
    def __init__(self, inputs, channels=96, dropout=.2):
        super().__init__()
        self.full_sequence = True
        self.receptive_radius = 0
        self.project = nn.Linear(inputs, channels)
        self.norm = nn.LayerNorm(channels)
        self.recurrent = nn.GRU(channels, channels, num_layers=2, batch_first=True,
                                bidirectional=True, dropout=dropout)
        self.head = nn.Linear(channels * 2, 3)

    def forward(self, x):
        x = torch.nn.functional.gelu(self.norm(self.project(x.transpose(1, 2))))
        x, _ = inference_gru(self.recurrent,x)
        return self.head(x).transpose(1, 2)


class ProbabilityEnsemble(nn.Module):
    def __init__(self, members, weights):
        super().__init__()
        if len(members) != len(weights) or min(weights) < 0 or abs(sum(weights)-1.) > 1e-6:
            raise ValueError('Ensemble weights must be nonnegative and sum to one')
        self.members = nn.ModuleList(members)
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        self.full_sequence = any(m.full_sequence for m in members)
        self.receptive_radius = max(m.receptive_radius for m in members)

    def forward(self, x):
        probability = sum(w * torch.softmax(m(x), dim=1) for m,w in zip(self.members,self.weights))
        return torch.log(probability.clamp_min(1e-9))


class PossessionHybrid(nn.Module):
    """Preserve local contact evidence alongside recurrent half-level context."""
    def __init__(self, inputs, channels=128, dropout=.2):
        super().__init__()
        self.full_sequence, self.receptive_radius = True, 0
        self.project = nn.Conv1d(inputs, channels, 1)
        self.local = nn.Sequential(*(TemporalBlock(channels, d, dropout) for d in (1, 2, 4)))
        self.recurrent = nn.GRU(channels, channels, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Conv1d(channels*3, channels, 1), nn.GELU(), nn.Dropout(dropout), nn.Conv1d(channels, 3, 1))

    def forward(self, x):
        local = self.local(self.project(x))
        context, _ = inference_gru(self.recurrent,local.transpose(1, 2))
        return self.head(torch.cat([local, context.transpose(1, 2)], dim=1))


class ClassifierAdapter(nn.Module):
    """Expose a frozen sklearn classifier to the common symmetric inference path."""
    def __init__(self, fitted):
        super().__init__()
        self.fitted = fitted
        self.full_sequence, self.receptive_radius = True, 0

    def forward(self, x):
        from threadpoolctl import threadpool_limits
        values = x.transpose(1, 2).detach().cpu().numpy()
        with threadpool_limits(limits=2):
            probability = self.fitted.predict_proba(values.reshape(-1, values.shape[-1]))
        if self.fitted.classes_.tolist() != [0, 1, 2]:
            raise ValueError('Classifier must emit H/A/D in that order')
        logits = np.log(np.maximum(probability, 1e-9)).reshape(values.shape[0], values.shape[1], 3)
        return torch.as_tensor(logits.transpose(0, 2, 1), device=x.device, dtype=x.dtype)


def make_model(inputs, config):
    architecture = config.get('architecture', 'tcn')
    if architecture == 'tcn':
        return PossessionTCN(inputs, config['hidden_channels'], config['dilations'], config.get('dropout', .1))
    if architecture == 'bigru':
        return PossessionBiGRU(inputs, config['hidden_channels'], config.get('dropout', .2))
    if architecture == 'hybrid':
        return PossessionHybrid(inputs, config['hidden_channels'], config.get('dropout', .2))
    if architecture == 'boundary_hybrid':
        from possession.boundary import BoundaryHybrid
        return BoundaryHybrid(inputs, config['hidden_channels'], config.get('dropout', .2), config.get('boundary_positive_weight', 10.), config.get('recurrent_layers',1),config.get('recurrent_channels'))
    if architecture == 'residual_boundary':
        from possession.residual import ResidualBoundaryHybrid
        if config.get('residual_kind')=='player_trajectories':
            from possession.player_trajectory_model import trajectory_mask_names
            if set(config.get('unit_feature_names',[]))!=set(trajectory_mask_names()):
                raise ValueError('Player trajectories require their explicit unscaled observation masks')
        return ResidualBoundaryHybrid(inputs,config['hidden_channels'],config.get('dropout',.2),
            config.get('boundary_positive_weight',20.),config.get('residual_channels',64),
            config.get('residual_dilations',[1,2,4,8,16,32]),config.get('residual_dropout',.1),
            config.get('residual_teacher_context',False),config.get('teacher_feature_count'),
            config.get('residual_kind','tcn'),config.get('player_channels',32),config.get('teacher_model_config'),config.get('player_pooling','mean_max'),config.get('trajectory_feature_start'))
    if architecture == 'height_hybrid':
        from possession.boundary import HeightBoundaryHybrid
        return HeightBoundaryHybrid(inputs, config['hidden_channels'], config.get('dropout', .2), config.get('boundary_positive_weight', 20.))
    if architecture == 'boundary_tcn':
        from possession.boundary import BoundaryTCN
        return BoundaryTCN(inputs,config['hidden_channels'],config['dilations'],config.get('dropout',.2),config.get('boundary_positive_weight',20.))
    if architecture == 'local_attention':
        from possession.local_attention import PossessionLocalAttention
        return PossessionLocalAttention(inputs, config['hidden_channels'], config.get('attention_heads',4),
                    config.get('attention_layers',4),config.get('attention_radius',16),config.get('dropout',.2),
                    config.get('boundary_positive_weight',10.))
    if architecture == 'recurrent_attention':
        from possession.local_attention import RecurrentAttention
        return RecurrentAttention(inputs,config['hidden_channels'],config.get('attention_heads',4),
                    config.get('attention_layers',2),config.get('attention_radius',8),config.get('dropout',.2),
                    config.get('boundary_positive_weight',10.))
    if architecture == 'ensemble':
        return ProbabilityEnsemble([make_model(inputs,c) for c in config['ensemble_members']],config['ensemble_weights'])
    raise ValueError('Unknown architecture: ' + architecture)


def symmetric_normalization(matrices, names):
    from possession.external import swap_features
    count = sum(len(x) for x in matrices)
    mean = sum(x.astype(np.float64).sum(0) for x in matrices) / count
    square = sum((x.astype(np.float64) ** 2).sum(0) for x in matrices) / count
    mean = (mean + swap_features(mean, names)) / 2
    square = (square + swap_features(square, names)) / 2
    scale = np.sqrt(np.maximum(square - mean ** 2, 1e-4))
    return mean.astype(np.float32), scale.astype(np.float32)


def normalized(values, mean, scale):
    return np.clip((values - mean) / scale, -10, 10).astype(np.float32, copy=False)


def training_weights(labels, power=.5):
    """Upweight short state runs without making single-frame noise dominate."""
    if not 0 <= power <= 1:
        raise ValueError('Run weighting power must lie in [0,1]')
    cuts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    weights = np.zeros(len(labels), np.float32)
    for a, b in zip(cuts[:-1], cuts[1:]):
        if labels[a] in 'HAD':
            weights[a:b] = 1 / max(b - a, 5) ** power
    if (weights > 0).any():
        weights /= weights[weights > 0].mean()
    return weights


def frame_loss(logits, labels, focal_gamma=0.):
    """Focal weighting must remain differentiable at zero CE/ignored targets."""
    if focal_gamma < 0:
        raise ValueError('Focal gamma must be nonnegative')
    loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none', ignore_index=-100)
    if focal_gamma:
        # A fractional power at exactly zero has an infinite derivative. Clamping
        # before the power keeps confident and ignored frames' gradients finite.
        focus = (-torch.expm1(-loss)).clamp_min(1e-6).pow(focal_gamma)
        loss = loss*focus
    return loss


def make_weight_average(model, config):
    decay=config.get('ema_decay',0.)
    if not decay:
        return None
    if not np.isfinite(decay) or not 0<decay<1:
        raise ValueError('EMA decay must lie strictly between zero and one')
    from torch.optim.swa_utils import AveragedModel,get_ema_multi_avg_fn
    average=AveragedModel(model,device=next(model.parameters()).device,multi_avg_fn=get_ema_multi_avg_fn(decay))
    average.eval()
    return average


def symmetric_probability(model, matrix, names, mean, scale, device, chunk=2048,teacher_cache=None):
    """Overlap chunks by the exact receptive radius to avoid inference seams."""
    from possession.external import swap_features
    model.eval()
    cached_outputs=None
    if teacher_cache is not None:
        if not model.full_sequence or not hasattr(model,'forward_with_cached_teacher'):
            raise ValueError('Teacher reuse requires a whole-sequence residual model')
        cached_outputs=teacher_cache.outputs(model,matrix,names,mean,scale,device)
    if model.full_sequence:
        # A recurrent model has unbounded context. Never silently reset its
        # hidden state at arbitrary chunks or claim finite-overlap equivalence.
        chunk, radius = len(matrix), 0
    else:
        radius = model.receptive_radius
    result = []
    with torch.inference_mode():
        for a in range(0, len(matrix), chunk):
            b = min(len(matrix), a + chunk)
            left, right = max(0, a - radius), min(len(matrix), b + radius)
            if getattr(model,'raw_inputs',False):
                if not np.allclose(mean,0.) or not np.allclose(scale,1.):
                    raise ValueError('Raw-feature ensemble requires identity outer normalization')
                original = matrix[left:right].astype(np.float32)
                swapped = swap_features(original,names)
            else:
                original = normalized(matrix[left:right], mean, scale)
                swapped = normalized(swap_features(matrix[left:right], names), mean, scale)
            batch = torch.as_tensor(np.stack([original.T, swapped.T]), device=device)
            change = None
            if getattr(model, 'has_boundary_head', False):
                logits, change_logits = (model.forward_with_boundary(batch) if cached_outputs is None else
                                        model.forward_with_cached_teacher(batch,*cached_outputs))
                change = model.calibrated_boundary(change_logits).float().cpu().numpy().mean(0)
            else:
                logits = model(batch)
            p = torch.softmax(logits, dim=1).float().cpu().numpy()
            p = (p[0] + p[1, [1, 0, 2]]) / 2
            part = p[:, a-left:b-left].T
            if change is not None:
                part = np.column_stack([part, change[a-left:b-left]])
            result.append(part)
    return np.concatenate(result).astype(np.float32)


def viterbi_labels(probability, team_cost=6., dead_cost=4., change_weight=0., change_bias=0.):
    """MAP path only; no posterior work when exporting or selecting checkpoints."""
    probability = np.asarray(probability, dtype=float)
    if probability.ndim != 2 or probability.shape[1] not in (3,4) or not len(probability):
        raise ValueError('Expected nonempty time x H/A/D probabilities')
    change = probability[:,3] if probability.shape[1] == 4 else None
    probability = probability[:,:3]
    if not np.isfinite(probability).all() or (probability < 0).any() or not np.allclose(probability.sum(1), 1., atol=1e-5):
        raise ValueError('Invalid probabilities')
    if not np.isfinite([team_cost, dead_cost, change_weight, change_bias]).all() or min(team_cost, dead_cost, change_weight) < 0:
        raise ValueError('Transition costs must be nonnegative')
    if change is not None and (not np.isfinite(change).all() or (change < 0).any() or (change > 1).any()):
        raise ValueError('Invalid change probabilities')
    if change_weight and change is None:
        raise ValueError('Learned transition weighting requires change probabilities')
    odds = None
    if change is not None and change_weight:
        bounded = change.clip(1e-6,1-1e-6)
        odds = (np.log(bounded)-np.log1p(-bounded)+change_bias)*change_weight
    emission = np.log(np.maximum(probability, 1e-9))
    transition = np.array([[0., -team_cost, -dead_cost], [-team_cost, 0., -dead_cost], [-dead_cost, -dead_cost, 0.]])
    from possession.fast_decode import path as decode_path
    path=decode_path(emission,transition,np.zeros(len(emission),np.float64) if odds is None else odds)
    return np.array(['H', 'A', 'D'])[path]
