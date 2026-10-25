"""Auxiliary change-point supervision on external training labels only."""
import math
import torch
from torch import nn
from possession.temporal import TemporalBlock
from possession.recurrent import inference_gru


class BoundaryHybrid(nn.Module):
    def __init__(self, inputs, channels=128, dropout=.2, positive_weight=10., recurrent_layers=1,recurrent_channels=None):
        super().__init__()
        self.full_sequence, self.receptive_radius = True, 0
        self.has_boundary_head = True
        self.positive_weight = float(positive_weight)
        self.project = nn.Conv1d(inputs, channels, 1)
        self.local = nn.Sequential(*(TemporalBlock(channels, d, dropout) for d in (1,2,4)))
        recurrent_channels=channels if recurrent_channels is None else recurrent_channels
        if not isinstance(recurrent_channels,int) or recurrent_channels<=0:
            raise ValueError('Recurrent channel count must be a positive integer')
        self.recurrent = nn.GRU(channels, recurrent_channels, num_layers=recurrent_layers, batch_first=True,
                                bidirectional=True, dropout=dropout if recurrent_layers>1 else 0.)
        embedding_channels=channels+2*recurrent_channels
        self.head = nn.Sequential(nn.Conv1d(embedding_channels, channels, 1), nn.GELU(), nn.Dropout(dropout), nn.Conv1d(channels, 3, 1))
        self.boundary = nn.Sequential(nn.Conv1d(embedding_channels, channels//2, 1), nn.GELU(), nn.Conv1d(channels//2, 1, 1))

    def representation(self, x):
        local = self.local(self.project(x))
        context, _ = inference_gru(self.recurrent,local.transpose(1, 2))
        return torch.cat([local, context.transpose(1, 2)], dim=1)

    def forward_with_boundary(self, x):
        embedding = self.representation(x)
        return self.head(embedding), self.boundary(embedding)[:,0]

    def forward(self, x):
        return self.forward_with_boundary(x)[0]

    def calibrated_boundary(self, logits):
        # Weighted BCE estimates shifted odds; remove the known training weight.
        return torch.sigmoid(logits-math.log(self.positive_weight))


class HeightBoundaryHybrid(BoundaryHybrid):
    """Infer possession from XY; height is an auxiliary training target only."""
    def __init__(self, inputs, channels=128, dropout=.2, positive_weight=20.):
        super().__init__(inputs, channels, dropout, positive_weight)
        self.has_height_head = True
        self.height = nn.Sequential(nn.Conv1d(channels*3, channels//2, 1), nn.GELU(), nn.Conv1d(channels//2, 1, 1))

    def forward_with_auxiliary(self, x):
        embedding = self.representation(x)
        return self.head(embedding), self.boundary(embedding)[:,0], self.height(embedding)[:,0]

    def forward_with_boundary(self, x):
        return self.forward_with_auxiliary(x)[:2]


class BoundaryTCN(nn.Module):
    """Long dilated context with exactly bounded state and transition heads."""
    def __init__(self,inputs,channels=128,dilations=(1,2,4,8,16,32,64,128,256),dropout=.2,positive_weight=20.):
        super().__init__()
        self.full_sequence=False
        self.receptive_radius=sum(dilations)
        self.has_boundary_head=True
        self.positive_weight=float(positive_weight)
        self.project=nn.Conv1d(inputs,channels,1)
        self.blocks=nn.Sequential(*(TemporalBlock(channels,d,dropout) for d in dilations))
        self.head=nn.Sequential(nn.Conv1d(channels,channels,1),nn.GELU(),nn.Dropout(dropout),nn.Conv1d(channels,3,1))
        self.boundary=nn.Sequential(nn.Conv1d(channels,channels//2,1),nn.GELU(),nn.Conv1d(channels//2,1,1))

    def forward_with_boundary(self,x):
        embedding=self.blocks(self.project(x))
        return self.head(embedding),self.boundary(embedding)[:,0]

    def forward(self,x):
        return self.forward_with_boundary(x)[0]

    def calibrated_boundary(self,logits):
        return torch.sigmoid(logits-math.log(self.positive_weight))


def boundary_targets(labels, soft_radius=0):
    if not isinstance(soft_radius,int) or soft_radius<0:
        raise ValueError('Boundary soft radius must be a nonnegative integer')
    valid=(labels[:,1:]>=0)&(labels[:,:-1]>=0)
    hard=((labels[:,1:]!=labels[:,:-1])&valid).float()
    targets=hard.clone()
    for offset in range(1,min(soft_radius+1,hard.shape[1])):
        weight=1.-offset/(soft_radius+1.)
        targets[:,offset:]=torch.maximum(targets[:,offset:],hard[:,:-offset]*weight)
        targets[:,:-offset]=torch.maximum(targets[:,:-offset],hard[:,offset:]*weight)
    return targets,valid


def boundary_loss(logits, labels, positive_weight=10., soft_radius=0):
    if logits.shape != labels.shape:
        raise ValueError('Boundary logits and targets must have matching shape')
    targets,valid=boundary_targets(labels,soft_radius)
    targets=targets.to(logits.dtype)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:,1:], targets, reduction='none',
                pos_weight=torch.as_tensor(positive_weight, device=logits.device))
    return (loss*valid).sum()/valid.sum().clamp_min(1)
