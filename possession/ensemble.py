"""Frozen-model averaging with each member's own feature map and normalization."""
import numpy as np
import torch
from torch import nn


class FeatureMember(nn.Module):
    def __init__(self, model, outer_names, names, mean, scale, stride=1):
        super().__init__()
        if len(set(outer_names)) != len(outer_names) or any(n not in outer_names for n in names):
            raise ValueError('Ensemble member needs a unique, available feature subset')
        if len(names) != len(mean) or len(names) != len(scale) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (np.asarray(scale)<=0).any():
            raise ValueError('Invalid member normalization')
        self.model = model
        if not isinstance(stride,int) or stride<1:
            raise ValueError('Member sampling stride must be a positive integer')
        self.stride=stride
        self.register_buffer('indices',torch.tensor([outer_names.index(n) for n in names],dtype=torch.long))
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32)[None,:,None])
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32)[None,:,None])

    def forward(self, raw):
        length=raw.shape[-1]
        x = ((raw[:,self.indices,::self.stride]-self.mean)/self.scale).clamp(-10,10)
        if getattr(self.model,'has_boundary_head',False):
            logits,change = self.model.forward_with_boundary(x)
            probability,boundary=torch.softmax(logits,dim=1),self.model.calibrated_boundary(change)
        else:
            probability,boundary=torch.softmax(self.model(x),dim=1),None
        if self.stride>1:
            probability=probability.repeat_interleave(self.stride,dim=2)[:,:,:length]
            if boundary is not None:
                boundary=boundary.repeat_interleave(self.stride,dim=1)[:,:length]
        return probability,boundary


class FeatureEnsemble(nn.Module):
    def __init__(self, models, outer_names, member_states, weights):
        super().__init__()
        if len(models)!=len(member_states) or len(models)!=len(weights) or not len(models) or not np.isfinite(weights).all() or min(weights)<0 or abs(sum(weights)-1.)>1e-6:
            raise ValueError('Invalid ensemble members or weights')
        steps=[s.get('config',{}).get('sample_step',1) for s in member_states]
        outer_step=min(steps)
        if any(step<1 or step%outer_step for step in steps):
            raise ValueError('Member time grids must be whole multiples of the outer grid')
        strides=[step//outer_step for step in steps]
        self.members = nn.ModuleList([FeatureMember(m,outer_names,s['names'],s['mean'],s['scale'],stride) for m,s,stride in zip(models,member_states,strides)])
        self.register_buffer('weights',torch.as_tensor(weights,dtype=torch.float32))
        boundary_weights=[s.get('ensemble_boundary_weight',w) for s,w in zip(member_states,weights)]
        if not np.isfinite(boundary_weights).all() or min(boundary_weights)<0:
            raise ValueError('Boundary mixture weights must be finite and nonnegative')
        self.register_buffer('boundary_weights',torch.as_tensor(boundary_weights,dtype=torch.float32))
        # Whole-half evaluation also keeps coarse-grid sampling anchored at zero.
        self.full_sequence = any(m.full_sequence for m in models) or max(strides)>1
        self.raw_inputs = True
        self.receptive_radius = max(m.receptive_radius for m in models)
        self.has_boundary_head = any(getattr(m,'has_boundary_head',False) and w>0 for m,w in zip(models,boundary_weights))

    def forward_with_boundary(self, x):
        state,change,total = None,None,0.
        for member,weight,boundary_weight in zip(self.members,self.weights,self.boundary_weights):
            probability,boundary = member(x)
            state = weight*probability if state is None else state+weight*probability
            if boundary is not None:
                change = boundary_weight*boundary if change is None else change+boundary_weight*boundary
                total = total+boundary_weight
        logits = torch.log(state.clamp_min(1e-9))
        if change is None or not self.has_boundary_head:
            return logits,None
        q = (change/total).clamp(1e-6,1-1e-6)
        return logits,torch.logit(q)

    def forward(self, x):
        return self.forward_with_boundary(x)[0]

    def calibrated_boundary(self, logits):
        return torch.sigmoid(logits)
