"""A frozen temporal teacher with a zero-initialized local correction model."""
import torch
from torch import nn
from possession.boundary import BoundaryHybrid,BoundaryTCN


class ResidualBoundaryHybrid(nn.Module):
    def __init__(self,inputs,channels=128,dropout=.2,positive_weight=20.,
                 residual_channels=64,residual_dilations=(1,2,4,8,16,32),residual_dropout=.1,teacher_context=False,teacher_feature_count=None,
                 residual_kind='tcn',player_channels=32,teacher_model_config=None,player_pooling='mean_max',trajectory_feature_start=None):
        super().__init__()
        self.full_sequence,self.receptive_radius=True,0
        self.has_boundary_head=True
        self.positive_weight=float(positive_weight)
        self.input_feature_count=inputs
        if not isinstance(teacher_context,bool):raise ValueError('Teacher context must be a boolean')
        self.teacher_context=teacher_context
        self.teacher_feature_count=inputs if teacher_feature_count is None else teacher_feature_count
        if not isinstance(self.teacher_feature_count,int) or not 0<self.teacher_feature_count<=inputs:
            raise ValueError('Teacher feature count must be a valid input prefix')
        if teacher_model_config is None:
            self.teacher=BoundaryHybrid(self.teacher_feature_count,channels,dropout,positive_weight)
        else:
            from possession.temporal import make_model
            self.teacher=make_model(self.teacher_feature_count,teacher_model_config)
            if not getattr(self.teacher,'has_boundary_head',False) or self.teacher.positive_weight!=self.positive_weight:
                raise ValueError('Frozen teacher must have matching boundary calibration')
        self.teacher.requires_grad_(False);self.teacher.eval()
        if residual_kind=='tcn':
            self.corrector=BoundaryTCN(inputs+(4 if teacher_context else 0),residual_channels,residual_dilations,residual_dropout,positive_weight)
        elif residual_kind=='player_trajectories':
            from possession.player_trajectory_model import PlayerTrajectoryCorrector
            feature_start=self.teacher_feature_count if trajectory_feature_start is None else trajectory_feature_start
            self.corrector=PlayerTrajectoryCorrector(inputs+(4 if teacher_context else 0),feature_start,
                4 if teacher_context else 0,residual_channels,player_channels,residual_dilations,residual_dropout,player_pooling)
        else:raise ValueError('Unknown residual correction architecture')
        nn.init.zeros_(self.corrector.head[-1].weight);nn.init.zeros_(self.corrector.head[-1].bias)
        nn.init.zeros_(self.corrector.boundary[-1].weight);nn.init.zeros_(self.corrector.boundary[-1].bias)

    @property
    def warm_start_model(self):return self.teacher

    def train(self,mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward_with_boundary(self,x):
        teacher_input=x if x.shape[1]==self.teacher_feature_count else x[:,:self.teacher_feature_count]
        with torch.no_grad():state,edge=self.teacher.forward_with_boundary(teacher_input)
        return self._correct(x,state,edge)

    def forward_with_cached_teacher(self,x,state,edge):
        if self.training:raise ValueError('Cached teacher outputs are for evaluation only')
        if state.shape!=(x.shape[0],3,x.shape[2]) or edge.shape!=(x.shape[0],x.shape[2]):
            raise ValueError('Cached teacher shape differs')
        if state.requires_grad or edge.requires_grad or state.device!=x.device or edge.device!=x.device:
            raise ValueError('Cached teacher outputs must be detached and on the input device')
        return self._correct(x,state,edge)

    def _correct(self,x,state,edge):
        if self.teacher_context:
            context=torch.cat([state.softmax(dim=1),self.teacher.calibrated_boundary(edge)[:,None]],dim=1)
            x=torch.cat([x,context],dim=1)
        correction,edge_correction=self.corrector.forward_with_boundary(x)
        return state+correction,edge+edge_correction

    def forward(self,x):return self.forward_with_boundary(x)[0]

    def calibrated_boundary(self,logits):return self.teacher.calibrated_boundary(logits)
