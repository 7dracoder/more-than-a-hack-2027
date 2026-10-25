"""Shared player-trajectory encoding with masked, order-independent team pooling."""
import torch
from torch import nn
from possession.temporal import TemporalBlock


def trajectory_mask_names():
    from possession.features_v25 import OFFSETS
    return [f'{side}_tube{rank}_o{offset}_known' for side in ('home','away') for rank in (1,2,3) for offset in OFFSETS]


class PlayerTrajectoryCorrector(nn.Module):
    def __init__(self,inputs,teacher_features,context_features=4,channels=64,player_channels=32,
                 dilations=(1,2,4,8,16,32),dropout=.1,pooling='mean_max'):
        super().__init__()
        if inputs!=teacher_features+180+context_features or context_features not in (0,4):
            raise ValueError('Player-trajectory correction requires the fixed six-player, five-point schema')
        self.teacher_features=teacher_features;self.context_features=context_features
        self.player_channels=player_channels
        if pooling not in ('mean_max','attention'):raise ValueError('Unknown player pooling')
        self.pooling=pooling
        self.encoder=nn.Sequential(nn.Conv1d(30,player_channels,1),nn.GELU(),nn.Conv1d(player_channels,player_channels,1),nn.GELU())
        if pooling=='attention':self.attention=nn.Conv1d(player_channels,1,1)
        self.project=nn.Conv1d((6 if pooling=='attention' else 4)*player_channels+context_features,channels,1)
        self.blocks=nn.Sequential(*(TemporalBlock(channels,d,dropout) for d in dilations))
        self.head=nn.Sequential(nn.Conv1d(channels,channels,1),nn.GELU(),nn.Dropout(dropout),nn.Conv1d(channels,3,1))
        self.boundary=nn.Sequential(nn.Conv1d(channels,channels//2,1),nn.GELU(),nn.Conv1d(channels//2,1,1))

    def trajectory_embedding(self,x):
        batch,_,frames=x.shape
        tubes=x[:,self.teacher_features:self.teacher_features+180].reshape(batch,2,3,5,6,frames)
        known=(tubes[:,:,:,2,5,:]>.5)[:,:,:,None,:]
        encoded=self.encoder(tubes.reshape(batch*6,30,frames)).reshape(batch,2,3,self.player_channels,frames)
        count=known.sum(dim=2)
        mean=(encoded*known).sum(dim=2)/count.clamp_min(1)
        maximum=encoded.masked_fill(~known,-torch.inf).amax(dim=2)
        maximum=torch.where(count>0,maximum,torch.zeros_like(maximum))
        summaries=[mean,maximum]
        if self.pooling=='attention':
            scores=self.attention(encoded.reshape(batch*6,self.player_channels,frames)).reshape(batch,2,3,1,frames)
            weights=scores.masked_fill(~known,-1e4).softmax(dim=2)*known
            weights=weights/weights.sum(dim=2,keepdim=True).clamp_min(1e-8)
            summaries.append((encoded*weights).sum(dim=2))
        return torch.cat(summaries,dim=2).reshape(batch,-1,frames)

    def forward_with_boundary(self,x):
        pooled=self.trajectory_embedding(x)
        if self.context_features:pooled=torch.cat([pooled,x[:,-self.context_features:]],dim=1)
        hidden=self.blocks(self.project(pooled))
        return self.head(hidden),self.boundary(hidden)[:,0]

    def forward(self,x):return self.forward_with_boundary(x)[0]
