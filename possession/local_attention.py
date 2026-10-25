"""Finite-context relative attention for brief contact evidence, without match time."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from possession.temporal import TemporalBlock
from possession.recurrent import inference_gru


class RelativeAttentionBlock(nn.Module):
    def __init__(self, channels=128, heads=4, radius=16, dropout=.2):
        super().__init__()
        if channels%heads or radius<1:
            raise ValueError('Attention needs divisible channels and a positive radius')
        self.heads,self.radius,self.width=heads,radius,channels//heads
        self.norm=nn.LayerNorm(channels)
        self.qkv=nn.Linear(channels,3*channels)
        self.relative=nn.Parameter(torch.zeros(heads,2*radius+1))
        self.output=nn.Linear(channels,channels)
        self.dropout=nn.Dropout(dropout)
        self.ff=nn.Sequential(nn.LayerNorm(channels),nn.Linear(channels,2*channels),nn.GELU(),
                              nn.Dropout(dropout),nn.Linear(2*channels,channels),nn.Dropout(dropout))

    def forward(self,x):
        b,t,c=x.shape
        qkv=self.qkv(self.norm(x)).view(b,t,3,self.heads,self.width).permute(2,0,3,1,4)
        q,k,v=qkv.unbind(0)
        window=2*self.radius+1
        keys=F.pad(k,(0,0,self.radius,self.radius)).unfold(2,window,1)
        values=F.pad(v,(0,0,self.radius,self.radius)).unfold(2,window,1)
        scores=(q.unsqueeze(-1)*keys).sum(-2)/math.sqrt(self.width)+self.relative[None,:,None,:]
        positions=torch.arange(t,device=x.device)[:,None]+torch.arange(-self.radius,self.radius+1,device=x.device)[None,:]
        scores=scores.masked_fill(((positions<0)|(positions>=t))[None,None],-torch.inf)
        attention=torch.softmax(scores,dim=-1)
        context=(attention.unsqueeze(-2)*values).sum(-1).transpose(1,2).reshape(b,t,c)
        x=x+self.dropout(self.output(context))
        return x+self.ff(x)


class PossessionLocalAttention(nn.Module):
    def __init__(self,inputs,channels=128,heads=4,layers=4,radius=16,dropout=.2,positive_weight=10.):
        super().__init__()
        self.full_sequence=False
        self.receptive_radius=layers*radius
        self.has_boundary_head=True
        self.positive_weight=float(positive_weight)
        self.project=nn.Sequential(nn.Linear(inputs,channels),nn.LayerNorm(channels),nn.GELU())
        self.blocks=nn.Sequential(*(RelativeAttentionBlock(channels,heads,radius,dropout) for _ in range(layers)))
        self.head=nn.Linear(channels,3)
        self.boundary=nn.Linear(channels,1)

    def forward_with_boundary(self,x):
        value=self.blocks(self.project(x.transpose(1,2)))
        return self.head(value).transpose(1,2),self.boundary(value)[:,:,0]

    def forward(self,x):
        return self.forward_with_boundary(x)[0]

    def calibrated_boundary(self,logits):
        return torch.sigmoid(logits-math.log(self.positive_weight))


class RecurrentAttention(nn.Module):
    """Keep full-half memory and attend directly to nearby contact evidence."""
    def __init__(self,inputs,channels=128,heads=4,layers=2,radius=8,dropout=.2,positive_weight=10.):
        super().__init__()
        self.full_sequence=True
        self.receptive_radius=0
        self.has_boundary_head=True
        self.positive_weight=float(positive_weight)
        self.project=nn.Conv1d(inputs,channels,1)
        self.local=nn.Sequential(*(TemporalBlock(channels,d,dropout) for d in (1,2,4)))
        self.recurrent=nn.GRU(channels,channels,batch_first=True,bidirectional=True)
        self.fuse=nn.Sequential(nn.Linear(3*channels,channels),nn.LayerNorm(channels),nn.GELU())
        self.refine=nn.Sequential(*(RelativeAttentionBlock(channels,heads,radius,dropout) for _ in range(layers)))
        self.head=nn.Linear(channels,3)
        self.boundary=nn.Linear(channels,1)

    def forward_with_boundary(self,x):
        local=self.local(self.project(x)).transpose(1,2)
        context,_=inference_gru(self.recurrent,local)
        value=self.refine(self.fuse(torch.cat([local,context],dim=-1)))
        return self.head(value).transpose(1,2),self.boundary(value)[:,:,0]

    def forward(self,x):
        return self.forward_with_boundary(x)[0]

    def calibrated_boundary(self,logits):
        return torch.sigmoid(logits-math.log(self.positive_weight))
