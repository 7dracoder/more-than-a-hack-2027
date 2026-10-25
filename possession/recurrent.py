"""Exact state-carrying evaluation for sequences exceeding cuDNN's supported size."""
import torch
from torch import nn


def inference_gru(gru, inputs, max_sequence=60000):
    batch=inputs if gru.batch_first else inputs.transpose(0,1)
    if max_sequence<1:
        raise ValueError('Recurrent evaluation chunk must be positive')
    if gru.training or batch.shape[1]<=max_sequence:
        return gru(inputs)
    if torch.is_grad_enabled():
        raise ValueError('Long GRU evaluation requires no_grad or inference_mode')
    if not isinstance(gru,nn.GRU):
        raise TypeError('This evaluator supports GRU only')
    value=batch
    hidden=[]
    directions=2 if gru.bidirectional else 1
    for layer in range(gru.num_layers):
        outputs=[]
        for direction in range(directions):
            suffix='_reverse' if direction else ''
            # Construct on CPU under a forked RNG, then copy parameters. These
            # cells neither alias/repack the training model nor consume its RNG.
            with torch.random.fork_rng(devices=[]):
                cell=nn.GRU(value.shape[-1],gru.hidden_size,bias=gru.bias,batch_first=True,
                            device='cpu',dtype=value.dtype)
            names=['weight_ih','weight_hh']+(['bias_ih','bias_hh'] if gru.bias else [])
            parameters={name+'_l0':getattr(gru,f'{name}_l{layer}{suffix}').detach().cpu() for name in names}
            cell.load_state_dict(parameters)
            cell=cell.to(value.device).eval()
            sequence=value.flip(1) if direction else value
            carry=None
            chunks=[]
            chunk=min(max_sequence,32768)
            for start in range(0,sequence.shape[1],chunk):
                part,carry=cell(sequence[:,start:start+chunk].contiguous(),carry)
                chunks.append(part)
            output=torch.cat(chunks,dim=1)
            outputs.append(output.flip(1) if direction else output)
            hidden.append(carry)
        value=torch.cat(outputs,dim=-1) if directions==2 else outputs[0]
    return (value if gru.batch_first else value.transpose(0,1)),torch.cat(hidden,dim=0)
