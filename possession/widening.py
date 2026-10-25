"""Duplicate hidden channels while preserving a boundary model's eval function."""
import torch


def widen_boundary_hybrid(weights,model,factor):
    from possession.boundary import BoundaryHybrid
    if type(model) is not BoundaryHybrid or model.recurrent.num_layers!=1:
        raise ValueError('Widening supports the single-layer BoundaryHybrid only')
    if not isinstance(factor,int) or isinstance(factor,bool) or factor<2:
        raise ValueError('Widening needs an integer factor of at least two')
    old=weights['project.weight'];channels=old.shape[0]
    if channels%2 or model.project.out_channels!=channels*factor or old.shape[1:]!=model.project.weight.shape[1:]:
        raise ValueError('Widening input/hidden channel dimensions differ')
    if weights['recurrent.weight_hh_l0'].shape[1]!=channels or model.recurrent.hidden_size not in (channels,channels*factor):
        raise ValueError('Widening recurrent hidden dimensions differ')
    expected=model.state_dict()
    if set(weights)!=set(expected):raise ValueError('Widening layer schema differs')
    index=torch.arange(channels,device=old.device).repeat(factor)
    recurrent_factor=model.recurrent.hidden_size//channels
    recurrent_index=torch.arange(channels,device=old.device).repeat(recurrent_factor)
    gates=torch.cat([recurrent_index+i*channels for i in range(3)])
    representation=torch.cat([index,recurrent_index+channels,recurrent_index+2*channels])
    edge_index=torch.arange(channels//2,device=old.device).repeat(factor)
    # Unequal shares sum to one, preserving every forward sum while allowing
    # the copied channels to receive different gradients from the first step.
    shares=torch.arange(1,2*factor,2,device=old.device,dtype=old.dtype)/(factor*factor)
    channel_shares=shares.repeat_interleave(channels)
    channel_input=(index,channel_shares)
    recurrent_shares=channel_shares if recurrent_factor==factor else torch.ones(channels,device=old.device,dtype=old.dtype)
    recurrent_input=(recurrent_index,recurrent_shares)
    representation_input=(representation,torch.cat([channel_shares,recurrent_shares,recurrent_shares]))
    edge_input=(edge_index,shares.repeat_interleave(channels//2))
    def linear(value,output=None,input=None):
        if output is not None:value=value.index_select(0,output)
        if input is not None:
            mapping,fraction=input
            value=value.index_select(1,mapping)*fraction.reshape(1,-1,*([1]*(value.ndim-2)))
        return value.clone()
    result={}
    for name,value in weights.items():
        if name=='project.weight':new=linear(value,output=index)
        elif name=='project.bias':new=value[index].clone()
        elif name.startswith('local.'):
            if '.conv.weight' in name:new=linear(value,index,channel_input)
            elif name.endswith(('.conv.bias','.norm.weight','.norm.bias')):new=value[index].clone()
            else:raise ValueError('Unsupported local widening layer: '+name)
        elif name.startswith('recurrent.'):
            if name.startswith('recurrent.weight_ih_l0'):new=linear(value,gates,channel_input)
            elif name.startswith('recurrent.weight_hh_l0'):new=linear(value,gates,recurrent_input)
            elif name.startswith(('recurrent.bias_ih_l0','recurrent.bias_hh_l0')):new=value[gates].clone()
            else:raise ValueError('Unsupported recurrent widening layer: '+name)
        elif name=='head.0.weight':new=linear(value,index,representation_input)
        elif name=='head.0.bias':new=value[index].clone()
        elif name=='head.3.weight':new=linear(value,input=channel_input)
        elif name=='head.3.bias':new=value.clone()
        elif name=='boundary.0.weight':new=linear(value,edge_index,representation_input)
        elif name=='boundary.0.bias':new=value[edge_index].clone()
        elif name=='boundary.2.weight':new=linear(value,input=edge_input)
        elif name=='boundary.2.bias':new=value.clone()
        else:raise ValueError('Unsupported widening parameter: '+name)
        if new.shape!=expected[name].shape:raise ValueError('Widening parameter dimensions differ: '+name)
        result[name]=new
    return result
