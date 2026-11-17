"""Learn added input columns while preserving all inherited model tensors."""
import numpy as np
import torch


def column_indices(weight, columns):
    values = np.asarray(columns)
    if (weight.ndim != 3 or weight.shape[2] != 1 or values.ndim != 1 or
            not len(values) or not np.issubdtype(values.dtype,np.integer) or
            len(set(values.tolist())) != len(values) or values.min()<0 or
            values.max()>=weight.shape[1] or len(values)==weight.shape[1]):
        raise ValueError('Need a proper nonempty subset of point-projection columns')
    return torch.as_tensor(values,device=weight.device,dtype=torch.long)


def configure_added_columns(model, columns):
    weight = model.project.weight
    added = column_indices(weight,columns)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    weight.requires_grad_(True)
    mask = torch.zeros_like(weight,dtype=torch.bool)
    mask[:,added,:] = True
    weight.register_hook(lambda gradient: gradient.masked_fill(~mask,0.))
    return weight


@torch.no_grad()
def decay_added_columns(weight, columns, learning_rate, weight_decay):
    added = column_indices(weight,columns)
    factor = 1.-learning_rate*weight_decay
    if (not np.isfinite(learning_rate) or not np.isfinite(weight_decay) or
            learning_rate<0 or weight_decay<0 or not 0<=factor<=1):
        raise ValueError('Invalid decoupled decay')
    # Advanced indexing returns a copy, so explicitly assign the decayed values.
    weight[:,added,:] = weight[:,added,:]*factor


def assert_inherited_unchanged(current, reference, columns):
    if set(current) != set(reference):
        raise AssertionError('Model state keys changed')
    added = column_indices(current['project.weight'],columns)
    keep = torch.ones(current['project.weight'].shape[1],dtype=torch.bool,
        device=current['project.weight'].device)
    keep[added] = False
    for key,value in current.items():
        expected = reference[key].to(value.device)
        if value.dtype != expected.dtype or value.shape != expected.shape:
            raise AssertionError(f'Inherited tensor schema changed: {key}')
        actual = value[:,keep,:] if key=='project.weight' else value
        expected = expected[:,keep,:] if key=='project.weight' else expected
        if not torch.equal(actual,expected):
            raise AssertionError(f'Inherited tensor changed: {key}')
    if not torch.isfinite(current['project.weight']).all():
        raise AssertionError('Nonfinite projection coefficients')
