import copy
import numpy as np
import pytest
import torch
from possession.frozen_projection import (configure_added_columns,
    decay_added_columns,assert_inherited_unchanged)


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.project = torch.nn.Conv1d(4,3,1)
        self.head = torch.nn.Conv1d(3,2,1)
        self.register_buffer('calibration',torch.tensor([1.,2.]))

    def forward(self,x):
        return self.head(torch.tanh(self.project(x)))


def step(model,optimizer):
    optimizer.zero_grad(set_to_none=True)
    x = torch.arange(32,dtype=torch.float32).reshape(2,4,4)/10
    loss = model(x).square().mean()
    loss.backward()
    decay_added_columns(model.project.weight,[1,3],optimizer.param_groups[0]['lr'],.1)
    optimizer.step()


def test_optimizer_changes_added_columns_and_preserves_inherited_tensors():
    torch.manual_seed(2)
    model = SmallModel()
    reference = copy.deepcopy(model.state_dict())
    weight = configure_added_columns(model,[1,3])
    optimizer = torch.optim.AdamW([weight],lr=.05,weight_decay=0.)
    for _ in range(4):
        step(model,optimizer)
        assert_inherited_unchanged(model.state_dict(),reference,[1,3])
        assert torch.count_nonzero(weight.grad[:,[0,2],:])==0
        assert torch.count_nonzero(optimizer.state[weight]['exp_avg'][:,[0,2],:])==0
        assert model.head.weight.grad is None and model.project.bias.grad is None
    assert not torch.equal(weight[:,[1,3],:],reference['project.weight'][:,[1,3],:])


def test_optimizer_resume_with_reinstalled_mask_reproduces_next_step():
    torch.manual_seed(2)
    model = SmallModel()
    optimizer = torch.optim.AdamW([configure_added_columns(model,[1,3])],lr=.05,weight_decay=0.)
    step(model,optimizer)
    resumed = SmallModel()
    resumed.load_state_dict(copy.deepcopy(model.state_dict()))
    resumed_optimizer = torch.optim.AdamW([configure_added_columns(resumed,[1,3])],lr=.05,weight_decay=0.)
    resumed_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    step(model,optimizer);step(resumed,resumed_optimizer)
    for name,value in model.state_dict().items():
        assert torch.equal(value,resumed.state_dict()[name])


def test_invalid_columns_and_unexpected_inherited_change_are_rejected():
    for columns in [[],[1,1],[-1],[4],[0,1,2,3],[1.5]]:
        with pytest.raises(ValueError):
            configure_added_columns(SmallModel(),columns)
    model = SmallModel()
    reference = copy.deepcopy(model.state_dict())
    model.calibration[0] = 9
    with pytest.raises(AssertionError,match='calibration'):
        assert_inherited_unchanged(model.state_dict(),reference,np.array([1,3]))
