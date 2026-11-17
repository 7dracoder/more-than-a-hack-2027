import numpy as np
import pytest
import torch
from possession.weak_supervision import conditional_ha_loss
from scripts.research.spatial_memory.prepare_real_broadcast_memory import conservative_labels


def test_fractional_ticks_require_agreement_without_inventing_dead_targets():
    source_low = np.array([0, 0, 0, 1, 1, 2])
    ticks = np.array([0, 2, 4])
    np.testing.assert_array_equal(conservative_labels(np.array(list('HAA')), source_low, ticks), list('HXA'))
    np.testing.assert_array_equal(conservative_labels(np.array(list('XAA')), source_low, ticks), list('XXA'))
    with pytest.raises(ValueError, match='never infer D'):
        conservative_labels(np.array(list('DAA')), source_low, ticks)


def test_conditional_weak_loss_has_no_direct_dead_logit_or_unknown_gradient():
    logits = torch.tensor([[[.3, .5, -.2], [.1, -.5, .4], [4., -3., 2.]]], requires_grad=True)
    targets = torch.tensor([[0, -100, 1]])
    loss = conditional_ha_loss(logits, targets)
    changed = logits.detach().clone(); changed[:, 2] += 100.
    torch.testing.assert_close(loss, conditional_ha_loss(changed, targets))
    loss.backward()
    torch.testing.assert_close(logits.grad[:, 2], torch.zeros_like(logits.grad[:, 2]))
    torch.testing.assert_close(logits.grad[:, :, 1], torch.zeros_like(logits.grad[:, :, 1]))
    assert logits.grad[:, :2, [0, 2]].abs().sum() > 0


def test_all_unknown_targets_are_finite_zero_and_dead_targets_are_rejected():
    logits = torch.randn(2, 3, 4, requires_grad=True)
    loss = conditional_ha_loss(logits, torch.full((2, 4), -100))
    assert torch.isfinite(loss) and loss == 0
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
    with pytest.raises(ValueError, match='H/A'):
        conditional_ha_loss(logits.detach(), torch.full((2, 4), 2))
