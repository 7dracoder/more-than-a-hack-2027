"""Conditional team supervision without inventing external dead-ball targets."""
import torch


def conditional_ha_loss(logits, targets, weights=None):
    if logits.ndim != 3 or logits.shape[1] != 3 or targets.shape != (logits.shape[0], logits.shape[2]):
        raise ValueError('Need HAD logits and aligned weak H/A targets')
    if not torch.all((targets == -100) | (targets == 0) | (targets == 1)):
        raise ValueError('Weak supervision must contain H/A or ignored targets only')
    if weights is None:
        weights = torch.ones_like(targets, dtype=logits.dtype)
    if weights.shape != targets.shape or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Invalid weak-target weights')
    valid_weights = weights * (targets >= 0)
    loss = torch.nn.functional.cross_entropy(logits[:, :2], targets, reduction='none', ignore_index=-100)
    return (loss * valid_weights).sum() / valid_weights.sum().clamp_min(1.)
