import unittest
import torch
from possession.training_context import mask_crop_context


class TrainingContextTests(unittest.TestCase):
    def test_artificial_padding_has_no_state_gradient(self):
        labels=torch.tensor([[0,1,2,0,1,2,0,1,2]])
        weights=torch.ones_like(labels,dtype=torch.float32)
        mask_crop_context(labels,weights,2)
        logits=torch.zeros(1,3,9,requires_grad=True)
        loss=(torch.nn.functional.cross_entropy(logits,labels,ignore_index=-100,reduction='none')*weights).sum()/weights.sum()
        loss.backward()
        self.assertEqual(float(logits.grad[:,:,:2].abs().sum()),0.)
        self.assertEqual(float(logits.grad[:,:,-2:].abs().sum()),0.)
        self.assertGreater(float(logits.grad[:,:,2:-2].abs().sum()),0.)

    def test_zero_margin_preserves_targets_and_weights(self):
        labels=torch.tensor([[0,-100,1,2]]);weights=torch.tensor([[1.,0.,2.,1.]])
        before=labels.clone(),weights.clone();mask_crop_context(labels,weights,0)
        torch.testing.assert_close(labels,before[0]);torch.testing.assert_close(weights,before[1])

    def test_crop_needs_a_real_interior(self):
        with self.assertRaises(ValueError):mask_crop_context(torch.zeros(1,4),torch.ones(1,4),2)
