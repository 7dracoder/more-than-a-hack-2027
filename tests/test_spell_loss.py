import unittest
import torch
from possession.spell_loss import spell_overlap_loss


class SpellOverlapTests(unittest.TestCase):
    @staticmethod
    def logits(labels):
        return (torch.nn.functional.one_hot(labels,num_classes=3).permute(0,2,1).double()*16-8).requires_grad_()

    def test_correct_spells_beat_shifted_and_merged_possessions(self):
        labels=torch.tensor([[2,0,0,1,1,0,0,2]])
        perfect=spell_overlap_loss(self.logits(labels),labels)
        shifted=spell_overlap_loss(self.logits(torch.tensor([[2,0,0,0,1,1,0,2]])),labels)
        merged=spell_overlap_loss(self.logits(torch.tensor([[2,0,0,0,0,0,0,2]])),labels)
        self.assertLess(float(perfect.detach()),1e-4)
        self.assertLess(float(perfect.detach()),float(shifted.detach()))
        self.assertLess(float(shifted.detach()),float(merged.detach()))

    def test_censored_runs_have_no_spurious_overlap_target(self):
        labels=torch.tensor([[-100,0,0,-100,1,1,-100]])
        logits=torch.randn(1,3,7,requires_grad=True)
        loss=spell_overlap_loss(logits,labels)
        self.assertEqual(float(loss.detach()),0.)
        loss.backward()
        self.assertTrue((logits.grad==0).all())

    def test_team_symmetry_and_gradient_encourage_missing_short_possession(self):
        labels=torch.tensor([[2,0,0,1,1,0,0,2]])
        logits=self.logits(torch.tensor([[2,0,0,0,0,0,0,2]]))
        loss=spell_overlap_loss(logits,labels)
        swapped=torch.where(labels==0,1,torch.where(labels==1,0,labels))
        other=spell_overlap_loss(logits[:,[1,0,2]],swapped)
        torch.testing.assert_close(loss,other,rtol=1e-12,atol=1e-12)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(float(logits.grad[0,1,3]),0.)
