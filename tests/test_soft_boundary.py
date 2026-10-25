import unittest
import torch
from possession.boundary import boundary_targets,boundary_loss


class SoftBoundaryTests(unittest.TestCase):
    def test_hard_default_and_symmetric_distance_targets(self):
        labels=torch.tensor([[0,0,0,0,1,1,1,1]])
        hard,valid=boundary_targets(labels)
        torch.testing.assert_close(hard,torch.tensor([[0.,0.,0.,1.,0.,0.,0.]]))
        soft,_=boundary_targets(labels,2)
        torch.testing.assert_close(soft,torch.tensor([[0.,1/3,2/3,1.,2/3,1/3,0.]]))
        self.assertTrue(valid.all())

    def test_unknown_boundaries_do_not_create_positive_targets(self):
        targets,valid=boundary_targets(torch.tensor([[0,0,-100,1,1]]),3)
        self.assertEqual(float(targets.sum()),0.)
        self.assertEqual(valid.tolist(),[[True,False,False,True]])

    def test_soft_training_backward_is_finite(self):
        labels=torch.tensor([[0,0,1,1,2,2,0]])
        logits=torch.zeros(labels.shape,requires_grad=True)
        loss=boundary_loss(logits,labels,4.,5)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(float(logits.grad[0,2]),0.)
