import math
import unittest
import torch
from possession.paired_training import symmetric_outputs


class PairedTrainingTests(unittest.TestCase):
    def test_matches_inference_arithmetic_probability_mean_and_calibration(self):
        torch.manual_seed(13)
        a=torch.randn(2,3,17,dtype=torch.double);b=torch.randn_like(a)
        q=torch.randn(2,17,dtype=torch.double);r=torch.randn_like(q)
        state,edge=symmetric_outputs(a,b,q,r,20.)
        expected=(a.softmax(1)+b.softmax(1)[:,[1,0,2]])/2
        torch.testing.assert_close(state.softmax(1),expected,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close((edge-math.log(20.)).sigmoid(),((q-math.log(20.)).sigmoid()+(r-math.log(20.)).sigmoid())/2,rtol=1e-12,atol=1e-12)

    def test_gradients_reach_both_state_and_boundary_views(self):
        torch.manual_seed(4)
        values=[torch.randn(*shape,dtype=torch.double,requires_grad=True) for shape in ((1,3,5),(1,3,5),(1,5),(1,5))]
        state,edge=symmetric_outputs(*values,20.)
        loss=-state[:,0].mean()+torch.nn.functional.binary_cross_entropy_with_logits(edge,torch.ones_like(edge))
        loss.backward()
        for value in values:
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()),0.)

    def test_swapping_views_only_swaps_state_classes(self):
        torch.manual_seed(7)
        a=torch.randn(2,3,9);b=torch.randn_like(a);q=torch.randn(2,9);r=torch.randn_like(q)
        state,edge=symmetric_outputs(a,b,q,r,1.)
        reverse,reverse_edge=symmetric_outputs(b,a,r,q,1.)
        torch.testing.assert_close(state[:,[1,0,2]],reverse)
        torch.testing.assert_close(edge,reverse_edge)

    def test_extreme_logits_keep_finite_outputs_and_gradients(self):
        a=torch.tensor([[[1000.,-1000.],[-1000.,1000.],[0.,0.]]],requires_grad=True)
        b=-a.detach().clone().requires_grad_(True)
        q=torch.tensor([[1000.,-1000.]],requires_grad=True);r=q.detach().clone().requires_grad_(True)
        state,edge=symmetric_outputs(a,b,q,r,20.)
        self.assertTrue(torch.isfinite(state).all() and torch.isfinite(edge).all())
        loss=-state.mean()+torch.nn.functional.binary_cross_entropy_with_logits(edge,torch.ones_like(edge))
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all() and torch.isfinite(q.grad).all())

    def test_invalid_shapes_and_calibration_are_rejected(self):
        a=torch.zeros(1,3,5);q=torch.zeros(1,5)
        with self.assertRaises(ValueError):symmetric_outputs(a,a,q,q,0.)
        with self.assertRaises(ValueError):symmetric_outputs(a,a,q[:,:4],q[:,:4],20.)

    def test_boundary_model_optimizer_step_updates_shared_and_both_heads(self):
        from possession.boundary import BoundaryHybrid,boundary_loss
        from possession.temporal import frame_loss
        torch.set_num_threads(2);torch.manual_seed(19)
        model=BoundaryHybrid(4,8,0.,20.)
        x=torch.randn(2,4,31);y=torch.randint(0,3,(2,31));y[:,:3]=-100
        mapping=torch.tensor([1,0,3,2]);optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        state,edge=model.forward_with_boundary(torch.cat([x,x[:,mapping]],0))
        logits,boundary=symmetric_outputs(state[:2],state[2:],edge[:2],edge[2:],20.)
        loss=frame_loss(logits,y).mean()+.3*boundary_loss(boundary,y,20.)
        loss.backward()
        for parameter in (model.project.weight,model.head[-1].weight,model.boundary[-1].weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()),0.)
        before=model.project.weight.detach().clone();optimizer.step()
        self.assertFalse(torch.equal(before,model.project.weight))
