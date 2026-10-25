import unittest
import torch
from possession.carrier_loss import carrier_attention_loss
from possession.player_trajectory_model import PlayerTrajectoryCorrector


class CarrierAttentionLossTests(unittest.TestCase):
    def test_non_nearest_carriers_can_receive_more_weight_without_changing_unknown_targets(self):
        features=torch.zeros(1,184,3)
        for i in range(5,180,6):features[:,i]=1.
        scores=torch.zeros(1,2,3,3,requires_grad=True);labels=torch.zeros(1,3,dtype=torch.long);targets=torch.tensor([[0,1,2]])
        loss=carrier_attention_loss(scores,features,labels,targets,torch.tensor([1.,3.,6.]));loss.backward()
        torch.testing.assert_close(scores.grad[0,0,2,2].abs(),scores.grad[0,0,0,0].abs()*6,rtol=1e-6,atol=1e-7)
        self.assertTrue(torch.isfinite(loss))

    def fixture(self):
        torch.set_num_threads(2);torch.manual_seed(72)
        features=torch.randn(2,184,17)
        for i in range(5,180,6):features[:,i]=1.
        labels=torch.randint(0,2,(2,17));targets=torch.randint(0,3,(2,17));scores=torch.randn(2,2,3,17,requires_grad=True)
        return scores,features,labels,targets

    def test_team_swap_preserves_the_same_physical_carrier_loss(self):
        scores,features,labels,targets=self.fixture();changed=features.clone()
        changed[:,:180]=features[:,:180].reshape(2,2,3,5,6,17).flip(1).reshape(2,180,17)
        a=carrier_attention_loss(scores,features,labels,targets)
        b=carrier_attention_loss(scores.flip(1),changed,1-labels,targets)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_unknown_groups_or_unobserved_target_players_have_zero_finite_gradients(self):
        scores,features,labels,targets=self.fixture();labels[:,0]=-100
        for i in range(5,180,6):features[:,i,1:]=0.
        loss=carrier_attention_loss(scores,features,labels,targets);loss.backward()
        self.assertEqual(float(loss),0.);self.assertEqual(int(scores.grad.count_nonzero()),0)
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_real_attention_forward_receives_carrier_and_encoder_gradients(self):
        _,features,labels,targets=self.fixture();model=PlayerTrajectoryCorrector(184,0,4,8,4,[1,2],0.,'attention')
        captured={};hook=model.attention.register_forward_hook(lambda module,args,output:captured.update(scores=output))
        state,_=model.forward_with_boundary(features);hook.remove()
        loss=torch.nn.functional.cross_entropy(state,labels)+.1*carrier_attention_loss(captured['scores'].reshape(2,2,3,17),features,labels,targets)
        loss.backward();self.assertGreater(float(model.attention.weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.encoder[0].weight.grad.abs().sum()),0.)
