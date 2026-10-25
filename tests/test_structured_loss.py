import itertools
import math
import unittest
import numpy as np
import torch
from scipy.special import logsumexp
from possession.structured_loss import structured_state_loss


def brute_force(logits,boundary,labels,team,dead):
    unary=logits-torch.logsumexp(logits,dim=1,keepdim=True)
    unary=unary.detach().numpy()[0].T;boundary=boundary.detach().numpy()[0];labels=labels.numpy()[0]
    known=(labels>=0)&(labels<3)
    cost=np.array([[0.,team,dead],[team,0.,dead],[dead,dead,0.]])
    def score(path):
        value=sum(unary[i,state] for i,state in enumerate(path) if known[i])
        for i in range(1,len(path)):
            if known[i] and known[i-1]:
                value+=boundary[i]*(path[i]!=path[i-1])-cost[path[i-1],path[i]]
        return value
    partition=logsumexp([score(path) for path in itertools.product(range(3),repeat=len(labels))])
    truth=score(np.where(known,labels,0))
    return (partition-truth-(~known).sum()*math.log(3.))/max(known.sum(),1)


class StructuredLossTests(unittest.TestCase):
    def test_transition_gradient_rewards_true_changes_and_suppresses_false_ones(self):
        state=torch.zeros(1,3,4,dtype=torch.float64,requires_grad=True)
        boundary=torch.zeros(1,4,dtype=torch.float64,requires_grad=True)
        structured_state_loss(state,boundary,torch.tensor([[0,0,1,1]]),0.,0.).backward()
        self.assertLess(float(boundary.grad[0,2]),0.)
        self.assertGreater(float(boundary.grad[0,1]),0.)
        self.assertGreater(float(boundary.grad[0,3]),0.)

    def test_confident_long_predictions_keep_finite_loss_and_gradients(self):
        state=torch.full((2,3,1025),-1000.,dtype=torch.float64)
        state[:,0]=1000.;state.requires_grad_()
        boundary=torch.full((2,1025),-1000.,dtype=torch.float64,requires_grad=True)
        labels=torch.zeros(2,1025,dtype=torch.long);labels[1,100:200]=-100
        loss=structured_state_loss(state,boundary,labels);loss.backward()
        self.assertTrue(torch.isfinite(loss) and torch.isfinite(state.grad).all() and torch.isfinite(boundary.grad).all())
        self.assertLess(abs(float(loss.detach())),1e-10)

    def test_balanced_likelihood_matches_every_possible_path(self):
        torch.set_num_threads(2);torch.manual_seed(28)
        for timeline in ([0],[0,1],[0,0,1,2],[0,-100,1,2],[0,-100,-100,1,2],[-100]*4):
            y=torch.tensor([timeline]);state=torch.randn(1,3,len(timeline),dtype=torch.float64)
            boundary=torch.randn(1,len(timeline),dtype=torch.float64)
            actual=structured_state_loss(state,boundary,y,.7,1.3)
            expected=brute_force(state,boundary,y,.7,1.3)
            self.assertAlmostEqual(float(actual),expected,places=11)

    def test_gradients_match_numerical_derivatives(self):
        torch.manual_seed(29)
        state=torch.randn(1,3,5,dtype=torch.float64,requires_grad=True)
        boundary=torch.randn(1,5,dtype=torch.float64,requires_grad=True)
        labels=torch.tensor([[0,1,-100,2,0]])
        self.assertTrue(torch.autograd.gradcheck(lambda a,b:structured_state_loss(a,b,labels),
                                               (state,boundary),eps=1e-6,atol=1e-5))

    def test_unknown_frames_and_adjacent_transitions_have_no_gradient(self):
        state=torch.randn(1,3,7,dtype=torch.float64,requires_grad=True)
        boundary=torch.randn(1,7,dtype=torch.float64,requires_grad=True)
        labels=torch.tensor([[0,1,-100,-100,2,0,1]])
        loss=structured_state_loss(state,boundary,labels);loss.backward()
        self.assertEqual(float(state.grad[:,:,2:4].abs().sum()),0.)
        self.assertEqual(float(boundary.grad[:,[0,2,3,4]].abs().sum()),0.)
        self.assertTrue(torch.isfinite(state.grad).all() and torch.isfinite(boundary.grad).all())

    def test_home_away_swap_preserves_likelihood(self):
        state=torch.randn(2,3,11,dtype=torch.float64)
        boundary=torch.randn(2,11,dtype=torch.float64)
        labels=torch.randint(0,3,(2,11));swapped=torch.where(labels==0,1,torch.where(labels==1,0,labels))
        torch.testing.assert_close(structured_state_loss(state,boundary,labels),
                                   structured_state_loss(state[:,[1,0,2]],boundary,swapped),rtol=0.,atol=1e-12)
