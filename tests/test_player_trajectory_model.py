import unittest
import torch
from possession.player_trajectory_model import PlayerTrajectoryCorrector


class PlayerTrajectoryModelTests(unittest.TestCase):
    def test_full_trajectory_teacher_can_be_frozen_with_a_corrector_reusing_the_existing_suffix(self):
        from possession.temporal import make_model
        from possession.player_trajectory_model import trajectory_mask_names
        torch.set_num_threads(2)
        parent=dict(architecture='residual_boundary',hidden_channels=8,dropout=0.,boundary_positive_weight=20.,
            residual_channels=8,residual_dilations=[1,2],residual_teacher_context=True,
            teacher_feature_count=4,residual_kind='player_trajectories',player_channels=4,unit_feature_names=trajectory_mask_names())
        config=dict(parent,teacher_feature_count=184,trajectory_feature_start=4,teacher_model_config=parent)
        model=make_model(184,config).eval();x=torch.randn(2,184,31)
        with torch.no_grad():a,q=model.teacher.forward_with_boundary(x);b,r=model.forward_with_boundary(x)
        torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(q,r,rtol=0,atol=0)
        copy=make_model(184,config).eval();copy.load_state_dict(model.state_dict())
        with torch.no_grad():c,s=copy.forward_with_boundary(x)
        torch.testing.assert_close(b,c);torch.testing.assert_close(r,s)
        self.assertTrue(all(not p.requires_grad for p in model.teacher.parameters()))

    def test_attention_pooling_preserves_permutation_and_missing_player_invariants(self):
        _,x=self.fixture()
        model=PlayerTrajectoryCorrector(188,4,4,8,4,(1,2),0.,'attention').eval()
        changed=x.clone();changed[:,4:184]=x[:,4:184].reshape(2,2,3,5,6,31)[:,:,[2,0,1]].reshape(2,180,31)
        with torch.no_grad():a=model.trajectory_embedding(x);b=model.trajectory_embedding(changed)
        self.assertEqual(tuple(a.shape),(2,24,31));torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
        x[:,21]=0.;changed=x.clone();changed[:,4:34]=99.;changed[:,21]=0.
        with torch.no_grad():a=model.trajectory_embedding(x);b=model.trajectory_embedding(changed)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        state,edge=model.forward_with_boundary(x);(state.square().mean()+edge.square().mean()).backward()
        self.assertGreater(float(model.attention.weight.grad.abs().sum()),0.)
        model.zero_grad(set_to_none=True)
        for index in range(9,184,6):x[:,index]=0.
        self.assertEqual(int(model.trajectory_embedding(x).count_nonzero()),0)
        state,edge=model.forward_with_boundary(x);(state.square().mean()+edge.square().mean()).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def fixture(self):
        torch.set_num_threads(2);torch.manual_seed(25)
        model=PlayerTrajectoryCorrector(188,4,4,8,4,(1,2),0.).eval()
        x=torch.randn(2,188,31)
        for index in range(4+5,4+180,6):x[:,index]=1.
        return model,x

    def test_shared_encoder_pooling_is_invariant_to_player_slot_order(self):
        model,x=self.fixture();changed=x.clone()
        tubes=x[:,4:184].reshape(2,2,3,5,6,31)
        changed[:,4:184]=tubes[:,:,[2,0,1]].reshape(2,180,31)
        with torch.no_grad():a=model.trajectory_embedding(x);b=model.trajectory_embedding(changed)
        torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)

    def test_unobserved_players_cannot_affect_team_embedding(self):
        model,x=self.fixture();x[:,4+17]=0.
        changed=x.clone();changed[:,4:34]=99.;changed[:,4+17]=0.
        with torch.no_grad():a=model.trajectory_embedding(x);b=model.trajectory_embedding(changed)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_all_missing_players_have_zero_embedding_and_finite_backward(self):
        model,x=self.fixture()
        for index in range(4+5,4+180,6):x[:,index]=0.
        embedded=model.trajectory_embedding(x)
        self.assertEqual(int(embedded.count_nonzero()),0)
        state,edge=model.forward_with_boundary(x);loss=state.square().mean()+edge.square().mean();loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_gradients_reach_trajectory_encoder_and_heads(self):
        model,x=self.fixture();model.train()
        state,edge=model.forward_with_boundary(x);loss=state.square().mean()+edge.square().mean();loss.backward()
        self.assertEqual(tuple(state.shape),(2,3,31));self.assertEqual(tuple(edge.shape),(2,31))
        self.assertGreater(float(model.encoder[0].weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.head[-1].weight.grad.abs().sum()),0.)

    def test_incompatible_feature_schema_is_rejected(self):
        with self.assertRaises(ValueError):PlayerTrajectoryCorrector(187,4,4)

    def test_complete_residual_model_is_portable_and_preserves_teacher_at_start(self):
        from possession.temporal import make_model,frame_loss
        from possession.player_trajectory_model import trajectory_mask_names
        from possession.boundary import boundary_loss
        torch.set_num_threads(2);torch.manual_seed(31)
        config=dict(architecture='residual_boundary',hidden_channels=8,dropout=0.,boundary_positive_weight=20.,
            residual_channels=8,residual_dilations=[1,2],residual_teacher_context=True,
            teacher_feature_count=4,residual_kind='player_trajectories',player_channels=4,unit_feature_names=trajectory_mask_names())
        model=make_model(184,config).eval();x=torch.randn(2,184,31)
        for index in range(9,184,6):x[:,index]=1.
        with torch.no_grad():
            a,q=model.teacher.forward_with_boundary(x[:,:4]);b,r=model.forward_with_boundary(x)
        torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(q,r,rtol=0,atol=0)
        model.train();labels=torch.randint(0,3,(2,31));optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True);state,edge=model.forward_with_boundary(x)
            loss=frame_loss(state,labels).mean()+.3*boundary_loss(edge,labels,20.);loss.backward()
            if step:self.assertGreater(float(model.corrector.encoder[0].weight.grad.abs().sum()),0.)
            self.assertTrue(all(p.grad is None for p in model.teacher.parameters()));optimizer.step()
        model.eval();copy=make_model(184,config).eval();copy.load_state_dict(model.state_dict())
        with torch.no_grad():a,q=model.forward_with_boundary(x);b,r=copy.forward_with_boundary(x)
        torch.testing.assert_close(a,b);torch.testing.assert_close(q,r)

    def test_model_requires_unscaled_observation_mask_policy(self):
        from possession.temporal import make_model
        with self.assertRaisesRegex(ValueError,'unscaled observation'):
            make_model(184,dict(architecture='residual_boundary',hidden_channels=8,
                residual_kind='player_trajectories',teacher_feature_count=4))
