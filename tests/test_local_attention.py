import unittest
import numpy as np
import torch
from possession.local_attention import PossessionLocalAttention,RecurrentAttention
from possession.temporal import symmetric_probability,frame_loss


class LocalAttentionTests(unittest.TestCase):
    def test_recurrent_attention_never_resets_at_inference_chunks(self):
        torch.set_num_threads(2); torch.manual_seed(14)
        model=RecurrentAttention(3,8,2,1,3,0.)
        names=['home_a','away_a','ball_seen']
        x=np.random.default_rng(14).normal(size=(70,3)).astype(np.float32)
        args=(model,x,names,np.zeros(3),np.ones(3),'cpu')
        a=symmetric_probability(*args,chunk=9)
        b=symmetric_probability(*args,chunk=1000)
        np.testing.assert_array_equal(a,b)
        self.assertTrue(model.full_sequence)

    def test_overlap_is_exact_for_state_and_change_heads(self):
        torch.set_num_threads(2); torch.manual_seed(12)
        model=PossessionLocalAttention(3,8,2,2,3,0.)
        names=['home_a','away_a','ball_seen']
        x=np.random.default_rng(12).normal(size=(100,3)).astype(np.float32)
        args=(model,x,names,np.zeros(3),np.ones(3),'cpu')
        a=symmetric_probability(*args,chunk=17)
        b=symmetric_probability(*args,chunk=1000)
        np.testing.assert_allclose(a,b,atol=1e-6)
        self.assertEqual(a.shape,(100,4))

    def test_joint_training_has_finite_gradients_and_ignores_unknowns(self):
        from possession.boundary import boundary_loss
        torch.set_num_threads(2); torch.manual_seed(13)
        model=PossessionLocalAttention(3,8,2,2,3,0.)
        state,change=model.forward_with_boundary(torch.randn(2,3,30))
        labels=torch.randint(0,3,(2,30)); labels[:,10:15]=-100
        loss=frame_loss(state,labels,.5).mean()+.3*boundary_loss(change,labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_padding_cannot_attend_to_nonexistent_frames(self):
        torch.set_num_threads(2)
        model=PossessionLocalAttention(3,8,2,2,3,0.)
        state,change=model.forward_with_boundary(torch.ones(1,3,2))
        self.assertEqual(tuple(state.shape),(1,3,2))
        self.assertTrue(torch.isfinite(state).all()); self.assertTrue(torch.isfinite(change).all())
