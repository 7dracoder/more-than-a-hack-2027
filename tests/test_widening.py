import unittest
import torch
from possession.boundary import BoundaryHybrid,boundary_loss
from possession.widening import widen_boundary_hybrid


class WideningTests(unittest.TestCase):
    def models(self,dtype=torch.double,dropout=0.):
        torch.set_num_threads(2);torch.manual_seed(13)
        parent=BoundaryHybrid(4,8,dropout,20.).to(dtype)
        child=BoundaryHybrid(4,16,dropout,20.).to(dtype)
        child.load_state_dict(widen_boundary_hybrid(parent.state_dict(),child,2))
        return parent,child

    def test_complete_state_and_boundary_eval_outputs_are_preserved(self):
        parent,child=self.models();parent.eval();child.eval()
        with torch.no_grad():
            x=torch.randn(3,4,67,dtype=torch.double)
            a,q=parent.forward_with_boundary(x);b,r=child.forward_with_boundary(x)
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(q,r,rtol=1e-10,atol=1e-10)

    def test_duplication_consumes_no_rng_and_does_not_mutate_parent(self):
        parent,child=self.models();weights=parent.state_dict();before={k:v.clone() for k,v in weights.items()}
        rng=torch.get_rng_state().clone();widened=widen_boundary_hybrid(weights,child,2)
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        for key in weights:
            torch.testing.assert_close(weights[key],before[key],rtol=0,atol=0)
            self.assertNotEqual(weights[key].data_ptr(),widened[key].data_ptr())

    def test_dropout_training_breaks_duplicate_symmetry_with_finite_gradients(self):
        parent,child=self.models(torch.float32,.2);child.train()
        optimizer=torch.optim.SGD(child.parameters(),lr=.01)
        x=torch.randn(3,4,31);labels=torch.randint(0,3,(3,31))
        state,edge=child.forward_with_boundary(x)
        loss=torch.nn.functional.cross_entropy(state,labels)+.3*boundary_loss(edge,labels,20.)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in child.parameters()))
        self.assertGreater(float((child.project.weight.grad[:8]-child.project.weight.grad[8:]).abs().max()),1e-8)
        optimizer.step()
        self.assertFalse(torch.equal(child.project.weight[:8],child.project.weight[8:]))

    def test_incompatible_architecture_factor_and_features_are_rejected(self):
        parent,child=self.models()
        for other,factor in ((BoundaryHybrid(4,16,0.,20.,2),2),(child,1),(BoundaryHybrid(5,16,0.,20.),2),(BoundaryHybrid(4,24,0.,20.),2)):
            with self.assertRaises(ValueError):widen_boundary_hybrid(parent.state_dict(),other,factor)

    def test_local_channels_can_widen_while_recurrent_width_stays_fixed(self):
        parent,_=self.models();parent.eval()
        child=BoundaryHybrid(4,16,0.,20.,recurrent_channels=8).double().eval()
        child.load_state_dict(widen_boundary_hybrid(parent.state_dict(),child,2))
        with torch.no_grad():
            x=torch.randn(2,4,47,dtype=torch.double)
            a,q=parent.forward_with_boundary(x);b,r=child.forward_with_boundary(x)
        self.assertEqual(child.recurrent.hidden_size,8)
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(q,r,rtol=1e-10,atol=1e-10)

    def test_unmatched_recurrent_width_is_rejected(self):
        parent,_=self.models()
        with self.assertRaisesRegex(ValueError,'recurrent hidden'):
            widen_boundary_hybrid(parent.state_dict(),BoundaryHybrid(4,16,0.,20.,recurrent_channels=12),2)
