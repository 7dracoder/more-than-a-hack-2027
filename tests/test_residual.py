import unittest
import torch
from possession.boundary import BoundaryHybrid,boundary_loss
from possession.residual import ResidualBoundaryHybrid


class ResidualModelTests(unittest.TestCase):
    def models(self):
        torch.set_num_threads(2);torch.manual_seed(13)
        parent=BoundaryHybrid(4,8,.2,20.).eval()
        model=ResidualBoundaryHybrid(4,8,.2,20.,4,(1,2),.1)
        model.teacher.load_state_dict(parent.state_dict())
        return parent,model

    def test_zero_correction_preserves_both_teacher_outputs_exactly(self):
        parent,model=self.models();model.eval();x=torch.randn(2,4,61)
        with torch.no_grad():
            a,q=parent.forward_with_boundary(x);b,r=model.forward_with_boundary(x)
            frozen,frozen_edge=model.teacher.forward_with_boundary(x)
        torch.testing.assert_close(frozen,b,rtol=0,atol=0)
        torch.testing.assert_close(frozen_edge,r,rtol=0,atol=0)
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(q,r,rtol=1e-5,atol=1e-6)

    def test_training_mode_keeps_teacher_frozen_and_dropout_disabled(self):
        parent,model=self.models();model.train()
        self.assertFalse(model.teacher.training)
        self.assertTrue(model.corrector.training)
        self.assertTrue(all(not p.requires_grad for p in model.teacher.parameters()))
        x=torch.randn(2,4,31)
        with torch.no_grad():a=model.teacher(x);b=model.teacher(x)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_two_updates_train_correction_encoder_without_changing_teacher(self):
        parent,model=self.models();model.train()
        before={k:v.clone() for k,v in model.teacher.state_dict().items()}
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        x=torch.randn(3,4,31);labels=torch.randint(0,3,(3,31));labels[:,:3]=-100
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            state,edge=model.forward_with_boundary(x)
            loss=torch.nn.functional.cross_entropy(state,labels,ignore_index=-100)+.3*boundary_loss(edge,labels,20.)
            loss.backward()
            self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
            self.assertGreater(float(model.corrector.head[-1].weight.grad.abs().sum()),0.)
            self.assertGreater(float(model.corrector.boundary[-1].weight.grad.abs().sum()),0.)
            if step:self.assertGreater(float(model.corrector.project.weight.grad.abs().sum()),0.)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        for key,value in model.teacher.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)

    def test_complete_state_is_portable_without_a_parent_file(self):
        parent,model=self.models();model.eval()
        from possession.temporal import make_model
        copy=make_model(4,dict(architecture='residual_boundary',hidden_channels=8,dropout=.2,
            boundary_positive_weight=20.,residual_channels=4,residual_dilations=[1,2],residual_dropout=.1)).eval()
        copy.load_state_dict(model.state_dict());x=torch.randn(1,4,41)
        with torch.no_grad():
            a,q=model.forward_with_boundary(x);b,r=copy.forward_with_boundary(x)
        torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(q,r,rtol=0,atol=0)

    def test_optional_context_uses_only_detached_teacher_predictions(self):
        from possession.temporal import make_model
        torch.set_num_threads(2);torch.manual_seed(17)
        config=dict(architecture='residual_boundary',hidden_channels=8,dropout=.2,boundary_positive_weight=20.,
            residual_channels=4,residual_dilations=[1,2],residual_teacher_context=True)
        model=make_model(4,config).eval();x=torch.randn(2,4,31);captured=[]
        hook=model.corrector.project.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
        with torch.no_grad():
            state,edge=model.teacher.forward_with_boundary(x);a,q=model.forward_with_boundary(x)
        hook.remove()
        expected=torch.cat([state.softmax(1),model.teacher.calibrated_boundary(edge)[:,None]],1)
        torch.testing.assert_close(captured[0][:,:4],x,rtol=0,atol=0)
        torch.testing.assert_close(captured[0][:,4:],expected,rtol=0,atol=0)
        torch.testing.assert_close(a,state,rtol=0,atol=0);torch.testing.assert_close(q,edge,rtol=0,atol=0)
        copy=make_model(4,config).eval();copy.load_state_dict(model.state_dict())
        with torch.no_grad():b,r=copy.forward_with_boundary(x)
        torch.testing.assert_close(a,b);torch.testing.assert_close(q,r)
