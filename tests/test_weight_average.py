import copy
import unittest
import torch
from possession.temporal import make_weight_average


class WeightAverageTests(unittest.TestCase):
    def test_shadow_does_not_change_primary_mode_weights_or_rng(self):
        torch.manual_seed(20)
        model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.2),torch.nn.Linear(4,2))
        before=copy.deepcopy(model.state_dict()); rng=torch.get_rng_state().clone()
        average=make_weight_average(model,dict(ema_decay=.9))
        average.update_parameters(model)
        self.assertTrue(model.training); self.assertFalse(average.module.training)
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        for name,value in model.state_dict().items():
            torch.testing.assert_close(value,before[name],rtol=0,atol=0)
        with torch.no_grad():
            next(average.module.parameters()).add_(1.)
        self.assertFalse(torch.equal(next(average.module.parameters()),next(model.parameters())))

    def test_resume_preserves_average_trajectory_and_update_counter(self):
        torch.manual_seed(21)
        model=torch.nn.Linear(3,2)
        optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9)
        average=make_weight_average(model,dict(ema_decay=.8))
        x,y=torch.randn(8,3),torch.arange(8)%2
        def step(m,o,a):
            o.zero_grad(); torch.nn.functional.cross_entropy(m(x),y).backward(); o.step(); a.update_parameters(m)
        for _ in range(3): step(model,optimizer,average)
        saved=copy.deepcopy(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),average=average.state_dict()))
        for _ in range(4): step(model,optimizer,average)
        resumed=torch.nn.Linear(3,2); resumed.load_state_dict(saved['model'])
        opt=torch.optim.SGD(resumed.parameters(),lr=.1,momentum=.9); opt.load_state_dict(saved['optimizer'])
        avg=make_weight_average(resumed,dict(ema_decay=.8)); avg.load_state_dict(saved['average'])
        for _ in range(4): step(resumed,opt,avg)
        self.assertEqual(int(avg.n_averaged),7)
        for name,value in average.state_dict().items():
            torch.testing.assert_close(value,avg.state_dict()[name],rtol=0,atol=0)

    def test_averaging_is_optional_and_invalid_decay_is_rejected(self):
        model=torch.nn.Linear(3,2)
        self.assertIsNone(make_weight_average(model,{}))
        with self.assertRaises(ValueError): make_weight_average(model,dict(ema_decay=1.))
