import copy
import unittest
import torch
from possession.learning_rate import make_scheduler


class LearningRateTests(unittest.TestCase):
    def build(self):
        parameter=torch.nn.Parameter(torch.tensor([1.,-2.]))
        optimizer=torch.optim.AdamW([parameter],lr=.0005,weight_decay=.01)
        scheduler=make_scheduler(optimizer,dict(epochs=12,learning_rate_schedule='cosine',minimum_learning_rate=.00001))
        return parameter,optimizer,scheduler

    def advance(self,p,optimizer,scheduler,count):
        rates=[]
        for _ in range(count):
            rates.append(optimizer.param_groups[0]['lr'])
            optimizer.zero_grad();p.square().sum().backward();optimizer.step();scheduler.step()
        return rates

    def test_resume_preserves_rates_optimizer_and_parameter_trajectory(self):
        p,opt,scheduler=self.build();self.advance(p,opt,scheduler,5)
        saved=copy.deepcopy(dict(parameter=p.detach(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict()))
        rates=self.advance(p,opt,scheduler,7)
        q,other,restored=self.build()
        with torch.no_grad():q.copy_(saved['parameter'])
        other.load_state_dict(saved['optimizer']);restored.load_state_dict(saved['scheduler'])
        self.assertEqual(rates,self.advance(q,other,restored,7))
        torch.testing.assert_close(p,q,rtol=0.,atol=0.)
        self.assertAlmostEqual(other.param_groups[0]['lr'],.00001)

    def test_schedule_descends_to_declared_floor_without_consuming_rng(self):
        before=torch.get_rng_state().clone();p,opt,scheduler=self.build()
        torch.testing.assert_close(before,torch.get_rng_state(),rtol=0.,atol=0.)
        rates=self.advance(p,opt,scheduler,12)+[opt.param_groups[0]['lr']]
        self.assertTrue(all(a>=b for a,b in zip(rates,rates[1:])))
        self.assertEqual(rates[0],.0005)
        self.assertAlmostEqual(rates[-1],.00001)

    def test_constant_default_and_invalid_settings(self):
        p=torch.nn.Parameter(torch.zeros(1));opt=torch.optim.AdamW([p],lr=.0005)
        self.assertIsNone(make_scheduler(opt,{}))
        for settings in (dict(learning_rate_schedule='unknown'),dict(learning_rate_schedule='cosine',epochs=0),
                         dict(learning_rate_schedule='cosine',epochs=12,minimum_learning_rate=.1),
                         dict(learning_rate_schedule='cosine',epochs=12,minimum_learning_rate=float('nan'))):
            with self.assertRaises(ValueError):make_scheduler(opt,settings)
