import unittest
import numpy as np
import torch
from possession.boundary import BoundaryHybrid
from possession.ensemble import FeatureEnsemble
from possession.temporal import PossessionBiGRU,symmetric_probability


class FeatureEnsembleTests(unittest.TestCase):
    def test_transition_mix_can_differ_from_state_mix(self):
        torch.set_num_threads(2);torch.manual_seed(51)
        names=['home_a','away_a','ball_seen']
        states=[dict(names=names,mean=np.zeros(3),scale=np.ones(3),ensemble_boundary_weight=1.),
                dict(names=names,mean=np.zeros(3),scale=np.ones(3),ensemble_boundary_weight=0.)]
        models=[BoundaryHybrid(3,8,0.),BoundaryHybrid(3,8,0.)]
        x=np.random.default_rng(51).normal(size=(101,3)).astype(np.float32)
        a=symmetric_probability(models[0],x,names,np.zeros(3),np.ones(3),'cpu')
        b=symmetric_probability(models[1],x,names,np.zeros(3),np.ones(3),'cpu')
        ensemble=FeatureEnsemble(models,names,states,[.25,.75])
        result=symmetric_probability(ensemble,x,names,np.zeros(3),np.ones(3),'cpu')
        np.testing.assert_allclose(result[:,:3],.25*a[:,:3]+.75*b[:,:3],atol=1e-6)
        np.testing.assert_allclose(result[:,3],a[:,3],atol=1e-6)

    def test_coarse_member_uses_original_grid_and_exact_hold_expansion(self):
        torch.set_num_threads(2); torch.manual_seed(13)
        names=['home_a','away_a','ball_seen']
        states=[dict(names=names,mean=np.zeros(3),scale=np.ones(3),config=dict(sample_step=2)),
                dict(names=names,mean=np.zeros(3),scale=np.ones(3),config=dict(sample_step=1))]
        models=[BoundaryHybrid(3,8,0.),BoundaryHybrid(3,8,0.)]
        x=np.random.default_rng(13).normal(size=(71,3)).astype(np.float32)
        coarse=symmetric_probability(models[0],x[::2],names,np.zeros(3),np.ones(3),'cpu')
        fine=symmetric_probability(models[1],x,names,np.zeros(3),np.ones(3),'cpu')
        ensemble=FeatureEnsemble(models,names,states,[.25,.75])
        actual=symmetric_probability(ensemble,x,names,np.zeros(3),np.ones(3),'cpu')
        expected=.25*np.repeat(coarse,2,axis=0)[:len(x)]+.75*fine
        np.testing.assert_allclose(actual,expected,atol=1e-6)

    def test_member_features_and_different_normalizations_are_preserved(self):
        torch.set_num_threads(2); torch.manual_seed(10)
        names = ['home_a','away_a','ball_seen','extra']
        # Large raw values catch accidental outer clipping before member scaling.
        x = (np.random.default_rng(10).normal(size=(70,4))*15).astype(np.float32)
        states = [dict(names=names[:3],mean=np.array([.2,.2,0.]),scale=np.array([2.,2.,1.])),
                  dict(names=names,mean=np.zeros(4),scale=np.ones(4)*1.5)]
        models = [PossessionBiGRU(3,8,0.),BoundaryHybrid(4,8,0.)]
        a = symmetric_probability(models[0],x[:,:3],states[0]['names'],states[0]['mean'],states[0]['scale'],'cpu')
        b = symmetric_probability(models[1],x,names,states[1]['mean'],states[1]['scale'],'cpu')
        model = FeatureEnsemble(models,names,states,[.25,.75])
        result = symmetric_probability(model,x,names,np.zeros(4),np.ones(4),'cpu')
        np.testing.assert_allclose(result[:,:3],.25*a+.75*b[:,:3],atol=1e-6)
        np.testing.assert_allclose(result[:,3],b[:,3],atol=1e-6)

    def test_feature_order_is_mapped_by_name(self):
        torch.set_num_threads(2); torch.manual_seed(11)
        names = ['home_a','away_a','ball_seen']
        state = dict(names=names,mean=np.zeros(3),scale=np.ones(3))
        model = PossessionBiGRU(3,8,0.)
        x = np.random.default_rng(11).normal(size=(50,3)).astype(np.float32)
        order = [2,0,1]
        outer_names = [names[i] for i in order]
        ensemble = FeatureEnsemble([model],outer_names,[state],[1.])
        p = symmetric_probability(model,x,names,np.zeros(3),np.ones(3),'cpu')
        q = symmetric_probability(ensemble,x[:,order],outer_names,np.zeros(3),np.ones(3),'cpu')
        np.testing.assert_allclose(p,q,atol=1e-6)
