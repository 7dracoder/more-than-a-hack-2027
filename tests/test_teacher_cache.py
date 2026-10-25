import unittest
import numpy as np
import torch
from possession.temporal import make_model,symmetric_probability,make_weight_average
from possession.teacher_cache import build_teacher_cache


class TeacherCacheTests(unittest.TestCase):
    def test_expanded_trajectory_inputs_and_both_player_pooling_modes_match_full_inference(self):
        from possession.features_v25 import OFFSETS,COMPONENTS
        from possession.player_trajectory_model import trajectory_mask_names
        for pooling in ('mean_max','attention'):
            parent,_,_,_,_,config=self.fixture()
            config.update(residual_kind='player_trajectories',teacher_feature_count=3,player_channels=4,
                unit_feature_names=trajectory_mask_names(),player_pooling=pooling)
            model=make_model(183,config).eval()
            with torch.no_grad():model.corrector.head[-1].weight.normal_();model.corrector.boundary[-1].weight.normal_()
            names=['home_a','away_a','ball_seen']+[f'{side}_tube{rank}_o{offset}_{component}'
                for side in ('home','away') for rank in (1,2,3) for offset in OFFSETS for component in COMPONENTS]
            raw=np.random.default_rng(7).normal(size=(71,183)).astype(np.float32)
            for name in trajectory_mask_names():raw[:,names.index(name)]=1.
            mean=np.zeros(183,np.float32);scale=np.ones(183,np.float32)
            cache=build_teacher_cache(model,raw,names,mean,scale,'cpu')
            np.testing.assert_array_equal(symmetric_probability(model,raw,names,mean,scale,'cpu'),
                symmetric_probability(model,raw,names,mean,scale,'cpu',teacher_cache=cache))

    def fixture(self):
        torch.set_num_threads(2);torch.manual_seed(61)
        config=dict(architecture='residual_boundary',hidden_channels=8,dropout=.2,boundary_positive_weight=20.,
            residual_channels=4,residual_dilations=[1,2],residual_teacher_context=True,ema_decay=.995)
        parent=dict(config)
        config['teacher_model_config']=parent
        model=make_model(3,config).eval()
        with torch.no_grad():model.corrector.head[-1].weight.normal_();model.corrector.boundary[-1].weight.normal_()
        matrix=np.random.default_rng(11).normal(size=(71,3)).astype(np.float32)
        names=['home_a','away_a','ball_seen'];mean=np.array([.2,.2,0.],np.float32);scale=np.array([2.,2.,1.],np.float32)
        return model,matrix,names,mean,scale,config

    def test_cached_and_complete_nested_teacher_probabilities_are_identical_without_rng_changes(self):
        model,x,names,mean,scale,_=self.fixture();before=torch.get_rng_state().clone()
        cache=build_teacher_cache(model,x,names,mean,scale,'cpu')
        np.testing.assert_array_equal(symmetric_probability(model,x,names,mean,scale,'cpu'),
            symmetric_probability(model,x,names,mean,scale,'cpu',teacher_cache=cache))
        torch.testing.assert_close(torch.get_rng_state(),before,rtol=0,atol=0)
        self.assertFalse(cache.state.flags.writeable);self.assertFalse(cache.edge.flags.writeable)

    def test_changed_input_normalization_and_teacher_are_rejected(self):
        model,x,names,mean,scale,_=self.fixture();cache=build_teacher_cache(model,x,names,mean,scale,'cpu')
        changed=x.copy();changed[3,0]+=1
        for raw,mu,sigma in ((changed,mean,scale),(x,mean+1,scale),(x,mean,scale+1)):
            with self.assertRaisesRegex(ValueError,'changed'):
                symmetric_probability(model,raw,names,mu,sigma,'cpu',teacher_cache=cache)
        with torch.no_grad():next(model.teacher.parameters()).add_(.001)
        with self.assertRaisesRegex(ValueError,'changed'):
            symmetric_probability(model,x,names,mean,scale,'cpu',teacher_cache=cache)

    def test_corrector_and_ema_updates_can_reuse_the_same_frozen_teacher(self):
        model,x,names,mean,scale,config=self.fixture();cache=build_teacher_cache(model,x,names,mean,scale,'cpu')
        average=make_weight_average(model,config)
        for _ in range(3):
            with torch.no_grad():model.corrector.head[-1].weight.add_(.01)
            average.update_parameters(model)
            for candidate in (model,average.module):
                np.testing.assert_array_equal(symmetric_probability(candidate,x,names,mean,scale,'cpu'),
                    symmetric_probability(candidate,x,names,mean,scale,'cpu',teacher_cache=cache))

    def test_trainable_teacher_and_training_cache_bypass_are_rejected(self):
        model,x,names,mean,scale,_=self.fixture();cache=build_teacher_cache(model,x,names,mean,scale,'cpu')
        model.train()
        with self.assertRaisesRegex(ValueError,'evaluation only'):
            model.forward_with_cached_teacher(torch.zeros(2,3,71),torch.from_numpy(cache.state.copy()),torch.from_numpy(cache.edge.copy()))
        model.teacher.requires_grad_(True)
        with self.assertRaisesRegex(ValueError,'completely frozen'):build_teacher_cache(model,x,names,mean,scale,'cpu')

    def test_changed_teacher_architecture_calibration_or_mode_is_rejected(self):
        model,x,names,mean,scale,_=self.fixture();cache=build_teacher_cache(model,x,names,mean,scale,'cpu')
        model.teacher.positive_weight=21.
        with self.assertRaisesRegex(ValueError,'changed'):cache.outputs(model,x,names,mean,scale,'cpu')
        model.teacher.positive_weight=20.;model.teacher.teacher.train()
        with self.assertRaisesRegex(ValueError,'evaluation mode'):cache.outputs(model,x,names,mean,scale,'cpu')
