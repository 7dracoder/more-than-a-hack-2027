import hashlib,json,tempfile,unittest,zipfile
from pathlib import Path
import torch
from possession.temporal import make_model
from possession.player_trajectory_model import PlayerTrajectoryCorrector,trajectory_mask_names
from possession.features_v25 import OFFSETS,COMPONENTS
from possession.pretrained_encoder import initialize_encoder


class PretrainedEncoderTests(unittest.TestCase):
    def fixture(self,root):
        torch.set_num_threads(2);torch.manual_seed(81)
        weak=dict(train_matches=['101','102'],validation_matches=['103'],label_kind='provider_HA_weak',epochs=2,
            residual_channels=8,player_channels=4,residual_dilations=[1,2],player_pooling='attention')
        pretrained=PlayerTrajectoryCorrector(184,0,4,8,4,[1,2],0.,'attention')
        names=[f'{side}_tube{rank}_o{offset}_{component}' for side in ('home','away') for rank in (1,2,3) for offset in OFFSETS for component in COMPONENTS]
        state=dict(model=pretrained.state_dict(),config=weak,epoch=2,fingerprint='fixture',names=names)
        torch.save(state,root/'selected.pt');digest=hashlib.sha256((root/'selected.pt').read_bytes()).hexdigest()
        (root/'selection.json').write_text(json.dumps(dict(selection_rule='fixed_final_epoch',checkpoint_sha256=digest)))
        source=b'weak coordinate adapter';source_hash=hashlib.sha256(source).hexdigest()
        (root/'fixed_protocol.json').write_text(json.dumps(dict(config=weak,fingerprint='fixture',sources={'possession/skillcorner.py':source_hash})))
        with zipfile.ZipFile(root/'source.zip','w') as archive:archive.writestr('possession/skillcorner.py',source)
        (root/'source_manifest.json').write_text(json.dumps(dict(files={'possession/skillcorner.py':source_hash},source_zip_sha256=hashlib.sha256((root/'source.zip').read_bytes()).hexdigest())))
        config=dict(architecture='residual_boundary',hidden_channels=8,dropout=0.,boundary_positive_weight=20.,
            residual_channels=8,player_channels=4,residual_dilations=[1,2],player_pooling='attention',residual_dropout=0.,
            teacher_feature_count=4,residual_kind='player_trajectories',residual_teacher_context=True,
            unit_feature_names=trajectory_mask_names(),trajectory_physical_units=True,encoder_pretrain_run=str(root),
            encoder_pretrain_training_matches=weak['train_matches'],train_matches=['M1'],validation_matches=['V1'])
        model=make_model(184,config).eval()
        return model,['home_a','away_a','ball_seen','other']+names,config,dict(encoder_pretrain_checkpoint_sha256=digest),pretrained

    def test_transfer_preserves_teacher_and_initial_outputs_but_copies_representation_without_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model,names,config,record,pretrained=self.fixture(root)
            frozen={key:value.clone() for key,value in model.teacher.state_dict().items()};x=torch.randn(2,184,31)
            with torch.no_grad():a,q=model.forward_with_boundary(x)
            before=torch.get_rng_state().clone();receipt=initialize_encoder(model,names,config,record,root,'cpu')
            torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)
            with torch.no_grad():b,r=model.forward_with_boundary(x)
            torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(q,r,rtol=0,atol=0)
            for key,value in model.teacher.state_dict().items():torch.testing.assert_close(value,frozen[key],rtol=0,atol=0)
            torch.testing.assert_close(model.corrector.encoder[0].weight,pretrained.encoder[0].weight,rtol=0,atol=0)
            self.assertEqual(int(model.corrector.head[-1].weight.count_nonzero()),0)
            self.assertEqual(int(model.corrector.project.weight[:,-4:].count_nonzero()),0)
            self.assertFalse(receipt['prediction_heads_transferred'])

    def test_undeclared_weak_cohort_or_changed_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model,names,config,record,_=self.fixture(root);config['encoder_pretrain_training_matches']=[]
            with self.assertRaisesRegex(ValueError,'declared cohort'):initialize_encoder(model,names,config,record,root,'cpu')
            config['encoder_pretrain_training_matches']=['101','102']
            with (root/'selected.pt').open('ab') as output:output.write(b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint differs'):initialize_encoder(model,names,config,record,root,'cpu')
