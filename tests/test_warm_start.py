import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
import torch
from possession.boundary import BoundaryHybrid
from possession.warm_start import load_warm_start


class WarmStartTests(unittest.TestCase):
    def test_complete_composite_teacher_survives_prefix_extension_and_portable_reload(self):
        from possession.temporal import make_model
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,names,record,current=self.fixture(root)
            path=root/'selected.pt';state=torch.load(path,weights_only=False)
            parent_config=dict(state['config'],architecture='residual_boundary',hidden_channels=8,
                dropout=0.,residual_channels=4,residual_dilations=[1,2])
            parent=make_model(3,parent_config).eval()
            with torch.no_grad():parent.corrector.head[-1].bias.fill_(.75)
            state.update(config=parent_config,model=parent.state_dict());torch.save(state,path)
            fixed=json.loads((root/'fixed_protocol.json').read_text());fixed['config']=parent_config
            (root/'fixed_protocol.json').write_text(json.dumps(fixed))
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            (root/'selection.json').write_text(json.dumps(dict(checkpoint_sha256=digest)))
            record.update(warm_start_checkpoint_sha256=digest,prefix_parent_caches=record['external_caches'])
            record['external_caches']={'M1':'expanded'}
            current.update(architecture='residual_boundary',hidden_channels=8,dropout=0.,
                boundary_positive_weight=20.,residual_channels=4,residual_dilations=[1,2],
                allow_feature_prefix_expansion=True,teacher_feature_count=3,teacher_model_config=parent_config)
            child=make_model(4,current).eval()
            receipt=load_warm_start(child,names+['new'],np.zeros(4,np.float32),np.ones(4,np.float32),current,record,root,'cpu')
            self.assertTrue(receipt['teacher_excludes_new_features'])
            self.assertTrue(all(not p.requires_grad for p in child.teacher.parameters()))
            copy=make_model(4,current).eval();copy.load_state_dict(child.state_dict())
            x=torch.randn(2,4,31)
            with torch.no_grad():
                a,q=parent.forward_with_boundary(x[:,:3]);b,r=child.forward_with_boundary(x);c,s=copy.forward_with_boundary(x)
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6);torch.testing.assert_close(q,r,rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(b,c,rtol=0,atol=0);torch.testing.assert_close(r,s,rtol=0,atol=0)
            current['teacher_model_config']=dict(parent_config,dropout=.3)
            with self.assertRaisesRegex(ValueError,'teacher configuration'):
                load_warm_start(child,names+['new'],np.zeros(4,np.float32),np.ones(4,np.float32),current,record,root,'cpu')

    def fixture(self,root,training=None):
        torch.set_num_threads(2);torch.manual_seed(101)
        parent=BoundaryHybrid(3,8,0.,20.).eval();names=['home_a','away_a','ball_seen']
        config=dict(train_matches=training or ['M1'],validation_matches=['V1'],boundary_positive_weight=20.)
        source=b'external adapter';source_hash=hashlib.sha256(source).hexdigest()
        fixed=dict(config=config,fingerprint='fixture',sources={'possession/external.py':source_hash},external_caches={m:'cache' for m in config['train_matches']})
        (root/'fixed_protocol.json').write_text(json.dumps(fixed))
        torch.save(dict(model=parent.state_dict(),config=config,fingerprint='fixture',names=names,mean=np.zeros(3,np.float32),scale=np.ones(3,np.float32)),root/'selected.pt')
        digest=hashlib.sha256((root/'selected.pt').read_bytes()).hexdigest();(root/'selection.json').write_text(json.dumps(dict(checkpoint_sha256=digest)))
        with zipfile.ZipFile(root/'source.zip','w') as archive:archive.writestr('possession/external.py',source)
        (root/'source_manifest.json').write_text(json.dumps(dict(files={'possession/external.py':source_hash},source_zip_sha256=hashlib.sha256((root/'source.zip').read_bytes()).hexdigest())))
        record=dict(warm_start_checkpoint_sha256=digest,external_caches=fixed['external_caches'])
        current=dict(warm_start_run=str(root),train_matches=['M1'],validation_matches=['V1'])
        return parent,names,record,current

    def load(self,root,names,record,current,mean=None):
        child=BoundaryHybrid(3,8,0.,1.).eval()
        before=torch.get_rng_state().clone()
        receipt=load_warm_start(child,names,np.zeros(3,np.float32) if mean is None else mean,np.ones(3,np.float32),current,record,root,'cpu')
        torch.testing.assert_close(before,torch.get_rng_state(),rtol=0.,atol=0.)
        return child,receipt

    def test_calibration_rebase_preserves_state_and_effective_boundary_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent,names,record,current=self.fixture(root);child,receipt=self.load(root,names,record,current)
            with torch.no_grad():
                x=torch.randn(2,3,31);a,qa=parent.forward_with_boundary(x);b,qb=child.forward_with_boundary(x)
            torch.testing.assert_close(a,b,rtol=0.,atol=0.)
            torch.testing.assert_close(parent.calibrated_boundary(qa),child.calibrated_boundary(qb),rtol=1e-6,atol=1e-7)
            self.assertTrue(receipt['normalization_exact'] and receipt['optimizer_reset'])

    def test_parent_cannot_contain_held_out_training_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent,names,record,current=self.fixture(root,['V1'])
            with self.assertRaisesRegex(ValueError,'held-out'):self.load(root,names,record,current)

    def test_public_match_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent,names,record,current=self.fixture(root,['BAR-ATH'])
            with self.assertRaisesRegex(ValueError,'external-only'):self.load(root,names,record,current)

    def test_normalization_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent,names,record,current=self.fixture(root)
            with self.assertRaisesRegex(ValueError,'normalization'):self.load(root,names,record,current,np.ones(3,np.float32))

    def test_changed_checkpoint_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent,names,record,current=self.fixture(root)
            with (root/'selected.pt').open('ab') as file:file.write(b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint differs'):self.load(root,names,record,current)

    def test_explicit_prefix_extension_ignores_new_inputs_at_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent,names,record,current=self.fixture(root)
            current['allow_feature_prefix_expansion']=True
            record['prefix_parent_caches']=dict(record['external_caches']);record['external_caches']={'M1':'expanded'}
            child=BoundaryHybrid(4,8,0.,1.).eval()
            receipt=load_warm_start(child,names+['new_contact'],np.zeros(4,np.float32),np.ones(4,np.float32),current,record,root,'cpu')
            self.assertEqual(receipt['feature_prefix_extension'],1)
            self.assertEqual(int(child.project.weight[:,3:].count_nonzero()),0)
            with torch.no_grad():
                x=torch.randn(2,3,31);extra=torch.randn(2,1,31)
                expected=parent(x);a=child(torch.cat([x,extra],dim=1));b=child(torch.cat([x,extra*1000],dim=1))
            torch.testing.assert_close(a,expected,rtol=1e-6,atol=1e-7)
            torch.testing.assert_close(a,b,rtol=0.,atol=0.)

    def test_prefix_extension_rejects_unlinked_parent_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent,names,record,current=self.fixture(root);current['allow_feature_prefix_expansion']=True
            record['prefix_parent_caches']={'M1':'wrong'}
            with self.assertRaisesRegex(ValueError,'caches differ'):
                load_warm_start(BoundaryHybrid(4,8,0.,1.),names+['new_contact'],np.zeros(4,np.float32),np.ones(4,np.float32),current,record,root,'cpu')

    def test_feature_expansion_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent,names,record,current=self.fixture(root)
            with self.assertRaisesRegex(ValueError,'feature schema'):
                load_warm_start(BoundaryHybrid(4,8,0.,1.),names+['new_contact'],np.zeros(4,np.float32),np.ones(4,np.float32),current,record,root,'cpu')

    def test_verified_parent_can_widen_and_rebase_calibration_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent,names,record,current=self.fixture(root)
            current['warm_start_widen_factor']=2
            child=BoundaryHybrid(3,16,0.,1.).eval()
            receipt=load_warm_start(child,names,np.zeros(3,np.float32),np.ones(3,np.float32),current,record,root,'cpu')
            with torch.no_grad():
                x=torch.randn(2,3,31);a,q=parent.forward_with_boundary(x);b,r=child.forward_with_boundary(x)
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(parent.calibrated_boundary(q),child.calibrated_boundary(r),rtol=1e-5,atol=1e-6)
            self.assertEqual(receipt['hidden_channel_widen_factor'],2)

    def test_residual_model_initializes_only_its_frozen_verified_teacher(self):
        from possession.residual import ResidualBoundaryHybrid
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent,names,record,current=self.fixture(root)
            child=ResidualBoundaryHybrid(3,8,0.,20.,4,(1,2),.1).eval()
            load_warm_start(child,names,np.zeros(3,np.float32),np.ones(3,np.float32),current,record,root,'cpu')
            self.assertTrue(all(not p.requires_grad for p in child.teacher.parameters()))
            self.assertEqual(int(child.corrector.head[-1].weight.count_nonzero()),0)
            with torch.no_grad():
                x=torch.randn(2,3,31);a=parent(x);b=child(x)
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)

    def test_residual_teacher_can_exclude_a_verified_new_feature_suffix(self):
        from possession.residual import ResidualBoundaryHybrid
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent,names,record,current=self.fixture(root)
            current.update(allow_feature_prefix_expansion=True,teacher_feature_count=3)
            record['prefix_parent_caches']=dict(record['external_caches']);record['external_caches']={'M1':'expanded'}
            child=ResidualBoundaryHybrid(4,8,0.,20.,4,(1,2),.1,teacher_feature_count=3).eval()
            receipt=load_warm_start(child,names+['new_contact'],np.zeros(4,np.float32),np.ones(4,np.float32),current,record,root,'cpu')
            with torch.no_grad():
                x=torch.randn(2,4,31);a=parent(x[:,:3]);b=child(x)
                changed=x.clone();changed[:,3]*=1000.;c=child(changed)
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6);torch.testing.assert_close(b,c,rtol=0,atol=0)
            self.assertTrue(receipt['teacher_excludes_new_features'])
            self.assertFalse(receipt['new_input_weights_zero_initialized'])
