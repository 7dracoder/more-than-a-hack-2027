import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import run_development_campaign as campaign


class DevelopmentCampaignTests(unittest.TestCase):
    def exercise(self,fresh_score,receipt=False):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root=Path(temporary.name);(root/'data/local-v08').mkdir(parents=True)
        for name,score in [('initial',.849),('after',.901)]:
            run=root/name;run.mkdir()
            (run/'selection.json').write_text(json.dumps(dict(validation=dict(spell_f1_mean=score))))
            (run/'selected.pt').write_bytes(b'frozen-checkpoint')
        if receipt:
            folder=root/'after/candidate';folder.mkdir()
            for name in ('CoreOrigin.csv','CoreOrigin.first-pass.csv'):
                (folder/name).write_bytes(b'frame,label\n0,H\n')
            (root/'after/reproducibility.json').write_text(json.dumps(dict(two_complete_passes_byte_identical=True,
                sha256=campaign.digest(folder/'CoreOrigin.csv'),selected_sha256=campaign.digest(root/'after/selected.pt') if receipt!='invalid' else 'changed')))
        plan=root/'plan.json'
        plan.write_text(json.dumps(dict(target_spell_f1=.9,scope='Fixed external development only',
            after_run='after',after_config='config.json',initial_best='initial',
            wait_for_feature_config='unused.json',runs=[])))
        calls=[]
        def execute(arguments,**kwargs):
            calls.append(arguments)
            if 'scripts/audit_local_errors.py' in arguments:
                out=Path(arguments[arguments.index('--out')+1]);out.mkdir()
                (out/'audit.json').write_text(json.dumps(dict(scores=dict(spell_f1_mean=fresh_score))))
            elif 'scripts/run_local.py' in arguments:
                self.assertIn('--infer',arguments)
                out=Path(arguments[arguments.index('--out')+1])/'candidate';out.mkdir(exist_ok=True)
                (out/'CoreOrigin.csv').write_bytes(b'frame,label\n0,H\n')
        with patch.object(campaign,'ROOT',root),patch('sys.argv',['campaign','--plan',str(plan)]),patch.object(campaign.subprocess,'run',side_effect=execute):
            if receipt=='invalid':
                with self.assertRaisesRegex(ValueError,'repeat receipt'):
                    campaign.main()
            elif fresh_score==.901:
                campaign.main()
            else:
                with self.assertRaisesRegex(ValueError,'Fresh development audit'):
                    campaign.main()
        return root,calls,json.loads((root/'data/local-v08/campaign_status.json').read_text())

    def test_target_requires_fresh_audit_and_two_exports_before_stopping(self):
        root,calls,state=self.exercise(.901)
        self.assertEqual(state['status'],'target_met_on_development_only')
        self.assertEqual(sum('scripts/run_local.py' in call for call in calls),2)
        receipt=json.loads((root/'after/reproducibility.json').read_text())
        self.assertTrue(receipt['two_complete_passes_byte_identical'])
        self.assertIsNone(receipt['official_competition_score'])

    def test_inconsistent_fresh_score_fails_without_claiming_target(self):
        root,calls,state=self.exercise(.85)
        self.assertEqual(state['status'],'failed')
        self.assertFalse(state.get('target_met',False))
        self.assertFalse(any('scripts/run_local.py' in call for call in calls))

    def test_valid_receipt_reuses_identical_frozen_artifacts(self):
        root,calls,state=self.exercise(.901,receipt=True)
        self.assertEqual(state['status'],'target_met_on_development_only')
        self.assertFalse(any('scripts/run_local.py' in call for call in calls))

    def test_changed_checkpoint_receipt_cannot_be_reused(self):
        root,calls,state=self.exercise(.901,receipt='invalid')
        self.assertEqual(state['status'],'failed')
        self.assertFalse(state.get('target_met',False))
