import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scripts.report_local_progress import verified_target


class ProgressEvidenceTests(unittest.TestCase):
    def test_selected_target_without_audit_and_exports_is_not_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(verified_target(Path(tmp),dict(validation=dict(spell_f1_mean=.901))))

    def test_certificate_links_score_checkpoint_and_both_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);(run/'campaign-audit').mkdir();(run/'candidate').mkdir()
            (run/'selected.pt').write_bytes(b'synthetic-model')
            (run/'candidate/CoreOrigin.csv').write_bytes(b'frame,label\n0,H\n')
            (run/'candidate/CoreOrigin.first-pass.csv').write_bytes(b'frame,label\n0,H\n')
            checkpoint=hashlib.sha256(b'synthetic-model').hexdigest();csv=hashlib.sha256(b'frame,label\n0,H\n').hexdigest()
            selection=dict(validation=dict(spell_f1_mean=.901),checkpoint_sha256=checkpoint)
            (run/'campaign-audit/audit.json').write_text(json.dumps(dict(selected_checkpoint_sha256=checkpoint,scores=dict(spell_f1_mean=.901))))
            (run/'reproducibility.json').write_text(json.dumps(dict(selected_sha256=checkpoint,sha256=csv,two_complete_passes_byte_identical=True,rows=199319)))
            self.assertTrue(verified_target(run,selection))
            (run/'selected.pt').write_bytes(b'changed-model')
            self.assertFalse(verified_target(run,selection))
            (run/'selected.pt').write_bytes(b'synthetic-model')
            (run/'candidate/CoreOrigin.csv').write_bytes(b'changed-output')
            self.assertFalse(verified_target(run,selection))
