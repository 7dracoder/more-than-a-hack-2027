import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from scripts.snapshot_local_run import verify_source_archive


class SourceSnapshotTests(unittest.TestCase):
    def make_archive(self,run,content=b'original'):
        with zipfile.ZipFile(run/'source.zip','w') as archive:archive.writestr('training.py',content)
        manifest=dict(files={'training.py':hashlib.sha256(content).hexdigest()},
                      source_zip_sha256=hashlib.sha256((run/'source.zip').read_bytes()).hexdigest())
        (run/'source_manifest.json').write_text(json.dumps(manifest))
        return dict(sources={'training.py':hashlib.sha256(b'original').hexdigest()})

    def test_historical_archive_stays_valid_when_working_copy_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);fixed=self.make_archive(run)
            (run/'training.py').write_bytes(b'new working copy')
            before=(run/'source.zip').read_bytes();result=verify_source_archive(run,fixed)
            self.assertTrue(result['source_archive_verified'])
            self.assertEqual(before,(run/'source.zip').read_bytes())

    def test_archive_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);fixed=self.make_archive(run)
            with (run/'source.zip').open('ab') as file:file.write(b'changed')
            with self.assertRaisesRegex(ValueError,'archive bytes'):verify_source_archive(run,fixed)

    def test_rebuilt_archive_cannot_substitute_new_sources_for_old_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);fixed=self.make_archive(run,b'new source')
            with self.assertRaisesRegex(ValueError,'run protocol'):verify_source_archive(run,fixed)

    def test_manifest_is_required_without_overwriting_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);fixed=self.make_archive(run)
            (run/'source_manifest.json').unlink()
            with self.assertRaises(FileNotFoundError):verify_source_archive(run,fixed)
