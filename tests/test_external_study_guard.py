import json
from pathlib import Path
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]


def test_explicit_external_derivatives_are_read_only_and_other_studies_stay_blocked():
    base = (ROOT / 'data/research' / ('guard-test-' + uuid.uuid4().hex)).resolve()
    assert base.is_relative_to((ROOT / 'data/research').resolve())
    base.mkdir()
    allowed = base / 'external-derivative.txt'
    allowed.write_text('external test metadata', encoding='utf-8')
    source = '''
import json,sys
from pathlib import Path
from scripts.research.spatial_memory.external_guard import restrict_external,ROOT
base=Path(sys.argv[1]); out=base/'out'; out.mkdir()
allowed=base/'external-derivative.txt'
denied=restrict_external(out,[allowed])
assert allowed.read_text()=='external test metadata'
for path,mode in [(allowed,'w'),(ROOT/'data/research/other-study/__probe__.csv','r'),(ROOT/'evaluation/gt/__probe__.csv','r')]:
 try:
  open(path,mode)
  raise AssertionError('Expected forbidden access')
 except PermissionError:
  pass
print(json.dumps({'blocked':len(denied)}))
'''
    try:
        result = subprocess.run([sys.executable, '-c', source, str(base)], cwd=ROOT,
            capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout)['blocked'] == 3
        assert allowed.read_text() == 'external test metadata'
    finally:
        # Only the checked test directory and two explicitly created entries.
        allowed.unlink()
        (base / 'out').rmdir()
        base.rmdir()
