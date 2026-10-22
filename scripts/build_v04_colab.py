"""Package the v0.4 source snapshot and generate notebooks/BlindSpot_v04_Colab.ipynb.

    python scripts/build_v04_colab.py

Writes data/drive-transfer/BlindSpot_v04_project.zip (source + public benchmark files only, no
match data) and a notebook pinned to that ZIP's SHA256. In Colab the user uploads the ZIP with a
file picker; the four tactical tracking inputs are fetched from the existing shared Drive copy and
checksum-verified. CPU runtime is enough.
"""

import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_colab import ASSETS, ROOT, digest  # noqa: E402

ZIP_NAME = 'BlindSpot_v04_project.zip'
DOCS = ('games.py', 'requirements.txt', 'requirements-v3.txt', 'README.md', 'LICENSE', 'DATA.md',
        'EVALUATION.md', 'CHALLENGE.md', 'RULES.md', 'CoreOrigin.md', 'AUDIT.md', 'EXPERIMENTS_v04.md',
        'WORK_DONE.md', 'RESEARCH.md', 'IMPLEMENTATION.md')


def bundle(target):
    selected = [ROOT / name for name in DOCS]
    for folder in ('possession', 'scripts', 'tests', 'starter', 'evaluation'):
        selected.extend((ROOT / folder).glob('*.py'))
    selected.extend((ROOT / 'configs').glob('*.json'))
    selected.extend((ROOT / 'tests').glob('*.js'))
    selected.extend((ROOT / 'evaluation/gt').glob('*_public*'))
    selected.extend((ROOT / 'evaluation/results').glob('baseline_*.csv'))
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(selected)):
            if path.is_file():
                info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026, 10, 21, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes())
    return digest(target)


def notebook(bundle_hash, output):
    cells = []

    def md(text):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)})

    def code(text):
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                      'source': text.splitlines(keepends=True)})

    md("""# BlindSpot v0.4: new possession features, held-out comparison

Run the cells top to bottom (**Runtime → Run all** works). A normal **CPU** runtime is enough; no GPU needed.

1. Installs pinned packages.
2. Asks you to upload `BlindSpot_v04_project.zip` (built locally in `data/drive-transfer/`). Its SHA256 is checked.
3. Downloads the four tactical tracking files (~220 MB, no video) from the shared Drive copy and checks their SHA256.
4. Runs the unit tests.
5. Runs every v0.4 experiment (about 15–25 min) and shows the results table.
6. Exports a candidate `CoreOrigin.csv` for the group you choose and validates it.
7. Downloads a ZIP with `EXPERIMENTS_v04.md` and the candidate. Send the results table back.

Scores reuse the same inspected public five minutes. Trying and selecting feature groups on this sample makes this exploratory development, not independent validation. New features use offline raw context that may extend across a full half; the 300-frame embargo excludes training labels, not every shared tracking observation. No private-match score is known.
""")
    code('%pip -q install numpy==2.2.6 pandas==2.2.3 scipy==1.17.1 scikit-learn==1.8.0 joblib==1.5.3 threadpoolctl==3.6.0 gdown==5.2.0\n')
    md('## 1. Upload and verify the source bundle\n\nA file picker appears. Choose `BlindSpot_v04_project.zip`. Skipped if already uploaded in this runtime.\n')
    code(f'''from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, zipfile

BUNDLE_SHA256 = {bundle_hash!r}
PROJECT = Path('/content/blindspot-v04')
package = Path('/content/{ZIP_NAME}')

def sha256(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

if not package.exists() or sha256(package) != BUNDLE_SHA256:
    from google.colab import files
    uploaded = files.upload()
    name = next(iter(uploaded))
    if Path(name).resolve() != package.resolve():
        shutil.move(name, package)
digest = sha256(package)
assert digest == BUNDLE_SHA256, f'Bundle checksum mismatch: got {{digest}}. Rebuild with scripts/build_v04_colab.py and use the matching notebook.'
PROJECT.mkdir(exist_ok=True)
with zipfile.ZipFile(package) as archive:
    for entry in archive.infolist():
        if not (PROJECT / entry.filename).resolve().is_relative_to(PROJECT.resolve()):
            raise ValueError('Unsafe bundle path: ' + entry.filename)
    archive.extractall(PROJECT)
os.chdir(PROJECT)
os.environ['LOKY_MAX_CPU_COUNT'] = '2'
print('Verified source bundle', digest, '->', PROJECT)
''')
    expected = {name: digest(ROOT / 'data/tactical' / name) for name in ASSETS}
    md('## 2. Download and verify the tactical tracking inputs\n')
    code(f'''import gdown
ASSETS = {ASSETS!r}
EXPECTED = {expected!r}
data_dir = PROJECT / 'data/tactical'
data_dir.mkdir(parents=True, exist_ok=True)
for name, file_id in ASSETS.items():
    path = data_dir / name
    if not path.exists() or sha256(path) != EXPECTED[name]:
        if gdown.download(id=file_id, output=str(path), quiet=False) is None:
            raise RuntimeError('Download failed (sharing or Drive quota): ' + name)
    assert sha256(path) == EXPECTED[name], 'Dataset checksum mismatch: ' + name
    print('Verified', name, path.stat().st_size, 'bytes')
''')
    md('## 3. Unit tests\n')
    code('''result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'], capture_output=True, text=True)
print(result.stdout[-3000:]); print(result.stderr[-3000:])
result.check_returncode()
''')
    md("""## 4. Run all experiments

Prints one line per feature group as it finishes. The first step reads the raw tracking once (a few minutes).
`baseline` must print whole-run **0.4457** and fixed-time **0.6200**; if not, stop and report it.
""")
    code('''def stream(command):
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
        status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)

stream([sys.executable, '-u', 'scripts/run_v04_experiments.py', '--note', 'Colab CPU run'])
stream([sys.executable, 'scripts/report_v04_results.py'])
''')
    md('## 5. Results\n')
    code('''from IPython.display import Markdown, display
text = (PROJECT / 'EXPERIMENTS_v04.md').read_text()
display(Markdown(text[text.index('## Results'):]))
''')
    md("""## 6. Export a candidate

The conservative development rule below retains `baseline` unless a group improves spell F1 by more than 0.06 in **both** protocols. This is a selection heuristic, not a statistical significance threshold. Selection still reuses the public sample.
Options: `baseline`, `contacts`, `asym`, `gap_out`, `ballmotion`, `shape`, `all`.
""")
    code('''scores = json.loads((PROJECT / 'data/experiments/blindspot-v04/scores.json').read_text())
baseline = scores['baseline']
eligible = [name for name, score in scores.items() if name != 'baseline' and all(
    score[protocol]['spell'] - baseline[protocol]['spell'] > .06
    for protocol in ('whole_run', 'fixed_time'))]
GROUP = max(eligible, key=lambda name: min(scores[name][p]['spell'] - baseline[p]['spell']
    for p in ('whole_run', 'fixed_time'))) if eligible else 'baseline'
print('Provisional export group:', GROUP, '| same-sample development selection')

stream([sys.executable, '-u', 'scripts/run_v04_experiments.py', '--candidate', GROUP])
candidate = PROJECT / f'data/experiments/blindspot-v04/{GROUP}/candidate/CoreOrigin.csv'
stream([sys.executable, 'evaluation/validate_submission.py', str(candidate)])
print('TRAINING FIT ONLY: the exported model used all public labels. The next score is not held-out validation.')
stream([sys.executable, 'evaluation/run_evaluation.py', '--gt', 'public', '--results', str(candidate.parent),
        '--out', str(PROJECT / 'data/experiments/blindspot-v04/training_fit_only'), '--no-detail'])
stream([sys.executable, 'scripts/report_v04_results.py'])
''')
    md('## 7. Download results\n\nThe ZIP holds the experiment log, fold predictions/scores, provenance, report and candidate. Keep these match-derived outputs within the permitted sharing scope.\n')
    code('''from google.colab import files
bundle_dir = Path('/content/BlindSpot_v04_results')
if bundle_dir.exists():
    shutil.rmtree(bundle_dir)
bundle_dir.mkdir()
shutil.copy(PROJECT / 'EXPERIMENTS_v04.md', bundle_dir)
shutil.copytree(PROJECT / 'data/experiments/blindspot-v04', bundle_dir / 'experiment')
files.download(shutil.make_archive('/content/BlindSpot_v04_results', 'zip', bundle_dir))
''')
    obj = {'nbformat': 4, 'nbformat_minor': 5,
           'metadata': {'colab': {'name': output.name}, 'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
           'cells': cells}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    target = ROOT / 'data/drive-transfer' / ZIP_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    checksum = bundle(target)
    output = ROOT / 'notebooks/BlindSpot_v04_Colab.ipynb'
    notebook(checksum, output)
    with zipfile.ZipFile(target) as archive:
        count = len(archive.namelist())
    print(json.dumps({'bundle': str(target), 'files': count, 'sha256': checksum, 'notebook': str(output)}, indent=2))
