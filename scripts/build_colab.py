"""Package only source/public benchmark files and create the project Colab notebook."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "BAR-ATH_tactical_atd.txt": "1JEPmnLYax4dnVh1ZlInJAH5calmS1tIz",
    "BAR-ATH_tactical_atd.xml": "1WBfy2sOwmJ2cX2w9XJq6y4VXo_kxKHjX",
    "BAR-ATH_tactical_homography.csv": "1tuq2EGFmfvnXl8u5dU7JLoN3eNbrWwDJ",
    "BAR-ATH_tactical_smart_tagging.xml": "1SvfntIRtFh6i7KAK1gVJk1NQbS_lus2K",
}


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def bundle(target):
    selected = [ROOT / name for name in ("games.py", "requirements.txt", "requirements-colab.txt",
                                         "README.md", "RESEARCH.md", "WORK_DONE.md", "LICENSE",
                                         "DATA.md", "EVALUATION.md", "CHALLENGE.md", "RULES.md")]
    for folder in ("possession", "scripts", "tests", "starter", "evaluation"):
        selected.extend((ROOT / folder).glob("*.py"))
    selected.extend((ROOT / "configs").glob("*.json"))
    selected.extend((ROOT / "evaluation/gt").glob("*_public*"))
    selected.extend((ROOT / "evaluation/results").glob("baseline_*.csv"))
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(selected)):
            if not path.is_file():
                continue
            # Fixed ZIP timestamps make identical code bundles reproducible.
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 10, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return digest(target)


def notebook(bundle_id, bundle_hash, output):
    cells = []
    def md(text):
        cells.append({"cell_type":"markdown", "metadata":{}, "source":text.splitlines(keepends=True)})
    def code(text):
        cells.append({"cell_type":"code", "metadata":{}, "execution_count":None, "outputs":[], "source":text.splitlines(keepends=True)})
    md("# BlindSpot: possession under missing evidence\n\nCPU-first research pipeline: alternative histories, explicit missingness, camera evidence, fixed ablations, and a temporally separated calibration experiment. No paid GPU or hosted LLM is required.\n\nRun cells in order. Only tactical tracking inputs are downloaded by default (~220 MB), not the 19 GB of video. Results are exploratory; public labels cover only five minutes. The calibration model is a separate experiment; the main CSV stays label-free.")
    code("%pip -q install numpy==2.2.6 pandas==2.2.3 gdown==5.2.0\n")
    code(f'''from pathlib import Path
import hashlib, json, os, zipfile, subprocess, sys
import gdown

PROJECT = Path('/content/blindspot')
PROJECT.mkdir(exist_ok=True)
BUNDLE_ID = {bundle_id!r}
BUNDLE_SHA256 = {bundle_hash!r}
def sha256(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()
package = Path('/content/BlindSpot_project.zip')
if not package.exists() or sha256(package) != BUNDLE_SHA256:
    result = gdown.download(id=BUNDLE_ID, output=str(package), quiet=False)
    if result is None:
        raise RuntimeError('Bundle download failed; check sharing or Drive bandwidth limits')
assert sha256(package) == BUNDLE_SHA256, 'Project bundle checksum mismatch'
with zipfile.ZipFile(package) as archive:
    for entry in archive.infolist():
        if not (PROJECT / entry.filename).resolve().is_relative_to(PROJECT.resolve()):
            raise ValueError('Unsafe bundle path')
    archive.extractall(PROJECT)
os.chdir(PROJECT)
print('Verified source bundle:', PROJECT)
''')
    expected = {name: digest(ROOT / "data/tactical" / name) for name in ASSETS}
    code(f'''ASSETS = {ASSETS!r}
EXPECTED = {expected!r}
data_dir = PROJECT / 'data/tactical'
data_dir.mkdir(parents=True, exist_ok=True)
for name, file_id in ASSETS.items():
    path = data_dir / name
    if not path.exists() or sha256(path) != EXPECTED[name]:
        result = gdown.download(id=file_id, output=str(path), quiet=False)
        if result is None:
            raise RuntimeError('Download failed: ' + name)
    assert sha256(path) == EXPECTED[name], 'Dataset checksum mismatch: ' + name
    print('Verified', name, path.stat().st_size, 'bytes')
''')
    md("## Mathematical and data invariants\n\nTests check exact decoding against exhaustive paths, posterior marginals, reproducibility, slot-order invariance, team symmetry, halftime resets, and separation of training labels.")
    code("subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], check=True)\n")
    md("## Run the experiment\n\nThe optional calibration uses frames 11185–14184 for training, leaves a 250-frame buffer, and evaluates frames 14435–18684. Its five-minute descriptive score includes training data and must not be treated as held-out accuracy. No parameter sweep is performed. Check runtime-local storage and save the output archive after execution.")
    code("subprocess.run([sys.executable, 'scripts/run_experiment.py', '--ablations', '--train-public'], check=True)\n")
    code("OUT = PROJECT / 'data/experiments/blindspot-v1'\nsubprocess.run([sys.executable, 'evaluation/validate_submission.py', str(OUT / 'blindspot.csv')], check=True)\nresults = json.loads((OUT / 'scores.json').read_text())\nimport pandas as pd\ndisplay(pd.DataFrame(results).T[['macro_f1','spell_f1_mean','n_pred_spells','scope']])\n")
    md("## Explore evidence\n\nThe self-contained HTML report includes sampled state histories, missing-ball observations, model uncertainty, source sensitivity, query selection, and scoring provenance. Agreement is conditional on the model, not a correctness certificate. Video motion measurements require the optional video input and dependencies; the default run does not download videos.")
    code("from IPython.display import HTML, display\ndisplay(HTML((OUT / 'report.html').read_text()))\n")
    md("## Save results before disconnecting\n\nThe archive contains the CSV, trained calibration weights, run manifest, aggregate scores, gap histories, boundary receipts, query controls, and interactive report. Keep match-derived diagnostics within the permitted sharing scope.")
    code("import shutil\nfrom google.colab import files\nresult_zip = shutil.make_archive('/content/BlindSpot_results', 'zip', OUT)\nfiles.download(result_zip)\n")
    obj = {"nbformat":4, "nbformat_minor":5, "metadata":{"colab":{"name":"BlindSpot_Colab.ipynb"}, "kernelspec":{"display_name":"Python 3", "language":"python", "name":"python3"}}, "cells":cells}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id", default="SET_AFTER_DRIVE_UPLOAD")
    parser.add_argument("--reuse-bundle", action="store_true")
    args = parser.parse_args()
    target = ROOT / "data/drive-transfer/BlindSpot_project.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    checksum = digest(target) if args.reuse_bundle else bundle(target)
    output = ROOT / "notebooks/BlindSpot_Colab.ipynb"
    notebook(args.bundle_id, checksum, output)
    print(json.dumps({"bundle":str(target), "sha256":checksum, "notebook":str(output), "bytes":target.stat().st_size},indent=2))
