"""Create a separate checksum-pinned v0.2 notebook; retain the executed v0.1 snapshot."""

import argparse
import json
from pathlib import Path

from build_colab import ROOT, bundle, digest, notebook


def build(bundle_id, reuse=False):
    target = ROOT / 'data/drive-transfer/BlindSpot_v02_project.zip'
    target.parent.mkdir(parents=True, exist_ok=True)
    checksum = digest(target) if reuse else bundle(target)
    output = ROOT / 'notebooks/BlindSpot_v02_Colab.ipynb'
    notebook(bundle_id, checksum, output)
    obj = json.loads(output.read_text(encoding='utf-8'))
    obj['metadata']['colab']['name'] = output.name
    for cell in obj['cells']:
        text = ''.join(cell['source'])
        if cell['cell_type'] == 'markdown':
            if text.startswith('# BlindSpot:'):
                text = '# BlindSpot v0.2: audit, evaluate, review\n\nCPU-only follow-up: whole-possession development split, fixed stoppage proposal, error analysis, label-independent query benchmark and hash-verified stage checkpoints. Public footage has already been examined; these results are development evidence, not an independent test. Video and broadcast are optional and are not downloaded by default.\n'
            elif text.startswith('## Run the experiment'):
                text = '## Run the follow-up\n\nComplete H/A/D runs stay together. A minimum 250-frame embargo is rounded out to group boundaries. Previously observed public scores are not a fresh holdout. The fixed stoppage candidate stays separate. Completed stages resume only when source/config/input provenance and artifact hashes match.\n'
            elif text.startswith('## Explore evidence'):
                text = '## Inspect development findings\n\nThe dashboard reports grouped development scores, stoppage errors and query error enrichment. Error discovery is not measured correction or reviewer time saved. The local video review lab supports synchronized clips; this default cloud run has no video.\n'
        else:
            text = text.replace("'/content/blindspot'", "'/content/blindspot-v02'").replace("'/content/BlindSpot_project.zip'", "'/content/BlindSpot_v02_project.zip'")
            if "'scripts/run_experiment.py'" in text:
                text = "subprocess.run([sys.executable, 'scripts/run_followup.py', '--out', str(OUT), '--resume', '--archive'], check=True)\n"
            elif text.startswith('OUT = PROJECT'):
                text = "subprocess.run([sys.executable, 'evaluation/validate_submission.py', str(OUT / 'predictions/evidence_v1.csv')], check=True)\nresults = json.loads((OUT / 'validation/scores.json').read_text())\nimport pandas as pd\ndisplay(pd.DataFrame(results['grouped_development']).T[['macro_f1','spell_f1_mean','scope']])\nprint(json.dumps(json.loads((OUT / 'validation/split.json').read_text()), indent=2))\nprint(json.dumps(json.loads((OUT / 'query_evaluation/query_benchmark.json').read_text())['random_30_seeds'], indent=2))\nprint(json.dumps(json.loads((OUT / 'predictions/checkpoint.json').read_text())['artifacts'], indent=2))\n"
            elif "display(HTML" in text:
                text = text.replace("'report.html'", "'index.html'")
            elif 'result_zip = shutil.make_archive' in text:
                text = text.replace("'/content/BlindSpot_results'", "'/content/BlindSpot_v02_results'")
        cell['source'] = text.splitlines(keepends=True)
    persist = [
        {'cell_type': 'markdown', 'metadata': {}, 'source': ['## Optional durable checkpoints\n', 'Default: temporary runtime storage, with result ZIP download at the end. To preserve stages across a runtime reset, set PERSIST_TO_DRIVE=True before running this cell and authorize your own Google Drive mount. This keeps generated results in your own Drive, without changing sharing. Downloaded source/inputs remain checksum-verified.\n']},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [
            'PERSIST_TO_DRIVE = False\n',
            'if PERSIST_TO_DRIVE:\n', '    from google.colab import drive\n', "    drive.mount('/content/drive')\n",
            "    OUT = Path('/content/drive/MyDrive/BlindSpot/checkpoints/v02')\n", 'else:\n',
            "    OUT = PROJECT / 'data/experiments/blindspot-v2'\n", 'OUT.mkdir(parents=True, exist_ok=True)\n',
            "print('Checkpoint destination:', OUT, '| durable:', PERSIST_TO_DRIVE)\n"]}]
    obj['cells'][4:4] = persist
    output.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'bundle': str(target), 'sha256': checksum, 'notebook': str(output)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-id', default='SET_AFTER_DRIVE_UPLOAD')
    parser.add_argument('--reuse-bundle', action='store_true')
    args = parser.parse_args()
    build(args.bundle_id, args.reuse_bundle)
