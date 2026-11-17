"""Check repeated frozen exports and official format without possession scoring."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from possession.data import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--folders', nargs='+', required=True)
    args = parser.parse_args()
    if len(args.folders) < 3:
        raise ValueError('Need three completed exports')
    manifests = []
    for folder in args.folders:
        directory = args.study / folder
        record = json.loads((directory / 'manifest.json').read_text())
        assert record['rows'] == 199319
        assert record['source_frozen_throughout'] and record['training_data_and_truth_guard_verified']
        assert not record['unexpected_forbidden_access']
        assert not record['training_on_provided_match'] and not record['public_or_private_possession_truth_read']
        for filename, key in [('raw.csv', 'raw_sha256'), ('CoreOrigin.csv', 'output_sha256')]:
            assert sha256(directory / filename) == record[key]
        for path, digest in record['source_sha256'].items():
            assert sha256(ROOT / path) == digest, path
        with (directory / 'CoreOrigin.csv').open(newline='') as stream:
            rows = csv.reader(stream)
            assert next(rows) == ['frame', 'label']
            count = 0
            for count, row in enumerate(rows, 1):
                assert len(row) == 2 and row[0] == str(count - 1) and row[1] in {'H', 'A', 'D', 'X'}
            assert count == record['rows']
        manifests.append(record)
    for key in ['raw_sha256', 'output_sha256', 'selected_model_sha256', 'selection_sha256',
                'parent_sha256', 'input_sha256', 'ball_hypotheses_sha256', 'source_sha256']:
        assert all(m[key] == manifests[0][key] for m in manifests), key
    original = json.loads((ROOT / 'models/coreorigin-video-v1/fixed_protocol.json').read_text())
    for path, digest in original['sources'].items():
        assert sha256(ROOT / path) == digest, path
    previous = json.loads((ROOT / 'data/research/observation-gate-20261114/verification.json').read_text())
    preserved = previous['preserved_v11_and_official_sources']
    # This explicit four-file receipt excludes public possession labels.
    assert len(preserved) == 4 and not any('gt' in Path(p).parts for p in preserved)
    for path, digest in preserved.items():
        assert sha256(ROOT / path) == digest, path
    command = [sys.executable, str(ROOT / 'evaluation/validate_submission.py'),
               str(args.study / args.folders[0] / 'CoreOrigin.csv')]
    checked = subprocess.run(command, capture_output=True, text=True, check=True)
    (args.study / 'official_format_validation.txt').write_text(checked.stdout, encoding='utf-8')
    manifest_path = args.model / 'model_manifest.json'
    bundle = json.loads(manifest_path.read_text())
    for name, digest in bundle['files'].items():
        assert sha256(args.model / name) == digest, name
    bundle['files'].update({p.name:sha256(p) for p in args.model.iterdir()
                            if p.is_file() and p.name != 'model_manifest.json'})
    bundle['source_sha256'] = {**original['sources'], **manifests[0]['source_sha256']}
    bundle['expected_csv_sha256'] = manifests[0]['output_sha256']
    manifest_path.write_text(json.dumps(bundle, indent=2) + '\n', encoding='utf-8')
    result = dict(complete=True, rows=199319, folders=args.folders,
        raw_and_final_byte_identical=True, packaged_inference_reproduced=True,
        training_data_and_truth_blocked=True, unexpected_forbidden_access=[],
        source_frozen_throughout=True, parent_original_source_checksums_verified=True,
        official_format_validator_exit_code=checked.returncode,
        output_sha256=manifests[0]['output_sha256'], bundle_manifest_sha256=sha256(manifest_path),
        preserved_v11_and_official_sources=preserved, public_possession_scoring_performed=False,
        full_match_spell_f1=None, competition_target_verified=False)
    (args.study / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(complete=True, output_sha256=result['output_sha256'])), flush=True)


if __name__ == '__main__':
    main()
