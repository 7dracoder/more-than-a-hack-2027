"""Verify packaged weights using synthetic input while denying all data/ access.

This checks that old local experiment folders are not hidden runtime
dependencies. No match input, possession labels or video is opened.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'models/coreorigin-video-v1')
    parser.add_argument('--out', type=Path, default=ROOT / 'data/research/target080/bundle_isolation_check.json')
    args = parser.parse_args()
    denied_root = (ROOT / 'data').resolve()
    active, denied = True, []

    def audit(event, values):
        if active and event == 'open' and values and isinstance(values[0], (str, bytes, os.PathLike)):
            name = Path(os.fsdecode(values[0])).resolve()
            if name.is_relative_to(denied_root):
                denied.append(str(name))
                raise PermissionError('Local data/ access is prohibited in isolated model verification')

    sys.addaudithook(audit)
    # Confirm the guard operates before importing or loading project inference.
    try:
        with open(denied_root / 'README.md', 'rb'):
            raise AssertionError('Guard unexpectedly allowed a data/ read')
    except PermissionError:
        pass
    denied.clear()
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    import numpy as np
    import torch
    from scripts.run_local import selected_model, symmetric_probability

    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((args.bundle / 'model_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        if hashlib.sha256((args.bundle / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Packaged model artifact differs from its manifest')
    model, state = selected_model(args.bundle, torch.device('cpu'))
    matrix = np.zeros((128, len(state['names'])), np.float32)
    output = symmetric_probability(model, matrix, state['names'], state['mean'], state['scale'], torch.device('cpu'))
    assert output.shape in ((128, 3), (128, 4))
    assert np.isfinite(output).all()
    assert np.allclose(output[:, :3].sum(1), 1, atol=1e-5)
    active = False
    result = dict(purpose='Synthetic isolated model-bundle load and forward pass',
                  complete=True, data_access_guard_test_passed=True,
                  data_access_attempts_during_inference=denied,
                  feature_count=len(state['names']), output_shape=list(output.shape),
                  model_manifest_sha256=hashlib.sha256((args.bundle / 'model_manifest.json').read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  provided_match_data_read=False, possession_score=None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
