"""Verify two-feed provenance, repeat coordinate fusion and compare CPU exports.

No possession truth, pixel rendering, model fitting or score calculation.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from games import asset_paths, half_video_blocks
from evaluation import possession_metrics as pm
from possession.data import sha256
from scripts.research.broadcast.dual_feed import fuse_ball, write_json
from scripts.research.video_ball.fuse_detections import read_exact_tracking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--motion', type=Path, default=ROOT / 'models/coreorigin-video-v1/motion.json')
    parser.add_argument('--tactical-hypotheses', type=Path,
                        default=ROOT / 'data/research/target080/ball_hypotheses.npy')
    parser.add_argument('--receipt-name', default='verification.json',
                        help='Use a new filename after verifier changes; old receipts are immutable')
    args = parser.parse_args()
    if Path(args.receipt_name).name != args.receipt_name:
        raise ValueError('Receipt name must be a filename inside the run directory')
    receipt = json.loads((args.run / 'ball_hypotheses.json').read_text(encoding='utf-8'))
    checked = []
    for source, assets in receipt['inputs'].items():
        paths = asset_paths(data_dir=str(args.data), source=source)
        for key, expected in assets.items():
            path = Path(paths[key])
            if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
                raise ValueError(f'Supplied input changed: {source}/{key}')
            checked.append(f'{source}/{key}')
    for path, expected in receipt['dependency_sha256'].items():
        if sha256(ROOT / path) != expected:
            raise ValueError('Coordinate dependency changed: ' + path)
    if sha256(ROOT / 'scripts/research/broadcast/dual_feed.py') != receipt['source_sha256']:
        raise ValueError('Fusion source differs from the executed protocol')
    for name, key in [('ball_hypotheses.npy', 'output_sha256'),
                      ('aligned_broadcast.npz', 'aligned_signal_sha256'),
                      ('coordinate_sources.npy', 'coordinate_sources_sha256'),
                      ('protocol.json', 'protocol_sha256')]:
        if sha256(args.run / name) != receipt[key]:
            raise ValueError('Derived input changed: ' + name)
    motion_path = args.motion
    if sha256(motion_path) != receipt['motion_sha256']:
        raise ValueError('External motion artifact changed')
    motion = json.loads(motion_path.read_text(encoding='utf-8'))
    prior_path = args.tactical_hypotheses
    prior = None
    if receipt['tactical_hypotheses_sha256'] is not None:
        if sha256(prior_path) != receipt['tactical_hypotheses_sha256']:
            raise ValueError('Tactical video hypotheses changed')
        prior = np.load(prior_path, allow_pickle=False)
    _, original, _ = read_exact_tracking(args.data)
    blocks = [(a, b + 1) for _, a, b in half_video_blocks('BAR-ATH')]
    with np.load(args.run / 'aligned_broadcast.npz', allow_pickle=False) as signal:
        repeated, sources, _ = fuse_ball(original, signal, blocks, motion['max_speed_mps'], prior)
    saved = np.load(args.run / 'ball_hypotheses.npy', allow_pickle=False)
    if not np.array_equal(saved, repeated, equal_nan=True):
        raise ValueError('Repeated coordinate fusion differs')
    if not np.array_equal(sources, np.load(args.run / 'coordinate_sources.npy', allow_pickle=False)):
        raise ValueError('Repeated source assignments differ')
    seen = np.isfinite(original).all(1)
    if not np.array_equal(saved[seen], original[seen]):
        raise ValueError('Original observations changed')
    manifests = [json.loads((args.run / name / 'manifest.json').read_text(encoding='utf-8'))
                 for name in ('candidate', 'candidate-repeat')]
    for name, manifest in zip(('candidate', 'candidate-repeat'), manifests):
        output = args.run / name / 'CoreOrigin.csv'
        actual = sha256(output)
        if actual != manifest['output_sha256'] or manifest['ball_hypotheses_sha256'] != receipt['output_sha256']:
            raise ValueError('Prediction provenance differs')
        if manifest['ground_truth_read'] or manifest['training_performed']:
            raise ValueError('Unexpected match supervision')
        problem = pm.validate_submission(pm.read_label_csv(output), len(original))
        if problem:
            raise ValueError(problem)
        if sha256(ROOT / 'models/coreorigin-video-v1/selected.pt') != manifest['parent_sha256']:
            raise ValueError('Frozen parent changed')
    if manifests[0]['output_sha256'] != manifests[1]['output_sha256'] or manifests[0]['raw_sha256'] != manifests[1]['raw_sha256']:
        raise ValueError('Repeated CPU export differs')
    result = dict(complete=True, input_files_verified=checked,
                  coordinate_fusion_reproduced=True, coordinate_source_assignments_reproduced=True,
                  original_coordinates_preserved=True, csv_rows=len(original),
                  official_format_validation_passed=True, cpu_export_repeated_identically=True,
                  csv_sha256=manifests[0]['output_sha256'],
                  frozen_parent_unchanged=True, training_performed=False, ground_truth_read=False,
                  possession_score=None, candidate_selected=False, source_sha256=sha256(__file__))
    write_json(args.run / args.receipt_name, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
