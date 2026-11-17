"""CPU export with the frozen external-only parent and optional ball hypotheses.

No fitting, possession scoring or manual/video comparison. The pre-kickoff rule
uses automatic kickoff-start tag intervals for each half rather than manually
chosen video-frame anchors. Out-of-bounds constants retain external provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy
from games import GAMES, asset_paths, half_video_blocks
from evaluation import possession_metrics as pm
from possession.data import sha256
from possession.dead_ball_rules import apply_oob
from possession.features_v25 import build_features
from possession.inference_cache import cached_block
from scripts.run_local import selected_model, symmetric_probability, viterbi_labels
from scripts.research.video_ball.fuse_detections import read_exact_tracking
from starter.load_smart_tagging import load as load_tagging


def automatic_prekickoff_dead(labels, events, halves):
    """Apply one generic event-derived rule, without frame constants or guards."""
    result = labels.copy()
    intervals = []
    for half, first, last in halves:
        name = '1st half' if half == 1 else '2nd half'
        starts = sorted(set(event['start_frame'] for event in events
                            if (event['code'] or '').strip().lower() == 'set pieces'
                            and (event['tags'].get('Type') or '').strip().lower() == 'kick off (start)'
                            and (event['tags'].get('Half') or '').strip().lower() == name))
        if not starts:
            continue
        if len(starts) != 1:
            raise ValueError('Ambiguous automatic kickoff-start tags for one half')
        stop = min(last + 1, max(first, starts[0]))
        if stop > first:
            mask = np.isin(result[first:stop], ['H', 'A'])
            result[first:stop][mask] = 'D'
            intervals.append([int(first), int(stop)])
    return result, intervals


def write_csv(path, labels):
    table = pd.DataFrame(dict(frame=np.arange(len(labels)), label=labels))
    problem = pm.validate_submission(table, GAMES['BAR-ATH']['n_video_frames'])
    if problem:
        raise ValueError(problem)
    part = path.with_suffix('.csv.partial')
    table.to_csv(part, index=False, lineterminator='\n')
    if path.is_file() and path.read_bytes() != part.read_bytes():
        raise ValueError('Refusing to replace a different existing candidate; use a new directory')
    part.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=ROOT / 'models/coreorigin-video-v1')
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--ball', type=Path)
    parser.add_argument('--verify-control', action='store_true')
    parser.add_argument('--control-reference', type=Path,
                        help='Fresh unchanged CPU exporter CSV and its adjacent manifest')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.verify_control and args.ball is not None:
        parser.error('Control verification requires original tracking, without ball hypotheses')
    if args.verify_control and args.control_reference is None:
        parser.error('Control verification requires an explicit fresh CPU reference')
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    device = torch.device('cpu')
    paths = asset_paths(data_dir=str(args.data))
    input_hashes = {k: sha256(paths[k]) for k in ('atd_raw', 'atd_meta', 'smart_tagging')}
    players, original_ball, teams = read_exact_tracking(args.data)
    ball = original_ball
    ball_hash = None
    if args.ball is not None:
        ball_hash = sha256(args.ball)
        receipt = json.loads(args.ball.with_suffix('.json').read_text(encoding='utf-8'))
        if (receipt['output_sha256'] != ball_hash or receipt['calibration_used_provided_match'] or
                receipt['original_tracking_sha256'] != input_hashes['atd_raw']):
            raise ValueError('Invalid or non-external ball-recovery provenance')
        ball = np.load(args.ball, allow_pickle=False)
        if ball.shape != original_ball.shape or ball.dtype != np.float32:
            raise ValueError('Ball-hypothesis schema mismatch')
        seen = np.isfinite(original_ball).all(1)
        if not np.array_equal(ball[seen], original_ball[seen]):
            raise ValueError('Observed supplied ball coordinates changed')
    model, state = selected_model(args.parent, device)
    config = state['config']
    if config['feature_version'] != 'v25' or config['sample_step'] != 1:
        raise ValueError('This exporter requires the full-rate v25 external parent schema')
    if any('BAR' in m.upper() for m in config['train_matches'] + config['validation_matches']):
        raise ValueError('Parent includes competition-match supervision')
    source_files = ['possession/external.py', 'possession/data.py', 'scripts/run_local.py',
                    'games.py', 'possession/inference_cache.py', 'possession/features_v08.py',
                    'possession/features_v09.py', 'possession/features_v13.py', 'possession/features_v25.py']
    feature_sources = {name: sha256(ROOT / name) for name in source_files}
    raw_inputs = {k: input_hashes[k] for k in ('atd_meta', 'atd_raw')}
    if ball_hash is not None:
        raw_inputs['ball_hypotheses'] = ball_hash
    n, pitch = GAMES['BAR-ATH']['n_video_frames'], GAMES['BAR-ATH']['pitch']
    labels = np.full(n, 'X', dtype='<U1')
    feature_receipts = []
    for half, first, last in half_video_blocks('BAR-ATH'):
        key = dict(raw_inputs=raw_inputs, sources=feature_sources, game='BAR-ATH',
                   first=first, last=last, pitch=list(pitch), sample_step=1,
                   feature_version='v25', numpy=np.__version__, scipy=scipy.__version__)
        matrix, names, cache = cached_block(ROOT / 'data/inference-features', key,
                lambda: build_features(players[first:last + 1], teams, ball[first:last + 1], pitch, step=1),
                last - first + 1)
        if names != state['names']:
            raise ValueError('Feature schema differs from frozen external parent')
        probability = symmetric_probability(model, matrix, names, state['mean'], state['scale'], device)
        labels[first:last + 1] = viterbi_labels(probability, *state['decoder_costs'])
        feature_receipts.append(cache)
        print(f'CPU inference complete for half {half}', flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / 'raw.csv', labels)
    verified = False
    reference_hash = None
    if args.verify_control:
        reference_hash = sha256(args.control_reference)
        reference_manifest = json.loads(args.control_reference.with_name('manifest.json').read_text(encoding='utf-8'))
        if (reference_manifest['device'] != 'cpu' or reference_manifest['sha256'] != reference_hash or
                reference_manifest['selected_sha256'] != sha256(args.parent / 'selected.pt') or
                reference_manifest['input_sha256'] != {k:input_hashes[k] for k in ('atd_meta','atd_raw')}):
            raise ValueError('Control reference must be an unchanged CPU export from the same weights and inputs')
        previous = pm.read_label_csv(args.control_reference)
        if pm.validate_submission(previous, n) or not np.array_equal(previous['label'].to_numpy(), labels):
            raise ValueError('Fresh control differs from the unchanged CPU exporter')
        verified = True
    after, intervals = automatic_prekickoff_dead(labels, load_tagging(data_dir=str(args.data)),
                                                 half_video_blocks('BAR-ATH'))
    graded = np.zeros(n, bool)
    for _, first, last in half_video_blocks('BAR-ATH'):
        graded[first:last + 1] = True
    # Retain the existing externally selected OOB rule on original observations only.
    final, oob_changed = apply_oob(after, original_ball, pitch, graded)
    output = args.out / 'CoreOrigin.csv'
    write_csv(output, final)
    if input_hashes != {k: sha256(paths[k]) for k in input_hashes}:
        raise ValueError('Supplied inputs changed during inference')
    result = dict(rows=n, output_sha256=sha256(output), raw_sha256=sha256(args.out / 'raw.csv'),
                  parent_sha256=sha256(args.parent / 'selected.pt'),
                  parent_fixed_protocol_sha256=sha256(args.parent / 'fixed_protocol.json'),
                  parent_training_matches=config['train_matches'],
                  parent_selection_matches=config['validation_matches'],
                  original_tracking_control_verified=verified, ball_hypotheses_sha256=ball_hash,
                  control_reference_sha256=reference_hash,
                  input_sha256=input_hashes, feature_sources=feature_sources,
                  coordinate_feature_cache=feature_receipts,
                  automatic_prekickoff_dead_intervals=intervals,
                  automatic_prekickoff_frames_changed=int((after != labels).sum()),
                  original_observation_oob_frames_changed=int(oob_changed.sum()),
                  source_sha256=sha256(__file__), device='cpu',
                  additional_source_sha256={name:sha256(ROOT / name) for name in
                      ['scripts/research/video_ball/fuse_detections.py', 'possession/temporal.py',
                       'possession/dead_ball_rules.py', 'starter/load_smart_tagging.py']},
                  versions=dict(torch=torch.__version__,numpy=np.__version__,
                                scipy=scipy.__version__,pandas=pd.__version__),
                  training_performed=False, ground_truth_read=False,
                  video_prediction_comparison_performed=False,
                  manual_video_frame_anchors_used=False,
                  possession_score=None, target_verified=False,
                  status='Experimental algorithmic candidate; no BAR-ATH accuracy or selection claim')
    (args.out / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('feature_sources','coordinate_feature_cache')},indent=2))


if __name__ == '__main__':
    main()
