"""Automatic detection-coordinate checks, never video tuning or possession scoring.

Reports agreement with supplied automatic tracking and raw candidate coverage.
No video pixels are read here. No parameters, detections, tracks or predictions
are changed. These diagnostics cannot establish possession accuracy or select
a detector/fusion configuration on BAR-ATH.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from games import asset_paths
from scripts.research.video_ball.audit_inputs import digest, quantiles
from scripts.research.video_ball.tracking import load_tracking


def coordinate_checks(selected, completed, det_frame, confidence, xy, ball):
    selected, completed = np.asarray(selected), np.asarray(completed)
    det_frame, confidence, xy = np.asarray(det_frame), np.asarray(confidence), np.asarray(xy)
    if not np.array_equal(selected, completed):
        raise ValueError('Detection run is incomplete')
    if (not np.issubdtype(selected.dtype, np.integer) or selected.ndim != 1 or
            not len(selected) or selected[0] < 0 or selected[-1] >= len(ball) or
            not np.all(np.diff(selected) > 0)):
        raise ValueError('Invalid selected video timeline')
    if (not np.issubdtype(det_frame.dtype, np.integer) or det_frame.ndim != 1 or
            not np.isin(det_frame, selected).all() or
            confidence.shape != det_frame.shape or xy.shape != (len(det_frame), 2)):
        raise ValueError('Detection columns do not match the selected timeline')
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError('Detection confidence must be finite in [0,1]')
    candidates = np.zeros(len(ball), np.int64)
    valid_projection = np.isfinite(xy).all(1)
    np.add.at(candidates, det_frame[valid_projection], 1)
    observed = np.isfinite(ball).all(1)
    comparable = valid_projection & observed[det_frame]
    error = np.linalg.norm(xy[comparable] - ball[det_frame[comparable]], axis=1)
    nearest = np.full(len(ball), np.inf)
    np.minimum.at(nearest, det_frame[comparable], error)
    known = selected[observed[selected]]
    missing = selected[~observed[selected]]
    with_candidate = known[np.isfinite(nearest[known])]
    return {
        'selected_frames': len(selected), 'raw_detections': len(det_frame),
        'nonfinite_projection_detections': int((~valid_projection).sum()),
        'selected_frames_with_projected_candidates': int((candidates[selected] > 0).sum()),
        'known_ball_selected_frames': len(known),
        'known_ball_frames_with_candidates': len(with_candidate),
        'known_ball_nearest_candidate_error_m_quantiles': quantiles(nearest[with_candidate]),
        'known_ball_frames_with_candidate_within_1m': int((nearest[known] <= 1).sum()),
        'known_ball_frames_with_candidate_within_3m': int((nearest[known] <= 3).sum()),
        'missing_ball_selected_frames': len(missing),
        'missing_ball_frames_with_projected_candidates': int((candidates[missing] > 0).sum()),
        'candidate_confidence_quantiles': quantiles(confidence),
        'possession_score': None, 'ground_truth_read': False,
        'parameter_or_candidate_selection_performed': False,
        'limitations': [
            'Supplied automatic coordinates are not corrected ball truth.',
            'Candidate coverage does not establish correct recovered ball positions.',
            'Nearest-candidate agreement is optimistic when multiple detections exist.',
            'Projection of an airborne ball need not match its ground-plane tracking point.',
            'One/three-metre counts are fixed descriptive summaries, never a selection gate.',
            'Do not tune detector, tracking, fusion or possession settings to these results.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.detections.with_suffix('.json').read_text(encoding='utf-8'))
    detection_hash = digest(args.detections)
    if not receipt['complete'] or receipt['output_sha256'] != detection_hash:
        raise ValueError('Incomplete or corrupted detector output')
    paths = asset_paths(data_dir=str(args.data))
    if receipt['protocol']['homography_sha256'] != digest(paths['homography']):
        raise ValueError('Detection geometry differs from the supplied input')
    _, ball, _ = load_tracking(args.data)
    with np.load(args.detections, allow_pickle=False) as data:
        result = coordinate_checks(data['frames'], data['completed_frames'], data['det_frame'],
                                   data['conf'], np.column_stack([data['x'], data['y']]), ball)
    result.update(detections_sha256=detection_hash, source_sha256=digest(__file__),
                  supplied_tracking_sha256=digest(paths['atd_raw']),
                  detector_protocol=receipt['protocol'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    part = args.out.with_suffix('.json.partial')
    part.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    part.replace(args.out)
    print(json.dumps({k: v for k, v in result.items() if k != 'detector_protocol'}, indent=2))


if __name__ == '__main__':
    main()
