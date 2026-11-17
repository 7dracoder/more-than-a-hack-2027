"""Produce conservative coordinate hypotheses using externally fixed settings.

Observed ATD points are preserved exactly. Only a unique confident candidate
inside a missing interval can be proposed; it must be reachable from supplied
anchors and have an adjacent physically consistent observation. No BAR-ATH
truth/video is used to fit or tune any threshold. No possession score is read.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from games import GAMES, asset_paths, half_video_blocks
from possession.data import slot_teams
from scripts.research.video_ball.audit_inputs import digest


def read_exact_tracking(data_dir):
    """Use the original inference reader's float64-multiply/float32-store order."""
    paths = asset_paths(data_dir=str(data_dir))
    teams = slot_teams(paths['atd_meta'])
    n, pitch = GAMES['BAR-ATH']['n_video_frames'], GAMES['BAR-ATH']['pitch']
    players = np.full((n, len(teams), 2), np.nan, np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    count = 0
    with open(paths['atd_raw'], encoding='utf-8') as stream:
        for i, line in enumerate(stream):
            frame, people, point = line.strip().split(':')
            if i >= n or int(frame) != i + 1:
                raise ValueError('ATD timeline is not the complete tactical video sequence')
            row, point = np.fromstring(people.replace(';', ','), sep=','), np.fromstring(point, sep=',')
            if row.size != len(teams) * 3 or point.size != 3:
                raise ValueError('Malformed supplied tracking')
            players[i] = row.reshape(-1, 3)[:, :2] * pitch
            ball[i] = point[:2] * pitch
            count += 1
    if count != n:
        raise ValueError('Incomplete supplied tracking')
    return players, ball, teams


def confidence_threshold(evidence):
    slope, intercept = evidence['fitted']['slope'], evidence['fitted']['intercept']
    if not np.isfinite([slope, intercept]).all() or slope <= 0:
        raise ValueError('Need a valid increasing external confidence-evidence fit')
    # Equal class-balanced evidence, bounded below by the library's standard
    # prediction threshold. Both choices precede application to provided video.
    boundary = float(1 / (1 + np.exp(np.clip(intercept / slope, -700, 700))))
    return max(.25, boundary)


def coordinate_hypotheses(original, det_frame, confidence, xy, threshold, speed, blocks, fps=25):
    original = np.asarray(original)
    n = len(original)
    det_frame, confidence, xy = np.asarray(det_frame), np.asarray(confidence), np.asarray(xy)
    if (not np.issubdtype(det_frame.dtype, np.integer) or det_frame.ndim != 1 or
            confidence.shape != det_frame.shape or xy.shape != (len(det_frame), 2) or
            not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any() or
            (det_frame < 0).any() or (det_frame >= n).any()):
        raise ValueError('Invalid detections')
    if not 0 <= threshold <= 1 or not np.isfinite(speed) or speed <= 0:
        raise ValueError('Invalid externally fitted thresholds')
    seen = np.isfinite(original).all(1)
    active = np.zeros(n, bool)
    for first, stop in blocks:
        active[first:stop] = True
    passed = np.isfinite(xy).all(1) & (confidence >= threshold) & active[det_frame] & ~seen[det_frame]
    count = np.bincount(det_frame[passed], minlength=n)
    unique = passed & (count[det_frame] == 1)
    proposed = original.copy()
    proposed[det_frame[unique]] = xy[unique]
    proposed_mask = np.zeros(n, bool)
    proposed_mask[det_frame[unique]] = True
    reachable = np.zeros(n, bool)
    for first, stop in blocks:
        index = np.arange(first, stop)
        before = np.maximum.accumulate(np.where(seen[first:stop], index, -1))
        after = np.minimum.accumulate(np.where(seen[first:stop], index, n)[::-1])[::-1]
        has_before, has_after = before >= 0, after < n
        selected = proposed_mask[first:stop].copy() & (has_before | has_after)
        for anchor, known in ((before, has_before), (after, has_after)):
            at = np.flatnonzero(selected & known)
            displacement = np.linalg.norm(proposed[index[at]].astype(float) - original[anchor[at]], axis=1)
            allowance = speed * np.abs(index[at] - anchor[at]) / fps
            selected[at] &= displacement <= allowance
        reachable[first:stop] = selected
    staged = original.copy()
    staged[reachable] = proposed[reachable]
    supported = np.zeros(n, bool)
    for first, stop in blocks:
        known = np.isfinite(staged[first:stop]).all(1)
        pair = known[1:] & known[:-1]
        pair &= np.linalg.norm(np.diff(staged[first:stop].astype(float), axis=0), axis=1) <= speed / fps
        supported[first:stop - 1] |= pair
        supported[first + 1:stop] |= pair
    accepted = reachable & supported
    fused = original.copy()
    fused[accepted] = proposed[accepted]
    if not np.array_equal(fused[seen], original[seen]):
        raise AssertionError('Observed supplied coordinates must never change')
    return fused, dict(confidence_passing_detections=int(passed.sum()),
                       unique_candidate_frames=int(proposed_mask.sum()),
                       reachable_candidate_frames=int(reachable.sum()),
                       coordinate_hypothesis_frames=int(accepted.sum()),
                       originally_observed_frames=int(seen.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--external-confidence', type=Path, default=ROOT / 'models/coreorigin-video-v1/confidence_evidence.json')
    parser.add_argument('--external-motion', type=Path, default=ROOT / 'models/coreorigin-video-v1/motion.json')
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    detector = json.loads(args.detections.with_suffix('.json').read_text(encoding='utf-8'))
    if not detector['complete'] or detector['output_sha256'] != digest(args.detections):
        raise ValueError('Need a complete, unchanged detector run')
    evidence = json.loads(args.external_confidence.read_text(encoding='utf-8'))
    motion = json.loads(args.external_motion.read_text(encoding='utf-8'))
    if evidence['provided_match_data_used'] or motion['provided_match_data_used']:
        raise ValueError('All calibration must be external only')
    if detector['protocol']['weights_sha256'] != evidence['detector_inference_protocol']['weights_sha256']:
        raise ValueError('Confidence calibration belongs to different detector weights')
    for key in ('imgsz', 'conf', 'class_id'):
        if detector['protocol']['settings'][key] != evidence['detector_inference_protocol'][key]:
            raise ValueError('Detector settings differ from external confidence calibration')
    _, original, _ = read_exact_tracking(args.data)
    paths = asset_paths(data_dir=str(args.data))
    active = np.zeros(len(original), bool)
    blocks = [(first, last + 1) for _, first, last in half_video_blocks('BAR-ATH')]
    for first, stop in blocks:
        active[first:stop] = True
    expected = np.flatnonzero(active & ~np.isfinite(original).all(1))
    if detector['protocol']['tracking_sha256'] != digest(paths['atd_raw']):
        raise ValueError('Detector frame selection used different supplied tracking')
    with np.load(args.detections, allow_pickle=False) as data:
        if not np.array_equal(data['completed_frames'], expected) or not np.array_equal(data['frames'], expected):
            raise ValueError('Require complete coverage of all originally missing-ball frames')
        fused, stats = coordinate_hypotheses(original, data['det_frame'], data['conf'],
                                             np.column_stack([data['x'], data['y']]),
                                             confidence_threshold(evidence), motion['max_speed_mps'], blocks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    part = args.out.with_suffix('.npy.partial')
    with part.open('wb') as stream:
        np.save(stream, fused, allow_pickle=False)
    part.replace(args.out)
    result = dict(purpose='Externally gated coordinate hypotheses, not corrected tracking truth',
                  output_sha256=digest(args.out), detections_sha256=digest(args.detections),
                  confidence_sha256=digest(args.external_confidence), motion_sha256=digest(args.external_motion),
                  source_sha256=digest(__file__), original_tracking_sha256=digest(paths['atd_raw']),
                  raw_confidence_threshold=confidence_threshold(evidence), max_speed_mps=motion['max_speed_mps'],
                  calibration_used_provided_match=False, original_observations_unchanged=True,
                  possession_labels_read=False, possession_score=None, **stats,
                  limitations=['External COCO calibration has limited data and a different image domain.',
                               'Physical feasibility and confidence do not establish correct ball identity.',
                               'No threshold was selected from the provided video or ground truth.'])
    args.out.with_suffix('.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
