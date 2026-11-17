"""Audit BAR-ATH input coverage and video-recovery opportunities without labels.

Known-ball masking measures coordinate reconstruction error, never possession accuracy.
No input is modified, no detector is trained, and no candidate is selected or scored.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from games import GAMES, asset_paths, half_video_blocks
from starter.load_homography import check_alignment, load as load_homography
from starter.load_smart_tagging import load as load_tagging
from scripts.research.video_ball.tracking import (
    bracketed_gaps, linear_gap, load_tracking, play_mask, project_points, runs,
)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def quantiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {str(q): float(np.quantile(values, q)) for q in (0.1, 0.5, 0.9, 0.99)}


def known_ball_masking(ball, max_samples=100, seed=804):
    """Fixed experiment, uniform eligible windows; no tuning against the result."""
    seen = np.isfinite(ball).all(1)
    rng = np.random.default_rng(seed)
    result = []
    for width in (5, 13, 25, 50, 125, 250):
        candidates = []
        for _, first, last in half_video_blocks('BAR-ATH'):
            visible = seen[first:last + 1]
            # A window includes the two retained anchor observations.
            sums = np.r_[0, np.cumsum(visible.astype(np.int64))]
            size = width + 2
            if len(visible) < size:
                continue
            starts = np.flatnonzero(sums[size:] - sums[:-size] == size) + first + 1
            candidates.extend(starts.tolist())
        rng.shuffle(candidates)
        selected = []
        occupied = np.zeros(len(ball), bool)
        for start in candidates:
            if not occupied[start - 1:start + width + 1].any():
                selected.append(start)
                occupied[start - 1:start + width + 1] = True
            if len(selected) == max_samples:
                break
        errors = []
        for start in selected:
            stop = start + width
            errors.extend(np.linalg.norm(linear_gap(ball, start, stop) - ball[start:stop], axis=1))
        errors = np.asarray(errors)
        result.append({
            'masked_frames_per_window': width, 'seconds_per_window': width / 25,
            'eligible_windows': len(candidates), 'sampled_windows': len(selected),
            'error_m_quantiles': quantiles(errors),
            'fraction_within_1m': float(np.mean(errors <= 1)) if len(errors) else None,
            'fraction_within_3m': float(np.mean(errors <= 3)) if len(errors) else None,
        })
    return result


def audit(data_dir, out):
    paths = asset_paths(data_dir=str(data_dir))
    inputs = {}
    for key in ('atd_raw', 'atd_meta', 'homography', 'smart_tagging'):
        path = Path(paths[key])
        inputs[key] = {'name': path.name, 'bytes': path.stat().st_size, 'sha256': digest(path)}
    manifest = Path(data_dir) / 'tactical/download_manifest.json'
    if manifest.is_file():
        expected = {item['name']: item for item in json.loads(manifest.read_text(encoding='utf-8'))}
        for item in inputs.values():
            pin = expected.get(item['name'])
            if pin is None or pin['sha256'] != item['sha256'] or pin['bytes'] != item['bytes']:
                raise ValueError(f"Supplied input differs from pinned manifest: {item['name']}")
    if not check_alignment(data_dir=str(data_dir))['agrees']:
        raise ValueError('Homography rows do not follow the tactical video timeline')
    players, ball, teams = load_tracking(data_dir)
    n = len(ball)
    matrices, error, valid = load_homography(data_dir=str(data_dir))
    if len(matrices) != n:
        raise ValueError('Homography and tracking row counts differ')
    seen = np.isfinite(ball).all(1)
    active = play_mask(n)
    pitch = GAMES['BAR-ATH']['pitch']
    pixels = project_points(matrices, ball, pitch)
    in_image = np.isfinite(pixels).all(1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < 1920) & (pixels[:, 1] >= 0) & (pixels[:, 1] < 1080)
    gaps = []
    for half, start, stop in bracketed_gaps(ball):
        line = linear_gap(ball, start, stop)
        projected = project_points(matrices[start:stop], line, pitch)
        in_view = np.isfinite(projected).all(1) & (projected[:, 0] >= 0) & (projected[:, 0] < 1920) & (projected[:, 1] >= 0) & (projected[:, 1] < 1080)
        gaps.append({
            'half': half, 'start_frame': start, 'stop_frame_exclusive': stop,
            'frames': stop - start, 'seconds': (stop - start) / 25,
            'anchor_displacement_m': float(np.linalg.norm(ball[stop] - ball[start - 1])),
            'linear_path_in_image_fraction': float(in_view.mean()),
            'homography_valid_fraction': float(valid[start:stop].mean()),
        })
    halves = []
    for half, first, last in half_video_blocks('BAR-ATH'):
        selected = slice(first, last + 1)
        starts, stops = runs(~seen[selected])
        halves.append({
            'half': half, 'frames': last - first + 1,
            'ball_observed_frames': int(seen[selected].sum()),
            'ball_missing_fraction': float((~seen[selected]).mean()),
            'ball_missing_run_count': len(starts),
            'ball_missing_run_seconds_quantiles': quantiles((stops - starts) / 25),
            'observed_ball_outside_image_fraction': float((~in_image[selected])[seen[selected]].mean()),
            'player_counts_quantiles': quantiles(np.isfinite(players[selected]).all(-1).sum(1)),
            'homography_error_quantiles': quantiles(error[selected]),
        })
    tags = load_tagging(data_dir=str(data_dir))
    video = Path(paths['video'])
    summary = {
        'match': 'BAR-ATH', 'purpose': 'Label-free coverage and ball-recovery feasibility only',
        'possession_labels_read': False, 'training_performed': False,
        'candidate_selection_performed': False, 'possession_score': None,
        'inputs': inputs, 'audit_source_sha256': digest(__file__),
        'pinned_input_manifest_sha256': digest(manifest) if manifest.is_file() else None,
        'tracking_source_sha256': digest(Path(__file__).with_name('tracking.py')),
        'video': {'path': str(video.resolve()), 'available': video.is_file(),
                  'bytes': video.stat().st_size if video.is_file() else None},
        'frames': n, 'track_slots': len(teams), 'half_summaries': halves,
        'graded_window_ball_missing_frames': int((active & ~seen).sum()),
        'bracketed_gap_frames': sum(g['frames'] for g in gaps),
        'bracketed_gaps': len(gaps),
        'known_ball_masking_linear_baseline': known_ball_masking(ball),
        'smart_tagging_codes': dict(Counter(t['code'] for t in tags)),
        'set_piece_types': dict(Counter(t['tags'].get('Type', '') for t in tags if t['code'].strip().lower() == 'set pieces')),
        'limitations': [
            'Known-ball masking is easier than natural gaps and cannot establish possession accuracy.',
            'An interpolated path in the image is a detector search opportunity, not proof the ball is visible.',
            'Tracking-half windows include the published pre-second-half dead-ball interval.',
            'The 0.80 full-match spell-F1 objective requires an authoritative score unavailable locally.',
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [('input_audit.json', summary), ('ball_gap_inventory.json', gaps)]:
        part = out / (name + '.partial')
        part.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
        part.replace(out / name)
    print(json.dumps({k: summary[k] for k in ('video', 'frames', 'half_summaries', 'graded_window_ball_missing_frames', 'bracketed_gaps', 'known_ball_masking_linear_baseline')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--out', type=Path, default=ROOT / 'data/research/target080')
    args = parser.parse_args()
    audit(args.data, args.out)
