"""Fixed, local two-feed preparation; never reads BAR-ATH possession truth.

Synchronize using team player geometry, reject mismatched views, and propose
broadcast ball coordinates only inside tactical gaps. These are experimental
inputs, not verified ball truth or an automatically selected submission.
"""
import argparse
import json
from pathlib import Path
import sys
import unicodedata
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from games import GAMES, asset_paths, half_video_blocks, source_frames
from possession.data import sha256, slot_teams
from scripts.research.video_ball.fuse_detections import coordinate_hypotheses
from starter.load_homography import check_alignment, load as load_homography


# Frozen engineering defaults, not selected on this match's labels/video.
# The geometric acceptance rule predates this run (apply_ball_fill.py).
PROTOCOL = dict(version=1, fps=25, coarse_tactical_stride=25,
                coarse_broadcast_stride=1000, refinement_stride=125,
                chamfer_clip_m=5.0, max_player_cost_m=1.5,
                min_visible_players=6, min_alignment_samples=3,
                hypothesis_priority='broadcast tracking, then existing tactical-video hypotheses',
                broadcast_video_used_for_ball_detection=False,
                selection='experimental only; never replace the frozen submission',
                parameter_selection_on_provided_match=False,
                ground_truth_read=False, video_prediction_comparison=False)


def write_json(path, value):
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise ValueError(f'Refusing to overwrite a different receipt: {path}')
    path.write_text(text, encoding='utf-8')


def canonical_name(value):
    return ''.join(c for c in unicodedata.normalize('NFKD', value).casefold()
                   if c.isalnum())


def read_source(data_dir, source):
    paths = asset_paths(data_dir=str(data_dir), source=source)
    teams = slot_teams(paths['atd_meta'])
    names = [canonical_name(t.findtext('Name') or t.get('name') or '')
             for t in ET.parse(paths['atd_meta']).getroot().iter('Team')]
    if len(set(names)) != 2 or not all(names):
        raise ValueError('Need two distinct named teams')
    n, pitch = source_frames(source), GAMES['BAR-ATH']['pitch']
    players = np.full((n, len(teams), 2), np.nan, np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    count = 0
    with open(paths['atd_raw'], encoding='utf-8') as stream:
        for i, line in enumerate(stream):
            frame, people, point = line.strip().split(':')
            if i >= n or int(frame) != i + 1:
                raise ValueError(f'{source}: incomplete or nonsequential video-frame counter')
            row = np.fromstring(people.replace(';', ','), sep=',')
            point = np.fromstring(point, sep=',')
            if row.size != len(teams) * 3 or point.size != 3:
                raise ValueError(f'{source}: malformed tracking record')
            players[i] = row.reshape(-1, 3)[:, :2] * pitch
            ball[i] = point[:2] * pitch
            count += 1
    if count != n or np.isinf(players).any() or np.isinf(ball).any():
        raise ValueError(f'{source}: invalid tracking length/coordinates')
    if not check_alignment(data_dir=str(data_dir), source=source)['agrees']:
        raise ValueError(f'{source}: homography timeline differs from video')
    matrices, _, valid = load_homography(data_dir=str(data_dir), source=source)
    if len(matrices) != n or not np.isfinite(matrices[valid]).all():
        raise ValueError(f'{source}: invalid homography matrices')
    return dict(players=players, ball=ball, teams=teams, names=names, valid=valid)


def remap_teams(broadcast_names, tactical_names, teams):
    if set(broadcast_names) != set(tactical_names) or len(set(tactical_names)) != 2:
        raise ValueError('The recordings do not identify the same two teams')
    mapping = np.array([tactical_names.index(name) for name in broadcast_names])
    return mapping[teams]


def paired_cost(query, qteams, candidate, cteams, clip=5.):
    """One-sided team Chamfer, ignoring absent query slots but penalizing absent targets.

    query/candidate may have matching batches, or a single broadcast query may
    be compared against many tactical frames. Track identity is never assumed.
    """
    if len(query) == 1 and len(candidate) != 1:
        query = np.broadcast_to(query, (len(candidate), *query.shape[1:]))
    if len(query) != len(candidate):
        raise ValueError('Paired batches differ')
    total, count = np.zeros(len(query)), np.zeros(len(query), np.int32)
    for team in (0, 1):
        q, c = query[:, qteams == team], candidate[:, cteams == team]
        if q.shape[1] == 0:
            continue
        seen = np.isfinite(q).all(-1)
        if c.shape[1]:
            d = np.linalg.norm(q[:, :, None].astype(float) - c[:, None].astype(float), axis=-1)
            nearest = np.where(np.isfinite(d), d, np.inf).min(-1)
        else:
            nearest = np.full(seen.shape, np.inf)
        total += np.where(seen, np.minimum(nearest, clip), 0.).sum(1)
        count += seen.sum(1)
    cost = np.divide(total, count, out=np.full(len(total), np.nan), where=count > 0)
    return cost, count


def estimate_offset(bp, bt, bv, tp, tt, first, stop, settings=PROTOCOL):
    """Global geometry search then integer refinement; no match-specific offset seeds."""
    stride = settings['coarse_tactical_stride']
    candidates = np.arange(first, stop, stride)
    samples = np.arange(0, len(bp), settings['coarse_broadcast_stride'])
    seen = np.isfinite(bp[samples]).all(-1).sum(1)
    samples = samples[bv[samples] & (seen >= settings['min_visible_players'])]
    offsets, coarse = [], []
    for b in samples:
        costs, _ = paired_cost(bp[b:b + 1], bt, tp[candidates], tt,
                               settings['chamfer_clip_m'])
        at = int(np.nanargmin(costs))
        if costs[at] < settings['max_player_cost_m']:
            offsets.append(int(candidates[at] - b))
            coarse.append([int(b), int(candidates[at]), float(costs[at])])
    if len(offsets) < settings['min_alignment_samples']:
        raise ValueError('Insufficient geometric matches to synchronize this half')
    offsets = np.asarray(offsets)
    support = (np.abs(offsets[:, None] - offsets) <= stride).sum(1)
    center = int(offsets[np.argmax(support)])
    consensus = offsets[np.abs(offsets - center) <= stride]
    if len(consensus) < settings['min_alignment_samples']:
        raise ValueError('No supported constant-offset alignment')
    center = int(np.median(consensus))
    sample = np.arange(0, len(bp), settings['refinement_stride'])
    visible = np.isfinite(bp[sample]).all(-1).sum(1)
    sample = sample[bv[sample] & (visible >= settings['min_visible_players'])]
    # Use the same eligible frames for every offset, not a changing boundary cohort.
    sample = sample[(sample + center - stride >= first) &
                    (sample + center + stride < stop)]
    if len(sample) < settings['min_alignment_samples']:
        raise ValueError('Too few common frames for alignment refinement')
    scores = []
    for offset in range(center - stride, center + stride + 1):
        costs, _ = paired_cost(bp[sample], bt, tp[sample + offset], tt,
                               settings['chamfer_clip_m'])
        scores.append((float(np.nanmedian(costs)), offset))
    cost, best = min(scores)
    if cost >= settings['max_player_cost_m']:
        raise ValueError('Alignment fails the predeclared geometric acceptance rule')
    # Replays/close-ups are rejected separately for every frame, not filled by offset alone.
    return best, dict(offset=best, coarse_consensus_frames=len(consensus),
                      refinement_samples=len(sample), median_player_cost_m=cost,
                      coarse_matches=coarse,
                      refinement_costs=[dict(offset=o, median_cost_m=c) for c, o in scores])


def align_signal(tactical, broadcast, blocks, settings=PROTOCOL):
    n = len(tactical['ball'])
    signal = dict(ball=np.full((n, 2), np.nan, np.float32),
                  broadcast_frame=np.full(n, -1, np.int32),
                  accepted_view=np.zeros(n, bool),
                  player_cost=np.full(n, np.nan, np.float32))
    receipts = []
    bt = remap_teams(broadcast['names'], tactical['names'], broadcast['teams'])
    for half, first, stop in blocks:
        offset, receipt = estimate_offset(broadcast['players'], bt, broadcast['valid'],
                                         tactical['players'], tactical['teams'], first, stop, settings)
        for start in range(first, stop, 1000):
            t = np.arange(start, min(stop, start + 1000))
            b = t - offset
            inside = (b >= 0) & (b < len(broadcast['ball']))
            t, b = t[inside], b[inside]
            costs, count = paired_cost(broadcast['players'][b], bt,
                                       tactical['players'][t], tactical['teams'],
                                       settings['chamfer_clip_m'])
            accepted = (broadcast['valid'][b] & tactical['valid'][t] &
                        (count >= settings['min_visible_players']) &
                        (costs < settings['max_player_cost_m']))
            signal['broadcast_frame'][t] = b
            signal['player_cost'][t] = costs
            signal['accepted_view'][t] = accepted
            signal['ball'][t[accepted]] = broadcast['ball'][b[accepted]]
        receipts.append(dict(half=half, **receipt))
        print(f'Geometric synchronization complete for half {half}', flush=True)
    return signal, receipts


def fuse_ball(original, signal, blocks, speed, prior=None):
    """Use only observed anchors and the existing external motion bound.

    Original coordinates have highest priority. Broadcast proposals require
    geometric view acceptance, reachable anchors and adjacent motion support.
    The existing tactical-video artifact is a lower-priority fallback.
    """
    frame = np.flatnonzero(signal['accepted_view'] & np.isfinite(signal['ball']).all(1))
    fused, stats = coordinate_hypotheses(original, frame, np.ones(len(frame)),
                                        signal['ball'][frame], 1., speed, blocks)
    source = np.zeros(len(original), np.uint8)
    source[np.isfinite(original).all(1)] = 1
    source[(source == 0) & np.isfinite(fused).all(1)] = 2
    if prior is not None:
        observed = np.isfinite(original).all(1)
        if prior.shape != original.shape or prior.dtype != np.float32:
            raise ValueError('Tactical video hypotheses have incompatible schema')
        if not np.array_equal(prior[observed], original[observed]):
            raise ValueError('Tactical video hypotheses changed original observations')
        missing = ~np.isfinite(fused).all(1) & np.isfinite(prior).all(1)
        # A new cross-source junction must also satisfy the same motion rule.
        frame = np.flatnonzero(missing)
        combined, extra = coordinate_hypotheses(fused, frame, np.ones(len(frame)),
                                               prior[frame], 1., speed, blocks)
        source[(source == 0) & np.isfinite(combined).all(1)] = 3
        fused = combined
        stats['tactical_video_fallback'] = extra
    return fused, source, stats


def audit_video(path, expected, require_exact=True):
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError('Video cannot be opened')
        frames, fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), capture.get(cv2.CAP_PROP_FPS)
        aligned = frames == expected and abs(fps - PROTOCOL['fps']) <= 1e-6
        if require_exact and not aligned:
            raise ValueError('Video/tracking frame count or frame rate differs')
        if frames <= 0 or fps <= 0:
            raise ValueError('Invalid video metadata')
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sampled = np.linspace(0, frames - 1, 9, dtype=int)
        for frame in sampled:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok, image = capture.read()
            if not ok or image.shape[:2] != (height, width):
                raise ValueError('Sampled video frame cannot be decoded')
        return dict(frames=frames, fps=fps, width=width, height=height,
                    sampled_decodes=len(sampled), previewed=False,
                    tracking_frames_expected=expected, frame_count_difference=frames - expected,
                    pixel_inference_timeline_verified=aligned,
                    used_for_new_pixel_inference=False)
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--motion', type=Path, default=ROOT / 'models/coreorigin-video-v1/motion.json')
    parser.add_argument('--tactical-hypotheses', type=Path)
    args = parser.parse_args()
    # All supplied data and generated coordinates stay in ignored data/.
    if not args.out.resolve().is_relative_to((ROOT / 'data').resolve()):
        raise ValueError('Match-derived output must stay under ignored data/')
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'protocol.json', PROTOCOL)
    motion = json.loads(args.motion.read_text(encoding='utf-8'))
    if motion['provided_match_data_used'] or motion['validation_match_read']:
        raise ValueError('Motion bound must be fitted on external training data only')
    inputs, video = {}, {}
    for source in ('tactical', 'broadcast'):
        paths = asset_paths(data_dir=str(args.data), source=source)
        inputs[source] = {key: dict(name=Path(path).name, bytes=Path(path).stat().st_size,
                                   sha256=sha256(path)) for key, path in paths.items()}
        # Broadcast inference here reads its ATD, not its pixels. A video/ATD
        # discrepancy is recorded and prohibits pixel inference; it does not
        # silently shift or truncate the independently checked tracking clock.
        video[source] = audit_video(paths['video'], source_frames(source), require_exact=source == 'tactical')
        print(f'{source} assets hashed and video metadata/decodes verified', flush=True)
    tactical, broadcast = read_source(args.data, 'tactical'), read_source(args.data, 'broadcast')
    blocks = [(h, first, last + 1) for h, first, last in half_video_blocks('BAR-ATH')]
    signal, alignment = align_signal(tactical, broadcast, blocks)
    np.savez_compressed(args.out / 'aligned_broadcast.npz', **signal)
    write_json(args.out / 'alignment.json', alignment)
    prior, prior_hash = None, None
    if args.tactical_hypotheses:
        receipt = json.loads(args.tactical_hypotheses.with_suffix('.json').read_text(encoding='utf-8'))
        prior_hash = sha256(args.tactical_hypotheses)
        if (receipt['output_sha256'] != prior_hash or receipt['calibration_used_provided_match'] or
                receipt['original_tracking_sha256'] != inputs['tactical']['atd_raw']['sha256'] or
                receipt['motion_sha256'] != sha256(args.motion)):
            raise ValueError('Unverified or incompatible tactical video hypothesis provenance')
        prior = np.load(args.tactical_hypotheses, allow_pickle=False)
    fused, source, stats = fuse_ball(tactical['ball'], signal, [(a, b) for _, a, b in blocks],
                                    motion['max_speed_mps'], prior)
    output = args.out / 'ball_hypotheses.npy'
    np.save(output, fused)
    np.save(args.out / 'coordinate_sources.npy', source)
    active = np.zeros(len(fused), bool)
    for _, a, b in blocks:
        active[a:b] = True
    original_seen = np.isfinite(tactical['ball']).all(1)
    jointly_seen = original_seen & np.isfinite(signal['ball']).all(1) & active
    error = np.linalg.norm(tactical['ball'][jointly_seen] - signal['ball'][jointly_seen], axis=1)
    result = dict(purpose='Experimental dual-feed coordinate hypotheses; not verified ball truth',
                  output_sha256=sha256(output), original_tracking_sha256=inputs['tactical']['atd_raw']['sha256'],
                  calibration_used_provided_match=False, original_observations_unchanged=True,
                  possession_labels_read=False, possession_score=None,
                  parameter_or_candidate_selection_performed=False,
                  inputs=inputs, video=video, protocol_sha256=sha256(args.out / 'protocol.json'),
                  source_sha256=sha256(__file__),
                  dependency_sha256={name:sha256(ROOT / name) for name in
                      ['games.py', 'possession/data.py', 'starter/load_homography.py',
                       'scripts/research/video_ball/fuse_detections.py']},
                  aligned_signal_sha256=sha256(args.out / 'aligned_broadcast.npz'),
                  coordinate_sources_sha256=sha256(args.out / 'coordinate_sources.npy'),
                  motion_sha256=sha256(args.motion), tactical_hypotheses_sha256=prior_hash,
                  diagnostic_only=dict(jointly_observed_frames=int(jointly_seen.sum()),
                      median_ball_disagreement_m=float(np.median(error)) if len(error) else None),
                  graded_frames=int(active.sum()), originally_missing_graded_frames=int((active & ~original_seen).sum()),
                  coordinate_source_counts_graded={name:int((active & (source == code)).sum()) for code, name in
                      enumerate(['missing', 'original_tactical', 'broadcast', 'tactical_video'])},
                  fusion_stats=stats, full_match_accuracy=None,
                  limitation='Historical naive broadcast filling reduced external proxy F1; this candidate is not promoted.')
    write_json(output.with_suffix('.json'), result)
    print(json.dumps({k:result[k] for k in ['graded_frames', 'originally_missing_graded_frames',
          'coordinate_source_counts_graded', 'possession_score', 'parameter_or_candidate_selection_performed']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
