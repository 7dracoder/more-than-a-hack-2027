"""Streaming diagnostics. Flagged jumps are not automatically tracking errors."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from games import SOURCES, asset_paths
from possession.data import slot_teams, tagging_features, sha256


def jump_flags(previous, current, threshold_m, pitch=(105.3, 68.)):
    valid = np.isfinite(previous).all(-1) & np.isfinite(current).all(-1)
    distance = np.linalg.norm((current - previous) * pitch, axis=-1)
    return valid & (distance > threshold_m), distance


def scan_source(data_dir, source, packets=False):
    paths = asset_paths(data_dir=str(data_dir), source=source)
    teams = slot_teams(paths['atd_meta'])
    previous = np.full((len(teams), 2), np.nan)
    previous_ball = np.full(2, np.nan)
    count = ball_jumps = slot_jumps = bad_order = malformed = 0
    examples = []
    previous_id = None
    with open(paths['atd_raw']) as stream:
        for line in stream:
            count += 1
            fields = line.strip().split(':')
            if len(fields) != 3:
                malformed += 1
                continue
            frame_id = int(fields[0])
            bad_order += int(frame_id != count)
            players = np.fromstring(fields[1].replace(';', ','), sep=',')
            ball = np.fromstring(fields[2], sep=',')[:2]
            if players.size != len(teams) * 3 or ball.size != 2:
                malformed += 1
                previous_id = None
                continue
            xy = players.reshape(-1, 3)[:, :2]
            if previous_id is not None and frame_id == previous_id + 1:
                flags, distance = jump_flags(previous, xy, 3.)
                bj, bd = jump_flags(previous_ball, ball, 5.)
                slot_jumps += int(flags.sum())
                ball_jumps += int(bj)
                if (flags.any() or bj) and len(examples) < 100:
                    examples.append({'frame': frame_id - 1, 'slot_indices': np.flatnonzero(flags).tolist(),
                                     'slot_jump_m': np.round(distance[flags], 3).tolist(),
                                     'ball_jump_m': round(float(bd), 3) if bj else None})
            previous, previous_ball, previous_id = xy, ball, frame_id
    homography = pd.read_csv(paths['homography'])
    valid = homography['valid'].astype(bool).to_numpy()
    matrices = homography[[f'h{i}{j}' for i in range(3) for j in range(3)]].to_numpy().reshape(-1, 3, 3)
    finite = np.isfinite(matrices).all((1, 2))
    determinants = np.full(len(valid), np.nan)
    determinants[finite] = np.linalg.det(matrices[finite])
    result = {'source': source, 'raw_rows': count, 'slots': len(teams), 'malformed_rows': malformed,
              'nonsequential_ids': bad_order, 'homography_rows': len(homography),
              'homography_contiguous_zero_based': bool(np.array_equal(homography['frame'], np.arange(len(homography)))),
              'homography_flagged_valid': int(valid.sum()),
              'valid_but_nonfinite_or_singular': int((valid & (~finite | (np.abs(determinants) < 1e-12))).sum()),
              'slot_jumps_over_3m_per_frame': slot_jumps, 'ball_jumps_over_5m_per_frame': ball_jumps,
              'jump_examples_first_100': examples,
              'thresholds': {'slot_m_per_40ms': 3., 'ball_m_per_40ms': 5.},
              'interpretation': 'Slot discontinuities may be reassignment or tracking errors; slots are not persistent identities. Jumps are review flags, not corrections.',
              'input_sha256': {key: sha256(paths[key]) for key in ('atd_raw', 'atd_meta', 'homography')}}
    if 'smart_tagging' in paths:
        # Count contradictory simultaneous team-phase support, separately from N/A.
        support = np.zeros((count + 1, 2), dtype=np.int16)
        unknown = 0
        offensive = {'build up', 'progression', 'creating chances', 'attacking transition', 'long ball',
                     'ball in final third', 'ball in the box', 'players in the box'}
        defensive = {'defending in attacking third', 'defending in middle third', 'defending in defensive third', 'defensive transition'}
        for item in ET.parse(paths['smart_tagging']).iter('instance'):
            code = (item.findtext('code') or '').lower().strip()
            tags = {(x.findtext('group') or '').lower().strip(): (x.findtext('text') or '').lower().strip() for x in item.findall('label')}
            team = tags.get('team', '')
            if team not in ('barcelona', 'athletic club', 'atletic club'):
                unknown += 1
                continue
            if code not in offensive | defensive:
                continue
            side = 0 if team == 'barcelona' else 1
            if code in defensive:
                side = 1 - side
            a, b = [max(0, min(count, round(float(item.findtext(key)) * 25))) for key in ('start', 'end')]
            if b > a:
                support[a, side] += 1
                support[b, side] -= 1
        active = np.cumsum(support[:-1], axis=0) > 0
        result['tag_conflict_frames'] = int(active.all(1).sum())
        result['tag_instances_without_team'] = unknown
    if Path(paths['video']).exists():
        import av
        with av.open(paths['video']) as container:
            video = container.streams.video[0]
            result['video'] = {'advertised_frames': video.frames, 'fps': str(video.average_rate),
                               'duration_seconds': float(video.duration * video.time_base) if video.duration else None,
                               'width': video.width, 'height': video.height,
                               'scope': 'Container metadata; full pixel decode not performed'}
            if packets:
                count_packets, first_pts, last_pts, max_pts = 0, None, None, None
                for packet in container.demux(video):
                    if packet.pts is not None and packet.size:
                        count_packets += 1
                        first_pts = packet.pts if first_pts is None else first_pts
                        last_pts = packet.pts
                        max_pts = packet.pts if max_pts is None else max(max_pts, packet.pts)
                result['video'].update({'timestamped_packets': count_packets,
                                        'first_pts': first_pts, 'last_packet_pts': last_pts, 'max_pts': max_pts,
                                        'scope': 'All compressed video packets scanned; not a full decode/integrity guarantee'})
        mismatch = count - result['video']['advertised_frames']
        result['alignment'] = {'tracking_minus_advertised_video_frames': mismatch,
                               'fusion_ready': mismatch == 0 and source == 'tactical',
                               'action': 'Broadcast remains unaligned to tactical; no guessed one-frame shift or padding applied.' if source == 'broadcast' else 'Tactical is the submission frame authority.'}
    return result
