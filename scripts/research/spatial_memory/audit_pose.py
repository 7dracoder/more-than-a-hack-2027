"""Stream an externally published pose sample/full match; never read BAR-ATH.

This audits signal availability and uncertainty, not head/gaze truth accuracy
or possession performance. The full-match ZIP is verified against its publisher
manifest before any record is opened; only aggregate diagnostics are saved.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from possession.data import sha256
from possession.pose_orientation import angular_change, orientation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/external/skillcorner-bodypose')
    parser.add_argument('--match', choices=['sample', '1925299', '1996435'], default='sample')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    provenance = json.loads((args.source / 'source.json').read_text())
    manifest = json.loads((args.source / 'MANIFEST.json').read_text())
    if args.match == 'sample':
        path = args.source / 'sample_1925299_phase406.jsonl.gz'
        expected = provenance['files']['data/bodypose/sample_1925299_phase406.jsonl.gz']['sha256']
    else:
        path = args.source / (args.match + '.jsonl.zip')
        expected = manifest['files']['raw/' + path.name]['sha256']
        if path.stat().st_size != manifest['files']['raw/' + path.name]['size_bytes']:
            raise ValueError('Incomplete external pose archive')
    if sha256(path) != expected:
        raise ValueError('External pose bytes differ from pinned provenance')
    counts, last, turns, separation = Counter(), {}, [], []
    periods, last_frame = set(), -1
    def consume(stream):
        nonlocal last_frame
        for line in stream:
            record = json.loads(line)
            frame, period = int(record['frame']), record.get('period')
            if frame <= last_frame:
                raise ValueError('Pose frames are not strictly ordered')
            last_frame = frame
            periods.add(period)
            counts['frames'] += 1
            if counts['frames'] % 25000 == 0:
                print(f"Audited {counts['frames']} external pose frames", flush=True)
            for player in record.get('player_data') or []:
                counts['player_frames'] += 1
                if not player.get('joints') or not player.get('is_detected'):
                    continue
                counts['detected_player_frames_with_pose'] += 1
                value = orientation(player['joints'])
                counts['head_direction_accepted'] += int(value['head_known'])
                counts['body_axis_accepted'] += int(value['body_axis_known'])
                angle = value['unsigned_head_body_angle_degrees']
                if angle is not None:
                    separation.append(angle)
                key = (period, player['player_id'])
                if value['head_known']:
                    previous = last.get(key)
                    if previous is not None and previous[0] == frame - 1:
                        turns.append(abs(np.degrees(angular_change(previous[1], value['head_yaw_radians']))))
                    last[key] = (frame, value['head_yaw_radians'])
    if args.match == 'sample':
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            consume(stream)
    else:
        with zipfile.ZipFile(path) as archive, archive.open(args.match + '.jsonl') as stream:
            consume(stream)
    def quantiles(values):
        return {str(q):float(np.quantile(values, q)) for q in (.1, .5, .9)} if values else None
    result = dict(complete=True, dataset='SkillCorner/opendata-bodypose', match=args.match,
                  source_sha256=sha256(path), source_commit=provenance['commit'],
                  license='MIT; credit SkillCorner', counts=dict(counts),
                  periods=sorted(periods, key=lambda p:str(p)),
                  accepted_consecutive_head_changes_degrees=quantiles(turns),
                  unsigned_head_body_angle_degrees=quantiles(separation),
                  sampling_fps=25, tracking_fps=10, frame_conversion='pose_frame = 2.5 * tracking_frame',
                  possession_labels_read=False, provided_match_read=False,
                  eye_gaze_or_peripheral_vision_measured=False,
                  orientation_accuracy_measured=False, possession_score=None,
                  deployed_in_competition_model=False,
                  feature_source_sha256=sha256(ROOT / 'possession/pose_orientation.py'),
                  limitation='Provider-predicted geometry/errors, not human orientation truth. No corresponding validated BAR-ATH head measurements.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
