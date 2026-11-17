"""Fit natural visibility statistics on the existing 18 external training matches.

The two external weak-label validation matches, every BAR-ATH asset/prediction,
and all possession ground truth are excluded by the I/O guard.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from possession.data import sha256
from possession.external_visibility import summarize, fit_profile
from possession.skillcorner import load_match
from scripts.research.spatial_memory.run_external import dump, isolate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    denied = isolate(args.out)
    config = json.loads((ROOT / 'configs/skillcorner_encoder_pretrain.json').read_text(encoding='utf-8'))
    matches = config['train_matches']
    assert not set(matches) & set(config['validation_matches'])
    dump(args.out / 'visibility_plan.json', dict(matches=matches, validation_matches_excluded=config['validation_matches'],
        source_fps=10, target_fps=25, possession_labels_used=False, provided_match_used=False,
        stationary_augmentation='All choices external/predeclared; no BAR-ATH aggregate calibration'))
    summaries, inputs = [], {}
    for match in matches:
        directory = ROOT / 'data/external/skillcorner-raw' / match
        source = json.loads((directory / 'source.json').read_text(encoding='utf-8'))
        if (sha256(directory / 'metadata.json') != source['metadata_sha256'] or
                sha256(directory / 'tracking.jsonl') != source['tracking_sha256']):
            raise ValueError('External source bytes differ')
        metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
        pitch = np.array([metadata['pitch_length'], metadata['pitch_width']], np.float32)
        halves, teams = load_match(metadata, directory / 'tracking.jsonl')
        for period in (1, 2):
            half = halves[period]
            native_players, native_ball = half['native']
            detected_players, detected_ball = half['detected']
            summaries.append(summarize(native_players + pitch / 2., native_ball + pitch / 2.,
                detected_players + pitch / 2., detected_ball + pitch / 2., teams, pitch, fps=10.))
        inputs[match] = source
        print(f'External visibility fit: {match}, {sum(x["frames"] for x in summaries)} source frames', flush=True)
    result = fit_profile(summaries, matches)
    result.update(source_sha256={name:sha256(ROOT / name) for name in
        ['possession/external_visibility.py', 'possession/skillcorner.py',
         'scripts/research/spatial_memory/calibrate_external_visibility.py']},
        input_provenance=inputs, validation_matches_used=False, data_guard_verified=True,
        unexpected_competition_access=denied, rights='SkillCorner open data, MIT; credit SkillCorner')
    dump(args.out / 'external_visibility.json', result)
    print(json.dumps({key:result[key] for key in ['observed_frames','natural_gap_count','missing_fraction','visible_players_per_team']}), flush=True)


if __name__ == '__main__':
    main()
