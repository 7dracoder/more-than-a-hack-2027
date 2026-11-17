"""Measure the ball-free adapter's coverage under external camera loss.

No possession labels, model outputs or competition data enter this diagnostic.
The visibility model is copied from the frozen, external-only calibration.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from possession.data import sha256
from possession.external import read_sportec
from possession.external_visibility import degrade
from possession.pathcrf_coordinates import player_features, windows
from scripts.research.spatial_memory.run_external import dump, isolate
from scripts.research.spatial_memory.train_natural_visibility import seed_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    # Read only the explicit external calibration artifact before isolation;
    # reject any artifact declaring target-derived statistics or label fitting.
    profile = json.loads(args.profile.read_text(encoding='utf-8'))
    if (profile['provided_match_used'] or profile['possession_labels_used'] or
            profile['validation_matches_used'] or not profile['data_guard_verified']):
        raise ValueError('Need the independent external visibility model')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'plan.json').exists():
        raise ValueError('Coverage audit is already frozen')
    dump(args.out / 'external_visibility.json', profile)
    profile_hash = sha256(args.profile)
    denied = isolate(args.out)
    plan = dict(match='J03WR9', views=['native', 'masked', 'mixed'],
        profile_sha256=profile_hash, source_sha256=sha256(Path(__file__)),
        adapter_sha256=sha256(ROOT / 'possession/pathcrf_coordinates.py'),
        visibility_source_sha256=sha256(ROOT / 'possession/external_visibility.py'),
        sample_step=5, window_steps=50, stride_steps=25,
        possession_labels_used=False, predictions_used=False, provided_match_used=False,
        purpose='Input coverage only, never an estimate of possession accuracy')
    dump(args.out / 'plan.json', plan)
    folder = ROOT / 'data/external/sportec'
    paths = [next(folder.glob('*' + kind + '*' + plan['match'] + '.xml')) for kind in ('matchinformation', 'positions')]
    pins = {x['name']:x['sha256'] for x in json.loads((folder / 'download_manifest.json').read_text())['files']}
    if any(sha256(p) != pins[p.name] for p in paths):
        raise ValueError('External coordinates changed')
    rows = []
    for i, half in enumerate(read_sportec(*paths)):
        for view in plan['views']:
            p, b = half.players, half.ball
            if view != 'native':
                p, _ = degrade(p, half.teams, b, half.pitch, profile,
                    seed_for(plan['match'], i, view), mode='none' if view == 'masked' else 'mixed')
            features = player_features(p, half.pitch)
            mask = np.zeros((len(p) + 4) // 5, bool)
            count = 0
            for start, _, _ in windows(features, half.teams):
                mask[start:start + 50] = True
                count += 1
            row = dict(half=i, view=view, complete_windows=count,
                input_coverage=float(mask.mean()),
                visible_players_per_team=[float(np.isfinite(p[:, half.teams == side]).all(-1).sum(1).mean()) for side in (0, 1)])
            rows.append(row)
            print(json.dumps(row), flush=True)
    dump(args.out / 'results.json', dict(complete=True, plan=plan, rows=rows,
        unexpected_competition_access=denied, possession_accuracy_measured=False))


if __name__ == '__main__':
    main()
