"""Prepare higher-resolution coordinate-only native and corrupted external views."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from possession.data import sha256
from possession.external import read_sportec
from possession.features_v08 import build_features
from scripts.prepare_external_gaps import missing_intervals


def corrupt_tracking(players, ball, seed):
    """Missing observations/noise only; labels never enter this function."""
    rng = np.random.default_rng(seed)
    players, ball = players.copy(), ball.copy()
    ball[missing_intervals(len(ball), seed)] = np.nan
    # Sparse per-slot losses, independently generated without state/label inputs.
    for slot in range(players.shape[1]):
        start = int(rng.integers(100, 500))
        while start < len(players):
            length = int(rng.integers(5, 100))
            players[start:start+length, slot] = np.nan
            start += length+int(rng.integers(200, 1500))
    players += rng.normal(0, .08, players.shape).astype(np.float32)
    ball += rng.normal(0, .08, ball.shape).astype(np.float32)
    return players, ball


def prepare(match, step, augmented):
    folder = ROOT/'data/external/sportec'
    cache = ROOT/('data/external/features-v08-corrupt' if augmented else 'data/external/features-v08')
    cache.mkdir(parents=True, exist_ok=True)
    path = cache/(match+'.npz')
    metadata = next(folder.glob('*matchinformation*'+match+'.xml'))
    tracking = next(folder.glob('*positions*'+match+'.xml'))
    provenance = dict(source_sha256=sha256(ROOT/'possession/external.py'),
                      feature_source_sha256=sha256(ROOT/'possession/features_v08.py'),
                      preparation_source_sha256=sha256(Path(__file__)),
                      tracking_sha256=sha256(tracking), metadata_sha256=sha256(metadata),
                      gap_mask_source_sha256=sha256(ROOT/'scripts/prepare_external_gaps.py'),
                      step=step, augmentation=augmented)
    if path.exists() and path.with_suffix('.json').exists():
        old = json.loads(path.with_suffix('.json').read_text())
        if old['provenance'] == provenance and old['sha256'] == sha256(path):
            return
    arrays, records = {}, []
    for i, half in enumerate(read_sportec(metadata, tracking)):
        seed = int(hashlib.sha256((match+'/'+str(i)+'/v08').encode()).hexdigest()[:8], 16)
        players, ball = (corrupt_tracking(half.players, half.ball, seed) if augmented else (half.players, half.ball))
        x, names = build_features(players, half.teams, ball, half.pitch, step=step)
        arrays.update({f'x{i}': x, f'y{i}': half.labels[::step], f'truth{i}': half.labels, f'frames{i}': half.frames})
        records.append(dict(period=half.period, frames=len(half.frames), seed=seed))
    arrays['names'] = np.array(names)
    partial = path.with_suffix('.partial.npz')
    np.savez_compressed(partial, **arrays); partial.replace(path)
    path.with_suffix('.json').write_text(json.dumps(dict(provenance=provenance, sha256=sha256(path), halves=records), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matches', nargs='+', default=['J03WMX','J03WN1','J03WOH','J03WOY','J03WPY'])
    p.add_argument('--step', type=int, default=2)
    p.add_argument('--corrupt', action='store_true')
    args = p.parse_args()
    if args.step < 1:
        p.error('--step must be positive')
    for match in args.matches:
        started = time.monotonic()
        print('Preparing v08 '+match+(' corrupt' if args.corrupt else ' native'), flush=True)
        prepare(match, args.step, args.corrupt)
        print(json.dumps(dict(match=match, seconds=round(time.monotonic()-started, 2))), flush=True)
