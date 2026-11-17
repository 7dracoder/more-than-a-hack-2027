"""Estimate a motion plausibility bound from the existing external training pool.

Only the six already-designated Sportec training matches are read. No validation
match, BAR-ATH input, video, or possession evaluation is opened.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from possession.external import read_sportec

TRAIN = ['J03WMX', 'J03WN1', 'J03WOH', 'J03WOY', 'J03WQQ', 'J03WR9']


def digest(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def live_speeds(ball, labels, fps=25):
    ball, labels = np.asarray(ball), np.asarray(labels)
    good = np.isfinite(ball).all(1) & np.isin(labels, ['H', 'A'])
    return np.linalg.norm(np.diff(ball.astype(float), axis=0), axis=1)[good[1:] & good[:-1]] * fps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sportec', type=Path, default=ROOT / 'data/external/sportec')
    parser.add_argument('--out', type=Path, default=ROOT / 'data/research/target080/external_calibration/motion.json')
    args = parser.parse_args()
    manifest = json.loads((args.sportec / 'download_manifest.json').read_text(encoding='utf-8'))
    pinned = {x['name']: x['sha256'] for x in manifest['files']}
    values, sources, counts = [], {}, {}
    for match in TRAIN:
        metadata = list(args.sportec.glob('*matchinformation*' + match + '.xml'))
        positions = list(args.sportec.glob('*positions*' + match + '.xml'))
        if len(metadata) != 1 or len(positions) != 1:
            raise ValueError('Need one published metadata and position file per external training match')
        for path in metadata + positions:
            actual = digest(path)
            if pinned.get(path.name) != actual:
                raise ValueError('External training input differs from publisher-verified manifest')
            sources[path.name] = actual
        speeds = [live_speeds(h.ball, h.labels) for h in read_sportec(metadata[0], positions[0])]
        counts[match] = sum(len(x) for x in speeds)
        values.extend(speeds)
        print('External motion fit:', match, counts[match], 'live transitions', flush=True)
    values = np.concatenate(values)
    values = values[np.isfinite(values)]
    bound = float(np.quantile(values, .999))
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError('No valid positive external motion bound')
    result = dict(kind='External-only motion plausibility bound', fps=25, speed_quantile=.999,
                  max_speed_mps=bound, training_matches=TRAIN, live_transition_counts=counts,
                  input_sha256=sources, source_sha256=digest(__file__),
                  adapter_sha256=digest(ROOT / 'possession/external.py'),
                  provided_match_data_used=False, validation_match_read=False,
                  rights='Sportec Open DFL dataset, Bassek et al. 2025, CC BY 4.0',
                  attribution_doi='10.1038/s41597-025-04505-y',
                  possession_score=None,
                  limitation='An external empirical speed bound is a plausibility filter, not proof of ball identity.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'}, indent=2))


if __name__ == '__main__':
    main()
