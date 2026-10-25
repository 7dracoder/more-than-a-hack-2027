"""Cache coordinates-only features and separate official external targets."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession.external import prepare_match


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/local_v07.json')
    parser.add_argument('--matches', nargs='*')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    matches = args.matches or sum((config[k] for k in ('train_matches', 'validation_matches', 'test_matches')), [])
    if len(matches) != len(set(matches)):
        raise ValueError('Match splits overlap')
    for match in matches:
        started = time.monotonic()
        print('Preparing external match ' + match, flush=True)
        path = prepare_match(ROOT / 'data/external/sportec', match, ROOT / 'data/external/features-v07', config['sample_step'])
        print(json.dumps(dict(match=match, seconds=round(time.monotonic() - started, 2), cache=str(path))), flush=True)


if __name__ == '__main__':
    main()
