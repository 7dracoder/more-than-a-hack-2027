"""Optional private video/pitch review for the v0.3 supervised candidate."""



import argparse

import json

from pathlib import Path

import sys



ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))



import numpy as np

import pandas as pd



from possession.data import sha256

from possession.review import build_review

from scripts.run_experiment import save_json





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--experiment', default=str(ROOT / 'data/experiments/blindspot-v3'))

    parser.add_argument('--data-dir', default=str(ROOT / 'data'))

    parser.add_argument('--budget', type=int, default=12)

    parser.add_argument('--no-video', action='store_true')

    args = parser.parse_args()

    if not 0 <= args.budget <= 12:

        parser.error('budget must be 0..12')

    experiment = Path(args.experiment)

    manifest = json.loads((experiment / 'manifest.json').read_text())

    csv = experiment / 'candidate/CoreOrigin.csv'

    expected = manifest['output_csv_sha256']['candidate/CoreOrigin.csv']

    if sha256(csv) != expected:

        raise ValueError('Candidate CSV hash mismatch')

    labels = pd.read_csv(csv)['label'].to_numpy()

    with np.load(experiment / 'supervised/full_public_inference.npz') as arrays:

        posterior = arrays['posterior']

    queries = json.loads((experiment / 'review_queries.json').read_text())

    notice = 'Displayed classifier fits all public labels. Queries follow exported state boundaries and model disagreement. Support is model-relative; v0.3.1 owner support is conditional on the decoded live/dead segmentation. Notes never alter the candidate.'

    result = build_review(args.data_dir, experiment / 'review', queries, labels, posterior,

                          encode=not args.no_video, budget=args.budget, experiment_id=expected,

                          title='Review the v' + manifest['version'] + ' candidate', notice=notice)

    result['candidate_csv_sha256'] = expected

    result['notice'] = notice

    save_json(experiment / 'review/summary.json', result)

    print(json.dumps(result, indent=2))

