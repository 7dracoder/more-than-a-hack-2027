"""Independently rescore saved v0.6 predictions and verify provenance and controls."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from possession.data import sha256
from scripts.run_champion import pooled


def main():
    out = ROOT / 'data/experiments/blindspot-v06'
    provenance = json.loads((out / 'run_provenance.json').read_text())
    for name, digest in provenance['source_sha256'].items():
        assert sha256(ROOT / name) == digest, name
    truth_path = ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'
    assert sha256(truth_path) == provenance['public_truth_sha256']
    truth = pm.label_codes(pm.read_label_csv(truth_path))
    scores = json.loads((out / 'scores.json').read_text())
    checked = controls = 0
    for group, score in scores.items():
        checkpoint = json.loads((out / group / 'scores.json').read_text())
        for name, digest in checkpoint['artifacts_sha256'].items():
            assert sha256(out / group / name) == digest
        for protocol in ('whole_run', 'fixed_time'):
            records = json.loads((out / group / f'{protocol}_folds.json').read_text())
            independent = []
            for record in records:
                name = f"{protocol}_fold_{record['fold']}.csv"
                path = out / group / name
                assert sha256(path) == record['prediction_sha256']
                frame = pm.read_label_csv(path)
                assert pm.validate_submission(frame, len(truth)) is None
                labels = pm.label_codes(frame)
                a, b = record['test_first'], record['test_stop']
                result = pm.evaluate(truth[a:b], labels[a:b], bootstrap_n=0)
                np.testing.assert_array_equal(result['confusion'], record['score']['confusion'])
                assert np.isclose(result['spell_f1_mean'], record['score']['spell_f1_mean'])
                independent.append(result)
                checked += 1
                if group in ('baseline', 'team_only_outplay'):
                    assert sha256(path) == sha256(ROOT / 'data/experiments/blindspot-v05' / group / name)
                    controls += 1
                if group in ('v06_gated', 'v06_liveteam'):
                    control = pm.label_codes(pm.read_label_csv(out / 'team_only_outplay' / name))
                    for state in (2, 3):
                        np.testing.assert_array_equal(labels == state, control == state)
            aggregate = pooled(independent)
            for key, metric in [('spell', 'spell_f1_mean'), ('dead', 'dead_f1'), ('macro', 'macro_f1')]:
                assert np.isclose(aggregate[metric], score[protocol][key])
    report = dict(fold_csvs_rescored=checked, controls_byte_identical=controls,
                  source_and_prediction_hashes_verified=True,
                  gated_and_live_team_dead_masks_preserved=True,
                  scope='Repeatedly inspected public development sample; no independent match or cloud verification.')
    (out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
