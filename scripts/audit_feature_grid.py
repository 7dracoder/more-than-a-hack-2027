"""Prove full-rate inputs preserve targets, coarse features and frozen-reference scores."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from possession.data import sha256
from scripts.run_local import load_halves,reference_score,save_json


def main():
    baseline_dir=ROOT/'data/local-v08/run-01-contacts'
    baseline=joblib.load(baseline_dir/'baseline.joblib')
    reference_names=json.loads((baseline_dir/'baseline_training.json').read_text())['feature_names']
    records={}
    for title,suffix in [('native',''),('corrupted','-corrupt')]:
        coarse,names=load_halves(['J03WPY'],ROOT/('data/external/features-v09'+suffix))
        fine,fine_names=load_halves(['J03WPY'],ROOT/('data/external/features-v10'+suffix))
        if names!=fine_names:
            raise ValueError('Full-rate feature schema changed')
        for a,b in zip(coarse,fine):
            np.testing.assert_array_equal(a['x'],b['x'][::2])
            np.testing.assert_array_equal(a['truth'],b['truth'])
            np.testing.assert_array_equal(a['y'],b['y'][::2])
        old=reference_score(dict(sample_step=2),baseline,coarse,names,reference_names)
        new=reference_score(dict(sample_step=1,baseline_sample_step=2),baseline,fine,names,reference_names)
        if old!=new:
            raise ValueError('Reference score changed when the new input grid became finer')
        records[title]=dict(coarse_features_exact=True,targets_exact=True,coarse_reference=old,full_rate_reference=new)
        del coarse,fine
    records.update(scope='External development alignment audit only; not a new model score',
                   baseline_sha256=sha256(baseline_dir/'baseline.joblib'),source_sha256=sha256(Path(__file__)))
    save_json(ROOT/'data/local-v08/full_rate_grid_audit.json',records)
    print(json.dumps(records,indent=2),flush=True)


if __name__=='__main__':
    main()
