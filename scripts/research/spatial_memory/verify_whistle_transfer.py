"""Rebuild TRAIN-only normalization and independently reproduce the frozen head."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from possession.data import sha256
from possession.whistle_transfer import temporal_features,frame_probabilities
from possession.whistle_events import extract_events,event_score
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();args.run=args.run.resolve();selection=json.loads((args.run/'selection_freeze.json').read_text());plan=json.loads((args.run/'plan.json').read_text());cfg=plan['config']
    paths=[ROOT/p for p in plan['inputs']];denied=restrict_external(args.run,paths)
    import joblib
    from scipy.special import expit
    from threadpoolctl import threadpool_limits
    for name,digest in {**plan['inputs'],**plan['sources']}.items():assert sha256(ROOT/name)==digest
    assert sha256(args.run/'selected.joblib')==selection['model_sha256']
    assert sha256(args.run/'plan.json')==selection['plan_sha256']
    saved=joblib.load(args.run/'selected.joblib');assert saved['config']==cfg
    train_path=next(p for p in paths if p.name==cfg['calibration']+'_embeddings.npz')
    dev_path=next(p for p in paths if p.name==cfg['development']+'_embeddings.npz')
    raw=json.loads((ROOT/'data/external/volleyball-whistles-5f23e58f6057/whistles_all.json').read_text())
    truth=np.array(sorted(set(r['t_raw'] for r in raw if r['match_id']==cfg['calibration'])))
    with np.load(train_path) as z:
        x=temporal_features(z['embedding']);times=z['time'];duration=float(z['duration'])
    distance=np.abs(times[:,None]-truth[None,:]).min(1)
    keep=((distance<=cfg['positive_radius_seconds'])|(distance>=cfg['negative_margin_seconds']))&(times<duration)
    values=x[keep].astype(np.float64);mean=values.mean(0);variance=values.var(0)
    np.testing.assert_allclose(mean,saved['scaler'].mean_,atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(variance,saved['scaler'].var_,atol=1e-9,rtol=1e-9)
    assert keep.sum()==selection['calibration_frames'] and (distance[keep]<=cfg['positive_radius_seconds']).sum()==selection['positive_calibration_frames']
    with np.load(dev_path) as z:dx=temporal_features(z['embedding']);frames=int(z['output_frames'])
    dx-=saved['scaler'].mean_;dx/=saved['scaler'].scale_
    with threadpool_limits(limits=2):probability=expit(dx@saved['model'].coef_[0]+saved['model'].intercept_[0])
    with np.load(args.run/'development_scores.npz') as z:
        max_probability_difference=float(np.max(np.abs(probability-z['native_probability'])))
        np.testing.assert_allclose(probability,z['native_probability'],atol=5e-7,rtol=3e-6);dtruth=z['truth']
    score=frame_probabilities(probability,frames);chosen=selection['chosen'];predicted=extract_events(score,chosen['threshold']);computed=event_score(dtruth,predicted)
    assert computed==chosen['score'] and not selection['final_test_accessed']
    result=dict(complete=True,calibration_only_normalization_recomputed=True,calibration_label_membership_verified=True,
        functional_development_probabilities_reproduced=True,max_float32_probability_difference=max_probability_difference,development_metrics_reproduced=True,model_sha256=selection['model_sha256'],selection_sha256=sha256(args.run/'selection_freeze.json'),
        source_sha256=sha256(Path(__file__)),provided_match_used=False,final_test_accessed=False,unexpected_competition_access=denied,competition_target_verified=False)
    dump(args.run/'verification.json',result);print(json.dumps(result))


if __name__=='__main__':main()
