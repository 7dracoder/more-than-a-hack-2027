"""Fit an external-only whistle head; freeze before acquiring a fresh test recording."""
import argparse
import json
from pathlib import Path
import sys
import warnings
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from possession.data import sha256
from possession.whistle_transfer import temporal_features,point_targets,frame_probabilities
from possession.whistle_events import extract_events,event_score
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--features',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'selection_freeze.json').exists():raise ValueError('Preserve frozen selection')
    config_path=ROOT/'configs/audio_whistle_transfer.json';cfg=json.loads(config_path.read_text())
    paths=[args.features/name for name in ['plan.json','embedding_freeze.json','feature_verification.json',cfg['calibration']+'_embeddings.npz',cfg['development']+'_embeddings.npz']]
    denied=restrict_external(args.out,paths)
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    raw=json.loads((ROOT/'data/external/volleyball-whistles-5f23e58f6057/whistles_all.json').read_text())
    freeze=json.loads((args.features/'embedding_freeze.json').read_text());verified=json.loads((args.features/'feature_verification.json').read_text())
    assert freeze['complete'] and verified['complete'] and verified['frontend_class_scores_reproduced'] and not freeze['unexpected_competition_access']
    for match in [cfg['calibration'],cfg['development']]:assert sha256(args.features/(match+'_embeddings.npz'))==freeze['records'][match]['sha256']
    sources=[Path(__file__),ROOT/'possession/whistle_transfer.py',ROOT/'possession/whistle_events.py']
    plan=dict(config=cfg,config_sha256=sha256(config_path),inputs={str(p.resolve().relative_to(ROOT)):sha256(p) for p in paths},sources={str(p.relative_to(ROOT)):sha256(p) for p in sources},provided_match_used=False,final_test_accessed=False)
    dump(args.out/'plan.json',plan)
    def load(match):
        with np.load(args.features/(match+'_embeddings.npz')) as z:
            x=temporal_features(z['embedding'],cfg['temporal_offsets']);times=z['time'];duration=float(z['duration']);frames=int(z['output_frames'])
        truth=np.array(sorted(set(float(row['t_raw']) for row in raw if row['match_id']==match)))
        return x,times,duration,frames,truth
    train,times,duration,frames,truth=load(cfg['calibration'])
    target=point_targets(times,truth,cfg['positive_radius_seconds'],cfg['negative_margin_seconds'])
    keep=(target>=0)&(times<duration)
    scaler=StandardScaler();x=scaler.fit_transform(train[keep]);y=target[keep]
    model=LogisticRegression(C=cfg['C'],class_weight=cfg['class_weight'],max_iter=cfg['max_iter'],random_state=cfg['seed'])
    with threadpool_limits(limits=2),warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning);model.fit(x,y)
    convergence=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
    if convergence:raise RuntimeError('Whistle head did not converge: '+convergence[0])
    development,dt,dd,df,dtruth=load(cfg['development'])
    with threadpool_limits(limits=2):native=model.predict_proba(scaler.transform(development))[:,1]
    score=frame_probabilities(native,df);table=[]
    for threshold in cfg['thresholds']:
        predicted=extract_events(score,threshold)
        table.append(dict(threshold=threshold,score=event_score(dtruth,predicted),positive_seconds=float((score>=threshold).sum()/100)))
    chosen=max(table,key=lambda row:(row['score']['f1'],row['threshold']))
    joblib.dump(dict(model=model,scaler=scaler,config=cfg),args.out/'selected.joblib')
    np.savez_compressed(args.out/'development_scores.npz',native_probability=native,truth=dtruth,frame_count=np.array(df))
    for name,digest in {**plan['inputs'],**plan['sources']}.items():assert sha256(ROOT/name)==digest
    result=dict(complete=True,chosen=chosen,development_grid=table,calibration_frames=int(keep.sum()),positive_calibration_frames=int((y==1).sum()),ignored_calibration_frames=int((~keep).sum()),features=x.shape[1],iterations=model.n_iter_.tolist(),
        model_sha256=sha256(args.out/'selected.joblib'),plan_sha256=sha256(args.out/'plan.json'),
        final_test_recording=cfg['fresh_test'],final_test_accessed=False,previous_test_reused=False,provided_match_used=False,
        unexpected_competition_access=denied,integrated_into_possession=False,competition_target_verified=False)
    dump(args.out/'selection_freeze.json',result);print(json.dumps({k:v for k,v in result.items() if k!='development_grid'}))


if __name__=='__main__':main()
