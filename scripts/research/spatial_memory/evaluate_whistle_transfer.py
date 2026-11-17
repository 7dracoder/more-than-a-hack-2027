"""Evaluate the frozen external whistle head once on its predeclared fresh recording."""
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--features',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'results.json').exists():raise ValueError('Preserve completed final test')
    selection_path=args.run/'selection_freeze.json';selection=json.loads(selection_path.read_text())
    train_plan=json.loads((args.run/'plan.json').read_text());cfg=train_plan['config'];match=cfg['fresh_test']
    paths=[selection_path,args.run/'plan.json',args.run/'verification.json',args.run/'selected.joblib',args.features/'embedding_freeze.json',args.features/'plan.json',args.features/(match+'_embeddings.npz')]
    denied=restrict_external(args.out,paths)
    import joblib
    from threadpoolctl import threadpool_limits
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    verification=json.loads((args.run/'verification.json').read_text())
    assert verification['complete'] and verification['selection_sha256']==sha256(selection_path)
    assert not verification['unexpected_competition_access'] and not selection['final_test_accessed']
    assert sha256(args.run/'selected.joblib')==selection['model_sha256']
    assert sha256(args.run/'plan.json')==selection['plan_sha256']
    freeze=json.loads((args.features/'embedding_freeze.json').read_text())
    assert freeze['complete'] and not freeze['unexpected_competition_access']
    assert sha256(args.features/(match+'_embeddings.npz'))==freeze['records'][match]['sha256']
    annotations=ROOT/'data/external/volleyball-whistles-5f23e58f6057/whistles_all.json'
    sources=[Path(__file__),ROOT/'possession/whistle_transfer.py',ROOT/'possession/whistle_events.py']
    plan=dict(match=match,threshold=selection['chosen']['threshold'],inputs={str(p.resolve().relative_to(ROOT)):sha256(p) for p in paths+[annotations]},sources={str(p.relative_to(ROOT)):sha256(p) for p in sources},provided_match_used=False,final_test_used_for_selection=False)
    dump(args.out/'plan.json',plan)
    saved=joblib.load(args.run/'selected.joblib');assert saved['config']==cfg
    with np.load(args.features/(match+'_embeddings.npz')) as z:
        x=temporal_features(z['embedding'],cfg['temporal_offsets']);frames=int(z['output_frames']);duration=float(z['duration'])
    with threadpool_limits(limits=2):native=saved['model'].predict_proba(saved['scaler'].transform(x))[:,1]
    signal=frame_probabilities(native,frames);predicted=extract_events(signal,plan['threshold'])
    raw=json.loads(annotations.read_text());truth=np.array(sorted(set(float(r['t_raw']) for r in raw if r['match_id']==match)))
    assert (truth>=0).all() and (truth<=duration).all()
    measured=event_score(truth,predicted)
    edges=csr_matrix(np.abs(truth[:,None]-predicted[None,:])<=.5)
    matched=int((maximum_bipartite_matching(edges,perm_type='column')>=0).sum());assert matched==measured['tp']
    np.savez_compressed(args.out/(match+'_events.npz'),native_probability=native,truth=truth,predicted=predicted)
    for name,digest in {**plan['inputs'],**plan['sources']}.items():assert sha256(ROOT/name)==digest
    result=dict(complete=True,match=match,threshold=plan['threshold'],final_test=measured,duration_seconds=duration,selection_sha256=sha256(selection_path),maximum_cardinality_matching_verified=True,final_test_used_for_selection=False,previous_test_reused=False,provided_match_used=False,unexpected_competition_access=denied,integrated_into_possession=False,competition_target_verified=False,limitation='One fresh volleyball recording. Whistle event F1 is not football possession spell F1; football transfer and pretraining overlap remain unverified.')
    dump(args.out/'results.json',result);print(json.dumps(result))


if __name__=='__main__':main()
