"""Calibrate external whistle thresholds, freeze selection, then score one recording."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from possession.data import sha256
from possession.whistle_events import extract_events,event_score
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scores',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'results.json').exists():raise ValueError('Preserve completed final-test result')
    config_path=ROOT/'configs/audio_whistle_pilot.json';cfg=json.loads(config_path.read_text())
    annotations_path=ROOT/'data/external/volleyball-whistles-5f23e58f6057/whistles_all.json'
    paths=[args.scores/'score_freeze.json',args.scores/'plan.json']
    for name in cfg['calibration']+cfg['development']+cfg['final_test']:paths += [args.scores/(name+'_scores.npz'),args.scores/(name+'_scores.json')]
    denied=restrict_external(args.out,paths)
    freeze=json.loads(paths[0].read_text());assert freeze['complete'] and not freeze['unexpected_competition_access']
    inference_plan=json.loads(paths[1].read_text());assert inference_plan['config']==cfg
    assert sha256(paths[1])==freeze['plan_sha256']
    for name,record in freeze['records'].items():assert sha256(args.scores/(name+'_scores.npz'))==record['sha256']
    annotations=json.loads(annotations_path.read_text())
    plan=dict(config=cfg,input_sha256={str(p.resolve().relative_to(ROOT)):sha256(p) for p in paths+[annotations_path,config_path]},
        sources={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'possession/whistle_events.py']},
        threshold_tie_break='Higher threshold on equal calibration F1.',detector_tie_break='Spectral baseline on equal development F1.',
        provided_match_used=False,public_possession_labels_used=False)
    dump(args.out/'plan.json',plan)
    def data(name,method):
        with np.load(args.scores/(name+'_scores.npz')) as z:
            assert float(z['fps'])==100
            signal=z['panns'].max(1) if method=='panns' else z['spectral']
            duration=float(z['duration'])
        times=np.array(sorted(set(float(x['t_raw']) for x in annotations if x['match_id']==name)))
        assert np.isfinite(times).all() and (times>=0).all() and (times<=duration).all()
        return signal,times,duration
    def measure(name,method,threshold,save=False):
        signal,truth,duration=data(name,method)
        predicted=extract_events(signal,threshold,100,cfg['merge_gap_seconds'],cfg['refractory_seconds'])
        score=event_score(truth,predicted,cfg['event_tolerance_seconds'])
        score.update(annotations=len(truth),predictions=len(predicted),duration_seconds=duration,positive_seconds=float((signal>=threshold).sum()/100))
        if save:np.savez_compressed(args.out/(name+'_events.npz'),truth=truth,predicted=predicted)
        return score
    assert len(cfg['calibration'])==len(cfg['development'])==len(cfg['final_test'])==1
    calibration={};development={};thresholds={}
    for method in ['spectral','panns']:
        rows=[dict(threshold=t,score=measure(cfg['calibration'][0],method,t)) for t in cfg['thresholds']]
        best=max(rows,key=lambda x:(x['score']['f1'],x['threshold']))
        thresholds[method]=best['threshold'];calibration[method]=rows
        development[method]=measure(cfg['development'][0],method,best['threshold'])
    chosen=max(['spectral','panns'],key=lambda m:(development[m]['f1'],m=='spectral'))
    selection=dict(complete=True,method=chosen,threshold=thresholds[chosen],all_thresholds=thresholds,
        calibration=calibration,development=development,plan_sha256=sha256(args.out/'plan.json'),
        final_test_labels_used_for_selection=False,provided_match_used=False)
    dump(args.out/'selection_freeze.json',selection)
    selected_hash=sha256(args.out/'selection_freeze.json')
    final=measure(cfg['final_test'][0],chosen,thresholds[chosen],save=True)
    # Independently verify point matching with a maximum bipartite matching.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    with np.load(args.out/(cfg['final_test'][0]+'_events.npz')) as z:truth,predicted=z['truth'],z['predicted']
    edges=csr_matrix(np.abs(truth[:,None]-predicted[None,:])<=cfg['event_tolerance_seconds'])
    matched=int((maximum_bipartite_matching(edges,perm_type='column')>=0).sum())
    assert matched==final['tp'] and len(predicted)-matched==final['fp'] and len(truth)-matched==final['fn']
    for name,digest in {**plan['input_sha256'],**plan['sources']}.items():assert sha256(ROOT/name)==digest
    assert sha256(args.out/'selection_freeze.json')==selected_hash
    result=dict(complete=True,selected_method=chosen,selected_threshold=thresholds[chosen],
        development=development,final_test=final,selection_freeze_sha256=selected_hash,
        maximum_cardinality_matching_verified=True,final_test_used_for_selection=False,
        limitation='Three externally labelled volleyball recordings, not football possession truth. Annotation completeness and pretraining overlap are not independently certified. Whistle-event F1 is not possession spell F1.',
        provided_match_used=False,public_possession_labels_used=False,unexpected_competition_access=denied,
        integrated_into_possession=False,competition_target_verified=False)
    dump(args.out/'results.json',result);print(json.dumps(result))


if __name__=='__main__':main()
