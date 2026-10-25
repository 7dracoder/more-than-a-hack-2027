"""Small frozen multi-rate ensemble study using previously audited development outputs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from possession.data import sha256
from scripts.run_local import provenance,save_json,save_torch,score_probabilities,validate_splits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',nargs=2,type=Path,required=True)
    parser.add_argument('--audits',nargs=2,type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--state-weights',type=float,nargs=2)
    parser.add_argument('--independent-boundaries',action='store_true')
    parser.add_argument('--boundary-biases',type=float,nargs='+')
    args=parser.parse_args()
    states=[]; probabilities=[]; truths=[]; parents=[]
    for source,audit in zip(args.sources,args.audits):
        selection=json.loads((source/'selection.json').read_text())
        checkpoint_hash=sha256(source/'selected.pt')
        if selection['checkpoint_sha256']!=checkpoint_hash:
            raise ValueError('Parent differs from completed checkpoint')
        state=torch.load(source/'selected.pt',map_location='cpu',weights_only=False)
        fixed=json.loads((source/'fixed_protocol.json').read_text())
        if state['fingerprint']!=fixed['fingerprint']:
            raise ValueError('Parent provenance differs')
        validate_splits(state['config'])
        if not state['config'].get('development_only'):
            raise ValueError('Explicit development-only parents required')
        report=json.loads((audit/'audit.json').read_text())
        if report.get('selected_checkpoint_sha256',checkpoint_hash)!=checkpoint_hash:
            raise ValueError('Cached audit checkpoint differs from parent')
        if Path(report['run']).resolve()!=source.resolve():
            raise ValueError('Cached audit belongs to another run')
        with np.load(audit/'probabilities.npz',allow_pickle=False) as z:
            p=[z[f'p{i}'] for i in range(2)]
            truth=[z[f'truth{i}'] for i in range(2)]
        if any(x.shape[1]!=4 for x in p):
            raise ValueError('Both parents need state and boundary probabilities')
        score=score_probabilities([dict(truth=t) for t in truth],p,state['decoder_costs'],state['config']['sample_step'])
        if score!=state['validation']:
            raise ValueError('Cached parent probabilities do not reproduce its frozen score')
        states.append(state); probabilities.append(p); truths.append(truth)
        parents.append(dict(run=str(source),checkpoint_sha256=checkpoint_hash,
                            audit_probability_sha256=sha256(audit/'probabilities.npz')))
    if states[0]['config']['validation_matches']!=states[1]['config']['validation_matches']:
        raise ValueError('Development cohorts differ')
    for a,b in zip(*truths):
        np.testing.assert_array_equal(a,b)
    finest=min(range(2),key=lambda i:(states[i]['config']['sample_step'],-len(states[i]['names'])))
    config=copy.deepcopy(states[finest]['config'])
    if config['sample_step']!=1:
        raise ValueError('This study explicitly uses a 25-Hz outer timeline')
    if any(name not in states[finest]['names'] for state in states for name in state['names']):
        raise ValueError('The finest outer bank must include every member feature')
    config.update(version='0.13-frozen-multi-rate-ensemble',architecture='feature_ensemble',
                  ensemble_parents=parents,member_sample_steps=[s['config']['sample_step'] for s in states])
    weights=[args.state_weights] if args.state_weights is not None else [[.25,.75],[.5,.5],[.75,.25]]
    if any(not np.isfinite(w).all() or min(w)<0 or abs(sum(w)-1)>1e-6 for w in weights):
        raise ValueError('State weights must be finite, nonnegative and sum to one')
    biases=args.boundary_biases if args.boundary_biases is not None else ([0.,.75,1.5] if args.independent_boundaries else [0.])
    if not np.isfinite(biases).all():raise ValueError('Boundary bias grid must be finite')
    boundary_grid=[[1.,0.],[.5,.5],[0.,1.]] if args.independent_boundaries else []
    costs=config['development_decoder_costs']
    args.out.mkdir(parents=True,exist_ok=False)
    save_json(args.out/'search_plan.json',dict(parents=parents,weights=weights,costs=costs,boundary_grid=boundary_grid,biases=biases,
              scope='Fixed external development only; coarse members keep their original time grid',test_opened=False))
    expanded=[[np.repeat(p,s['config']['sample_step'],axis=0)[:len(t)] for p,t in zip(ps,truths[0])]
              for s,ps in zip(states,probabilities)]
    halves=[dict(truth=t) for t in truths[0]]
    records=[]
    for weight in weights:
        boundary_choices=[weight]+[w for w in boundary_grid if w!=weight]
        for boundary_weight in boundary_choices:
            mixed=[np.column_stack([sum(w*p[i][:,:3] for w,p in zip(weight,expanded)),
                                    sum(w*p[i][:,3] for w,p in zip(boundary_weight,expanded))]) for i in range(2)]
            for base_cost in costs:
                for bias in (biases if base_cost[2] else [0.]):
                    cost=list(base_cost)+([bias] if bias else [])
                    score=score_probabilities(halves,mixed,cost,1)
                    records.append(dict(weights=weight,boundary_weights=boundary_weight,costs=cost,score=score))
                    save_json(args.out/'search_results.json',records)
    chosen=max(records,key=lambda r:r['score']['spell_f1_mean'])
    config['ensemble_weights']=chosen['weights']
    config['ensemble_boundary_weights']=chosen['boundary_weights']
    if args.independent_boundaries:
        config['version']='0.16-frozen-independent-transition-blend'
    for state,weight in zip(states,chosen['boundary_weights']):
        state['ensemble_boundary_weight']=weight
    record=provenance(config,config['train_matches']+config['validation_matches'])
    record['sources']['scripts/select_rate_ensemble.py']=sha256(Path(__file__))
    record['fingerprint']=hashlib.sha256(json.dumps({k:v for k,v in record.items() if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
    save_json(args.out/'fixed_protocol.json',record); save_json(args.out/'config.json',config)
    names=states[finest]['names']
    state=dict(members=states,weights=chosen['weights'],config=config,names=names,
               mean=np.zeros(len(names),np.float32),scale=np.ones(len(names),np.float32),
               decoder_costs=chosen['costs'],validation=chosen['score'],fingerprint=record['fingerprint'],epoch='Frozen multi-rate ensemble')
    save_torch(args.out/'selected.pt',state)
    for filename in ('baseline.joblib','baseline_training.json','baseline_validation.json'):
        shutil.copyfile(args.sources[finest]/filename,args.out/filename)
    parent_environment=json.loads((args.sources[finest]/'environment.json').read_text())
    save_json(args.out/'environment.json',dict(device='cpu',operation='Frozen cached probability selection; no refitting',
              member_training_environment=parent_environment,torch=torch.__version__,numpy=np.__version__))
    save_json(args.out/'selection.json',dict(validation=chosen['score'],decoder_costs=chosen['costs'],weights=chosen['weights'],boundary_weights=chosen['boundary_weights'],
              checkpoint_sha256=sha256(args.out/'selected.pt'),baseline_sha256=sha256(args.out/'baseline.joblib'),parents=parents,
              test_opened=False,official_competition_score=None,target_met_on_development=chosen['score']['spell_f1_mean']>=.9))
    print(json.dumps(chosen),flush=True)


if __name__=='__main__': main()
