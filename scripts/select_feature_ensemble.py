"""Select a small frozen ensemble on the fixed external development match only."""
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
from possession.temporal import symmetric_probability
from scripts.run_local import (load_halves,provenance,save_json,save_torch,score_probabilities,
                               selected_model,validate_splits)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if len(args.sources) not in (2,3):
        parser.error('Use two or three frozen members')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    device=torch.device('cuda')
    models,states,parents=[],[],[]
    for source in args.sources:
        selection=json.loads((source/'selection.json').read_text())
        checkpoint_hash=sha256(source/'selected.pt')
        if selection['checkpoint_sha256']!=checkpoint_hash:
            raise ValueError('Parent differs from completed selection receipt')
        model,state=selected_model(source,device)
        validate_splits(state['config'])
        models.append(model); states.append(state)
        parents.append(dict(run=str(source),checkpoint_sha256=checkpoint_hash,fingerprint=state['fingerprint']))
    if any(s['config']['validation_matches']!=states[0]['config']['validation_matches'] or s['config']['sample_step']!=states[0]['config']['sample_step'] for s in states):
        raise ValueError('Members need identical development matches and temporal resolution')
    largest=max(states,key=lambda s:len(s['names']))
    config=copy.deepcopy(largest['config'])
    config.update(architecture='feature_ensemble',development_only=True,test_matches=[],
                  train_matches=list(dict.fromkeys(m for state in states for m in state['config']['train_matches'])),
                  ensemble_parents=parents,test_policy='Development-only frozen ensemble; no independent test available.')
    validate_splits(config)
    weights=([[.25,.75],[.5,.5],[.75,.25]] if len(states)==2 else [[.75,.25,0.],[.5,.5,0.],[.5,.25,.25],[.75,0.,.25]])
    costs=[[4,6,0],[2,3,0],[0,1,.5],[1,2,.5],[1,2,1]]
    args.out.mkdir(parents=True,exist_ok=False)
    save_json(args.out/'search_plan.json',dict(parents=parents,weights=weights,costs=costs,
              selection_matches=config['validation_matches'],test_opened=False,
              scope='Exploratory fixed external development match only'))
    halves,names=load_halves(config['validation_matches'],ROOT/config.get('features_dir','data/external/features-v07'))
    if names!=largest['names']:
        raise ValueError('Largest member does not match outer feature cache')
    probabilities=[]
    for model,state in zip(models,states):
        columns=[names.index(n) for n in state['names']]
        probabilities.append([symmetric_probability(model,h['x'][:,columns],state['names'],state['mean'],state['scale'],device) for h in halves])
    records=[]
    for weight in weights:
        mixed=[]
        for half_index in range(len(halves)):
            state_p=sum(w*p[half_index][:,:3] for w,p in zip(weight,probabilities))
            boundary=[(w,p[half_index][:,3]) for w,p in zip(weight,probabilities) if w>0 and p[half_index].shape[1]==4]
            if boundary:
                state_p=np.column_stack([state_p,sum(w*q for w,q in boundary)/sum(w for w,_ in boundary)])
            mixed.append(state_p)
        for cost in costs:
            if cost[2] and mixed[0].shape[1]!=4:
                continue
            score=score_probabilities(halves,mixed,cost,config['sample_step'])
            records.append(dict(weights=weight,costs=cost,score=score))
            save_json(args.out/'search_results.json',records)
            print(json.dumps(records[-1]),flush=True)
    chosen=max(records,key=lambda r:r['score']['spell_f1_mean'])
    config['ensemble_weights']=chosen['weights']
    record=provenance(config,config['train_matches']+config['validation_matches'])
    record['sources']['scripts/select_feature_ensemble.py']=sha256(Path(__file__))
    record['fingerprint']=hashlib.sha256(json.dumps({k:v for k,v in record.items() if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
    save_json(args.out/'fixed_protocol.json',record); save_json(args.out/'config.json',config)
    # Reload immutable parent states onto CPU for a portable self-contained file.
    portable=[]
    for source,parent in zip(args.sources,parents):
        if sha256(source/'selected.pt')!=parent['checkpoint_sha256']:
            raise ValueError('Parent changed while selecting ensemble')
        portable.append(torch.load(source/'selected.pt',map_location='cpu',weights_only=False))
    state=dict(members=portable,weights=chosen['weights'],config=config,names=names,
               mean=np.zeros(len(names),np.float32),scale=np.ones(len(names),np.float32),
               decoder_costs=chosen['costs'],validation=chosen['score'],fingerprint=record['fingerprint'],epoch='Frozen feature ensemble')
    save_torch(args.out/'selected.pt',state)
    reference=ROOT/config['frozen_baseline_dir']
    for filename in ('baseline.joblib','baseline_training.json','baseline_validation.json'):
        shutil.copyfile(reference/filename,args.out/filename)
    save_json(args.out/'environment.json',dict(torch=torch.__version__,device=str(device),gpu=torch.cuda.get_device_name(0),operation='Frozen-model development selection; no refitting'))
    save_json(args.out/'selection.json',dict(validation=chosen['score'],decoder_costs=chosen['costs'],weights=chosen['weights'],
              checkpoint_sha256=sha256(args.out/'selected.pt'),baseline_sha256=sha256(args.out/'baseline.joblib'),parents=parents,
              test_opened=False,official_competition_score=None,target_0_90_met_on_development=chosen['score']['spell_f1_mean']>=.9))


if __name__=='__main__':
    main()
