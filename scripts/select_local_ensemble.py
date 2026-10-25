"""Select a small probability ensemble on the external development match only."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_local import (load_halves,provenance,save_json,save_torch,
                               score_probabilities,selected_model)
from possession.temporal import ProbabilityEnsemble,symmetric_probability
from possession.data import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first',type=Path,default=Path('data/local-v07/run-02'))
    p.add_argument('--second',type=Path,default=Path('data/local-v07/run-05-bigru'))
    p.add_argument('--out',type=Path,default=Path('data/local-v07/run-06-ensemble'))
    args=p.parse_args()
    if args.out.exists(): raise ValueError('Preserve existing ensemble evidence; use a new output directory')
    torch.use_deterministic_algorithms(True)
    device=torch.device('cuda')
    first,a=selected_model(args.first,device); second,b=selected_model(args.second,device)
    for key in ('train_matches','validation_matches','test_matches','sample_step'):
        if a['config'][key]!=b['config'][key]: raise ValueError('Member protocols disagree: '+key)
    if a['names']!=b['names'] or not np.array_equal(a['mean'],b['mean']) or not np.array_equal(a['scale'],b['scale']):
        raise ValueError('Members require identical train-only normalization/schema')
    if sha256(args.first/'baseline.joblib')!=sha256(args.second/'baseline.joblib'):
        raise ValueError('Frozen baseline differs')
    weights=[.25,.5,.75]
    costs=[[1.,2.],[2.,3.],[3.,3.],[6.,4.]]
    args.out.mkdir(parents=True)
    parents={str(path):sha256(path/'selected.pt') for path in (args.first,args.second)}
    save_json(args.out/'ensemble_plan.json',dict(first_weights=weights,decoder_costs=costs,
              parents=parents,selection_matches=a['config']['validation_matches'],test_opened=False))
    halves,names=load_halves(a['config']['validation_matches'])
    pa=[symmetric_probability(first,h['x'],names,a['mean'],a['scale'],device) for h in halves]
    pb=[symmetric_probability(second,h['x'],names,b['mean'],b['scale'],device) for h in halves]
    records=[]
    for weight in weights:
        probabilities=[weight*x+(1-weight)*y for x,y in zip(pa,pb)]
        for cost in costs:
            score=score_probabilities(halves,probabilities,cost,a['config']['sample_step'])
            records.append(dict(first_weight=weight,costs=cost,score=score))
            save_json(args.out/'ensemble_results.json',records)
            print(json.dumps(records[-1]),flush=True)
    chosen=max(records,key=lambda r:r['score']['spell_f1_mean'])
    config=copy.deepcopy(a['config'])
    config.update(architecture='ensemble',ensemble_members=[a['config'],b['config']],
                  ensemble_weights=[chosen['first_weight'],1-chosen['first_weight']],development_decoder_costs=costs)
    record=provenance(config,config['train_matches']+config['validation_matches'])
    record.update(parent_checkpoints=parents,selection_source_sha256=sha256(Path(__file__)))
    save_json(args.out/'fixed_protocol.json',record); save_json(args.out/'config.json',config)
    model=ProbabilityEnsemble([first,second],config['ensemble_weights'])
    state=dict(model=model.state_dict(),mean=a['mean'],scale=a['scale'],names=names,config=config,
               decoder_costs=chosen['costs'],validation=chosen['score'],epoch='frozen-member ensemble',fingerprint=record['fingerprint'])
    save_torch(args.out/'selected.pt',state)
    for name in ('baseline.joblib','baseline_training.json','baseline_validation.json','environment.json'):
        shutil.copyfile(args.first/name,args.out/name)
    save_json(args.out/'selection.json',dict(validation=chosen['score'],decoder_costs=chosen['costs'],
              checkpoint_sha256=sha256(args.out/'selected.pt'),baseline_sha256=sha256(args.out/'baseline.joblib'),
              parents=parents,weights=config['ensemble_weights'],test_opened=False,official_competition_score=None))
    print('Selected ensemble '+json.dumps(chosen),flush=True)


if __name__=='__main__': main()
