"""Four global event-prior probes on a frozen external-development model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from possession.data import sha256
from possession.temporal import symmetric_probability
from scripts.run_local import load_halves,provenance,save_json,save_torch,score_probabilities,selected_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    selection=json.loads((args.source/'selection.json').read_text())
    parent_hash=sha256(args.source/'selected.pt')
    if selection['checkpoint_sha256']!=parent_hash:
        raise ValueError('Parent checkpoint differs from completed fit')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    model,state=selected_model(args.source,'cuda')
    if not getattr(model,'has_boundary_head',False):
        raise ValueError('A trained boundary head is required')
    original=list(state['decoder_costs'])[:3]
    if len(original)!=3 or original[2]<=0:
        raise ValueError('Start from a selected active-boundary decoder')
    log_weight=math.log(model.positive_weight)
    costs=[original+[factor*log_weight] for factor in (0.,.5,1.,1.5)]
    args.out.mkdir(parents=True,exist_ok=False)
    save_json(args.out/'search_plan.json',dict(parent_sha256=parent_hash,costs=costs,
          scope='Fixed external development only; no weights fitted and no test opened'))
    config=dict(state['config'],development_decoder_costs=costs)
    halves,names=load_halves(config['validation_matches'],ROOT/config.get('features_dir','data/external/features-v07'))
    probabilities=[symmetric_probability(model,h['x'],names,state['mean'],state['scale'],'cuda') for h in halves]
    records=[]
    for cost in costs:
        score=score_probabilities(halves,probabilities,cost,config['sample_step'])
        records.append(dict(costs=cost,score=score))
        save_json(args.out/'search_results.json',records)
        print(json.dumps(records[-1]),flush=True)
    if records[0]['score']!=state['validation']:
        raise ValueError('Zero-bias probe did not exactly reproduce the frozen parent')
    chosen=max(records,key=lambda r:r['score']['spell_f1_mean'])
    record=provenance(config,config['train_matches']+config['validation_matches'])
    record['sources']['scripts/select_boundary_prior.py']=sha256(Path(__file__))
    record['parent_checkpoint_sha256']=parent_hash
    record['fingerprint']=hashlib.sha256(json.dumps({k:v for k,v in record.items() if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
    save_json(args.out/'fixed_protocol.json',record); save_json(args.out/'config.json',config)
    state.update(config=config,decoder_costs=chosen['costs'],validation=chosen['score'],fingerprint=record['fingerprint'])
    save_torch(args.out/'selected.pt',state)
    for filename in ('baseline.joblib','baseline_training.json','baseline_validation.json','environment.json'):
        shutil.copyfile(args.source/filename,args.out/filename)
    save_json(args.out/'selection.json',dict(validation=chosen['score'],decoder_costs=chosen['costs'],
              parent_checkpoint_sha256=parent_hash,checkpoint_sha256=sha256(args.out/'selected.pt'),
              baseline_sha256=sha256(args.out/'baseline.joblib'),test_opened=False,official_competition_score=None))


if __name__=='__main__':main()
