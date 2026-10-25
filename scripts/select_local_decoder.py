"""Refine a frozen network's decoder on the external development match only.

Preserves the parent run, model weights and baseline. No training or test labels
are opened. Every considered cost pair and its development result is retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_local import (load_halves, provenance, save_json, save_torch,
                                score_probabilities, selected_model)
from possession.data import sha256
from possession.temporal import symmetric_probability
import torch
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('data/local-v07/run-02'))
    p.add_argument('--out', type=Path, default=Path('data/local-v07/run-03-decoder'))
    p.add_argument('--device', default='cuda', choices=['cuda','cpu'])
    p.add_argument('--audit',type=Path,help='Use a checksum-linked frozen development probability cache')
    p.add_argument('--costs-json',type=Path,help='Predeclared JSON list of team/dead/boundary-weight[/boundary-bias] costs')
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('Use a new output directory; preserve earlier selection evidence')
    device = torch.device('cpu' if args.audit else args.device)
    parent_hash=sha256(args.source/'selected.pt')
    selection=json.loads((args.source/'selection.json').read_text())
    if selection['checkpoint_sha256']!=parent_hash:
        raise ValueError('Parent differs from its completed selection')
    if args.audit:
        state=torch.load(args.source/'selected.pt',map_location='cpu',weights_only=False)
        if state['fingerprint']!=json.loads((args.source/'fixed_protocol.json').read_text())['fingerprint']:
            raise ValueError('Parent protocol differs')
    else:
        model, state = selected_model(args.source, device)
    config = dict(state['config'])
    costs = [[team,dead,0.] for team in (.5,1.,1.5,2.,3.) for dead in (2.,3.,4.)]
    if args.costs_json:
        costs=json.loads(args.costs_json.read_text())
        if not isinstance(costs,list) or not costs or any(not isinstance(c,list) or len(c) not in (2,3,4) or not all(isinstance(v,(int,float)) and np.isfinite(v) for v in c) or min(c[:3])<0 for c in costs):
            raise ValueError('Decoder costs must be a nonempty finite transition grid')
    if list(state['decoder_costs']) not in costs:
        costs.append(list(state['decoder_costs']))
    config['development_decoder_costs'] = costs
    args.out.mkdir(parents=True)
    # Write the entire search before reading any development predictions/scores.
    save_json(args.out / 'decoder_search_plan.json', dict(costs=costs,
              parent_checkpoint_sha256=sha256(args.source/'selected.pt'),
              selection_matches=config['validation_matches'], test_matches_opened=False,
              reason=('Predeclared calibration grid for a frozen boundary model.' if args.costs_json else
                      'Initial model underpredicts development possessions; test lower transition penalties.'),
              supplied_cost_grid_sha256=sha256(args.costs_json) if args.costs_json else None))
    save_json(args.out / 'config.json', config)
    if args.audit:
        audit=json.loads((args.audit/'audit.json').read_text())
        if audit['selected_checkpoint_sha256']!=parent_hash or Path(audit['run']).resolve()!=args.source.resolve():
            raise ValueError('Cached probabilities belong to another checkpoint')
        with np.load(args.audit/'probabilities.npz',allow_pickle=False) as z:
            probabilities=[z[f'p{i}'] for i in range(2)]
            halves=[dict(truth=z[f'truth{i}']) for i in range(2)]
        names=state['names']
        if score_probabilities(halves,probabilities,state['decoder_costs'],config['sample_step'])!=state['validation']:
            raise ValueError('Cached predictions do not reproduce the frozen parent score')
    else:
        halves,names = load_halves(config['validation_matches'],ROOT/config.get('features_dir','data/external/features-v07'))
        probabilities = [symmetric_probability(model,h['x'],names,state['mean'],state['scale'],device) for h in halves]
    records = []
    for cost in costs:
        score = score_probabilities(halves,probabilities,cost,config['sample_step'])
        records.append(dict(costs=cost,score=score))
        save_json(args.out/'decoder_search_results.json',records)
        print(json.dumps(records[-1]),flush=True)
    chosen = max(records,key=lambda r:r['score']['spell_f1_mean'])
    record = provenance(config,config['train_matches']+config['validation_matches'])
    record['decoder_selection_source_sha256'] = sha256(Path(__file__))
    record['parent_checkpoint_sha256'] = sha256(args.source/'selected.pt')
    if record['parent_checkpoint_sha256']!=parent_hash:
        raise ValueError('Parent changed during decoder selection')
    if args.audit:
        record['development_probability_cache_sha256']=sha256(args.audit/'probabilities.npz')
    record['fingerprint']=hashlib.sha256(json.dumps({k:v for k,v in record.items() if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
    save_json(args.out/'fixed_protocol.json',record)
    state.update(config=config,decoder_costs=chosen['costs'],validation=chosen['score'],fingerprint=record['fingerprint'])
    save_torch(args.out/'selected.pt',state)
    for name in ('baseline.joblib','baseline_training.json','baseline_validation.json','environment.json'):
        shutil.copyfile(args.source/name,args.out/name)
    save_json(args.out/'decoder_environment.json',dict(device=str(device),torch=torch.__version__,numpy=np.__version__,
              cached_probabilities=bool(args.audit),operation='Fixed external development decoder selection'))
    save_json(args.out/'selection.json',dict(epoch=state['epoch'],validation=chosen['score'],
              decoder_costs=chosen['costs'],checkpoint_sha256=sha256(args.out/'selected.pt'),
              baseline_sha256=sha256(args.out/'baseline.joblib'),feature_names=names,
              parent_checkpoint_sha256=record['parent_checkpoint_sha256'],test_opened=False,
              official_competition_score=None))
    print('Selected external-development decoder '+json.dumps(chosen),flush=True)


if __name__ == '__main__':
    main()
