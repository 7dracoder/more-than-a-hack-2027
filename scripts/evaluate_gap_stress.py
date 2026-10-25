"""Development-only clean/synthetic-gap comparison, never a new independent test."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import torch
from scripts.run_local import load_halves,save_json,score_probabilities,selected_model,reference_score
from possession.champion import classifier_probability
from possession.external import swap_features
from possession.temporal import symmetric_probability


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args(); device=torch.device('cuda')
    model,state=selected_model(args.run,device); config=state['config']
    baseline=joblib.load(args.run/'baseline.joblib')
    reference_names=json.loads((args.run/'baseline_training.json').read_text())['feature_names']
    native,names=load_halves(config['validation_matches'],ROOT/config.get('features_dir','data/external/features-v07'))
    version=config.get('feature_version','v07')
    default_stress=(config.get('features_dir',f'data/external/features-{version}')+'-corrupt'
                    if version in ('v08','v09','v13') else 'data/external/features-v07-gaps')
    stress_dir=config.get('augmentation_features_dir',default_stress)
    gaps,gap_names=load_halves(config['validation_matches'],ROOT/stress_dir)
    if names!=gap_names: raise ValueError('Schema differs')
    if any(not np.array_equal(a['truth'],b['truth']) for a,b in zip(native,gaps)):
        raise ValueError('Stress view changed target labels')
    records={}
    for title,halves in [('native',native),('synthetic_missing_ball',gaps)]:
        tp=[symmetric_probability(model,h['x'],names,state['mean'],state['scale'],device) for h in halves]
        records[title]=dict(baseline=reference_score(config,baseline,halves,names,reference_names),
                           temporal=score_probabilities(halves,tp,state['decoder_costs'],config['sample_step']))
    records['scope']='Previously used external development match only; artificial coordinate gaps, unchanged official targets. Not natural-gap or independent-test evidence.'
    save_json(args.run/'development_gap_stress.json',records)
    print(json.dumps(records,indent=2),flush=True)


if __name__=='__main__': main()
