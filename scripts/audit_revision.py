"""Compare the fixed v0.3.1 changes against frozen v0.3 with two split protocols."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from possession.data import sha256
from possession.metric_features import extract_metric, design
from possession.champion import (classifier_probability, decode_factorized, fit_classifier,
                                 purged_folds, fixed_time_folds)
from scripts.run_champion import evaluate, pooled
from scripts.run_experiment import save_json, save_submission


def decode(arrays, config, probability):
    p=probability.clip(1e-6,1.)
    return decode_factorized(arrays,dict(config,history_samples=0),p[:,2],
        np.log(p[:,:2]/p[:,:2].sum(1)[:,None]))[0]


def run(reference, revision):
    out=revision/'audit';out.mkdir(parents=True,exist_ok=True)
    old=json.loads((ROOT/'configs/champion_v3.json').read_text())
    new=json.loads((ROOT/'configs/champion_v031.json').read_text())
    arrays,_=extract_metric(ROOT/'data',revision.parent/'cache',new)
    x0,n0=design(arrays,old);x1,n1=design(arrays,new)
    truth=pd.read_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv')['label'].to_numpy()
    valid=truth!='X'
    smooth=pd.read_csv(ROOT/'evaluation/results/baseline_smooth125.csv')['label'].to_numpy()
    manifest=json.loads((reference/'manifest.json').read_text())
    old_scores=json.loads((reference/'supervised/pooled_scores.json').read_text())
    new_scores=json.loads((revision/'supervised/pooled_scores.json').read_text())
    reset_scores=[];reproduction={}
    for train,test,info in purged_folds(truth,4,300,130):
        i=info['fold'];model=joblib.load(reference/f'supervised/fold_{i}_model.joblib')
        probability=classifier_probability(model,x0,n0)
        labels=decode(arrays,old,probability)
        digest=save_submission(out/f'reproduced_v3_fold_{i}.csv',labels)
        expected=manifest['output_csv_sha256'][f'supervised/fold_{i}_predictions.csv']
        reproduction[str(i)]={'sha256':digest,'matches_frozen_reference':digest==expected}
        if digest!=expected:raise ValueError('Historical feature/decoder reproduction failed')
        reset=decode(arrays,dict(old,reset_owner_at_stoppage=True),probability)
        save_submission(out/f'owner_reset_only_fold_{i}.csv',reset)
        reset_scores.append(evaluate(truth,reset,test))
    fixed={name:[] for name in ('frozen_v3_method','audit_revision','smooth125')};fold_audits=[]
    for train,test,info in fixed_time_folds(valid):
        fold_audits.append(dict(info,train_frames=int(train.sum()),test_frames=int(test.sum())))
        for name,matrix,names,config in [('frozen_v3_method',x0,n0,old),('audit_revision',x1,n1,new)]:
            model,_=fit_classifier(matrix,names,truth,train,config)
            labels=decode(arrays,config,classifier_probability(model,matrix,names))
            save_submission(out/f"fixed_time_{name}_{info['fold']}.csv",labels)
            fixed[name].append(evaluate(truth,labels,test))
        fixed['smooth125'].append(evaluate(truth,smooth,test))
        print('Fixed-time audit completed fold',info['fold'],flush=True)
    fixed_pooled={name:dict(pooled(scores),scope='Pooled counts over four equal-duration, 300-frame-embargo public development folds') for name,scores in fixed.items()}
    result={'whole_run_development':{'frozen_v3':old_scores['cross_fitted_classifier'],
             'owner_reset_only':pooled(reset_scores),'audit_revision':new_scores['cross_fitted_classifier'],
             'smooth125':old_scores['baseline_smooth125']},'fixed_time_development':fixed_pooled,
             'historical_reproduction':reproduction,'feature_counts':{'v3':len(n0),'revision':len(n1)},
             'scope':'Same previously inspected 7500-frame public sample. Additional split audit is not new data, independent validation or statistical proof. No hyperparameter sweep.',
             'method':'Frozen v3 model files used only for whole-run reproduction/decoder ablation; both methods retrained separately for each fixed-time fold.',
             'new_candidate_sha256':sha256(revision/'candidate/CoreOrigin.csv')}
    save_json(out/'comparison.json',result);save_json(out/'fixed_time_fold_scores.json',fixed);save_json(out/'fixed_time_fold_audits.json',fold_audits)
    for scope in ('whole_run_development','fixed_time_development'):
        print(scope,{k:round(v['spell_f1_mean'],4) for k,v in result[scope].items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=ROOT/'data/experiments/blindspot-v3')
    parser.add_argument('--revision',type=Path,default=ROOT/'data/experiments/blindspot-v031')
    args=parser.parse_args();run(args.reference,args.revision)
