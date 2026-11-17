"""Independently rebuild TRAIN statistics and frozen boundary-grid metrics."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from possession.data import sha256
from possession.observation_boundary import mix_boundaries
from possession.uncertain_decode import decode_labels
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump,score_parts
from scripts.research.spatial_memory.train_real_broadcast_memory import VIEWS
from scripts.research.spatial_memory.train_partial_had_memory import selection_result
from scripts.research.spatial_memory.audit_broadcast_transfer import weak_counts,summarize_counts


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--features',type=Path,required=True);args=parser.parse_args()
    selection=json.loads((args.run/'selection.json').read_text());plan=selection['plan'];cfg=plan['config']
    assert selection['complete'] and selection['backbone_frozen'] and not selection['unexpected_competition_access']
    denied=restrict_external(args.run,[ROOT/name for name in plan['inputs']])
    import joblib
    import torch
    from scipy.special import expit
    from threadpoolctl import threadpool_limits
    for name,digest in {**plan['inputs'],**plan['sources']}.items():assert sha256(ROOT/name)==digest
    assert sha256(args.run/'selected.joblib')==selection['model_sha256']
    saved=joblib.load(args.run/'selected.joblib');assert saved['config']==cfg
    feature_plan=json.loads((args.features/'plan.json').read_text());freeze=json.loads((args.features/'embedding_freeze.json').read_text())
    assert sha256(args.features/'embedding_freeze.json')==plan['preparation_sha256']
    assert sha256(ROOT/cfg['initial_model']/'selected.pt')==feature_plan['model_sha256']
    inventory=json.loads((args.run/'training_inventory.json').read_text());by_stem={r['stem']:r for r in inventory['rows']}
    total=0.;first=np.zeros(384);second=np.zeros(384);class_counts=np.zeros(2);nrows=npositive=0
    for r in freeze['records']:
        if not r['train']:continue
        assert r['domain']=='strong' and r['stem'] in by_stem
        labels=np.load(ROOT/r['truth'])[::2]
        valid=np.zeros(len(labels),bool);valid[1:]=np.isin(labels[1:],['H','A','D'])&np.isin(labels[:-1],['H','A','D'])
        positive=np.zeros(len(labels),bool);positive[1:]=labels[1:]!=labels[:-1];positive &= valid
        negatives=np.flatnonzero(valid&~positive);selected_negatives=negatives[::cfg['negative_stride']]
        indices=np.sort(np.r_[np.flatnonzero(positive),selected_negatives]);multiplier=len(negatives)/len(selected_negatives)
        expected=by_stem[r['stem']]
        assert expected['positive']==int(positive.sum()) and expected['negative']==len(negatives) and expected['excluded']==int((~valid).sum())
        assert expected['selected_negative']==len(selected_negatives) and expected['negative_sampling_multiplier']==multiplier
        embedding=np.load(args.features/(r['stem']+'.npy'),mmap_mode='r')
        values=embedding[indices].reshape(-1,384).astype(np.float64)
        target=np.repeat(positive[indices],2)
        weight=np.repeat(np.where(positive[indices],1.,multiplier)*cfg['train_views'][r['view']]/2,2)
        # StandardScaler validates sample weights in the float32 input dtype.
        # Keep the classifier's separate balancing calculation in float64.
        scaler_weight=weight.astype(np.float32).astype(np.float64)
        total+=scaler_weight.sum();first+=(values*scaler_weight[:,None]).sum(0);second+=(values*values*scaler_weight[:,None]).sum(0)
        class_counts+=np.bincount(target.astype(int),weights=weight,minlength=2);nrows+=len(values);npositive+=int(target.sum())
    mean=first/total;variance=second/total-mean**2
    np.testing.assert_allclose(mean,saved['scaler'].mean_,atol=1e-9,rtol=1e-9)
    np.testing.assert_allclose(variance,saved['scaler'].var_,atol=1e-8,rtol=1e-8)
    assert nrows==inventory['total_rows'] and npositive==inventory['positive_rows']
    shift=float(np.log(class_counts[0]/class_counts[1]));assert abs(shift-saved['log_weight_odds'])<1e-10
    print('TRAIN membership, sampling, normalization and prior correction verified',flush=True)
    parts=[];max_difference=0.
    for r in freeze['records']:
        if r['train']:continue
        embedding=np.load(args.features/(r['stem']+'.npy'),mmap_mode='r');shape=embedding.shape
        z=embedding.reshape(-1,384).copy();z-=saved['scaler'].mean_;z/=saved['scaler'].scale_
        with threadpool_limits(limits=2):logits=z@saved['model'].coef_[0]+saved['model'].intercept_[0]
        manual=expit(logits-shift).reshape(shape[:2]).mean(1)
        with np.load(args.run/(r['stem']+'_boundary.npz')) as f:native=f['native_probability']
        np.testing.assert_allclose(manual,native,atol=1e-6,rtol=3e-6);max_difference=max(max_difference,float(np.max(np.abs(manual-native))))
        with np.load(ROOT/r['reference']) as f:h={k:f[k] for k in ('probability','expert','gate','memory','memory_names')}
        h.update(domain=r['domain'],view=r['view']);h['expert']=h['expert'].copy();h['expert'][:,3]=manual
        if r['domain']=='strong':h['truth']=np.load(ROOT/r['truth'])
        else:
            with np.load(ROOT/r['truth']) as f:h.update(truth=f['truth'],ticks=f['ticks'],seen=f['detected_ball_seen'])
        parts.append(h)
    fallback=json.loads((ROOT/cfg['fallback']/'selection.json').read_text());assert sha256(ROOT/cfg['fallback']/'selection.json')==plan['fallback_sha256']
    state=torch.load(ROOT/cfg['initial_model']/'selected.pt',map_location='cpu',weights_only=False)
    computed=[]
    for expected in selection['candidates']:
        strength=expected['boundary_strength'];predicted=[]
        for h in parts:
            p=mix_boundaries(h['probability'],h['expert'],h['memory'],h['memory_names'].tolist(),strength)
            predicted.append(decode_labels(p,h['gate'],cfg['continuity_penalty'],*state['decoder_costs']))
        scores={}
        for view in VIEWS:
            indices=[i for i,h in enumerate(parts) if h['domain']=='strong' and h['view']==view]
            scores[view]=score_parts([parts[i]['truth'] for i in indices],[predicted[i] for i in indices])
        weak={'all_known':[],'ball_missing':[]}
        for i,h in enumerate(parts):
            if h['domain']!='weak':continue
            for scope,mask in [('all_known',np.ones(len(h['ticks']),bool)),('ball_missing',~h['seen'])]:weak[scope].append(weak_counts(h['probability'][h['ticks']],predicted[i][h['ticks']],h['truth'],mask))
        weak={scope:summarize_counts(values) for scope,values in weak.items()}
        row=dict(boundary_strength=strength,**selection_result(scores,weak,fallback['selected']))
        assert row['eligible']==expected['eligible'] and row['guards']==expected['guards']
        assert abs(row['objective']-expected['objective'])<1e-12
        for view in VIEWS:
            for metric in ('spell_f1_mean','macro_f1'):assert abs(scores[view][metric]-expected['scores'][view][metric])<1e-12
        for scope in weak:
            for metric in ('weak_ha_accuracy','conditional_ha_accuracy'):assert abs(weak[scope][metric]-expected['weak_scores'][scope][metric])<1e-12
        computed.append(row)
    assert max((r for r in computed if r['eligible']),key=lambda r:r['objective'])['boundary_strength']==selection['selected']['boundary_strength']
    for name,digest in {**plan['inputs'],**plan['sources']}.items():assert sha256(ROOT/name)==digest
    result=dict(complete=True,training_statistics_independently_recomputed=True,training_membership_and_unknown_masks_verified=True,class_odds_correction_verified=True,functional_probabilities_verified=True,max_float32_probability_difference=max_difference,all_candidates_reproduced=True,selected_strength=selection['selected']['boundary_strength'],selected_objective=selection['selected']['objective'],selection_sha256=sha256(args.run/'selection.json'),verifier_sha256=sha256(Path(__file__)),backbone_frozen=True,provided_match_used=False,unexpected_competition_access=denied,independent_test=False,competition_target_verified=False)
    dump(args.run/'verification.json',result);print(json.dumps(result))


if __name__=='__main__':main()
