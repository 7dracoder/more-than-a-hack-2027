"""Describe frozen v4's external spell errors without changing model or decoder."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from possession.data import sha256
from possession.uncertain_decode import decode_labels
from scripts.research.spatial_memory.frozen_development import load,ORIGIN
from scripts.research.spatial_memory.run_external import dump,score_parts
from scripts.research.spatial_memory.train_real_broadcast_memory import VIEWS
from evaluation import possession_metrics as pm


def describe(truth,predicted,gate):
    t,p=pm.label_codes(truth),pm.label_codes(predicted);runs=pm.valid_runs(t)
    p=pm.fill_abstentions(p,runs);gt=pm.possession_spells(t,runs);pr=pm.possession_spells(p,runs)
    overlaps=pm.spell_overlaps(gt,pr);matched=pm.matched_gt_flags(overlaps,len(gt),.5)
    rows=[]
    for i,(start,stop,code) in enumerate(gt):
        same=[j for j,(a,b,k) in enumerate(pr) if k==code and min(b,stop)>max(a,start)]
        merged=any(sum(k==code and min(b,gb)>max(a,ga) for ga,gb,k in gt)>1 for j in same for a,b,_ in [pr[j]])
        rows.append(dict(start=start,stop=stop,team=str(pm.LABELS[code]),seconds=(stop-start)/25,
            observation_loss_fraction=float(np.mean(gate[start:stop])),matched_iou_05=bool(matched[i]),
            same_team_overlapping_predictions=len(same),merged_across_gt_spells=bool(merged),
            best_same_team_iou=max([value for value,g,j in overlaps if g==i],default=0.),
            predicted_dead_fraction=float(np.mean(p[start:stop]==pm.D)),
            predicted_opponent_fraction=float(np.mean((p[start:stop]!=code)&(p[start:stop]!=pm.D)))))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'results.json').exists():raise ValueError('Preserve completed diagnostic')
    origin,fallback,parts,hashes,denied=load(args.out)
    import torch
    state=torch.load(ORIGIN/'selected.pt',map_location='cpu',weights_only=False)
    sources=[Path(__file__),ROOT/'possession/uncertain_decode.py',ROOT/'evaluation/possession_metrics.py',ROOT/'scripts/research/spatial_memory/frozen_development.py']
    plan=dict(inputs=hashes,sources={str(p.relative_to(ROOT)):sha256(p) for p in sources},decoder_costs=state['decoder_costs'],penalty=8.,provided_match_used=False,independent_test=False)
    dump(args.out/'plan.json',plan);results={};all_rows=[]
    for view in VIEWS:
        current=[h for h in parts if h['domain']=='strong' and h['view']==view]
        predictions=[decode_labels(h['probability'],h['gate'],8.,*state['decoder_costs']) for h in current]
        score=score_parts([h['truth'] for h in current],predictions)
        assert abs(score['spell_f1_mean']-fallback['selected']['scores'][view]['spell_f1_mean'])<1e-12
        rows=[]
        for half,(h,predicted) in enumerate(zip(current,predictions)):
            rows.extend(dict(view=view,half=half,**row) for row in describe(h['truth'],predicted,h['gate']))
        missed=[r for r in rows if not r['matched_iou_05']]
        bins={}
        for name,condition in [
            ('observation_loss_majority',lambda r:r['observation_loss_fraction']>=.5),
            ('observation_loss_minority',lambda r:0<r['observation_loss_fraction']<.5),
            ('fully_observed',lambda r:r['observation_loss_fraction']==0),
            ('under_2_seconds',lambda r:r['seconds']<2),
            ('2_to_5_seconds',lambda r:2<=r['seconds']<5),
            ('5_seconds_or_longer',lambda r:r['seconds']>=5)]:
            subset=[r for r in rows if condition(r)]
            bins[name]=dict(n_gt=len(subset),matched=sum(r['matched_iou_05'] for r in subset))
        results[view]=dict(score=score,missed_at_iou_05=len(missed),
            missed_with_no_same_team_overlap=sum(r['same_team_overlapping_predictions']==0 for r in missed),
            missed_with_multiple_same_team_fragments=sum(r['same_team_overlapping_predictions']>1 for r in missed),
            missed_with_merged_prediction=sum(r['merged_across_gt_spells'] for r in missed),
            missed_majority_dead=sum(r['predicted_dead_fraction']>.5 for r in missed),
            missed_majority_opponent=sum(r['predicted_opponent_fraction']>.5 for r in missed),strata=bins)
        all_rows.extend(rows)
    for name,digest in {**plan['inputs'],**plan['sources']}.items():assert sha256(ROOT/name)==digest
    dump(args.out/'spells.json',all_rows)
    result=dict(complete=True,views=results,plan_sha256=sha256(args.out/'plan.json'),unexpected_competition_access=denied,provided_match_used=False,independent_test=False,model_or_decoder_changed=False,competition_target_verified=False,
        interpretation='Diagnostic on reused external development fixture. Categories overlap and are descriptive, not additive causes or an independent performance estimate.')
    dump(args.out/'results.json',result);print(json.dumps(result))


if __name__=='__main__':main()
