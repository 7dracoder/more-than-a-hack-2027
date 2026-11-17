"""Calibrate both detectors fairly on development; keep the final test closed."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from possession.data import sha256
from possession.external_object_metrics import summary
from scripts.research.video_ball.evaluate_openimages_detectors import coco_ap
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--inference',type=Path,required=True);parser.add_argument('--initial-evaluation',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'selection.json').exists():raise ValueError('Preserve frozen operating-point selection')
    freeze_path=args.inference/'prediction_freeze.json';freeze=json.loads(freeze_path.read_text());assert freeze['complete'] and freeze['split']=='development' and not freeze['unexpected_competition_access']
    paths=[freeze_path,args.inference/'plan.json',args.initial_evaluation/'results.json']+[args.inference/r['model']/(r['image_id']+'.npz') for r in freeze['records']]
    denied=restrict_external(args.out,paths)
    initial=json.loads((args.initial_evaluation/'results.json').read_text());assert initial['complete'] and initial['official_coco_api_ap_verified'] and not initial['final_test_evaluated']
    manifest_path=ROOT/'data/external/openimages-football-20261115/manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    inference_plan=json.loads((args.inference/'plan.json').read_text());assert sha256(manifest_path)==inference_plan['manifest_sha256'] and sha256(args.inference/'plan.json')==freeze['plan_sha256']
    config_path=ROOT/'configs/openimages_detector_operating_points.json';cfg=json.loads(config_path.read_text())
    sources=[Path(__file__),ROOT/'possession/external_object_metrics.py',ROOT/'scripts/research/video_ball/evaluate_openimages_detectors.py']
    plan=dict(config=cfg,config_sha256=sha256(config_path),inputs={str(p.resolve().relative_to(ROOT)):sha256(p) for p in paths},sources={str(p.relative_to(ROOT)):sha256(p) for p in sources},manifest_sha256=sha256(manifest_path),provided_match_used=False)
    dump(args.out/'plan.json',plan);grid={};best={};control=initial['metrics']['yolo11m']
    for model in ('yolo11m','rfdetr_large'):
        rows=[]
        for item in manifest['images']:
            if item['split']!='development':continue
            path=args.inference/model/(item['ImageID']+'.npz');record=next(r for r in freeze['records'] if r['model']==model and r['image_id']==item['ImageID']);assert sha256(path)==record['sha256']
            with np.load(path) as z:boxes,confidence=z['boxes'],z['confidence']
            truth=np.asarray(item['ball_boxes_normalized'],float).reshape(-1,4)*[item['width'],item['height'],item['width'],item['height']]
            rows.append(dict(boxes=boxes,confidence=confidence,truth=truth))
        table=[]
        for threshold in cfg['thresholds']:
            filtered=[dict(truth=r['truth'],boxes=r['boxes'][r['confidence']>=threshold],confidence=r['confidence'][r['confidence']>=threshold]) for r in rows]
            measured=summary(filtered);table.append(dict(threshold=threshold,metrics=measured,recall_guard=measured['recall']>=control['recall']-1e-12))
        chosen=max((r for r in table if r['recall_guard']),key=lambda r:(r['metrics']['f1'],r['threshold']))
        filtered=[dict(truth=r['truth'],boxes=r['boxes'][r['confidence']>=chosen['threshold']],confidence=r['confidence'][r['confidence']>=chosen['threshold']]) for r in rows]
        assert abs(coco_ap(filtered)-chosen['metrics']['ap50'])<1e-10
        grid[model]=table;best[model]=chosen
    challenger=best['rfdetr_large']['metrics'];baseline=best['yolo11m']['metrics']
    guards=dict(f1=challenger['f1']>baseline['f1']+1e-12,raw_ap50=initial['metrics']['rfdetr_large']['ap50']>control['ap50']+1e-12,precision=challenger['precision']>=control['precision']-1e-12,recall=challenger['recall']>=control['recall']-1e-12)
    model='rfdetr_large' if all(guards.values()) else 'yolo11m'
    for name,digest in {**plan['inputs'],**plan['sources']}.items():assert sha256(ROOT/name)==digest
    result=dict(complete=True,selected_model=model,selected_threshold=best[model]['threshold'] if model=='rfdetr_large' else .05,calibrated=best,grid=grid,guards=guards,control=control,
        final_test_accessed=False,access_scope='Final-test accuracy metrics; annotations were used for qualification/splitting.',provided_match_used=False,unexpected_competition_access=denied,official_coco_api_calibrated_ap_verified=True,plan_sha256=sha256(args.out/'plan.json'),possession_model_promoted=False,competition_target_verified=False)
    dump(args.out/'selection.json',result);print(json.dumps({k:v for k,v in result.items() if k!='grid'}))


if __name__=='__main__':main()
