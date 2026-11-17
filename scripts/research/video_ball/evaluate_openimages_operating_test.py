"""One final external check of previously frozen detector operating points."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from possession.data import sha256
from possession.external_object_metrics import summary,matches
from scripts.research.video_ball.evaluate_openimages_detectors import coco_ap
from scripts.research.video_ball.calibrate_external_confidence import detection_targets
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--inference',type=Path,required=True);parser.add_argument('--selection',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'results.json').exists():raise ValueError('Preserve final-test result')
    choice=json.loads(args.selection.read_text());freeze=json.loads((args.inference/'prediction_freeze.json').read_text());inference_plan=json.loads((args.inference/'plan.json').read_text())
    assert choice['complete'] and choice['selected_model']=='rfdetr_large' and not choice['final_test_accessed']
    assert freeze['complete'] and freeze['split']=='final_test' and not freeze['unexpected_competition_access']
    assert sha256(args.selection)==inference_plan['selection_sha256'] and sha256(args.inference/'plan.json')==freeze['plan_sha256']
    paths=[args.selection,args.inference/'prediction_freeze.json',args.inference/'plan.json']+[args.inference/r['model']/(r['image_id']+'.npz') for r in freeze['records']]
    denied=restrict_external(args.out,paths)
    manifest_path=ROOT/'data/external/openimages-football-20261115/manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'));assert sha256(manifest_path)==inference_plan['manifest_sha256']
    sources=[Path(__file__),ROOT/'possession/external_object_metrics.py',ROOT/'scripts/research/video_ball/evaluate_openimages_detectors.py']
    plan=dict(inputs={str(p.resolve().relative_to(ROOT)):sha256(p) for p in paths},sources={str(p.relative_to(ROOT)):sha256(p) for p in sources},manifest_sha256=sha256(manifest_path),selected_model=choice['selected_model'],thresholds={model:row['threshold'] for model,row in choice['calibrated'].items()},provided_match_used=False,final_test_used_for_selection=False)
    dump(args.out/'plan.json',plan);metrics={}
    for method,model,threshold in [('unchanged_control','yolo11m',.05),('calibrated_yolo11m','yolo11m',plan['thresholds']['yolo11m']),('calibrated_rfdetr','rfdetr_large',plan['thresholds']['rfdetr_large'])]:
        rows=[]
        for item in manifest['images']:
            if item['split']!='final_test':continue
            path=args.inference/model/(item['ImageID']+'.npz');record=next(r for r in freeze['records'] if r['model']==model and r['image_id']==item['ImageID']);assert sha256(path)==record['sha256']
            with np.load(path) as z:keep=z['confidence']>=threshold;boxes,confidence=z['boxes'][keep],z['confidence'][keep]
            truth=np.asarray(item['ball_boxes_normalized'],float).reshape(-1,4)*[item['width'],item['height'],item['width'],item['height']]
            xywh=truth.copy();xywh[:,2:]-=xywh[:,:2]
            np.testing.assert_array_equal(matches(boxes,confidence,truth)[0],detection_targets(boxes,confidence,xywh).astype(bool))
            rows.append(dict(boxes=boxes,confidence=confidence,truth=truth))
        result=summary(rows);official=coco_ap(rows);assert abs(official-result['ap50'])<1e-10
        result['official_coco_api_ap50']=official;result['threshold']=threshold;metrics[method]=result
    for name,digest in {**plan['inputs'],**plan['sources']}.items():assert sha256(ROOT/name)==digest
    result=dict(complete=True,metrics=metrics,model_and_thresholds_selected_before_test=True,selection_sha256=sha256(args.selection),official_coco_api_verified=True,matching_independently_verified=True,provided_match_used=False,unexpected_competition_access=denied,final_test_used_for_selection=False,parameters_changed_after_test=False,possession_model_promoted=False,competition_target_verified=False,
        limitation='Only six football-positive images in 32 author-separated test images. This is a limited external detector pilot, not match-video small-ball performance, possession F1 or proof of absence from all backbone pretraining data.')
    dump(args.out/'results.json',result);print(json.dumps(result))


if __name__=='__main__':main()
