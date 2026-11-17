"""Select on development only, cross-checking AP with the official COCO API."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from possession.data import sha256
from possession.external_object_metrics import summary,matches
from scripts.research.video_ball.calibrate_external_confidence import detection_targets
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump


def coco_ap(images):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    annotations=[];detections=[];records=[]
    for identity,row in enumerate(images):
        records.append(dict(id=identity))
        for box in row['truth']:
            a,b,c,d=map(float,box);annotations.append(dict(id=len(annotations)+1,image_id=identity,category_id=1,bbox=[a,b,c-a,d-b],area=(c-a)*(d-b),iscrowd=0))
        for box,confidence in zip(row['boxes'],row['confidence']):
            a,b,c,d=map(float,box);detections.append(dict(image_id=identity,category_id=1,bbox=[a,b,c-a,d-b],score=float(confidence)))
    if not detections:return 0.
    with contextlib.redirect_stdout(io.StringIO()):
        truth=COCO();truth.dataset=dict(info={},images=records,annotations=annotations,categories=[dict(id=1,name='sports ball')]);truth.createIndex()
        predicted=truth.loadRes(detections);evaluation=COCOeval(truth,predicted,'bbox')
        evaluation.params.iouThrs=np.array([.5]);evaluation.params.maxDets=[1,10,300]
        evaluation.params.areaRng=[[0,1e10]];evaluation.params.areaRngLbl=['all']
        evaluation.evaluate();evaluation.accumulate()
    precision=evaluation.eval['precision'][0,:,0,0,2]
    return float(precision[precision>=0].mean())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--inference',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--selection',type=Path);args=parser.parse_args()
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'results.json').exists():raise ValueError('Preserve completed evaluation')
    freeze=json.loads((args.inference/'prediction_freeze.json').read_text());inference_plan=json.loads((args.inference/'plan.json').read_text());split=freeze['split']
    assert freeze['complete'] and not freeze['unexpected_competition_access'] and freeze['first_image_repeated_exactly_for_both_models']
    assert sha256(args.inference/'plan.json')==freeze['plan_sha256']
    paths=[args.inference/'prediction_freeze.json',args.inference/'plan.json']+[args.inference/r['model']/(r['image_id']+'.npz') for r in freeze['records']]
    if args.selection:paths.append(args.selection)
    denied=restrict_external(args.out,paths)
    manifest_path=ROOT/'data/external/openimages-football-20261115/manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'));assert sha256(manifest_path)==inference_plan['manifest_sha256']
    cfg_path=ROOT/'configs/openimages_football_detector.json';cfg=json.loads(cfg_path.read_text());assert sha256(cfg_path)==inference_plan['config_sha256']
    if split=='final_test':
        assert args.selection is not None
        selected=json.loads(args.selection.read_text());assert selected['complete'] and selected['selected_model']=='rfdetr_large' and not selected['final_test_accessed']
        assert sha256(args.selection)==inference_plan['selection_sha256']
    sources=[Path(__file__),ROOT/'possession/external_object_metrics.py',ROOT/'scripts/research/video_ball/calibrate_external_confidence.py']
    plan=dict(split=split,inputs={str(p.resolve().relative_to(ROOT)):sha256(p) for p in paths},sources={str(p.relative_to(ROOT)):sha256(p) for p in sources},manifest_sha256=sha256(manifest_path),metric='101-point interpolated AP50 on predictions retained at confidence >=0.05, not full low-threshold COCO AP.',provided_match_used=False)
    dump(args.out/'plan.json',plan)
    items=[r for r in manifest['images'] if r['split']==split];results={}
    for model in ('yolo11m','rfdetr_large'):
        rows=[]
        for item in items:
            path=args.inference/model/(item['ImageID']+'.npz');record=next(r for r in freeze['records'] if r['model']==model and r['image_id']==item['ImageID']);assert sha256(path)==record['sha256']
            with np.load(path) as data:boxes,confidence=data['boxes'],data['confidence']
            truth=np.asarray(item['ball_boxes_normalized'],float).reshape(-1,4)*[item['width'],item['height'],item['width'],item['height']]
            xywh=truth.copy();xywh[:,2:]-=xywh[:,:2]
            expected=detection_targets(boxes,confidence,xywh).astype(bool);actual,_=matches(boxes,confidence,truth);np.testing.assert_array_equal(actual,expected)
            rows.append(dict(boxes=boxes,confidence=confidence,truth=truth))
        result=summary(rows);official=coco_ap(rows);assert abs(result['ap50']-official)<1e-10
        result['official_coco_api_ap50']=official;results[model]=result
    selected_model='yolo11m';baseline=results['yolo11m'];challenger=results['rfdetr_large']
    guards=dict(ap50=challenger['ap50']>baseline['ap50']+1e-12,precision=challenger['precision']>=baseline['precision']-1e-12,recall=challenger['recall']>=baseline['recall']-1e-12)
    if all(guards.values()):selected_model='rfdetr_large'
    for name,digest in {**plan['inputs'],**plan['sources']}.items():assert sha256(ROOT/name)==digest
    if split=='development':
        selection=dict(complete=True,selected_model=selected_model,development=results,guards=guards,final_test_accessed=False,access_scope='Final-test accuracy metrics; annotations were used for qualification and splitting, never test-metric selection.',provided_match_used=False,plan_sha256=sha256(args.out/'plan.json'),prediction_freeze_sha256=sha256(args.inference/'prediction_freeze.json'))
        dump(args.out/'selection.json',selection)
    result=dict(complete=True,split=split,metrics=results,selected_on_development=selected_model if split=='development' else selected['selected_model'],development_selection_guards=guards if split=='development' else None,
        official_coco_api_ap_verified=True,matching_independently_verified=True,provided_match_used=False,unexpected_competition_access=denied,final_test_evaluated=split=='final_test',possession_model_promoted=False,competition_target_verified=False,
        limitation='External still-image detector component only, different native input sizes, limited football images and incomplete certification of backbone pretraining overlap. Detector AP/F1 is not possession spell F1.')
    dump(args.out/'results.json',result);print(json.dumps(result))


if __name__=='__main__':main()
