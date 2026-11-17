"""Select preserving native/tiled detector unions using development images only."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from possession.data import sha256
from possession.external_object_metrics import summary,matches
from possession.ball_detector_ensemble import compose
from scripts.research.video_ball.calibrate_external_confidence import fit_evidence,detection_targets
from scripts.research.video_ball.evaluate_openimages_detectors import coco_ap
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--native',type=Path,required=True);parser.add_argument('--tiles',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'selection.json').exists():raise ValueError('Preserve frozen development selection')
    native=json.loads((args.native/'prediction_freeze.json').read_text());tiles=json.loads((args.tiles/'prediction_freeze.json').read_text())
    assert all(f['complete'] and f['split']=='development' and not f['unexpected_competition_access'] for f in (native,tiles))
    paths=[args.native/'prediction_freeze.json',args.native/'plan.json',args.tiles/'prediction_freeze.json',args.tiles/'plan.json']
    paths += [args.native/r['model']/(r['image_id']+'.npz') for r in native['records']]+[args.tiles/(r['image_id']+'.npz') for r in tiles['records']]
    denied=restrict_external(args.out,paths)
    nplan=json.loads((args.native/'plan.json').read_text());tplan=json.loads((args.tiles/'plan.json').read_text())
    assert sha256(args.native/'plan.json')==native['plan_sha256'] and sha256(args.tiles/'plan.json')==tiles['plan_sha256']
    assert nplan['challenger_source_sha256']==tplan['model_source_sha256']
    manifest_path=ROOT/'data/external/openimages-football-20261115/manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    assert sha256(manifest_path)==nplan['manifest_sha256']==tplan['manifest_sha256']
    config_path=ROOT/'configs/openimages_tinyball_ensemble.json';cfg=json.loads(config_path.read_text());assert sha256(config_path)==tplan['config_sha256']
    assert sha256(ROOT/'possession/tiled_ball_detection.py')==tplan['geometry_sha256']
    assert sha256(ROOT/'scripts/research/video_ball/infer_tiled_rfdetr.py')==tplan['source_sha256']
    sources=[Path(__file__),ROOT/'possession/tiled_ball_detection.py',ROOT/'possession/ball_detector_ensemble.py',ROOT/'possession/external_object_metrics.py',ROOT/'scripts/research/video_ball/calibrate_external_confidence.py',ROOT/'scripts/research/video_ball/evaluate_openimages_detectors.py']
    plan=dict(config=cfg,config_sha256=sha256(config_path),inputs={str(p.resolve().relative_to(ROOT)):sha256(p) for p in paths},sources={str(p.relative_to(ROOT)):sha256(p) for p in sources},manifest_sha256=sha256(manifest_path),provided_match_used=False,fresh_test_evaluated=False)
    dump(args.out/'plan.json',plan);items=[];pooled={name:[[],[]] for name in ('yolo11m','rfdetr_large','rfdetr_tiled')}
    for item in manifest['images']:
        if item['split']!='development':continue
        truth=np.asarray(item['ball_boxes_normalized'],float).reshape(-1,4)*[item['width'],item['height'],item['width'],item['height']]
        xywh=truth.copy();xywh[:,2:]-=xywh[:,:2];branches={}
        for name in pooled:
            if name=='rfdetr_tiled':
                path=args.tiles/(item['ImageID']+'.npz');record=next(r for r in tiles['records'] if r['image_id']==item['ImageID'])
            else:
                path=args.native/name/(item['ImageID']+'.npz');record=next(r for r in native['records'] if r['model']==name and r['image_id']==item['ImageID'])
            assert sha256(path)==record['sha256']
            with np.load(path) as z:branches[name]=dict(boxes=z['boxes'],confidence=z['confidence'])
            labels=detection_targets(branches[name]['boxes'],branches[name]['confidence'],xywh)
            np.testing.assert_array_equal(labels.astype(bool),matches(branches[name]['boxes'],branches[name]['confidence'],truth)[0])
            pooled[name][0].extend(branches[name]['confidence']);pooled[name][1].extend(labels)
        items.append(dict(image_id=item['ImageID'],truth=truth,branches=branches))
    evidence={name:fit_evidence(*[np.asarray(v) for v in data]) for name,data in pooled.items()}
    dump(args.out/'evidence.json',evidence)
    candidates=[dict(kind='yolo',branches=1,tile_threshold=None),dict(kind='rfdetr',branches=1,tile_threshold=None),dict(kind='union',branches=2,tile_threshold=None)]
    candidates += [dict(kind='union_tiled',branches=3,tile_threshold=t) for t in cfg['tile_thresholds']]
    table=[]
    for specification in candidates:
        rows=[]
        for item in items:
            boxes,confidence=compose(item['branches'],evidence,specification['kind'],specification['tile_threshold'])
            rows.append(dict(boxes=boxes,confidence=confidence,truth=item['truth']))
        measured=summary(rows);assert abs(coco_ap(rows)-measured['ap50'])<1e-10
        table.append(dict(**specification,metrics=measured))
    yolo,rf=table[0]['metrics'],table[1]['metrics']
    for row in table:
        m=row['metrics'];row['guards']=dict(precision=m['precision']>=yolo['precision']-1e-12,recall=m['recall']>=rf['recall']-1e-12,small_recall=m['small_recall']>=yolo['small_recall']-1e-12)
        row['eligible']=all(row['guards'].values())
    assert table[1]['eligible']
    best=max((r for r in table if r['eligible']),key=lambda r:(r['metrics']['f1'],-r['branches'],r['tile_threshold'] or 0.))
    for name,digest in {**plan['inputs'],**plan['sources']}.items():assert sha256(ROOT/name)==digest
    result=dict(complete=True,selected=best,candidates=table,evidence=evidence,evidence_sha256=sha256(args.out/'evidence.json'),plan=plan,
        official_coco_api_all_candidates_verified=True,fresh_test_evaluated=False,previous_pilot_test_reused=False,provided_match_used=False,unexpected_competition_access=denied,possession_model_promoted=False,competition_target_verified=False)
    dump(args.out/'selection.json',result);print(json.dumps(dict(selected=best,complete=True,fresh_test_evaluated=False)),flush=True)


if __name__=='__main__':main()
