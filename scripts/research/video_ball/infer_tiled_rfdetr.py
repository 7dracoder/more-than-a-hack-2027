"""Frozen local RF-DETR corner-crop inference on an external image manifest."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from possession.data import sha256
from possession.tiled_ball_detection import crop_windows,translate_boxes,nms
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',type=Path,default=ROOT/'data/external/openimages-football-20261115');parser.add_argument('--split',choices=['development','external_test'],default='development');parser.add_argument('--out',type=Path,required=True);parser.add_argument('--selection',type=Path);args=parser.parse_args()
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'prediction_freeze.json').exists():raise ValueError('Preserve completed inference')
    if args.split=='external_test' and args.selection is None:raise ValueError('Freeze development selection before new test')
    denied=restrict_external(args.out,[args.selection] if args.selection else [])
    sys.addaudithook(lambda event,values: (_ for _ in ()).throw(PermissionError('External inference is offline')) if event=='socket.connect' else None)
    manifest_path=args.dataset/'manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'));assert manifest['complete'] and not manifest['provided_match_used']
    config_path=ROOT/'configs/openimages_tinyball_ensemble.json';cfg=json.loads(config_path.read_text())
    source_path=ROOT/'data/external/rf-detr-large-source.json';source=json.loads(source_path.read_text());model_dir=ROOT/source['folder']
    for name,record in source['files'].items():assert sha256(model_dir/name)==record['sha256']
    if args.selection:
        choice=json.loads(args.selection.read_text());assert choice['complete'] and not choice['fresh_test_evaluated']
    for name,value in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','HF_HOME':str(args.out/'hf-cache'),'TORCH_HOME':str(args.out/'torch-cache')}.items():os.environ[name]=value
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoImageProcessor,AutoModelForObjectDetection
    torch.set_num_threads(2);torch.manual_seed(0);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    processor=AutoImageProcessor.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
    model=AutoModelForObjectDetection.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,attn_implementation='eager').to('cuda').eval()
    assert model.config.disable_custom_kernels and processor.size=={'height':704,'width':704}
    ids=[int(k) for k,v in model.config.id2label.items() if v=='sports ball'];assert len(ids)==1;ball_id=ids[0]
    plan=dict(config=cfg,config_sha256=sha256(config_path),manifest_sha256=sha256(manifest_path),source_sha256=sha256(Path(__file__)),geometry_sha256=sha256(ROOT/'possession/tiled_ball_detection.py'),model_source_sha256=sha256(source_path),split=args.split,
        versions=dict(torch=torch.__version__,numpy=np.__version__,transformers=transformers.__version__),provided_match_used=False,hosted_inference=False)
    if args.selection:plan['selection_sha256']=sha256(args.selection)
    dump(args.out/'plan.json',plan)
    def infer(item):
        path=args.dataset/'images'/(item['ImageID']+'.jpg');assert sha256(path)==item['sha256']
        all_boxes,all_scores=[],[]
        with Image.open(path) as opened,torch.inference_mode():
            image=opened.convert('RGB')
            for window in crop_windows(item['width'],item['height']):
                left,top,right,bottom=window;crop=image.crop(window)
                inputs=processor(images=crop,return_tensors='pt').to('cuda');outputs=model(**inputs)
                result=processor.post_process_object_detection(outputs,threshold=0.,target_sizes=[(bottom-top,right-left)],top_k=300)[0]
                keep=(result['labels']==ball_id)&(result['scores']>=cfg['confidence_floor'])
                boxes=translate_boxes(result['boxes'][keep].cpu().numpy(),window,item['width'],item['height']);scores=result['scores'][keep].cpu().numpy()
                all_boxes.append(boxes);all_scores.append(scores)
        return nms(np.concatenate(all_boxes),np.concatenate(all_scores),cfg['tile_nms_iou'],cfg['max_detections'])
    images=[r for r in manifest['images'] if r['split']==args.split];assert images;records=[];first=None
    for index,item in enumerate(images):
        torch.cuda.synchronize();started=time.monotonic();boxes,scores=infer(item);torch.cuda.synchronize()
        assert boxes.shape==(len(scores),4) and np.isfinite(boxes).all() and np.isfinite(scores).all()
        path=args.out/(item['ImageID']+'.npz');np.savez_compressed(path,boxes=boxes,confidence=scores)
        records.append(dict(image_id=item['ImageID'],sha256=sha256(path),detections=len(scores),seconds=time.monotonic()-started))
        if index==0:first=(boxes.copy(),scores.copy())
        if index%25==0 or index==len(images)-1:print(json.dumps(dict(completed=index+1,total=len(images))),flush=True)
    boxes,scores=infer(images[0]);np.testing.assert_array_equal(boxes,first[0]);np.testing.assert_array_equal(scores,first[1])
    assert sha256(Path(__file__))==plan['source_sha256'] and sha256(ROOT/'possession/tiled_ball_detection.py')==plan['geometry_sha256'] and sha256(manifest_path)==plan['manifest_sha256']
    if args.selection:assert sha256(args.selection)==plan['selection_sha256']
    dump(args.out/'prediction_freeze.json',dict(complete=True,split=args.split,records=records,plan_sha256=sha256(args.out/'plan.json'),first_image_reproduced_exactly=True,unexpected_competition_access=denied,provided_match_used=False))
    print(json.dumps(dict(complete=True,predictions=len(records))),flush=True)


if __name__=='__main__':main()
