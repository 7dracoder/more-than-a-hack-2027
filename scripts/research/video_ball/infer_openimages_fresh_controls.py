"""Local frozen detector controls on the disjoint official external test corpus."""
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
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--split',choices=['external_test'],default='external_test');parser.add_argument('--selection',type=Path);args=parser.parse_args()
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'prediction_freeze.json').exists():raise ValueError('Preserve completed inference')
    if args.split=='external_test' and args.selection is None:raise ValueError('Freeze development selection first')
    allowed=[args.selection] if args.selection else []
    denied=restrict_external(args.out,allowed)
    sys.addaudithook(lambda event,values: (_ for _ in ()).throw(PermissionError('Offline external inference forbids network connections')) if event=='socket.connect' else None)
    folder=ROOT/'data/external/openimages-football-fresh-test-20261115';manifest_path=folder/'manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['complete'] and not manifest['provided_match_used']
    assert sha256(folder/'frozen_selection.json')==manifest['selection_sha256']
    config_path=ROOT/'configs/openimages_football_fresh_test.json';frozen=json.loads((folder/'frozen_selection.json').read_text(encoding='utf-8'))
    assert sha256(config_path)==frozen['config_sha256']
    if args.selection:
        choice=json.loads(args.selection.read_text());assert choice['complete'] and not choice['fresh_test_evaluated'] and choice['selected']['eligible']
    source_path=ROOT/'data/external/rf-detr-large-source.json';source=json.loads(source_path.read_text());model_dir=ROOT/source['folder']
    for name,record in source['files'].items():assert sha256(model_dir/name)==record['sha256']
    for name,value in {'YOLO_CONFIG_DIR':str(args.out/'yolo-config'),'YOLO_OFFLINE':'true','YOLO_AUTOINSTALL':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','HF_HOME':str(args.out/'hf-cache'),'TORCH_HOME':str(args.out/'torch-cache')}.items():os.environ[name]=value
    import cv2
    import torch
    import ultralytics
    import transformers
    from PIL import Image
    from ultralytics import YOLO
    from transformers import AutoImageProcessor,AutoModelForObjectDetection
    torch.set_num_threads(2);cv2.setNumThreads(2);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ultralytics.settings.update({k:False for k in ('sync','clearml','comet','dvc','mlflow','raytune','tensorboard','wandb')})
    baseline_path=ROOT/'yolo11m.pt'
    plan=dict(split=args.split,manifest_sha256=sha256(manifest_path),config_sha256=sha256(config_path),source_sha256=sha256(Path(__file__)),baseline_sha256=sha256(baseline_path),challenger_source_sha256=sha256(source_path),challenger_revision=source['revision'],
        parameters=dict(confidence=.05,baseline_imgsz=1920,challenger_published_processor=True,max_detections=300,dtype='float32',device='cuda:0',attention='eager',custom_kernels=False),
        versions=dict(torch=torch.__version__,numpy=np.__version__,ultralytics=ultralytics.__version__,transformers=transformers.__version__),provided_match_used=False,hosted_inference=False)
    if args.selection:plan['selection_sha256']=sha256(args.selection)
    dump(args.out/'plan.json',plan)
    images=[r for r in manifest['images'] if r['split']==args.split];records=[]
    for model_name in ('yolo11m','rfdetr_large'):
        target=args.out/model_name;target.mkdir(exist_ok=True)
        if model_name=='yolo11m':
            model=YOLO(str(baseline_path));assert model.names[32]=='sports ball'
        else:
            processor=AutoImageProcessor.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
            model=AutoModelForObjectDetection.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,attn_implementation='eager').to('cuda').eval()
            assert model.config.disable_custom_kernels and processor.size=={'height':704,'width':704}
            ids=[int(key) for key,label in model.config.id2label.items() if label=='sports ball'];assert len(ids)==1;ball_id=ids[0]
        first_result=None
        def infer(item):
            image_path=folder/'images'/(item['ImageID']+'.jpg');assert sha256(image_path)==item['sha256']
            with torch.inference_mode():
                if model_name=='yolo11m':
                    result=model.predict(str(image_path),imgsz=1920,conf=.05,classes=[32],device=0,half=False,max_det=300,verbose=False,save=False)[0]
                    boxes=result.boxes.xyxy.cpu().numpy();scores=result.boxes.conf.cpu().numpy()
                else:
                    with Image.open(image_path) as image:inputs=processor(images=image.convert('RGB'),return_tensors='pt').to('cuda')
                    outputs=model(**inputs)
                    result=processor.post_process_object_detection(outputs,threshold=0.,target_sizes=[(item['height'],item['width'])],top_k=300)[0]
                    keep=(result['labels']==ball_id)&(result['scores']>=.05)
                    boxes=result['boxes'][keep].cpu().numpy();scores=result['scores'][keep].cpu().numpy()
            boxes=boxes.copy();boxes[:,[0,2]]=boxes[:,[0,2]].clip(0,item['width']);boxes[:,[1,3]]=boxes[:,[1,3]].clip(0,item['height'])
            assert boxes.shape==(len(scores),4) and np.isfinite(boxes).all() and np.isfinite(scores).all() and ((scores>=.05)&(scores<=1)).all()
            return boxes,scores
        for index,item in enumerate(images):
            torch.cuda.synchronize();started=time.monotonic();boxes,scores=infer(item);torch.cuda.synchronize();elapsed=time.monotonic()-started
            path=target/(item['ImageID']+'.npz');np.savez_compressed(path,boxes=boxes,confidence=scores)
            records.append(dict(model=model_name,image_id=item['ImageID'],sha256=sha256(path),detections=len(scores),seconds=elapsed))
            if index==0:first_result=(boxes.copy(),scores.copy())
            if index%25==0 or index==len(images)-1:print(json.dumps(dict(model=model_name,completed=index+1,total=len(images))),flush=True)
        boxes,scores=infer(images[0]);np.testing.assert_array_equal(boxes,first_result[0]);np.testing.assert_array_equal(scores,first_result[1])
        del model
        if model_name=='rfdetr_large':del processor
        torch.cuda.empty_cache()
    assert sha256(Path(__file__))==plan['source_sha256'] and sha256(manifest_path)==plan['manifest_sha256'] and sha256(baseline_path)==plan['baseline_sha256']
    if args.selection:assert sha256(args.selection)==plan['selection_sha256']
    dump(args.out/'prediction_freeze.json',dict(complete=True,split=args.split,plan_sha256=sha256(args.out/'plan.json'),records=records,first_image_repeated_exactly_for_both_models=True,provided_match_used=False,unexpected_competition_access=denied,hosted_inference=False))
    print(json.dumps(dict(complete=True,split=args.split,predictions=len(records))),flush=True)


if __name__=='__main__':main()
