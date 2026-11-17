"""Fixed detector inference and confidence-evidence fitting on external COCO only.

The fitted scalar is a class-balanced likelihood-ratio proxy, not a calibrated
probability for BAR-ATH. No provided-match pixels, coordinates or possession
labels enter inference, fitting, splitting or settings selection here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.video_ball.prepare_coco_calibration import ZIP_URL, allowed_license


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def detection_targets(boxes, confidence, ground_truth):
    """COCO-style greedy, unique GT matching at fixed IoU 0.5."""
    boxes, confidence = np.asarray(boxes, float), np.asarray(confidence, float)
    gt = np.asarray(ground_truth, float).reshape(-1, 4).copy()
    if boxes.shape != (len(confidence), 4):
        raise ValueError('Detection boxes/confidence shape mismatch')
    if (not np.isfinite(boxes).all() or not np.isfinite(confidence).all() or
            ((confidence < 0) | (confidence > 1)).any()):
        raise ValueError('Invalid detector output')
    target = np.zeros(len(boxes), np.int8)
    if not len(gt) or not len(boxes):
        return target
    gt[:, 2:] += gt[:, :2]  # COCO xywh -> xyxy
    intersection = np.maximum(0, np.minimum(boxes[:, None, 2:], gt[None, :, 2:]) -
                              np.maximum(boxes[:, None, :2], gt[None, :, :2])).prod(2)
    area = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(1)
    gt_area = np.maximum(0, gt[:, 2:] - gt[:, :2]).prod(1)
    iou = intersection / np.maximum(area[:, None] + gt_area[None] - intersection, 1e-12)
    assigned = np.zeros(len(gt), bool)
    for i in np.lexsort((np.arange(len(confidence)), -confidence)):
        score = np.where(assigned, -1, iou[i])
        j = int(score.argmax())
        if score[j] >= .5:
            target[i] = 1
            assigned[j] = True
    return target


def fit_evidence(confidence, target):
    from sklearn.linear_model import LogisticRegression
    confidence, target = np.asarray(confidence, float), np.asarray(target, int)
    if len(confidence) != len(target) or set(np.unique(target)) != {0, 1}:
        raise ValueError('External development detections must contain both target classes')
    clipped = np.clip(confidence, 1e-6, 1 - 1e-6)
    feature = np.log(clipped / (1 - clipped))[:, None]
    model = LogisticRegression(C=1., class_weight='balanced', solver='lbfgs',
                               max_iter=1000, random_state=0).fit(feature, target)
    slope, intercept = float(model.coef_[0, 0]), float(model.intercept_[0])
    if slope <= 0:
        raise ValueError('External calibration is not monotone; do not apply it')
    return dict(slope=slope, intercept=intercept,
                transform='slope * logit(clamp(confidence,1e-6,1-1e-6)) + intercept',
                interpretation='Class-balanced log-odds evidence; not deployment probability')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external', type=Path, default=ROOT / 'data/external/coco2017-calibration')
    parser.add_argument('--weights', type=Path, default=ROOT / 'yolo11m.pt')
    parser.add_argument('--out', type=Path, default=ROOT / 'data/research/target080/external_calibration')
    args = parser.parse_args()
    manifest_path = args.external / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    protocol = manifest['protocol']
    if protocol['source'] != ZIP_URL or protocol['provided_match_data_used']:
        raise ValueError('Only the frozen external COCO protocol is accepted')
    if any(not allowed_license(x['license_record']) for x in manifest['images']):
        raise ValueError('Image license outside the explicit allowed set')
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str((args.out / 'yolo-config').resolve())
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '2'
    import cv2
    import torch
    import ultralytics
    import sklearn
    from ultralytics import YOLO

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    ultralytics.settings.update({'sync': False, 'clearml': False, 'comet': False,
                                 'dvc': False, 'mlflow': False, 'raytune': False,
                                 'tensorboard': False, 'wandb': False})
    inference_protocol = dict(weights_sha256=digest(args.weights), manifest_sha256=digest(manifest_path),
                              source_sha256=digest(__file__), device='cpu', imgsz=1920,
                              conf=.05, class_id=32, training_detector=False,
                              versions=dict(torch=torch.__version__, ultralytics=ultralytics.__version__,
                                            numpy=np.__version__, sklearn=sklearn.__version__))
    source_record = args.out / 'inference_protocol.json'
    if source_record.is_file() and json.loads(source_record.read_text(encoding='utf-8')) != inference_protocol:
        raise ValueError('External inference protocol changed; use a new output directory')
    source_record.write_text(json.dumps(inference_protocol, indent=2), encoding='utf-8')
    predictor = YOLO(str(args.weights))
    predictions = args.out / 'predictions'; predictions.mkdir(exist_ok=True)
    pooled = {'development': [[], []], 'validation': [[], []]}
    image_counts = {'development': 0, 'validation': 0}
    for k, item in enumerate(manifest['images']):
        image = args.external / 'images' / item['file_name']
        if digest(image) != item['sha256']:
            raise ValueError('External image checksum mismatch')
        destination = predictions / (str(item['id']) + '.npz')
        receipt = destination.with_suffix('.json')
        if destination.is_file():
            record = json.loads(receipt.read_text(encoding='utf-8'))
            if record['output_sha256'] != digest(destination) or record['image_sha256'] != item['sha256']:
                raise ValueError('Corrupted external prediction cache')
            with np.load(destination, allow_pickle=False) as data:
                boxes, confidence = data['boxes'], data['confidence']
        else:
            result = predictor.predict(str(image), imgsz=1920, conf=.05, classes=[32],
                                       device='cpu', verbose=False, save=False)[0]
            boxes, confidence = result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()
            part = destination.with_suffix('.npz.partial')
            with part.open('wb') as stream:
                np.savez_compressed(stream, boxes=boxes, confidence=confidence)
            part.replace(destination)
            receipt.write_text(json.dumps(dict(output_sha256=digest(destination),
                                                image_sha256=item['sha256'])), encoding='utf-8')
        target = detection_targets(boxes, confidence, item['ball_boxes'])
        pooled[item['split']][0].extend(confidence.tolist())
        pooled[item['split']][1].extend(target.tolist())
        image_counts[item['split']] += 1
        if (k + 1) % 10 == 0:
            print(f'{k + 1}/{len(manifest["images"])} external images processed', flush=True)
    confidence, target = map(np.asarray, pooled['development'])
    fitted = fit_evidence(confidence, target)
    vc, vy = map(np.asarray, pooled['validation'])
    validation = dict(images=image_counts['validation'], detections=len(vy),
                      true_detections=int(vy.sum()), false_detections=int((vy == 0).sum()))
    if len(vy) and len(np.unique(vy)) == 2:
        from sklearn.metrics import roc_auc_score
        validation['ranking_auc'] = float(roc_auc_score(vy, vc))
    result = dict(kind='External-only detector confidence evidence', provided_match_data_used=False,
                  detector_inference_protocol=inference_protocol, fitted=fitted,
                  development=dict(images=image_counts['development'], detections=len(target),
                                   true_detections=int(target.sum()), false_detections=int((target == 0).sum())),
                  validation=validation, target_definition='Unique external COCO sports-ball box match at IoU 0.5',
                  parameters_fixed_before_inference=dict(C=1., class_weight='balanced', solver='lbfgs'),
                  possession_score=None,
                  limitations=['External COCO image calibration is not video-domain validation.',
                               'Class-balanced evidence must not be interpreted as a BAR-ATH posterior.',
                               'No possession metric or candidate selection is established.'])
    (args.out / 'confidence_evidence.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='detector_inference_protocol'}, indent=2))


if __name__ == '__main__':
    main()
