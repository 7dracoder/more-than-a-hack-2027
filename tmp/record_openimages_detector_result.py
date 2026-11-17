import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];study=root/'data/research/openimages-football-detectors-20261115'
final=json.loads((study/'final-evaluation/results.json').read_text());selection_path=study/'operating-point-calibration/selection.json';selection=json.loads(selection_path.read_text())
assert final['complete'] and final['official_coco_api_verified'] and not final['unexpected_competition_access']
assert hashlib.sha256(selection_path.read_bytes()).hexdigest()==final['selection_sha256']
assert selection['selected_model']=='rfdetr_large' and selection['selected_threshold']==.8
manifest=json.loads((root/'data/external/openimages-football-20261115/manifest.json').read_text(encoding='utf-8'))
assert len({r['photo_id'] for r in manifest['images']})==len(manifest['images'])
assert len({r['sha256'] for r in manifest['images']})==len(manifest['images'])
assert not ({r['author_group'] for r in manifest['images'] if r['split']=='development'}&{r['author_group'] for r in manifest['images'] if r['split']=='final_test'})
for row in manifest['images']:
    path=root/'data/external/openimages-football-20261115/images'/(row['ImageID']+'.jpg')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
receipt=dict(complete=True,component='External still-image football ball detector',model='Roboflow/rf-detr-large',revision='f62f7dd5252b61097cbace33886045816dadbde9',license='Apache-2.0',
    raw_development_ap50=dict(yolo11m=.697195438811536,rfdetr_large=.8859032964930511),
    calibrated_development_f1=dict(yolo11m=.7654320987654321,rfdetr_large=.8809523809523809),
    final_test_metrics=final['metrics'],final_football_images=6,final_total_images=32,
    dataset_photo_ids_bytes_and_authors_disjoint=True,model_and_thresholds_frozen_before_test=True,official_coco_api_verified=True,
    promoted_to_possession=False,reason='Limited pilot; RF-DETR misses the only final small ball, which the retained detector detects. Need fresh external tiny-ball/ensemble validation.',
    supplied_video_decoded=False,provided_match_used=False,competition_target_verified=False)
(study/'run_status.json').write_text(json.dumps(receipt,indent=2)+'\n')
progress_path=root/'data/research/resumed-improvement-20261114/progress.json';progress=json.loads(progress_path.read_text());progress['external_football_detector_comparison']=receipt
progress['external_football_detector_comparison']['study']='data/research/openimages-football-detectors-20261115'
progress['last_goal_turn_classification']='progress';progress_path.write_text(json.dumps(progress,indent=2)+'\n')
files={
 'data/research/resumed-improvement-20261114/submission-ready/CoreOrigin.csv':'cbc30311a6c9fdbe8cc7198e3a25f38d444c07c437e491528da5a924ecca408d',
 'data/bar-ath-check-20261103/submission/CoreOrigin.csv':'496b42abcb0aec5ec0d4dffa6a1d8ca3c5b386dd0ccb9e4c697e877dfa1e2c80',
 'data/bar-ath-check-20261103/model/selected.pt':'d2449c5a47534c702a17853a093dee8fc60a5e90ef571e848bd9bd70d4bcfd20',
 'evaluation/run_evaluation.py':'4c09764c06768fa0e3c661a93f86a14e17389a4a01f05370c96035190a1da66d',
 'evaluation/possession_metrics.py':'32c878630c74fb52a90c301485d6dff806509d8e933b1177b6035ee8e8f42b80'}
for name,expected in files.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected
(study/'retained_artifact_check.json').write_text(json.dumps(dict(complete=True,sha256=files),indent=2)+'\n')
with (root/'WORK_DONE.md').open('ab') as output:
    output.write(b'\n\n## 2026-11-15: new licensed football detector comparison\n\nRechecked full pose audit: no accepted head directions, not a missed full-data opportunity. Built new Open Images football benchmark with 46 positive plus 138 verified-negative images, explicit CC BY 2.0 metadata, author-separated DEV/TEST, no source-photo overlap with all COCO 2017 TRAIN/VAL records, no duplicate image hashes. Acquired official Apache-2.0 RF-DETR Large safetensors at f62f7dd5252b61097cbace33886045816dadbde9 and verified primary LFS SHA. Local offline inference repeats first image exactly for both models. Fixed-0.05 RF trial improves AP50 0.697195 -> 0.885903 but fails precision guard and is preserved. Separate fair development calibration selects YOLO threshold 0.5, RF threshold 0.8, F1 0.765432 vs 0.880952. Selection frozen before final metrics. On 32-image TEST with six balls, calibrated YOLO has 4 TP/0 FP/2 FN, F1 0.8; RF has 5 TP/0 FP/1 FN, F1 0.909091. RF misses the sole small ball detected by YOLO, so no possession promotion or BAR inference/export. COCO API cross-checks AP; independent greedy matching agrees; two tests pass. Added only pycocotools without dependencies; Torch/NumPy/Transformers unchanged. All source/model/image/metric receipts retained locally; no match uploads/reads or target scoring. Retained CSV, v11 and official source hashes unchanged. Report: scripts/research/video_ball/OPENIMAGES_FOOTBALL.md. Next useful test needs fresh external tiny-ball/ensemble evidence, not reuse of this opened pilot. BAR-ATH goal active and unverified.\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='final_test_metrics'}))
