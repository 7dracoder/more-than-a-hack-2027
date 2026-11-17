# External football detector comparison

RF-DETR Large is a promising complementary detector, but it does not replace
the retained possession pipeline. On the frozen final pilot it detects five of
six balls without a false positive at its development-selected threshold:
detection F1 0.909091. Calibrated YOLO11m detects four of six, also without a
false positive: F1 0.800000. RF-DETR misses the test's only small ball, which
YOLO detects. No BAR-ATH possession improvement follows from these results.

## Rights and provenance

The [Open Images](https://storage.googleapis.com/openimages/web/download_v7.html)
validation annotations are published by Google LLC under CC BY 4.0. The subset
uses only images whose individual publisher metadata explicitly records
CC BY 2.0. Original Flickr URLs, landing pages, authors and license URLs are
retained for attribution. Cite Kuznetsova et al., *The Open Images Dataset V4*,
IJCV 2020. Images are fetched from the publisher's CVDF S3 bucket; payloads
are checked against available ETag MD5 and local SHA256 is recorded.

The challenger is the official
[Roboflow RF-DETR Large](https://huggingface.co/Roboflow/rf-detr-large), pinned
to `f62f7dd5252b61097cbace33886045816dadbde9`. Its code and weights are
Apache-2.0. The license text, model card, config and primary LFS weight hash are
retained locally. Only safetensors and model data are downloaded; no remote
Python code or hosted inference is used. The existing YOLO11m baseline retains
its existing Ultralytics AGPL-3.0 provenance; this experiment does not relicense
the project or publish any files.

## Cohort and leakage checks

Select football-positive images using Open Images' explicit Football class.
Include all physical-ball boxes on those images, including the declared ball
subclasses and golf balls. Merge exactly identical boxes and exclude ambiguous
overlapping parent/subclass boxes. Exclude ball group-of and depiction images,
unknown/nonzero rotation, missing author/photo identity and nonmatching license
records. Negative images require a human-verified absent Ball(Object) label;
exclude any image with a positive ball label or box. Select three deterministic
negative images per positive.

The official COCO archive's image-record prefixes supply 118,287 training and
5,000 validation image identities. Exclude their Flickr photo IDs and deduplicate
the selected Open Images photos. This checks source-photo overlap, not only
JPEG byte equality. It does not certify absence from every DINOv2 or other
backbone pretraining dataset. Only about 5.7 MB of the official COCO archive
needed downloading to extract the training image-record prefix; no training
images or instance annotations were acquired for this check.

Split by a fixed hash of photographer profile, so no author crosses development
and final test. The final cohort has 184 images: development has 40 positive and
112 negative images, final test has six positive and 26 negative images.
There are 43 development GT boxes and six test GT boxes. The metadata and split
were frozen before detector comparison. Final-test annotations were available
for qualification and splitting; final-test accuracy metrics were not computed
until model and operating points were frozen.

## Fixed inference and the preserved first trial

Run the existing YOLO11m at its fixed 1920-pixel setting. RF-DETR uses its
published 704-square processor, float32 eager attention and built-in kernels.
Both run locally on CUDA, retain sports-ball detections with confidence at least
0.05, clip boxes to image bounds and cap output at 300 detections. No detector
weights are trained. Different native resolutions limit architecture-only
interpretations of the comparison.

The initial fixed-threshold protocol gets development AP50 0.697195 for YOLO11m
and 0.885903 for RF-DETR, but RF-DETR's precision at 0.05 falls from 0.254098 to
0.141379. Its recall rises from 0.720930 to 0.953488. The predeclared precision
guard fails, so that first trial retains YOLO11m. It is preserved unchanged.

## Separate operating-point follow-up

Before any final-test metric, freeze a follow-up that calibrates both models
fairly on the same eleven-threshold development grid. Maximize detection F1,
requiring recall at least the unchanged control's development recall. Break
ties toward a higher threshold. RF-DETR must beat the separately optimized
YOLO11m F1 and its raw AP50, while preserving the unchanged control's precision
and recall. Otherwise leave the final test unscored.

YOLO11m selects 0.5: 31 TP, seven FP, twelve FN; precision 0.815789, recall
0.720930, F1 0.765432. RF-DETR selects 0.8: 37 TP, four FP, six FN; precision
0.902439, recall 0.860465, F1 0.880952. Freeze that selection before final
inference and evaluation. No threshold changes follow the final result.

Final results are limited to six balls in 32 images:

- Unchanged YOLO11m at 0.05: four TP, seventeen FP, two FN; F1 0.296296.
- Calibrated YOLO11m at 0.5: four TP, zero FP, two FN; F1 0.800000.
- Calibrated RF-DETR at 0.8: five TP, zero FP, one FN; F1 0.909091.

Only three development GT balls and one final GT ball have area below 32x32
pixels in the downloaded images. At the calibrated thresholds, each detector
matches one of the three development small balls. On final test, YOLO matches
the small ball and RF-DETR misses it. The sample cannot establish small-ball
improvement, game-video performance or possession accuracy. The opened final
pilot must not be reused to select a new tiling or ensemble strategy; that
requires a fresh test.

## Verification and reproduction

All image, weight, manifest, source and prediction hashes are recorded under
ignored `data/research/openimages-football-detectors-20261115/`. Both detectors
exactly reproduce their first image's boxes and scores in a repeated inference.
External isolation guards record zero competition-asset reads, and network
connections are blocked during inference. No supplied video is decoded.

Matching is independently checked against the existing greedy IoU matcher.
The 101-point interpolated AP50 implementation agrees with the official
`pycocotools` COCO evaluator. AP is measured on the predictions retained above
the fixed confidence floor, not on an unrestricted low-threshold COCO run.
Two tests cover duplicate detections, negative-image false positives,
one-to-one matching and empty predictions. Torch, NumPy and Transformers stay
unchanged; only the small pycocotools 2.0.11 wheel was installed without dependencies.

Preparation and inference scripts are `prepare_openimages_football.py` and
`infer_openimages_detectors.py`. Evaluation is split into
`evaluate_openimages_detectors.py`, `select_openimages_operating_points.py`
and `evaluate_openimages_operating_test.py`. Each supports CLI help. Protocols
are `configs/openimages_football_detector.json` and the separate
`configs/openimages_detector_operating_points.json`. Source acquisition scripts
and the COCO-prefix parser are retained under `tmp/` with this run's provenance.

The v4 CSV and frozen v11 stay unchanged. Any next tiny-ball or ensemble study
needs new external validation; these component F1 values are not BAR-ATH spell F1.
