# Frozen external camera-loss adaptation

This experimental model was trained and selected using external data only.
BAR–ATH spell F1 **above 0.740 remains unverified**. No BAR–ATH possession truth
was opened for this study or used to choose its models/parameters. The historical
v11 weights and CSV are unchanged. This bundle contains externally trained
weights and external research receipts; it contains no supplied-match data,
predictions or labels.

The selected expert is a 128-channel temporal convolution/recurrent model with
664 established geometry features and 128 spatial-history/passing-option
features. It starts from external run 13 and adapts on six Sportec matches using
native and synthetic camera-loss views. Camera-loss statistics were fitted
without possession labels on eighteen separate SkillCorner A-League matches.
It is not a head/gaze estimator and does not measure cognitive memory or
peripheral vision.

The frozen choice is epoch 20, mixed at 25% with the existing parent. Its
external development spell F1 is 0.847343 native, 0.415979 masked and 0.443593
mixed-augmentation. Mean 0.568972 compares with parent 0.545693 and prior
temporal-fusion 0.546466 on those same views. Native performance declines
0.009662, within the predeclared 0.01 guard. The development fixture J03WPY has
been used repeatedly; these are exploratory results, not independent-test or
BAR–ATH performance. They are not comparable with the older, easier artificial
gap benchmark. Replacing the prior gated expert using either epoch 20 or the
fixed final epoch did not beat this frozen choice.

Use `requirements-video-ball.txt` in the recorded CUDA environment (PyTorch
2.11.0+cu128, NumPy 2.5.3, SciPy 1.18.1). The parent bundle at
`models/coreorigin-video-v1` is also required. No external training datasets
are required for final inference. `model_manifest.json` pins all copied
receipts/weights. The exporter denies access to training data and all
possession ground truth.

With the existing verified coordinate artifact:

```powershell
.\.venv\Scripts\python.exe scripts/research/spatial_memory/export_natural_candidate.py --study models/coreorigin-natural-visibility-v1 --ball data/research/dual-feed-20261113-v2/ball_hypotheses.npy --out data/research/natural-visibility-20261114/candidate-packaged --device cuda
.\.venv\Scripts\python.exe evaluation/validate_submission.py data/research/natural-visibility-20261114/candidate-packaged/CoreOrigin.csv
```

To reconstruct coordinate hypotheses from the supplied tactical/broadcast
bundles, use the existing pinned `yolo11m.pt` (hash and attribution in the parent
bundle), externally fitted confidence/motion JSONs from that bundle, and these
unchanged commands. Detection remains local; supplied video is never used for
training or human prediction checks. Broadcast pixels are not used because
their frame count differs from the tracking timeline.

```powershell
.\.venv\Scripts\python.exe scripts/research/video_ball/detect.py --device 0 --frames missing --imgsz 1920 --conf 0.05 --checkpoint-every 1000 --out data/research/natural-inputs/ball_missing_gpu.npz
.\.venv\Scripts\python.exe scripts/research/video_ball/fuse_detections.py --detections data/research/natural-inputs/ball_missing_gpu.npz --out data/research/natural-inputs/ball_hypotheses.npy
.\.venv\Scripts\python.exe scripts/research/broadcast/dual_feed.py --out data/research/natural-dual-feed --tactical-hypotheses data/research/natural-inputs/ball_hypotheses.npy
.\.venv\Scripts\python.exe scripts/research/spatial_memory/export_natural_candidate.py --study models/coreorigin-natural-visibility-v1 --ball data/research/natural-dual-feed/ball_hypotheses.npy --out data/research/natural-regeneration --device cuda
.\.venv\Scripts\python.exe evaluation/validate_submission.py data/research/natural-regeneration/CoreOrigin.csv
```

Expected final CSV SHA-256:
`e5bd12c22fbeecf85ceefe5de13161e00a154c25e12711e1ef4b29368f893e86`.
It has 199,319 rows. Format validation establishes file correctness, not label
accuracy. No new public-window possession scoring is part of this workflow.
Full-match target confirmation requires the organisers' permitted evaluation.

Research methods, negative results, external data/model attribution and study
commands are documented in
`scripts/research/spatial_memory/NATURAL_VISIBILITY.md` and the neighbouring
README. Sportec data: Bassek et al. 2025, CC BY 4.0. SkillCorner open data: MIT.
The audited PathCRF model (MPL-2.0) is not used in this exporter: its complete
roster adapter accepted no externally occluded windows.
