# BAR-ATH ball recovery: unfinished experiment toward 0.80 spell F1

Current handoff: `models/coreorigin-continuity-v4`, documented in the root
`CoreOrigin.md`. The historical experiment notes below do not supersede it.
The active BAR-ATH target is spell F1 above 0.740, still unverified.
The new [external football detector comparison](OPENIMAGES_FOOTBALL.md)
qualifies RF-DETR as a promising complementary detector, with insufficient
tiny-ball evidence to replace the retained pipeline. No new supplied-match
scoring or detector-based possession promotion follows that comparison.

The [participant clarification](../../../RULES_CLARIFICATION.md) supplied by the
user overrides earlier guidance: no visual comparison of predictions against
video, no match-data uploads to AI services, and no training, fitting,
calibration or parameter tuning on BAR-ATH truth/video. Run local automatic
inference/checks only. Do not use pilot diagnostics to choose detector settings,
weights, fusion parameters, or possession candidates.

The objective is **0.80 spell F1 on BAR-ATH**, not an external development
score. It has **not been achieved or verified**. The frozen v11 submission
remains the fallback. Its public-window pipeline check is 0.54054 spell F1;
the private full-match score is unavailable. No new possession candidate has
been selected and no new match was downloaded for this work. After the final
CSV was frozen, the unchanged official scorer performed one public-window
pipeline check: **0.5200 spell F1 / 0.630699 macro F1**. No tuning or selection
followed that result. See the score-scope section below.

## First completed step: input audit

```powershell
.\.venv\Scripts\python.exe scripts/research/video_ball/audit_inputs.py
```

Reads the four supplied tactical non-video inputs and writes private output to
ignored `data/research/target080/`. The audit validates the one-based video-frame
counter in the raw ATD, separates halves, inventories missing runs, and checks
homography coverage. It never reads possession labels, fits a model or selects
possession parameters.

The local run on 4 November 2026 found 64,643 missing-ball frames in the two
tracking-half windows, with 147 gaps bracketed by observations in the same half.
The median gap is 11.56 s in the first half and 9.80 s in the second. These
windows include the published pre-second-half dead-ball interval.

A fixed coordinate-masking diagnostic used 100 nonoverlapping windows per
duration. Linear interpolation across ten-second masked windows has median
coordinate error 9.11 m and only 18.72% of masked positions within 3 m of the
supplied automatic ball coordinates. This is not ground-truth ball accuracy,
natural-gap accuracy, or a possession score. The interpolated gap inventory
only identifies image regions worth searching; it is not a tracking correction.

## Detector setup

The local checkout now has the supplied tracking, metadata, homography, Smart
Tagging and original tactical video, plus the pre-existing `yolo11m.pt`. The
video was obtained through the participant data folder supplied by the user,
using a normal authenticated browser download. It is 5,222,307,334 bytes; SHA256
`ff249f072aa7cd89a4a7f603a08f399aa0d34bd429982c847cd4a9373f645f14`.
Automatic checks verify 199,319 frames, 25 fps, 1920x1080 and 17 sampled decodes.
No video preview, frame export or human prediction comparison was performed.
Download provenance is in ignored `data/tactical/video_download_manifest.json`.

Use an existing externally pretrained detector. Organizer guidance in
[issue 2](https://github.com/metrica-sports/more-than-a-hack-2027/issues/2)
disallows training a detector on this match using its tracking as labels.
The local organizer-response snapshot is under the ignored audit directory.
The supplied participant clarification above is controlling. Checkpoint metadata
identifies the existing detector as COCO-trained, exported 25 September 2024,
under AGPL-3.0; its hash is recorded in the private provenance receipt. No
BAR-ATH training or parameter calibration is performed.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-video-ball.txt
# Fixed local GPU pilot, existing detector/settings:
.\.venv\Scripts\python.exe scripts/research/video_ball/detect.py --device 0 --clips 40 --clip-length 50 --imgsz 1920 --conf 0.05 --checkpoint-every 100 --out data/research/target080/ball_pilot_gpu.npz
# Deterministic CPU reproduction uses a separate output path:
.\.venv\Scripts\python.exe scripts/research/video_ball/detect.py --video <path-to-supplied-video> --device cpu --out data/research/target080/ball_pilot_cpu.npz
```

The pilot samples clips across both halves without crossing boundaries. Missing
frames can be selected with `--frames missing`, using supplied tracking loaded
from `--data` (default: this repository's `data` directory). The old hard-coded
Mac `/tmp` paths and MPS device default have been removed.

The runner saves raw pixel and projected pitch detections and periodically
checkpoints completed frames, including frames with zero detections. It hashes
the video, weights, geometry, relevant source and outputs. `--resume` refuses
changed inputs, settings, library versions, corrupt outputs or non-prefix frame
timelines. CPU/GPU results are separate protocols and must not be interchanged.
Automatic library downloads, telemetry and experiment integrations are disabled;
library settings stay under the private output directory.

The fixed 2,000-frame GPU pilot completed with 14,418 raw detections and no
nonfinite pitch projections. These are unverified candidates, not recovered
ball truth. Automatic tracking agreement was recorded privately as a correctness
diagnostic; it did not select any parameter. A repeat of that pilot produced
identical arrays and the same artifact SHA256. The separate pass over all
64,643 originally missing-ball frames has completed with 372,758 raw detections.
The fixed external filters retained 18,168 coordinate hypotheses and preserved
every originally observed coordinate. These counts do not establish correct
ball identity or possession performance. The full pass has also been repeated
successfully: every array and the exact artifact hash match the original run.
Repeated fusion produces identical coordinates, and CPU export from those
repeated inputs produces identical raw/final CSVs and passes the official
validator. The CPU repeats reuse the verified feature cache; no separate
cold-cache feature recomputation is claimed.

```powershell
.\.venv\Scripts\python.exe scripts/research/video_ball/detect.py --device 0 --frames missing --imgsz 1920 --conf 0.05 --checkpoint-every 1000 --out data/research/target080/ball_missing_gpu.npz
```

## External-only coordinate-hypothesis protocol

Confidence evidence is fitted on a frozen set of 140 COCO 2017 still images
(35 sports-ball images and 105 deterministic negatives), restricted to explicit
CC BY / CC BY-SA image licenses. The COCO Consortium's annotations are CC BY
4.0. Flickr attribution/license metadata and image hashes are retained in the
private manifest. Twenty-eight images are reserved for external validation.
The small calibration pool is not football-video-domain validation or evidence
of BAR-ATH possession performance.

A fixed class-balanced logistic fit maps raw detector confidence to an
evidence score, not a BAR-ATH probability. Its zero-evidence boundary, with the
library's standard 0.25 floor, defines the proposal threshold. A 99.9th-percentile
live-ball speed bound is fitted on the six existing external training matches.
No provided-match inputs enter either fit.

Observed tracking points remain exact. Missing positions receive hypotheses only
when one candidate passes the external confidence rule, is physically reachable
from supplied anchor observations and has adjacent physically consistent support.
Ambiguity is left missing. All thresholds are external or fixed in advance;
none is chosen against the provided video or possession truth.

```powershell
.\.venv\Scripts\python.exe scripts/research/video_ball/prepare_coco_calibration.py
.\.venv\Scripts\python.exe scripts/research/video_ball/calibrate_external_confidence.py
.\.venv\Scripts\python.exe scripts/research/video_ball/fit_external_motion.py
# Requires the missing-frame pass to be complete:
.\.venv\Scripts\python.exe scripts/research/video_ball/fuse_detections.py --detections data/research/target080/ball_missing_gpu.npz --out data/research/target080/ball_hypotheses.npy
```

The CPU exporter uses the frozen external-only run-58 parent, the unchanged
externally selected out-of-bounds rule on original observations, and a generic
pre-kickoff rule derived from automatic kickoff-start tags. It does not use the
earlier manually specified second-half kickoff frame/guard. A fresh raw control
must reproduce the parent before any video-augmented export:

```powershell
# Create an unchanged CPU reference from the packaged external parent. Keep
# match-derived outputs under ignored data/, outside the published model bundle.
$controlDir = 'data/research/target080/core_external_control/model'
New-Item -ItemType Directory -Force -Path $controlDir | Out-Null
Copy-Item -LiteralPath models/coreorigin-video-v1/selected.pt -Destination $controlDir
Copy-Item -LiteralPath models/coreorigin-video-v1/fixed_protocol.json -Destination $controlDir
.\.venv\Scripts\python.exe scripts/run_local.py --config configs/local_v26_pretrained_attention.json --out $controlDir --infer --device cpu
.\.venv\Scripts\python.exe scripts/research/video_ball/export_candidate.py --verify-control --control-reference data/research/target080/core_external_control/model/candidate/CoreOrigin.csv --out data/research/target080/external_control_cpu
.\.venv\Scripts\python.exe scripts/research/video_ball/export_candidate.py --ball data/research/target080/ball_hypotheses.npy --out data/research/target080/video_candidate
```

These exports are experimental artifacts. They do not replace v11
automatically and do not demonstrate the 0.80 target.

The video-augmented CPU export is complete and passes the official format
validator with 199,319 rows. A second CPU export using the same verified
coordinate artifact and feature cache produces identical raw and final CSVs.
The final SHA256 is
`63d8e7412f9b29937183b3883df9639bed598d76e77482de9919985126f0a635`.
Private receipts are `candidate_reproduction.json` and the candidate manifests
under `data/research/target080/`. The full detector and fusion repeats also match
exactly. The receipt is `full_reproduction_verification.json`; both the detector
and verification helper have confirmed successful terminal exits.

## Frozen public-window pipeline check and score scope

One final CSV, with the hash above, was frozen before the unchanged official
scorer opened the public labels. The isolated scoring directory contained no
raw/control alternatives. The check reports **0.5200 spell F1** and **0.630699
macro F1** on 7,500 graded frames and 17 ground-truth possessions. With 1,000
bootstrap draws, the spell-F1 5th–95th percentile interval is 0.38–0.66.
The 0.80 target was not reached on this window. The public check cannot verify
private full-match performance. No parameter change, candidate selection or
per-transition/video review followed the result.

The private freeze receipt, official scoreboard and evidence are under
`data/research/target080/public_pipeline_check/`. The command was:

```powershell
.\.venv\Scripts\python.exe evaluation/run_evaluation.py --gt public --results data/research/target080/public_pipeline_check/submission --out data/research/target080/public_pipeline_check/official --bootstrap 1000 --no-detail
```

This is a completed one-time check, not a tuning loop. The organizer's
[submission instructions](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md)
state that there is no live leaderboard or pre-deadline scoring feedback.
Do not seek or use private ground truth to work around this limitation.

The raw CPU control agrees with the unchanged CPU exporter at every frame.
Both differ from the historical CUDA run-58 CSV on two individual boundary
frames; this is recorded as a cross-device reproduction difference, without
manual corrections. Control verification requires an explicit fresh CPU
reference whose manifest matches the exact checkpoint, raw inputs and bytes.

The packaged parent also passes an isolated synthetic forward pass while an
audit guard denies all access to this checkout's `data/` directory:

```powershell
.\.venv\Scripts\python.exe scripts/research/video_ball/verify_model_bundle.py
```

This establishes that older local experiment folders are not hidden model-load
dependencies. It does not validate possession accuracy or reproduce video
inference.

## Required next evidence

1. Completed: restore and automatically verify the supplied original tactical video.
2. Completed: run the fixed pretrained-detector pilot and measure coordinate agreement on
   already observed ball frames and additional coverage in missing intervals.
   These are diagnostics, not possession labels or possession accuracy.
3. Use any coordinate agreement only to check correctness. Any integration and
   parameter selection must be fixed independently or established on external
   data, never tuned to these pilot results. Preserve v11 unchanged.
4. Completed: repeat the 199,319-row CPU export from the same coordinate artifact
   and feature cache, verify exact CSV bytes and pass the official validator.
   Completed: repeat the full detector pass and regenerate the candidate from
   those repeated inputs, with exact detector, coordinate and CSV matches.
   Pending: an authoritative full-match BAR-ATH score is required before
   claiming the 0.80 target.
   Do not use the public five minutes for threshold or candidate selection.

## Automatic phase-tag audit

```powershell
.\.venv\Scripts\python.exe scripts/research/video_ball/audit_tagging.py
```

This compares automatic phase descriptions with the frozen prediction CSV and
nearest-player geometry without reading possession truth or changing predictions.
Build-up/progression/chance-creation/transition and their opposing defensive tags
often describe exactly the same interval. They are deduplicated by interval and
implied possessor. Spatial tags (players or ball in a box/third), goal tags and
restart windows are not interpreted as live possession labels. Conflicting teams
remain unknown.

The supplied file yields 169 canonical intervals from 309 mapped descriptions;
140 descriptions are semantic duplicates. Unambiguous phase cues cover 12,266
of 64,643 missing-ball frames (18.97%). Of these, the model predicts live play on
4,896 frames and disagrees with the tagged team on 1,270. During observed-ball
disagreements, nearest-player geometry supports the phase side on 41.18% of
comparable frames. None of these agreement rates measures possession accuracy.
They do not establish which source is correct, and no tag override was promoted.

## Fresh fallback reproduction

A new CPU inference and the existing published-anchor/out-of-bounds corrections
were run in the updated environment. Intermediate artifacts are retained under
`data/research/target080/reproduction/`. The final LF-normalized CSV passes the
official validator and is byte-identical to frozen v11, SHA256
`496b42abcb0aec5ec0d4dffa6a1d8ca3c5b386dd0ccb9e4c697e877dfa1e2c80`.
The receipt is `reproduction/reproduction.json`. Neither public labels nor
external-match labels were read, and this verification establishes reproducibility,
not a new possession score. The 0.80 target remains unverified.

## Earlier asset-access check, before the participant folder was supplied

The user identified the Drive ZIP as a backup of this codebase. Reading only
65,536 bytes at the end plus the 3,511,638-byte ZIP directory confirmed 21,524
entries and **no video/container files**. Its tactical input members are the
same four non-video files already present locally. No archive member was
extracted and the full 34,687,979,399-byte backup was not downloaded.

Checked the current upstream/fork README and data instructions, the fork's
download code/notebook, release assets, and organizer download discussions.
No original video URL was found in those sources. The official instructions
require enrollment and a separately supplied participant download link. A
filename-only Drive search for BAR-ATH returned no files in the available
account. The private evidence record is
`data/research/target080/asset_access_audit.json`.

The user subsequently approved searching competition registration/data emails.
Restricted organizer/competition-subject/download searches found no data-access
message; the only topic match was the existing GitHub repository invitation.
No registration/download message was opened or saved. An attempted sender-only
query was rejected by automatic approval review as broader than this permission;
the narrower topic queries were used instead. Metadata for an already-supplied
public tracking file exposed no parent-folder navigation. The next video
experiment at that point needed the original participant download link. The
user subsequently supplied that link and the original video was retrieved as
documented above. The 0.80 target remains unverified.

Focused checks:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_video_ball.py tests/test_dead_ball_rules.py tests/test_anchors.py -q --basetemp=data/test-tmp/target080
.\.venv\Scripts\python.exe -m pytest tests/test_tagging_audit.py -q --basetemp=data/test-tmp/target080-tags
```
