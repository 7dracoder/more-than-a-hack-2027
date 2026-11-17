# CoreOrigin: experimental video-assisted possession reconstruction

This describes the current experimental pipeline, not an eligible-submission
certification or a selected replacement for v11. Its frozen final candidate
scores 0.5200 spell F1 on the public five-minute pipeline check. Full-match F1
is unknown. The 0.80 full-match spell-F1 objective has not been verified.

## Approach and supplied assets

The pipeline combines supplied automatic player/ball tracking with conservative
ball-position hypotheses from a fixed, externally pretrained image detector.
An existing external-only temporal possession model then processes both halves
independently. A frozen sequence decoder produces H, A or D inside the supplied
half windows; other frames remain X. No new possession model was trained for
this experiment.

The original tactical video is used only for local automatic detector inference
at frames where the supplied ball position is missing. The automatic tracking
provides original observations, player/team geometry, missing-observation masks
and the video timeline. Metadata supplies team membership and pitch dimensions.
The provided frame-specific homographies project detections onto the pitch.
Automatic Smart Tagging supplies kickoff-start intervals for one generic
pre-kickoff dead-ball rule. Other phase tags were audited but are not used as
possession overrides. The broadcast video and human match-event data are not
inputs.

Raw tracking has a one-based video-frame counter; inference uses zero-based
video indices. Decoding checks verify this alignment. Nothing was manually
labelled or corrected on the match. No video frames or prediction overlays were
rendered for human validation, and no supplied assets were uploaded to hosted
AI services. The participant clarification in
[RULES_CLARIFICATION.md](../../RULES_CLARIFICATION.md) controls this work.

## Ball-position hypotheses

The detector is the repository's existing COCO-pretrained Ultralytics YOLO11m
checkpoint, used without retraining. Inference uses class 32, image size 1920
and raw confidence floor 0.05, with integrations, automatic downloads and
telemetry disabled. The current match pass uses the pinned CUDA environment.
Every input, relevant source file, selected frame and output is hashed; resumable
checkpoints include frames with no detection.

Raw detector confidences pass through an externally fitted scalar evidence
model. The acceptance boundary is its class-balanced zero-evidence point,
bounded below by 0.25. It is not a calibrated BAR-ATH probability. A proposed
position must be the unique qualifying detection on that frame, physically
reachable from supplied observations in the same half, and supported by a
physically consistent adjacent position. The motion bound is the external
training-set 99.9th percentile of live-ball speed. Ambiguous or unsupported
positions remain missing. Original observed positions are preserved exactly.
No supplied-match diagnostic chose these settings.

## External fitting and training history

The packaged `selected.pt` is an exact copy of the earlier run-58 external
parent. Its full training configuration and input/source lineage remain in
`fixed_protocol.json`; the document does not replace that provenance record.
The checkpoint records epoch zero, `warm_initial`: it retains the preceding
teacher with the new correction heads initially zero. It is not evidence of
a newly improved trained model.

Strong supervision came from six Sportec Open DFL matches: J03WMX, J03WN1,
J03WOH, J03WOY, J03WQQ and J03WR9. Provider BallPossession and BallStatus fields
provide the targets, without project hand labels. J03WPY supplied historical
development selection only. Earlier test matches were explicitly retired into
training; this configuration has no independent test cohort. Repeated
development selection limits the interpretation of historical development
scores. Credit Bassek et al. (2025),
[Scientific Data paper](https://doi.org/10.1038/s41597-025-04505-y), and the
[published data](https://doi.org/10.6084/m9.figshare.28196177), CC BY 4.0.

The configuration also declares a representation pretrained using eighteen
public SkillCorner matches with provider H/A and carrier-rank weak labels.
Unknown provider ownership was kept unknown; absence of ownership did not
create D targets. This weak supervision is distinct from dense possession
ground truth. Source: [SkillCorner Open Data](https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656),
MIT, copyright 2020 SkillCorner. All eighteen IDs and the encoder-checkpoint
hash are retained in the packaged protocol. No new matches were downloaded for
the current video-input experiment.

The detector's confidence evidence was fitted locally on 112 images from a
fixed, licensed COCO 2017 selection; 28 further images were reserved for external
validation. Selection comprised 35 sports-ball images and 105 deterministic
negative images, restricted to explicit CC BY or CC BY-SA image licenses.
COCO Consortium annotations are CC BY 4.0. `coco_attribution.json` retains image
references and licenses. A fixed class-balanced logistic fit uses unique
ground-truth box matches at IoU 0.5. This small still-image pool is not validation
of football video or possession accuracy. Calibration inference used CPU;
match detector inference uses GPU, a separate numerical protocol.

The motion bound uses only the same six Sportec training matches. The existing
out-of-bounds constants likewise retain their earlier external-only provenance.
Neither fit reads BAR-ATH possession truth or video. Detector weights and
Ultralytics code retain their AGPL-3.0 licensing; model and fit hashes are in
`model_manifest.json`.

## Possession model and rules

The existing 960-feature v25 schema combines coordinate-only physical features
with player-trajectory inputs. The packaged state retains training normalization,
the external temporal teacher/correction architecture and frozen decoder costs.
Possession inference runs deterministically on CPU at 25 Hz, with team-swap
probability symmetrization. The image detector does not directly assign team
possession. Geometric proximity, motion and temporal context supply model
evidence for control, loose balls and challenges; they are not independent
proof that a player controlled the ball.

Two postprocessing rules are explicit. Before an automatically tagged
kickoff-start interval in a half, live H/A predictions become D. Separately,
original observed ball positions more than one metre outside the pitch for at
least thirteen consecutive frames cause H/A predictions to become D. The latter
uses original tracking only, never recovered coordinates. No manually selected
kickoff frame or match-specific guard is used in this pipeline.

## Verification, limitations and reproduction

The first complete detector pass covered 64,643 missing-ball frames. Fixed
filtering retained 18,168 coordinate hypotheses. These are coverage counts,
not correct-ball labels or accuracy measurements. The complete candidate passes
the official 199,319-row format validator. Two CPU exports from the same
coordinate artifact and feature cache are byte-identical, with final SHA256
`63d8e7412f9b29937183b3883df9639bed598d76e77482de9919985126f0a635`.
The fixed 2,000-frame GPU pilot also reproduced exactly. The complete 64,643-frame
detector repeat subsequently exited successfully, with every array and its exact
artifact hash matching. Fusion of the repeated detections produced an identical
coordinate artifact. CPU export from those repeated inputs, reusing the verified
feature cache, produced identical raw/final CSVs and passed the official validator.
This verifies the repeated extraction, fusion and export; it is not a separate
cold-cache feature recomputation or an accuracy measurement. The full receipt is
`data/research/target080/full_reproduction_verification.json`.

Small sports balls, false detections, projection of airborne balls, camera-domain
differences and long missing intervals can defeat recovery. Physical consistency
does not prove ball identity. External possession conventions and missingness
differ from BAR-ATH. One unchanged-official-scorer public-window check of the
frozen final CSV reports 0.5200 spell F1 and 0.630699 macro F1 on 7,500 graded
frames and 17 ground-truth possessions. Its spell-F1 bootstrap 5th–95th
percentile interval is 0.38–0.66 (1,000 draws). Only the final CSV was scored;
no raw/control alternatives were included and no parameters or candidate
selection changed afterward. No per-transition/video review was performed.
This does not reach 0.80 on the public window and cannot establish full-match
performance. Private freeze/scoring receipts are under
`data/research/target080/public_pipeline_check/`. No accuracy improvement is
claimed. The historical v11 artifact is preserved;
its separate earlier tracking-based calibration requires eligibility review
under the new external-only tuning clarification.

See [the execution instructions](../../scripts/research/video_ball/README.md)
for the pinned environment, detector, fusion, unchanged CPU control and candidate
export commands. The bundle passes an isolated synthetic load/forward check
while all checkout `data/` access is denied. Supplied assets, detections,
coordinates, feature caches and CSVs remain private under ignored `data/`.
An operator with the supplied data can run the published commands locally.
No competition upload or repository publication has been performed.
