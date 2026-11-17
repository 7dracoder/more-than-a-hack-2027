# Controlling participant clarification

Supplied by the user on 4 November 2026. The participant email states that this
clarification takes precedence over documentation, earlier emails, and individual
answers. Apply it to all new BAR-ATH work.

## Prohibited

- Checking algorithm output against the provided video by inspection.
- Hand labels, human-produced data, or other manually created data about this match.
- Uploading supplied match data to AI services.
- Training, fitting, or calibrating an algorithm on possession ground truth or
  on the provided video.

## Permitted

- Automatic checks using supplied data.
- Inspecting supplied data/video to understand possession, without using it to
  validate predictions or tune parameters.
- Using video, automatic tracking, Smart Tagging, and generated data as algorithm
  inputs, with a model not trained on the match truth or supplied video.
- Providing intermediate outputs within the competition's data-sharing terms.
- External datasets for which the team has the necessary rights.

## Training and parameter selection

Training, retraining, and parameter tuning must use external data only. Neither
the BAR-ATH ground-truth possession data nor its video may drive those choices.

## Current experiment

The ball-recovery runner performs local inference with existing externally
pretrained weights. No training command is used. Video pixels, clips, tracking
exports, and frame-level predictions must stay local/ignored and must not be
sent to hosted models. Do not render video/prediction overlays for human
validation. Automatic metadata/decoding and coordinate-consistency checks are
correctness diagnostics, never a parameter or candidate-selection criterion.
Keep frozen v11 unchanged and report no new possession score without valid
score evidence.

## Earlier v10/v11 calibration requires review

The historical realistic-gap workflow selected synthetic-gap parameters by
matching aggregate BAR-ATH automatic-tracking statistics. It did not use
BAR-ATH possession labels or video training. Nevertheless, the new instruction
that tuning must use external data only means this parameter-selection history
must be reviewed before claiming eligibility. Preserve its exact weights and
CSV as historical evidence; do not treat successful format validation or past
reproduction as certification under the new clarification. New calibration
must use externally sourced data only.
