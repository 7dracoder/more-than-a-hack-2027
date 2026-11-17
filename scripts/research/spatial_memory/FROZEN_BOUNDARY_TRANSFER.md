# Whole-half frozen boundary transfer

This experiment fits a new transition classifier on frozen recurrent features
extracted with full-half inference context. It tests whether that representation
can support a better transition signal than a head trained on short crops.
The existing possession emissions, backbone weights, input normalization,
observation gate and continuity penalty remain fixed.

It is external development only. No supplied-match assets, predictions, video
or possession truth are used. Development fixtures were previously opened;
these results are not independent or BAR-ATH performance.

## Protocol

`configs/frozen_boundary_transfer.json` fixes one classifier fit and boundary
blend strengths 0, 0.5 and 1, with continuity penalty 8. The zero-strength
control retains v4. Use six strong training matches in native/masked/mixed
views with view weights 0.5/0.25/0.25. No weak training labels enter this fit.
The existing strong and weak development memberships remain unchanged.

The v3 boundary hybrid produces a 384-dimensional representation combining
local features and both recurrent directions. Extract each whole half in both
the original and team-exchanged input orders, using exactly the existing
inference normalization and GRU handling. The original state and transition
heads reproduce all ten cached development outputs exactly, before fitting.
All 46 feature files and the frozen backbone are checksummed.

Targets describe a net state change between consecutive 12.5 Hz input-clock
endpoints, matching the old boundary supervision. Both endpoints must be
labelled H/A/D. Exclude each half's first frame and unknown endpoint pairs;
never concatenate halves before constructing targets. Multiple changes within
one sampling interval cannot be resolved by these endpoint targets.

Keep every positive and every fourth negative. Give selected negatives the
ratio of total to selected negatives within each half/view, preserving their
total sampling weight. Each input orientation receives half the view weight.
Fit StandardScaler with these sampling weights on TRAIN examples only. Fit
balanced logistic regression with C=0.01 and subtract the known class-weight
log-odds shift (4.876127) before interpreting output probabilities. Average
the two orientation probabilities, convert the native interval probability to
full-rate hazards using the existing conservation rule, and blend only during
observation loss. Unbracketed endpoints retain parent boundaries.

The model converged in 81 iterations on 656,928 selected orientation examples.
Its joblib SHA256 is
`e3bcc68c1fb40b4ca697dc41d29cdd501378a48e0725aecf262de8224e4891c9`.
The original neural backbone remains frozen; its weights are never refit.

## Result

The unchanged control reproduces v4's mean spell F1 **0.678165**. The 50% blend
scores **0.677658** and the full blend **0.677398**. Both pass the existing
native and weak-data guards, but neither beats the control. Selection keeps
zero blend and no competition CSV is exported from the new head.

The full blend improves mixed-view spell F1 from 0.632938 to 0.637467 and weak
decoded H/A accuracy on ball-missing ticks from 0.799851 to 0.815447, but masked
spell F1 falls from 0.544552 to 0.537723. These development component changes
do not establish an overall improvement. They do not justify promoting the
head or claiming the recurrent context mismatch caused v4's errors.

## Reproduction and verification

Run `prepare_frozen_boundary_transfer.py`, `train_frozen_boundary_transfer.py`
and `verify_frozen_boundary_transfer.py` from this directory. Each supports
CLI help. Receipts are under ignored
`data/research/frozen-boundary-transfer-20261115/`.
The independent verifier reconstructs training membership, negative-sampling
weights, weighted normalization and the class-odds correction. It reconstructs
functional classifier probabilities and all three candidate scores/guards.
Its receipt determines whether verification passed; training output alone is
not the verification receipt. Two tests cover unknown endpoint masks, half
boundaries, input clocks and analytic odds recovery.

Verification completed successfully: TRAIN statistics, sampling and class-odds
correction reconstruct independently; all candidate scores and guards reproduce.
Functional probability differences are at most 3.88e-7 from float32 operations.
The weighted normalization reconstruction explicitly accounts for scikit-learn's
conversion of sampling weights to the float32 input dtype.

A public release metadata check pins SkillCorner to
`4340d274572876239c154c90bc507a9b3250a656`, the same existing revision. Its
20 published match IDs are exactly the existing eighteen TRAIN and two DEV
matches. It supplies no new untouched match for this experiment. No tracking
files were downloaded in this check; receipts are under `source-review/`.

Sportec strong data and the inherited SkillCorner-trained backbone keep their
existing attribution and rights. All derived arrays and predictions stay local
and ignored. The retained v4 submission and frozen v11 remain the handoff.
