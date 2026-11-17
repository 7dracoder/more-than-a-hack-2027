# External missing-ball curriculum

This study gives labelled missing-ball moments more training exposure. It uses
only the existing external training split: six Sportec strong-label matches
and eighteen SkillCorner weak H/A matches. Unknown SkillCorner targets remain
masked; they do not become dead-ball targets. No supplied match, public sample
labels, new target-video review or target-derived visibility statistics enter
training or selection.

The control is the occurrence-safe expanded expert. Its team exchange acts on
raw features and preserves duplicate feature occurrences before normalization.
The curriculum keeps the same initialization, 1,088 inputs, training split,
train-only normalization, main random seed, 30 epochs, loss coefficients and
fixed observation-gated continuity decoder. New training sources and receipts
are isolated so completed experiments and the v4 handoff remain reproducible.

The fixed policy selects external training halves uniformly. For each weak
crop, an independent random stream has probability 0.5 of centering the crop
on a provider-labelled H/A frame with the ball absent. It clamps the crop to
its half and falls back to uniform sampling if no eligible center exists.
Known weak targets with the ball absent receive four times the loss weight;
visible frames retain their weight and unknown frames retain zero weight.
The conditional H/A loss cannot train D from weak labels. The additional RNG
state is saved with the optimizer for reproducible resumption. The control
sampling stream still consumes its uniform crop draw, keeping source-half
selection, strong crops and team exchange paired with the control experiment.

The trainer records per-half TRAIN counts and a hash of every eligible-center
array before fitting. Unit checks cover half boundaries, unknown masking,
control-stream preservation, disabled-policy equivalence and preservation of
the strong portion of mixed batches. GPU smoke training discards updated
weights. Selection uses the existing external development objective and
guards. Any candidate needs separate metric/hash verification and the same
paired possession uncertainty audit before replacing the retained handoff.

All development fixtures have already been opened repeatedly. Scores and
within-fixture uncertainty do not establish independent generalization or the
BAR-ATH target. No new BAR-ATH score is requested by this workflow.

```powershell
.\.venv\Scripts\python.exe scripts/research/spatial_memory/train_missing_ball_curriculum.py --out data/research/missing-ball-curriculum-20261114/run-02 --device cuda
.\.venv\Scripts\python.exe scripts/research/spatial_memory/verify_missing_ball_curriculum.py --run data/research/missing-ball-curriculum-20261114/run-02 --device cuda
```

The submission bundle continues to use continuity-v4 until a verified gain is
supported. The exact upload CSV must pass the official format validator.

## Completed training outcome

The fixed run completed all 30 epochs. The selected checkpoint is epoch zero,
the zero-extended fallback, with mean external spell F1 0.678165. The highest
trained-checkpoint mean is 0.674481 at epoch 20, which also fails the
missing-ball guard. The best eligible trained checkpoint is epoch 30 at
0.672768. No trained checkpoint beats the fallback. The final epoch increases
weak overall H/A agreement but does not improve the declared primary
criterion; this secondary result does not justify promotion.

The TRAIN-only inventory has 36 halves, 1,329,677 input ticks, 715,315 known
H/A targets and 55,421 known missing-ball targets. Every half has eligible
centers. All remaining unknown targets stay masked. No new BAR-ATH inference
or export is run for this rejected training policy. The fresh submission
bundle retains continuity-v4 and its existing verified predictions.

The separate verifier passed: it rebuilt every eligible-center hash and count
from TRAIN inputs, checked the frozen configuration and all input/source
checksums, recomputed added normalization from TRAIN only, preserved the old
normalization and 24-match training membership, and reproduced the selected
scores and guards. The fallback tensors exactly equal the zero-extended
initial checkpoint. No paired gain audit is needed because no trained
checkpoint exceeded the fallback. These checks establish the recorded
negative result, not the BAR-ATH numerical target.
