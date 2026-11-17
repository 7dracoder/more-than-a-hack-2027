# Adaptation restricted to added geometry coefficients

The preceding expanded-input and occurrence-safe studies produced small gains
on the reused development fixture, without positive paired uncertainty support.
Giving known missing-ball weak targets extra weight did not improve the
retained model. This study tests a narrower training update: use the 296 added
motion/trajectory features while preserving every inherited coefficient and
buffer. This is a capacity-control hypothesis, not evidence of improvement.

The configuration fixes the same six Sportec strong-label and eighteen
SkillCorner weak H/A training fixtures, development membership, seed,
initialization, 1,088-feature schema, occurrence-preserving raw team exchange,
30 epochs, batches, loss coefficients and observation-gated continuity decoder
as the occurrence-safe control. Normalization for the 296 new columns is fitted
only to external TRAIN inputs; all original normalization remains exact.
Unknown weak targets remain masked and supply no dead-ball targets.

Only 37,888 coefficients may change: 128 projection channels times 296 new
input columns. All other parameters have gradients disabled. The projection
gradient is masked to those columns, and its AdamW optimizer has zero built-in
weight decay. Explicit decoupled decay applies to the added columns before
each optimizer update, leaving inherited columns untouched. The model keeps
the original architecture and saves standard state dictionaries. Training-mode
dropout is retained; inherited weights and buffers remain fixed.

The trainer checks inherited-tensor equality after every epoch and on resume.
The separate verifier checks selected and final checkpoint tensors against the
zero-extended original, verifies zero inherited-column optimizer moments,
recomputes TRAIN-only normalization and reproduces scores and selection guards.
Unit checks exercise actual optimizer updates, frozen buffers/biases, invalid
columns and exact resumed-step agreement. A GPU preflight discards its update.

The primary selection criterion remains mean strong native/masked/mixed spell
F1, with native preservation and weak H/A guards. A candidate that beats the
fallback still needs the paired possession uncertainty check before promotion.
These are repeatedly used development fixtures. Neither the metric nor that
within-fixture check is an independent result or proof of BAR-ATH performance.
The organisers' permitted evaluation is still needed to verify the target.

No supplied match, its video, public possession labels or target-derived
visibility statistics enter training or selection. No new BAR-ATH possession
score is computed. Prior model bundles and the current submission remain
preserved until there is a verified reason to replace them. Research data and
receipts remain local under ignored `data/` paths.

```powershell
.\.venv\Scripts\python.exe scripts/research/spatial_memory/train_frozen_projection.py --out data/research/frozen-projection-adaptation-20261114/run-02 --device cuda
.\.venv\Scripts\python.exe scripts/research/spatial_memory/verify_frozen_projection.py --run data/research/frozen-projection-adaptation-20261114/run-02 --device cuda
```

## Completed result

All 30 epochs completed. Epoch 1 is selected, with native/masked/mixed spell
F1 0.857005 / 0.548352 / 0.636196 (mean 0.680517), versus the retained mean
0.678165. Weak decoded H/A agreement with the ball missing is 0.809135,
versus 0.799851. These are different development metrics, not BAR-ATH scores.

The separate verifier reproduced scores and guards, recomputed TRAIN-only
normalization and confirmed all inherited tensors remain exact in both the
selected and final checkpoints. Inherited-column optimizer moments remain
zero. All source/input checksums and the 24-match training membership passed.

The paired 1,000-draw possession audit gives a mean difference of 0.002352
with a 5th-95th percentile interval [-0.001645, 0.006507]. It does not support
replacing v4. This is a post-selection check on the reused 421-possession
fixture, without epoch-search correction or match-level generalization
evidence. The new checkpoint is not promoted or exported on BAR-ATH; the
current CSV, model bundle and technical PDF remain selected and unchanged.

Private receipts are retained under
`data/research/frozen-projection-adaptation-20261114/run-02/`.
