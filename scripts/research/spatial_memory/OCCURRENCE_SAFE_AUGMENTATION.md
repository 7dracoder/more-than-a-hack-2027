# Occurrence-preserving raw team exchange

The 1,088-input expert contains repeated names for two common availability
features. A name-only dictionary aliases the later occurrence during training
augmentation. Those raw values agree, but their stored normalization differs.
The isolated control therefore exchanges teams on raw features using
`team_swap_indices`, which pairs by name and occurrence and verifies a
bijective involution. Normalization then matches symmetric inference. Prior
trainers, checkpoints and the retained 792-input model remain unchanged.

The fixed 30-epoch experiment uses the same six strong and eighteen weak
external training matches, expanded geometry, initialization, normalization,
main seed and selection guards as the preceding expanded-input study. It
does not read supplied match inputs or public sample possession truth.

The selected epoch is 10. External native/masked/mixed spell F1 reproduces as
0.857005 / 0.543407 / 0.642614 (mean 0.681009), compared with the retained v4
mean 0.678165. Decoded weak H/A agreement is 0.828072 overall and 0.799480 with
the ball missing, satisfying the declared guards. The independent verifier
rehashes inputs and sources, recomputes added normalization from TRAIN only,
checks original normalization and training membership, and reproduces scores.

The observed mean difference is 0.002844. A paired 1,000-draw resampling of
the same 421 ground-truth possessions across both models and all three views
gives a 5th-95th percentile interval of [-0.004278, 0.009620]. This does not
support replacing the v4 handoff. The audit is conditional on the selected
epoch, has no adjustment for repeated model selection and uses an already
opened fixture; it cannot establish independent generalization or BAR-ATH F1.

Private receipts are retained at
`data/research/occurrence-safe-adaptation-20261114/run-02/`. No BAR-ATH CSV is
generated from this unpromoted model. The freshly regenerated handoff remains
the verified continuity-v4 CSV.

```powershell
.\.venv\Scripts\python.exe scripts/research/spatial_memory/train_occurrence_safe_geometry.py --out data/research/occurrence-safe-adaptation-20261114/run-02 --device cuda
.\.venv\Scripts\python.exe scripts/research/spatial_memory/verify_geometry_extension.py --run data/research/occurrence-safe-adaptation-20261114/run-02 --device cuda
.\.venv\Scripts\python.exe scripts/research/spatial_memory/audit_geometry_extension_uncertainty.py --run data/research/occurrence-safe-adaptation-20261114/run-02 --device cuda
```
