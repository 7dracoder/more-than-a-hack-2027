# Expanded geometry expert

This separate external-only experiment completed training and reproduction but
did not establish a reliable gain; the verified continuity-v4 handoff remains
selected. BAR–ATH spell F1 above 0.740 is not established. No supplied-match
truth, video inspection or public possession scoring enters this study.

## Hypothesis and input construction

The 792-input camera-loss expert uses 664 established geometry inputs and 128
spatial-history/receiver inputs. The full parent additionally uses 116 fine-motion
and 180 centre-player trajectory inputs. This experiment adds those 296 inputs
to the expert, yielding the existing 960-input parent bank followed by memory:
1,088 columns at 12.5 Hz. The added inputs are coordinate-derived; they do not
measure head direction, peripheral vision or intent. Trajectory offsets of
-8/-4/0/4/8 refer to 25 Hz source frames, not seconds.

Strong native/masked/mixed caches already contain all columns. For eighteen
weak SkillCorner training fixtures, preparation computes only the missing 296
columns from the pinned raw detected-only coordinates. It reuses the verified
792-column bank and unchanged H/A/X targets. All 36 half caches are complete.
The two weak development fixtures are excluded from preparation. The six strong
training fixtures, strong J03WPY development fixture and weak development pair
remain the same as the preceding study. No Metrica labels enter this fit.

## Initialization and normalization

Existing projection weights and their normalization are mapped to the new
positions; added columns start with zero weights. Two inherited names occur in
both the base and memory banks (`ball_seen` and `ball_seen_known`). The mapping
preserves each occurrence separately, including its own weights and statistics.
A first smoke run rejected an incorrect unique-name assumption before updating
weights; the corrected run reproduces all starting strong/weak metrics.

Only the added columns receive newly fitted normalization. Moments use external
training inputs exclusively, with group weights 0.25 strong-native, 0.125 masked,
0.125 mixed and 0.5 weak-detected. Means/second moments are symmetrized under team
exchange and standard deviations have the existing 0.01 floor. All original
normalization values remain unchanged. Five tests cover zero-weight projection
equivalence, duplicate occurrences, retained statistics, domain/team moments,
column placement and invalid clocks/schemas.

## Fixed training and selection

`configs/full_geometry_memory_adaptation.json` declares 30 epochs, 80 batches
per epoch, batch size 16, 512-step sequences and cosine learning rate
0.0001 to 0.00001. Half the batch uses strong HAD/boundary supervision and half
uses weak conditional H/A supervision (weight 0.25); X is ignored and no weak
D target is inferred. Only strong crops supervise the boundary head.

The existing observation-loss gate, expert strength 1, parent boundaries,
continuity penalty 8 and zero D-odds offset remain fixed. Selection maximizes
mean strong three-view spell F1, requiring unchanged native performance and
weak decoded H/A decline limits of 0.005 overall / 0.01 with the ball missing.
The zero-extended initial model stays eligible as fallback; ties retain it.
These are reused external development fixtures, not an independent test.

The corrected GPU preflight passed an optimizer step and discarded its updated
weights. Epoch zero reproduces continuity-v4's mean 0.678165 and all declared
weak control metrics. The actual fit is `run-03`; `run-01-smoke` failed before
initialization, and `run-02-smoke` is preflight only.

## Completed result

All 30 epochs completed and selected epoch 10 under the declared guards. Native
spell F1 is 0.857005, masked 0.545753 and mixed 0.642614, mean **0.681791** versus
0.678165 initially. Weak decoded H/A agreement is 0.828072 overall and 0.799480
with the ball missing. Missing-ball conditional H/A accuracy declines from
0.797995 to 0.783884, despite the decoded metric passing its guard; the gain is
therefore not universal. A separate execution verifies every input/source hash,
recomputes added normalization, preserves original statistics and training
membership, and reproduces all selected metrics.

The paired uncertainty audit does **not** support replacing the v4 handoff.
The observed mean spell-F1 difference is 0.003626, with a 1,000-draw paired
5th–95th percentile interval **[-0.003650, 0.010709]**. Resampling uses the same
421 GT possessions across both models and all three correlated views. This is
a post-selection check on one reused fixture, without epoch-search adjustment
or evidence of match-to-match generalization.

The first audit attempt stopped on a relative-path bookkeeping error before
producing comparison results. The corrected attempt is retained separately in
`run-03/paired_uncertainty_02/`; the first attempt remains preserved. No supplied
match was scored or inspected, and no BAR–ATH CSV was generated for this
unpromoted expert. The current v4 CSV and technical handoff remain unchanged.

## Commands and evidence

```powershell
.\.venv\Scripts\python.exe scripts/research/spatial_memory/prepare_geometry_extension.py --out data/research/full-geometry-adaptation-20261114
.\.venv\Scripts\python.exe scripts/research/spatial_memory/train_geometry_extension.py --out data/research/full-geometry-adaptation-20261114/run-03 --device cuda
.\.venv\Scripts\python.exe scripts/research/spatial_memory/verify_geometry_extension.py --run data/research/full-geometry-adaptation-20261114/run-03 --device cuda
```

The trainer verifies every input receipt and source hash and supports resuming
its recorded optimizer/RNG state. Preserve completed runs. The separate verifier
rehashes all inputs, recomputes new normalization from training inputs, confirms
original statistics and training membership, and reproduces selected metrics.
No new BAR–ATH file should be promoted before that verification and frozen
inference/export reproduction. Raw data, features and research receipts remain
local under ignored `data/` paths. Sportec is CC BY 4.0; SkillCorner is MIT.

The full-match numerical target still requires the organisers' permitted
evaluation. The user has confirmed no full-match score or endpoint is available
yet; the open submission form does not itself provide accuracy feedback.
