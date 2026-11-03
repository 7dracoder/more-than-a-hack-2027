# BlindSpot: local development toward 0.90 spell F1

**Latest study completed, October 25:** Metrica sample Games 1/2 added 94773 partial labelled training frames; Game 3 was reserved from training and evaluated only after model selection froze. Both matched 40-epoch control/treatment runs retained their epoch-zero 0.8570048309178743 control. Run 48 remains selected; the 0.90 spell-F1 target is unmet. All checks and full exports passed, and no further jobs are queued. See data/local-v08/metrica-expansion-v1/status.json and metrica_data_audit.json.

Latest verified result: **0.8570048309178743** in run 48. Its fresh audit agrees,
and two full CUDA tactical exports give the same valid 199,319-row CSV. The
0.90 target remains unmet. The experiments below are chronological external
development results.

The requested target is 0.90 spell F1. The starting best is 0.7867515923566878
on the fixed external development match J03WPY. Repeated selection on that
match is exploratory; reaching 0.90 there would not establish 0.90 on unseen
matches or the private competition ground truth. No official score is known.

The earlier independent test belongs to the earlier frozen v0.7 ensemble,
not a later candidate: 0.7317388493859082, versus 0.6429693076374019 for its
reference. J03WQQ/J03WR9 remain retired as independent tests. BAR-ATH human
labels never enter feature generation, training, checkpoint or decoder selection.

## Diagnosis and experiments

The initial frozen-model audit reproduces 0.7867515923566878. At IoU 0.5 it
matches 97/97 development possessions longer than 10 seconds, 67/71 between
5 and 10 seconds, 100/129 between 2 and 5 seconds, 35/67 between 1 and 2
seconds, and 13/57 shorter than 1 second. Total true/predicted spells are
421/364. This motivates more precise short-spell evidence rather than
claiming that running more epochs alone will solve the problem.

The fixed training matches remain J03WMX/J03WN1/J03WOH/J03WOY. New features
add past and future geometric contact evidence at several distance thresholds,
relative player/ball velocity and short smoothing windows. They accept only
coordinates and team membership. Sampling changes from 5 Hz to 12.5 Hz.
Whole-half offline inference may legitimately use future observations.

The corrupted-coordinate view masks the ball, drops intervals of individual
player tracks and adds small coordinate noise. Masks depend on match/half
seeds only. Targets remain unchanged. Simulation is not natural-gap ground truth.

Planned candidate families: higher-resolution contact BiGRU, the same model
trained on native and corrupted inputs, and a hybrid model that preserves
local contact evidence alongside recurrent context. All considered development
results and regressions will be retained under ignored `data/local-v08`.

## Local commands

```powershell
.\.venv\Scripts\python.exe scripts/prepare_external_v08.py
.\.venv\Scripts\python.exe scripts/prepare_external_v08.py --corrupt
.\.venv\Scripts\python.exe scripts/run_local.py --train --config configs/local_v08_contacts.json --out data/local-v08/run-01-contacts
```

Only infer/export a selected candidate using its exact saved configuration.
The official scorer is unchanged. Restricted match assets, feature caches,
models and predictions stay outside Git.

## Verified intermediate results (October 24)

- Contact BiGRU: 0.8009592326139089 at epoch 35. Its full-match CSV passes
  the official validator (199,319 rows). A fresh development audit reproduces
  the score and finds 70/124 sub-two-second spells matched at IoU 0.5,
  compared with 48/124 in the starting model.
- Larger CPU histogram trees: 0.7018963337547408; CUDA XGBoost:
  0.7017721518987342. Both are retained as regressions rather than promoted.
- Hybrid, stable fractional focal loss: 0.803513174404015.
- Hybrid with corrupted-coordinate training: 0.805. On the same corrupted
  development inputs it scores 0.6322237017310254, versus 0.386447638603696
  for the clean-trained contact model. These stress scores use a different
  mask/noise setup from the old v0.7 gap test and are not directly comparable.
- Six-match training and auxiliary boundary supervision are in progress.
  Neither has an independent test cohort.

The first focal-loss hybrid attempt failed at epoch 63 because a fractional
power at zero cross-entropy produced non-finite gradients. A related run
was stopped before its first epoch. Their original source ZIPs and failure
receipts are retained. The loss now clamps its focal factor before the power,
with regression coverage and finite-loss/gradient guards. The corrected runs
finish normally. The 94-test suite passed after boundary-decoder integration;
four additional physical-feature tests also pass.

The user requested continued work toward 0.90. A new durable goal could not
be armed because the earlier blocked goal remains unfinished. Work continues
in this chat; this is not a claim that a background automation exists.

For the six-match development experiments, J03WQQ/J03WR9 explicitly join
training after their earlier test was retired. J03WPY remains held out from
training. The four-match v0.8 reference is preserved byte-for-byte. The new
`development_only` flag forbids claiming an independent test for these runs.

An auxiliary boundary head learns change-point targets from external training
labels only. Its calibrated probabilities can modify Viterbi transition scores.
An exhaustive path-enumeration test verifies the new decoder objective; the
unchanged constant-cost path retains parity with the earlier decoder.

The next feature bank adds coordinate-derived ball acceleration, direction
changes, projected interception distances and team-relative motion. Its v0.8
prefix is checked exactly, and the same corruption seeds preserve comparison
across feature banks. The proposed bank adds no possession/status inputs.

GPU tree reproduction uses `requirements-v08-gpu.txt` (XGBoost 3.4.1).
The actual local build reports CUDA 13.3 and training device cuda:0. See the
[official XGBoost installation guide](https://xgboost.readthedocs.io/en/stable/install.html).

## Current completed best and ongoing full-rate work

Run 13 (`run-13-physical-boundary`, config `local_v09_boundary.json`) finishes
with 0.8374845869297165 development spell F1 at epoch 95, decoder 1/2 with
boundary log-odds weight 1. Macro F1 is 0.969831421523582 and alive/dead F1
0.9992264433584638. A fresh audit reproduces the score. The full tactical
CUDA export validates with 199,319 rows; CSV SHA256 is
0d0c3f60b61b9c186ad3f294c2c9071b36776c2ebad2cd33c1fe96810b6585fb.
On the same corrupted development inputs, it scores 0.6870588235294118 versus
0.644077134986226 for run 09. Neither is an independent-match score.

Completed comparisons are retained: six-match hybrid 0.8106925880923452;
boundary supervision 0.8270337922403004; normalized feature ensemble
0.8225563909774435; TCN 0.7950556242274414; equal-run boundary training
0.8134161490683229. The local attention and recurrent-attention comparisons
do not redefine the benchmark or turn previously opened matches into tests.

The device has 15.2 GiB usable system RAM and a 12-GB RTX 5070 Ti Laptop GPU.
Training now releases each raw feature half as its normalized replacement is
created, avoiding an unnecessary full-bank copy. Recurrent-attention inference
keeps whole-half state; finite-attention models have tested overlap parity.

Full-rate caches (`features-v10` and `features-v10-corrupt`) preserve all 25 fps
using the same physical feature formulas, targets and corruption seeds. They
are uncompressed to avoid extra ZIP work; approximately 5.5 GB of ignored local
storage is used. `audit_feature_grid.py` verifies exact equality at the old
sampled frames and identical frozen-reference scores on native and corrupted
development views. The reference keeps its original 12.5-Hz grid even when
candidate inference becomes finer. The full-rate configuration records this
explicitly as `baseline_sample_step=2` and `sample_step=1`.

Run 16 (`run-16-fullrate-ema`) tests full-rate boundary reconstruction, including
ordinary and exponentially averaged weights (decay 0.995), selected only by
development spell F1. Averaging uses the installed official PyTorch helper;
CUDA smoke tests and exact averaging-resume tests pass. Selected averaged
weights need no extra model at inference. All decoder/weight candidates are
recorded in future training histories. Its declared early stopping condition
is 0.90 development spell F1; it must not be called independent or official.
The full suite passes 108 tests before this run.

`scripts/report_local_progress.py` creates an evidence ledger at
`data/local-v08/progress.json`. It distinguishes completed and unfinished fits;
it does not infer that a process is running or arm a goal/automation.

## Full-rate result and further experiments

Run 16 failed at its first full-half validation: the installed cuDNN GRU path
rejects that sequence length. Its failure and original sources are preserved.
The repair evaluates each GRU direction in chunks with carried hidden states,
then combines both directions before any next layer. CPU double-precision
parity, CUDA parity and a real full-half shape check pass. No hidden state is
reset at a chunk boundary and the original shorter inference path is retained.

Run 17 (`run-17-fullrate-safe`) completes all 60 epochs. Its best raw weights
at epoch 24 reach **0.8490291262135923** development spell F1, macro F1
0.9738386687020464 and alive/dead F1 0.9994227391640772. Decoder: team cost 0,
stoppage cost 2, boundary weight 1. A fresh audit reproduces the score. Its
199,319-row full tactical export passes format validation; SHA256:
61eef72996f895d2ce94faf964483e2dd4242cb9a8377c0caed4d768f3c0c65d.
Corrupted development spell F1 is 0.6605504587155963; run 13 remains the
stronger stress fallback at 0.6870588235294118. The 0.90 target is not met.

Run 18 tests a predeclared boundary-prior calibration grid; zero bias wins,
leaving run 13 unchanged. Run 19 tests a differentiable spell-overlap training
loss against the same full-rate control. It completes all 60 epochs with best
0.8415327564894932 at epoch 6, below run 17. Regressions are retained.

Run 20 (`local_v11_height_aux.json`) tests ball-height auxiliary supervision.
The Sportec paper documents the source ball Z field. Height is read only for
the six training matches, aligned by exact frame IDs, checksum recorded and
transformed with log1p. It is a regression target, never an input feature.
Inference and development use the same 664 coordinate-only features. Missing
height values contribute no regression gradient. Native and corrupted training
views retain identical targets. The base possession and boundary objectives
remain present. Reader/alignment/masking tests and CUDA deterministic backward
checks pass. The preceding full suite passes 119 tests; two additional model
tests verify inference without height and gradients into shared features.

## Next fixed development batch

The optional Numba 0.67.0 / llvmlite 0.49.0 decoder executes the same float64
MAP recurrence, including tie order, without fast-math. Random/near-tie tests
and the actual frozen run-17 development probabilities give identical paths
and score. A 150,000-frame local benchmark takes 3.736 s in the reference path
and 0.00390 s compiled, excluding compilation. The unchanged official scorer
still calculates every reported score. Reproduction pins are in
`requirements-v11-gpu.txt`; the NumPy reference remains the import fallback.
[Numba's release notes](https://numba.readthedocs.io/en/latest/release/0.67.0-notes.html)
document compatibility with the installed NumPy 2.5 series.

Run 21 blends frozen run-13 and run-17 outputs across their original time
grids. Coarse members sample the outer features on their own grid, then use
hold expansion; normalization stays member-specific. Three weight settings
and the six existing decoder settings are declared before scoring. It selects
0.8449877750611247, below run 17. Unit tests cover odd-length expansion and
exact standalone-member probability agreement.

The next batch is recorded in `configs/development_campaign_v13.json`. It
waits for the active fit and feature preparation, repeats the existing best
export, then trains fine motion, deeper/longer context, soft transition targets
and native-only longer context sequentially. No planned score is a result.
An improvement must reproduce in a fresh audit and two complete tactical
exports before the batch can claim 0.90 development success. Its ledger is
`data/local-v08/campaign_status.json`. This is an active local subprocess,
not a Codex goal or scheduled automation. Tests reject inconsistent fresh
scores and exercise the target-stop/export sequence using simulated outputs;
their simulated 0.901 is not an experiment score.

Fine motion appends 116 coordinate-derived inputs to the exact 664-feature
prefix: two temporal scales of ball speed/impulse/turn and motion among each
team's nearest few players. Geometric ties include all tied players, preserving
slot-order invariance. Preparation reuses the audited base features, checks raw
tracking/frame/target equality and retains the same corruption seeds. Missing
values remain explicit; no identity codes or labels enter the new feature API.
The full suite passes 128 tests before the two campaign-control tests, which
also pass independently. The official scorer remains unchanged.

Run 20 finishes all 60 epochs with best 0.8444444444444444 at epoch 24 (raw
weights), below run 17. Run 17's second complete CUDA tactical export is
byte-identical to its first: SHA256 remains
61eef72996f895d2ce94faf964483e2dd4242cb9a8377c0caed4d768f3c0c65d.
Its `reproducibility.json` records both passes and all 199,319 valid rows.

All 14 fine-motion caches pass an actual-byte prefix, label/frame, finiteness
and checksum audit (`fine_motion_cache_audit.json`). The expanded bank has 780
features. The integrated suite now passes 131 tests. A long-context boundary
TCN is prepared for possible later testing; it is not in the current batch.
Its finite-overlap probability test and standalone full-shape CUDA
forward/backward check pass, including process exit code 0. Two earlier smoke
processes that cycled through several architectures produced finite gradients
but failed during shutdown with native code -1073740791. Those failures are
retained rather than counted as clean process passes. The campaign uses one
training architecture per process; completed actual training/export processes
have exited cleanly. The precise cause of those shutdown failures is unconfirmed.

## Verified blend and revised local batch

Run 22 completes 60 epochs at 0.8423357664233576 (raw epoch 18). A fresh audit
reproduces it and links its probability cache to the exact checkpoint. Run 26
combines 25% run 17 and 75% run 22, using each member's feature bank and
normalization. The fixed 4/6 decoder wins; the boundary contribution is zero.
Its development spell F1 is **0.851770451770452**, macro F1 0.9720948306931426
and alive/dead F1 0.9995394845911544. A fresh audit agrees. Both full CUDA
tactical exports validate and have SHA256
82ca7340c54344d0eb3420ac8c91f2c2c56a54b06166d6588c71c506487445a8.
Checkpoint SHA256 is
1a6b1590a8422e25d8625598d700e8bfc5afeff3bf9724fe375c51ef533355b2.
Its corrupted-coordinate score is 0.6588235294117647, so run 13 remains the
stronger stress fallback. This small latest development gain is not independent
evidence or an official score.

Run 23 is deliberately stopped after three completed epochs because each
costs 124â€“139 seconds. Its partial best is 0.8291457286432159; the configured
60 epochs are not claimed complete. Its checkpoint, history and exact source
archive are preserved, and failure.json explicitly says stopped_for_runtime.
The revised finite batch is configs/development_campaign_v14.json. It reuses
existing export certificates only when the checkpoint and both CSV hashes
still match. Invalid certificates fail rather than allowing a target claim.

Run 24 (soft transition targets) completes at 0.842962962962963. Run 25
(native-only longer context) completes at 0.8429268292682928. Run 27's longer
boundary TCN completes at 0.8211031175059953. Its training masks artificial crop
context: 511 frames at each edge receive no loss, leaving the real interior of
the 4096-frame crop. Inference still covers every frame with tested overlap.
None improves run 26. Runs 28/29 test cosine learning-rate reduction to 1e-5;
the scheduler, optimizer and parameter trajectories have exact resume tests.

Runs 30/31 are frozen post-fit studies. Independent state/boundary blend weights,
global boundary biases and lower constant switching costs all retain run 26's
existing settings and score. The full search grids, including regressions,
remain saved. Their source archives are separate; no fresh test cohort is used.

The integrated suite passes **152 tests**. Progress reporting distinguishes a
selected fit crossing 0.90 from a certified result: target_met requires a fresh
audit linked to the same checkpoint plus two matching full-export hashes.
Synthetic 0.901 values in those tests exercise the guard and are not model scores.

Two later training prototypes are prepared but are not yet integrated into the
active runner. Short-spell crop sampling uses only complete H/A runs from the
six training matches: 345 are at most one second, 805 at most two seconds.
Unknown/censored runs supply no anchors; jitter avoids a fixed crop position,
and the disabled mode preserves the original RNG draws exactly.

The structured-loss prototype fits state unaries and transition potentials
jointly using exact chain likelihood. A balanced log-semiring reduction avoids
a thousand sequential Python/GPU steps. Double-precision exhaustive path
enumeration, numerical gradient checks, unknown-label resets, team-swap symmetry
and confident long-sequence checks pass. A deterministic CUDA backward check
for batch 8 by length 1024 averages 0.0212 seconds and exits cleanly. This is an
implementation/performance result, not a prediction score. Both prototypes
still need actual training and fixed-development evaluation.

An optional float16 probe for the oversized GRU did not provide sufficient
speed benefit and failed during native shutdown; it is not adopted. A failed
probe using mean-reduced spatial cross entropy is also retained; the actual
runner uses the deterministic unreduced loss. Actual campaign training and
inference remain float32.


### Completed cosine and short-spell studies; contact geometry next

Run 29 reaches 0.8533007334963326 at raw epoch 36. Run 33, initialized from it and trained with complete short-spell crop sampling, completes 40 epochs and reaches 0.8533333333333333 at raw epoch 1. This is a negligible numerical increase. Fresh scoring and two complete CUDA tactical exports agree: 199,319 rows, CSV SHA256 87615ce4d005465e2a1ce2081f6f30d60d4a536908d4a839a06bfc3efe076b5e, checkpoint b9d7d2f7116e811d8b28349e9969d9a08d78677727d7bb08f7ac9d29dd1456ae. Its artificial-gap score is 0.6740157480314961, below run 13 at 0.6870588235294118.

The earlier structured-loss and sampling prototypes are now integrated. Cold structured training (run 32) completes 60 epochs at 0.832258064516129; structured warm fine-tuning (run 34) retains its epoch-zero teacher at 0.8533007334963326. The combined structured/sampling experiment (run 35) is being completed. Failed alternatives and source archives remain preserved. Archives are verified against their frozen protocol rather than overwritten by later working-copy changes.

The next experiment adds 56 coordinate-only contact geometry inputs to the existing 780-feature prefix. Added-input coefficients start at zero and reuse the parent training-only prefix normalization. Fourteen native/corrupted caches are being audited for exact prefix, label, frame and checksum lineage. All 169 tests pass at this integration checkpoint. The unchanged official scorer still reports spell F1; 0.90 remains unmet and there is no current independent or official score.

Run 35 also completes 40 epochs and retains the warm epoch-zero teacher at 0.8533007334963326; no structured-loss variant improves the selected candidate. The 14 contact-geometry cache audits pass with exact 780-feature prefixes and unchanged labels/frames. Run 36 is active: its epoch-zero 836-input model reproduces 0.8533333333333333 exactly. Its source archive is frozen before later trainer changes (SHA256 393bbce4504a75cf8b0b85cc6321fd6af507c62bca4f55848c36440097513c05).

An optional next experiment trains through the same arithmetic average of original/team-swapped predictions used at inference. Stable log-domain averaging preserves the weighted-boundary calibration. Six tests verify arithmetic parity, team symmetry, both-view gradients, extreme logits, validation and a complete model optimizer step. A deterministic float32 CUDA optimizer smoke test exits successfully, and the full suite passes 175 tests. These checks are implementation evidence; paired training has not yet produced a measured score.

Two additional cached-output studies preserve their full search evidence. A blend of frozen runs 26 and 33 reaches only 0.8447761194029851 (run 37), below both parents. A six-setting positive boundary-bias grid on frozen run 33 retains its zero-bias decoder and exact 0.8533333333333333 (run 38). Neither is promoted. Both source snapshots are preserved.

Run 36 completes all 40 epochs and retains its unchanged epoch-zero warm control (0.8533333333333333); the extra contact geometry does not improve this study. Its exact source snapshot remains verified despite later optional trainer changes. A new finite sequential batch tests inference-aligned paired training on the existing feature bank (run 39), followed by the same objective with contact geometry (run 40). Both keep the same six training matches, held-out development match, scoring code and warm control.

Further controlled candidates are prepared, not scored: dropout removal, focal frame loss, stronger boundary-positive weighting, and 90% native/10% corrupted gradient sampling. The latter retains both training views for the parent normalization and changes only the sampled training-view balance. Default view sampling consumes exactly the original RNG stream; endpoint, coverage, frequency and resume checks pass.

A 256-channel warm model is also prepared. Hidden channels and gate/head mappings are duplicated with unequal outgoing shares that sum to one. Tests verify complete state/boundary evaluation parity, unchanged parent tensors/RNG, finite gradients and independent learning of copied channels. An initial equal-share version preserved predictions but did not separate the projection gradients on its first step; it is replaced before any real fit. No score gain is attributed to these tests.

Run 39 completes all 40 epochs and retains the epoch-zero teacher at 0.8533333333333333. Run 40 is the paired geometry counterpart and is still training. The full integrated suite now passes 185 tests. A deterministic CUDA width-doubling probe exits successfully with a finite float32 optimizer update and peak allocated GPU memory of 416.31 MB. Mathematical evaluation parity is established in CPU double precision, while the local default cuDNN TF32 mode yields maximum logit difference 0.0062571 in the synthetic comparison. Disabling TF32 reduces that to 0.00003397. No global precision mode is changed for real runs. A wider candidate must therefore measure its actual epoch-zero development control, rather than assume exact numerical score parity. This probe is not a measured model improvement.

Run 40 also completes all 40 epochs without beating its 0.8533333333333333 warm control. The next predeclared batch (development_campaign_v22.json) has started: a widened 256-channel model, followed by lower dropout, focal frame weighting, stronger positive transition weighting, and native-view emphasis. The wider model reproduces the exact development score before training; its two first epochs take about 58 seconds each, substantially slower than the smaller control. Its source archive is frozen at SHA256 d64e25ad82bab4e98c727447b5165dae909261be616de4067d814a8a98ef2aac. These are running/prepared candidates, not improvements.

A further frozen-teacher/local-corrector prototype is being integrated. Only the corrector receives gradients; the teacher remains in evaluation mode even during training. The final correction heads start at zero. Initial constructor and overly strict separate-instance float32 equality checks were corrected before any real fit. Its configuration remains untrained.

The integrated suite passes 192 tests after adding a local-width variant that retains a 128-channel GRU. Its deterministic float32 CUDA probe exits successfully: warmed update times 0.02833/0.02703 seconds, peak allocated memory 369.12 MB. The fully widened 256-channel GRU probe also exits cleanly, but warmed updates take 0.36171/0.34307 seconds, peak 426.41 MB; the active full fit takes roughly 58–71 seconds per epoch. This is a measured runtime difference, not a hardware diagnosis. The frozen-teacher correction probe exits cleanly too: zero initial correction, no teacher gradients, finite corrector gradients, warmed updates 0.02749/0.02774 seconds and peak 161.13 MB. Synthetic probes establish implementation feasibility only.

A descriptive audit of the retained model at IoU 0.5 finds all 32 missed sub-second development spells have ball tracking throughout; 30 have a labelled-team player within 1.5 m and 22 have a clear nearest-team approach under the stated geometric proxy. The matched group has 25 spells, all visible, with 24/22 respectively. These existing development targets are used only for diagnosis; no BAR-ATH labels are read or changed.

Run 41 is intentionally stopped after nine saved epochs for runtime efficiency, with all checkpoints and source retained. Its best post-initialization development score is 0.8506024096385542; the selected epoch-zero control remains 0.8533333333333333. It is explicitly a partial trial, not a completed 40-epoch fit. The verified task-owned native process was stopped, and the interrupted campaign ledger is preserved separately. The revised finite batch prioritizes frozen correction and local-only widening before the four remaining controlled loss/data-emphasis candidates.

Run 46 completes its fit with a provisional 0.8559036144578313 development spell F1. The campaign is checking fresh scoring, artificial-gap stress and two complete tactical exports before accepting the improvement. It remains below 0.90. The optional teacher-context correction variant passes its CUDA optimizer check at the higher 0.001 learning rate with zero initial correction, no teacher gradients and finite corrector updates; the full suite passes 193 tests. That follow-up remains untrained.


### Verified frozen-teacher correction result

Run 46 completes 40 epochs and selects EMA epoch 24 at 0.8559036144578313 spell F1, macro F1 0.974063290142703 and alive/dead F1 0.9994104497776903, with 421 true/409 predicted spells. A fresh audit agrees, and two complete CUDA tactical inference passes give identical validated 199,319-row CSVs. CSV SHA256 b605437dff7edd60441b7b61f6da1cf6a12dcef3ca98b2a1d662136c65493c21; checkpoint 0d8209cd3e1c99ccf48b2e765bef286e43efd78820dfd8c1e6a59865f10fb151; source ZIP 83c62ac4d2586b7074368c9494d75e27da0ce07bcdcfc0f6a17d9343a7182d97. Its artificial-gap score is 0.66282722513089, below both the previous run 33 (0.6740157480314961) and retained robustness fallback run 13 (0.6870588235294118).

The clean gain is 0.26 points. At IoU 0.5, sub-second matches remain 25/57; matches improve by one in each of the 1–2 s, 2–5 s and 5–10 s groups. This does not establish a new independent or official score. The 0.90 development target remains unmet. The local-width candidate is now running, followed by four predeclared loss/data-emphasis candidates. Higher-rate and teacher-context correctors are prepared but untrained; all 193 current tests pass.


### Continuation: 24 October 2026

Campaign v23 completes without further gains: runs 42-45 and 47 retain their 0.8533333333333333 warm control. Run 46 remains best at verified development spell F1 0.8559036144578313; the 0.90 target is unmet. Its completed ledger is preserved in data/local-v08/campaign-v23-final-status.json.

Runs 48-51 now freeze the entire run 46 model and train another zero-initialized correction branch. Run 48 reproduces the exact 0.8559036144578313 epoch-zero control. Its source ZIP is 5a6777c16df9d5459ab4ce768447e93de77cbf20cc40fc6964b1ce112c45f5de. The full integrated suite passes 212 tests, including strict verified-parent configuration matching, nested frozen teachers and portable reload.

All fourteen player-trajectory caches pass actual-byte, bounded-memory auditing: exact 780-feature prefix, unchanged labels/frame indices, finite 180-feature suffix and checksum lineage. Player trajectories follow central-frame identities across five offsets, mask discontinuous observations and pool through a shared, player-order-independent encoder. The real-crop CUDA optimizer smoke test preserves the initial teacher outputs exactly and all teacher weights after three updates; warmed steps take 0.04735/0.03813 seconds at float32. These are implementation checks, not score gains.

Runs 52-54 are queued after the current batch, using its best verified parent and the audited trajectory inputs. Execution is sequential on the local GPU. No trajectory fit has completed yet. No new dataset or BAR-ATH human labels are used, and there is no new independent or official score.


The trajectory suffix now retains the producer's common physical units across all player slots (metres divided by ten and binary observation masks); the original teacher prefix remains unchanged. The complete suite passes 214 tests in 22.293 seconds. This avoids slot-specific centering before a shared player encoder. The frozen official scorer remains SHA256 32c878630c74fb52a90c301485d6dff806509d8e933b1177b6035ee8e8f42b80.

A predeclared 36-setting decoder refinement of run 46 yields provisional development spell F1 0.8565217391304347 at costs [0,1,1.25] (run 55). It is not promoted or independently verified, since the active fit already has a higher provisional result. Its full grid and source archive are preserved.

A coordinate-only training-context diagnostic compares a 1,024-frame crop with a 4,096-frame reference on J03WMX without reading targets. Class disagreements are 0.1953 percent overall and zero after removing 64-frame crop margins in this one sample. This does not establish equivalence for every crop. The current study remains focused on player trajectories rather than introducing a full teacher-cache pipeline. Run 56 additionally tests 3x state-loss weighting within 120 ms of training transitions, using the existing exact binary transition targets.


### Verified higher-rate correction result

Run 48 completes 40 epochs and selects raw epoch 12, decoder [0,2,1], at development spell F1 0.8570048309178743, macro F1 0.9742746076990892 and alive/dead F1 0.9994288837440662 (421 true/407 predicted spells). A fresh audit agrees exactly. Two complete CUDA exports give the same validated 199,319-row CSV: SHA256 602b7f1beefc538eac04ed881028ec609de4dacb95f6db33b425b38a599aca19. Checkpoint SHA256 52bc76488d046f0d87347f09dd1d4d4cafe77a29b82238509baa92a233c0de03; source ZIP 5a6777c16df9d5459ab4ce768447e93de77cbf20cc40fc6964b1ce112c45f5de.

The gain over run 46 is 0.11 points. Sub-second matches at IoU 0.5 remain 25/57; one fewer 2-5 s spell matches. At IoU 0.7 one additional spell matches. Artificial-gap score is 0.6640419947506562, below the 0.6870588235294118 fallback. This is a small development-selection improvement, with no independent or official result. The 0.90 target remains 4.30 points away.

Run 49 is training with detached teacher context. The queued trajectory comparison adds learned attention pooling (run 57), retaining mean/max summaries and adding a shared masked attention summary. All 215 tests pass. Its float32 real-crop CUDA probe preserves the frozen teacher and completes finite updates, with warmed steps 0.05045/0.05708 seconds. No trajectory score is claimed.


A frozen-teacher validation-reuse prototype passes the complete 221-test suite, including nested/prefix teachers, player-trajectory pooling, EMA parity and rejection of altered inputs, normalization, architecture, weights, calibration or training mode. A full 69,131-frame real development half gives identical float32 probability bytes with and without reuse. Warm full/reused timings are 1.4775/1.4521 seconds in this probe; the initial 19.82-second full call includes startup and is not a fair speed comparison. No useful runtime gain is established, so the optional flag remains disabled in every actual fit. The receipt and source checksums are preserved. This experiment contributes no score gain.


### Overnight continuation, 25 October 2026

The remaining v24 studies complete without surpassing run 48: teacher context (49) and native emphasis (50) retain 0.8559036144578313; equal-spell weighting (51) reaches 0.8565900846432889. The v24 final ledger is preserved. Run 52 begins the 960-input trajectory study and reproduces the exact verified 0.8570048309178743 warm control. Its source ZIP is 60a1e3bf95a65a6ccdd48595711ebc598b6dee48a695eadb2bd4ace2e3cf7098. No trajectory gain has been established yet.

The larger feature bank causes host-memory pressure on this 15.21-GiB machine: 93-98 percent RAM used, with the active trainer committing roughly 12 GiB. Public-data preparation is therefore intentionally stopped by its verified task-owned worker identity after ten native and nine detected-only caches; all completed files remain intact, with no orphan final caches. Preparation and its raw-frame audit are queued sequentially after v25 if the target remains unmet. This is a runtime scheduling decision, not a failed numerical fit.

A lossless float32 file-mapping path is enabled only for unstarted runs 53/54/56/57 and future weak pretraining. Raw NPZ array members are extracted with CRC/checksum lineage and mapped read-only. Normalized arrays are written in bounded chunks, also read-only and checksum-linked to exact normalization/source bytes. A complete 69,131-frame, 960-feature development half has bit-identical raw and normalized values compared with the original pipeline; there is no quantization or mixed precision. All 235 tests pass, including normalization statistics, deterministic optimizer-update parity and tamper rejection. Initial Windows temporary-file cleanup errors in the new tests are fixed by closing their maps. Actual speed/memory gains still need measurement.

Twenty public SkillCorner tracking files are fetched at pinned commit 4340d274572876239c154c90bc507a9b3250a656 and verified against their Git LFS SHA256 values. An adapter keeps provider home/away groups as weak H/A targets; null always remains X, never D. It separates labels from coordinate-only features, masks detected-only views, and resamples 10-Hz coordinates to 25 Hz only across adjacent finite observations without speed jumps. Metadata processing resumes with explicit Python UTF-8 mode after a Windows locale error; numerical producers and completed caches are unchanged.

If v25 remains below 0.90, the next queued stage requires all forty weak caches to pass an independent raw-frame alignment audit, then pretrains an attention trajectory encoder for a fixed 30 epochs on eighteen public weak-label matches. Two separate weak matches provide monitoring only; those metrics are not development spell F1. A strict transfer helper copies representation weights into a new corrector, retains zero final output heads and the complete frozen best teacher, declares the extra weak supervision, and checks portable source/checkpoint provenance. Full tests and a future real-crop CUDA gate protect this path. No selected/current model has used these weak pretraining weights yet, and no new independent or official score is claimed.


### Runtime and weak-carrier preparation follow-up

Runs 52 (trajectory correction), 53 (native emphasis), 54 (equal-spell weighting) and 56 (3x state-loss emphasis near transitions) complete without surpassing the 0.8570048309178743 warm control. Run 57 is the remaining attention-pooling trial. The verified best remains run 48.

Lossless file mappings reduce committed anonymous feature storage. The post-mapping memory snapshot is 65 percent load / 5.21 GiB available, although later training snapshots remain tight (94 percent). Observed epochs in run 53 are roughly 12-13 seconds versus 18-20 seconds in the preceding trajectory run; view sampling also differs, so this is not an isolated speed benchmark. Public-data preparation stays deferred. Source ZIPs: run 53 c6e9018fc1e46450ccff6c37c5b27e75dafc0a0d58df2d48f98eb91d3dc98030; run 54 4182ecf40afae5afa85c91070b7cb05e4dcc7b9a18e1f186fa2de436a9d24545; run 56 c4f5b50dde08fc32fa04d12e88f40f3ad3a65cc1fedd80d64f54d8b23983122b.

Optional coordinate-feature caching is enabled only for unstarted runs 56/57 and the downstream template. It keys complete blocks by raw tactical/meta checksums, all feature/input-parser sources, frame bounds, pitch, sampling and library versions. Every stored float32 value and file is checksum-verified. Raw inputs are checked before/after parsing and again before export. Each model export still recomputes every prediction on CUDA; only unchanged deterministic features may be reused. No actual candidate score is attributed to this optimization.

The queued weak pretraining additionally uses provider carrier identities as separate rank targets among the three nearest candidates on the labelled team. Ties, missing candidates, unknown identities and team disagreements remain unlabelled. Numeric identities never enter coordinate features. Attention supervision is weighted [1,3,6] for ranks [0,1,2], emphasizing non-nearest controllers, with loss coefficient 0.1. A separate audit checks every target's observation mask and sampled source-frame identities/distances before publication. The pretrainer waits for that audit.

A real 32-source-frame pilot produces 78 resampled frames and eight usable carrier targets, all matching coordinate-feature distances; encoder gradients are finite and no optimizer update is performed. A deterministic float32 CUDA prototype then completes three synthetic weighted-carrier optimizer steps with finite attention/encoder gradients; warmed steps are 0.03671/0.03809 seconds, peak allocated GPU memory 102.83 MiB. These checks are not score improvements. All 245 tests pass. Audit completion markers are published atomically to avoid readers seeing a partial report.

A finer duration audit of the unchanged run 48 development result finds 12/32 missed sub-second spells below 0.2 seconds and 20 at 0.2-1.0 seconds (IoU 0.5). This supports investigating control-vs-proximity signals, without implying a guaranteed gain or changing any targets.


User stop limit (2026-10-25): Finish two attempts counting the already active run 57 as attempt 1, then only pretrained run 58 as attempt 2. Stop regardless of 0.90. The old three-fit weak-training coordinator was cancelled and verified absent. See data/local-v08/user_attempt_limit.json and finish_two_attempts.py.


Two-attempt cap reached (2026-10-25): Run 57 completed 40 epochs and retained its epoch-zero control (0.8570048309178743). The second method completed the fixed 30-epoch public weak encoder pretraining, passed the real CUDA transfer/frozen-teacher gate, then completed run 58 for 40 strong-label fine-tuning epochs; it also retained the epoch-zero control. No learned checkpoint improved the development score. Run 48 remains the verified best. The user limit overrides the earlier unlimited 0.90 objective. No runs 59/60 or additional tuning were launched. Evaluation/export verification only follows; it performs no optimization.

Final two-attempt verification complete: fresh development spell F1 exactly 0.8570048309178743; artificial-gap F1 0.6640419947506562; two complete format-valid 199319-row GPU exports were byte-identical (CSV SHA256 602b7f1beefc538eac04ed881028ec609de4dacb95f6db33b425b38a599aca19), also matching retained best run 48. Run 58 selected epoch 0 (warm_initial), checkpoint SHA256 07b86f226a5898c0db1f681176d7e3d3d4a4dddb07ab827b47bbfe213ef1e2fa; source archive 5f544d3ea4007938dacc40bb0bdc485a6b7f87e19b449e70112c9cb459154419. All training and verification are complete. STOP under the user two-attempt limit; no automatic continuation is armed.

New authorization: after the prior two-attempt campaign stopped, the user asked for other approaches and then said commence. One bounded transition-repair study is now authorized (configs/transition_repair_v1.json). It creates three fixed-30-epoch EMA reference models, each excluding two of the six strong training matches. Within each excluded pair a repair model trains on one match and is checked on the other, in both directions. H/A edit proposals use no targets, and D/X predictions remain unchanged. A fixed five-threshold comparison must improve mean spell F1 and at least four of six match checks to enable repairs; otherwise the policy is identity. J03WPY remains repeatedly used development, never independent evidence. No open-ended 0.90 loop or other suggested approach is started.


Transition-repair study complete (new authorization, October 25): Three reference models each trained from scratch for a fixed 30 epochs on four strong matches, excluding a pair; selected fixed-final EMA without excluded/development checkpoint selection. Six repair cross-checks trained on one excluded match and evaluated the other. The fixed threshold 1.0 passed four of six checks, mean gain 0.0011850645173366314 spell F1 (+0.1185 points). The final repair MLP trained on all six matches using out-of-fold reference predictions. On J03WPY, one deletion changed 407 to 405 predicted spells; short matches remained 25/57. Spell F1 fell from 0.8570048309178743 to 0.8561743341404359, despite macro F1 improving slightly from 0.9742746076990892 to 0.9744441133561373. Alive/dead F1 was unchanged. No promotion: run 48 remains selected. Seven focused tests passed; the real float32 CUDA preflight passed; portable reload reproduced exact predictions and verified archived sources. Two complete tactical exports passed format validation, were byte-identical, and each applied 12 algorithmic edits (no BAR-ATH targets read). Candidate CSV SHA256 57ce6460b34c9acffe0c6a8a69397710bd0cd1c34e76bba841603e3b0b079359, repair checkpoint e197ad8bc699d3443937b440cb076b57aa14182f38823bf60f35ddf8553002f2. Full-match uncorrected controls exactly matched retained run 48 CSV SHA256 602b7f1beefc538eac04ed881028ec609de4dacb95f6db33b425b38a599aca19. Sources and all three reference archives verified. This is development evidence, not a new independent or official score. Study complete and stopped; the visibility-dependent blend or other approaches were not started.


Metrica data expansion started (new user authorization, October 25): Public metrica-sports/sample-data pinned to commit e706dd506b360d69d9d123d5b8026e7294b13996, released 2021-04-15, before the provided 2024-2025 BAR-ATH match. All eleven files verified against Git blob hashes and SHA256; provider attribution and event definitions preserved. Games 1 and 2 supply training only; Game 3 targets are sealed until all model selection is frozen. Only tracking was parsed for Game 3. Additional targets are partial event-derived supervision, not dense possession ground truth: supported completed non-header passes/carries, short same-receiver bridges with no intervening hazard, and explicit ball-out to confirmed restarts; no labels from event absence, conflicting claims unknown, two-frame boundary margins masked. Training counts: H 39689, A 33245, D 21839, X 191389 (94773 known frames). Event-start/ball median distances are 0.341/0.354 m, p95 0.623/0.628 m with the published frame counters, no timing offset tuning. All five 780-feature caches pass source/byte, finite-float32, label-claim support, native/corrupt target equality and held-out target-absence audits. Seven focused policy tests pass. A bounded paired comparison has exactly two 40-epoch fits, identical frozen run 48 teacher, seed, optimizer/schedule, feature normalization and 210 updates per epoch; treatment replaces 20 percent of sampled crops with Games 1/2 partial-label crops. Both retain the epoch-zero best-model control. Existing development selects checkpoints; one later Game 3 audit reports event-label frame agreement, never dense spell F1 or official score. No unlimited 0.90 loop. Live ledger: data/local-v08/metrica-expansion-v1/status.json; reproduction: scripts/fetch_metrica_samples.py, scripts/prepare_metrica_samples.py, scripts/audit_metrica_samples.py, scripts/train_metrica_expansion.py.


Metrica data expansion complete (October 25): Added two training matches with 94773 known source-supported partial labels (H 39689, A 33245, D 21839; X 191389). Game 3 remained excluded from normalization, training and checkpoint selection. All five coordinate-only caches passed source/frame/claim/finiteness checks, corrupted target equality, and held-out target-array absence. Nine focused parser/policy tests passed. The matched control and additional-data runs each completed 40 epochs with identical parent, seed, schedule and 210 updates per epoch. Treatment used 13456 Metrica crops at the predeclared 20 percent mixture. Both selected warm_initial epoch 0 at 0.8570048309178743; data gain versus control was zero. The original run 48 is preserved and remains selected, with no added-data-trained checkpoint promoted. After model selections froze, Game 3 event labels were opened once: 94729/143761 frames (65.893 percent) had supported partial labels, and the H/A macro-F1 agreement was 0.9635533291778218 for parent/control/treatment alike. This is a DIFFERENT partial-label frame metric, not the 0.90 spell target, dense ground truth, or an official score. Game 3 release/ball alignment median 0.10191 m, p95 0.74789 m passed the fixed checks. The holdout registry is completed and prevents claiming a later adaptive reuse as fresh. Fresh development scores and artificial-gap score 0.6640419947506562 reproduced exactly for both selected models. Each produced two complete format-valid, byte-identical 199319-row tactical exports matching the retained parent CSV (602b7f1beefc538eac04ed881028ec609de4dacb95f6db33b425b38a599aca19). Model SHA256: control 771ed0c69a64bbab335cfd5dad2697f46e79c314272a4ba3a106aca8e4c31750; treatment 61abda83f9abf5ac9659e1c2fb22b2bda4dcddf611e9674978fce43d0f331c12. Both source archives and unchanged official scorer verified. Study complete and stopped; no additional fits or test reuse are queued.


### Ceiling audit and submission correction (29 October 2026)

Recovered the working tree from the archived bundle `more-than-a-hack-2027.zip`
(34,687,979,399 bytes) with HTTP range requests: the archive's central directory
was parsed and only the four tactical non-video assets and the run-58
artifacts were fetched, not the 19 GB of video. The archive's stored CRC fields
do not match the decompressed bytes, so extraction was verified by exact
uncompressed size and by agreement with the repository for tracked files; this
is a weakness of that archive, not of the data. No match asset was uploaded
anywhere.

Rebuilt a CPU-only macOS arm64 environment (`.venv`, torch 2.15.0) and ran the
suite: **266 tests pass**. Decoding the saved run-58 development probabilities
with the recorded costs [0, 2, 1] reproduces the recorded selection exactly:
spell F1 **0.8570048309178743**, macro F1 0.9742746076990892, 421 true / 407
predicted spells. Nothing was retrained.

**The plateau is not the decoder.** A declared 144-setting grid over team cost,
stoppage cost, boundary log-odds weight and boundary bias spans 0.724145 to
**0.858041**; the best setting gains 0.001. `scripts/audit_decoder_ceiling.py`
reproduces both numbers and writes `data/local-v08/decoder_ceiling.json`.

**The ceiling is the evidence, not the decoding.** Two oracles bound the two
halves of the problem:

| substitution | spell F1 |
|---|---:|
| model as is | 0.857005 |
| ground-truth spell boundaries, model's own team decisions | 0.900635 |
| model's spell boundaries, ground-truth teams | 0.890076 |

So even perfect segmentation leaves ~0.90, because the team call is wrong on
about 76 of 421 possessions, and even perfect team calls leave ~0.89. Both are
needed for anything above 0.95, which rules out further decoder or
post-processing work as a route to the target.

**Where the error lives.** At IoU 0.5, 359/421 possessions match. 124 spells are
shorter than 2 s and only 79 of those match; 113 of the 124 are *brief changes
inside the other team's possession* -- a touch and back, which the labels record
as two changes of possession. The training history says the same thing from the
other side: over 40 epochs the training loss falls 0.06361 to 0.04563 while
development spell F1 falls 0.8570 to 0.8432. Frame-level fitting beyond the warm
start is anti-correlated with the ranking metric, so more epochs on this
objective cannot be the lever.

**Submission correction, from published boundaries only.** The exported
candidate called 1,609 home and 1,200 away frames inside the pre-second-half
window. FAQ.md publishes that the second half is graded from frame 110808 but
does not kick off until 113742, and that those 2934 frames are all dead ball.
`possession/anchors.py` and `scripts/apply_published_anchors.py` rewrite frames
110808-113717 (one second of guard before the anchor, the same convention v0.3.1
used) and nothing else. 2,784 frames change, predicted possessions fall 413 to
399, and **14 possessions that cannot exist in the ground truth are removed**.
No label and no tuning set is read, so the direction of the change is certain
even though its size on the private match is not measurable here. The official
validator accepts 199,319 rows; the public-window score is unchanged because the
window is in the first half. New candidate:
`data/submissions/CoreOrigin-v08/CoreOrigin.csv`, SHA256
42b0a75bf954a887ea84615bdcd312c78ab0be315ff58d2c03d2f232b1e919d3.

**Public-window check (correctness, not a tuning signal).** 0.505263 spell F1 /
0.651007 macro F1 on the 5 published minutes. Four dead-ball runs sit in that
window; two of them -- a 20.4 s free-kick and a 14.5 s goal-kick setup with the
ball visible and inside the pitch -- are called alive, which is 1,198 of the
1,469 dead frames. Smart Tagging brackets both, but adopting a set-piece-to-dead
rule because the public window likes it would be tuning on the labels the rules
reserve for checking the pipeline, so it was not adopted.

**Verdict on 0.95.** It is not reachable with the current 960-feature bank: the
measured oracle for segmentation is 0.900635, and the decoder grid plus the
training history show the probabilities -- not the decision layer -- are the
limit. The remaining routes all require evidence the geometry bank does not
contain: recovering the ball on the 41% of graded frames where tracking loses it
(the tactical video is the only source), detecting ball-player contact at frame
resolution rather than through windows, or substantially more labelled matches.
None of those is a tuning change, and no fit was launched here.

Reproduce on a CPU-only Mac: `python -m venv .venv --system-site-packages`,
`.venv/bin/pip install torch` (2.15.0 here; the repository's pinned CUDA wheels
are for the Windows GPU host), then `python -m pytest tests -q` (266 pass),
`python scripts/audit_decoder_ceiling.py` and
`python scripts/apply_published_anchors.py`.

### 0.95 push: rules re-check, short-spell oracle and out-of-bounds rule (29 October 2026, evening)

**Rules re-checked against upstream** (metrica-sports/more-than-a-hack-2027 at f8f2629,
identical to the local RULES/FAQ/EVALUATION/CHALLENGE) and its organiser answers:
#2 a ball detector trained on the provided match's video with the provided tracking as
automatic labels is *not allowed* (train detectors on other matches only); #10 no BAR-ATH
labels for training, validation or choosing parameter values; #11 commit weights, CPU
inference must regenerate the CSV byte for byte; #14 rules from watching are fine if not
tied to specific frames; #15 future frames allowed; #16 IDSSE/Sportec DFL use confirmed,
all thresholds must be chosen on other games. Hosted LLMs must not receive match assets.

**Why 0.95 on J03WPY is blocked (new oracle, saved run-58 probabilities, no fitting).**
421 true spells; 254 are shorter than 5 s, 124 shorter than 2 s, 59 shorter than 1 s.
Replacing the model's frames with the truth for every true spell shorter than T:

| T | spell F1 |
|---|---:|
| model | 0.857005 |
| < 1 s perfect | 0.8971 |
| < 2 s perfect | 0.9173 |
| < 3 s perfect | 0.9324 |
| < 5 s perfect | 0.9581 |
| dead ball perfect | 0.8554 |

0.95 needs near-perfect recovery of every possession under 5 s. Dead ball is already
solved on DFL (alive/dead F1 0.9994), so set-piece / dead-ball work cannot move J03WPY.
A deterministic proximity rule (switch to a team when its nearest player is within r of
the ball below height z) replicates DFL BallPossession at only 0.684 spell F1 even with
true dead ball (best of a 60-setting grid on the six training matches), so the label is
not a simple function of the tracking that a rule could copy.

**Implemented: out-of-bounds dead-ball rule** (`possession/dead_ball_rules.py`,
`scripts/apply_dead_ball_rules.py`, `scripts/eval_oob_rule_dfl.py`,
`tests/test_dead_ball_rules.py`, 4 tests pass). Ball more than 1.0 m outside the
pitch for at least 13 frames -> D. Constants fixed on the six DFL training matches only:
99.40 percent of 54,760 flagged frames are BallStatus 0, 78/80 runs at least 90 percent dead.
J03WPY development: 0.857005 -> **0.858041** spell F1 (macro 0.974275 -> 0.974079,
406 vs 407 predicted spells) -- within noise, as expected because DFL dead ball is solved.
BAR-ATH (no labels read, public window not scored for selection): 1,993 graded frames in 32
runs change H/A -> D; predicted possessions 399 -> 420. Candidate
`data/submissions/CoreOrigin-v09-oob/CoreOrigin.csv`, SHA256
014e26717a50b7ea72166437cede7dab9f613bfb0b0ee7954a9776abe6bde2a1, official validator OK
(199,319 rows), receipt `dead_ball_rules.json`. Risk: BAR-ATH ball positions are automatic
and may include off-pitch false detections (for example a spare ball); the DFL precision
does not measure that. Not promoted automatically -- Tanmay decides v08 vs v09-oob.

**Label-free Smart Tagging inspection** (aggregates only): set-piece intervals are fixed 35 s
windows (kickoffs 5 s). For throw-ins the ball stays out of bounds until a median 6.3 s
after the tag start, so a tag starts near the stoppage, not at the restart. A tag-gated
throw-in bridge rule is possible but has no external labels to check it on (DFL/Metrica
samples have no Smart Tagging), so it was not adopted.

**Remaining routes to above 0.90** all need new evidence, not tuning: (1) a ball detector
trained on other matches' video (issue #2) run on the 19 GB tactical video, which is not on
this Mac, with a GPU host; (2) dense possession labels on more *other* matches, hand labels
allowed there, targeted at sub-5 s spells; (3) frame-level contact detection from (1).


### Touch-chain / dead-ball study toward 0.95 (29 October 2026, CPU-only)

Rules re-checked against parent `metrica-sports/more-than-a-hack-2027` (latest commit f8f2629,
23 Oct 2026): unchanged. Any algorithmic method is allowed; no hand labels or tuning on BAR-ATH;
external human labels on other games are fine. All work below uses the six DFL training matches
(leave-one-match-out) and J03WPY as development. No BAR-ATH file was read. Code:
`scripts/research/touch_chain/` (README inside).

**Where the J03WPY gap actually is (oracles on run 48, J03WPY):**

| substitution | spell F1 |
|---|---:|
| run 48 as is | 0.857005 |
| ground-truth dead-ball mask, run-48 teams | 0.858525 |
| run-48 dead-ball mask, ground-truth teams | **0.994787** |

Dead ball is effectively solved on DFL tracking; a set-piece/long dead-ball detector cannot move
J03WPY by more than ~0.0015 (the earlier OOB rule gave +0.0010, 0.858041). The entire remaining gap
is H/A switching inside live play. (On BAR-ATH the balance differs: dead-ball errors dominate the
public window, so the set-piece work still matters there, but it cannot be measured on J03WPY.)

**Touch-chain model.** Possession changes on DFL happen at a ball approach by the new team (new-team
nearest-player distance median 0.6 m, p90 1.2 m, offset median 0 frames). Candidates = local minima
(Â±2 frames) of each team's nearest-player distance under 2 m: 95% of all 2,027 training H<->A
changes have a candidate within 5 frames. With perfect candidate labels the chain decoder reaches
**0.9714** mean spell F1 on train (ceiling of the representation). A two-stage LightGBM
(65 geometry + 27 ball-kinematics + 65 Â±12-frame sequence features, then neighbour-candidate
context), candidate-level 2-state Viterbi:

| system (J03WPY unless noted) | spell F1 |
|---|---:|
| frame-level LightGBM (132 feats) + Viterbi, best of grid | 0.6895 |
| touch chain, stage 2, Î»=0.5 (chosen on train OOF 0.7745), truth dead mask | 0.8204 |
| run 48 + chain posterior fusion (any weight tried) | 0.8385â€“0.8498 |
| run 48 + chain short-spell insertion (thr 0.95 from train precision 0.771) | 0.8564 |
| run 48 + OOB dead-ball rule (prior) | 0.8580 |

The decisive sub-task is "an opponent comes within 2 m: did it win the ball?" (1,140 such cases on
J03WPY, 25% positive). Best tracking-only classifier: AUC 0.931, accuracy 0.88 (train OOF 0.916 /
0.851). Spell F1 > 0.95 needs roughly 98%+. Run 48 missed 62/421 J03WPY possessions at IoU 0.5;
32 are under 1 s, 28 have <10% of their frames predicted for the right team, and the touching
player was within 0.65 m (median) of the ball â€” the touch is in the tracking geometry but not
separable from a near-miss with 25 fps centre-of-mass positions.

**Verdict.** No allowed CPU-side change lifts J03WPY above 0.858; run 48 stays selected. >0.95 on
J03WPY is not reachable with tracking-only evidence. Remaining routes (none runnable here): (1) much
more labelled tracking (e.g. PFF FC World Cup 2022 open tracking+events, ~64 matches) to train the
contact classifier; (2) video-based contact/touch detection â€” impossible on DFL (no video), so it
cannot be validated on J03WPY at all; (3) ball height/foot-level pose. The Mac has no tactical video
and no CUDA host is registered, so no video work was attempted. Nothing committed or pushed.
