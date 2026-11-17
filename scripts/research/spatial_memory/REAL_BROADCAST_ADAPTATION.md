# Real broadcast adaptation and continuity calibration

The BAR–ATH target is spell F1 above 0.740. It remains unverified. All training,
parameter selection and diagnostics described here use external data only.
The reused development cohorts are not independent tests. No new BAR–ATH public
possession scoring was performed; the official format validator was used.

## Real detected-only training

`configs/real_broadcast_memory_adaptation.json` freezes six Sportec strong
training fixtures, eighteen SkillCorner weak training fixtures and separate
development fixtures (Sportec J03WPY and SkillCorner 2016236/2017461). There are
856,433 unique native strong 25 Hz labels; augmented views reuse those labels.
The weak preparation contains 1,329,677 half-rate ticks: H 361,038; A 354,277;
X 614,362. Weak labels never include inferred D. At fractional source times both
adjacent labels must agree. Missing-ball examples do not require a visible carrier.

The 792-feature expert warm-starts from observation-gate-v2, retaining its input
normalization. Half of each batch uses strong HAD/boundary supervision and half
uses conditional weak H/A cross entropy (weight 0.25), which excludes the D logit
and ignores X. Strong replay constrains indirect changes to D. Training uses
40 epochs, 80 batches per epoch, batch size 16, 512-step sequences, cosine learning
rate from 0.0001 to 0.00001, and the fixed external seed. No BAR–ATH inputs enter
training or development.

Epoch zero reproduced initial metrics. Epoch 30 was selected by the declared
joint criterion and passed native, strong-mean and weak-domain guards. The
unchanged epoch-zero fallback remained eligible. Selected strong spell F1 is
0.857005 native, 0.532105 masked and 0.617817 mixed (mean **0.668976**). Weak
H/A agreement is 0.822591 overall and 0.798366 with the ball missing, versus
0.735982 and 0.610100 initially. The joint criterion's **0.744490 is neither a
spell-F1 result nor BAR–ATH performance**.

An independent execution rehashed every training/development cache and reference,
verified unchanged normalization and reproduced the selected metrics. The bundle
is `models/coreorigin-real-broadcast-v3`; private receipts are under
`data/research/real-broadcast-adaptation-20261114/run-02/`. Three exports, including
packaged and original run checkpoints, reproduce the same raw and final bytes.
The final CSV hash is `ed4c84360e265330d2f1a2c277d1446477c167a907cb562b58e29d810361e068`.

## Observation-loss continuity

A separate frozen grid (0, 2, 4, 8) tests an additional transition penalty only
when the established observation-loss gate activates. It is multiplied by
`clip(1.5 * (1 - max(H,A,D)), 0, 1)`. Zero penalty or complete observations use
the original decoder exactly. Neural weights, parent boundary probabilities
and all input normalization remain unchanged. Selection maximizes mean strong
three-view spell F1, requiring unchanged native performance and weak H/A decline
limits of 0.005 overall / 0.01 on missing-ball ticks. Ties retain zero.

Penalty 8 yields **0.857005 native, 0.544552 masked and 0.632938 mixed**, mean
**0.678165**. Weak H/A agreement is **0.825815 overall / 0.799851 missing-ball**.
All guards pass. A separate execution reproduces every grid candidate from
checksum-verified frozen probabilities and external targets. Three meaningful
decoder tests cover zero-penalty equivalence, exact brute-force path agreement,
and uncertain-flip suppression while retaining confident changes.

The bundle is `models/coreorigin-continuity-v4`; external selection, reproduction
and three identical exports are under `data/research/continuity-calibration-20261114/`.
The official validator accepts the 199,319-row file, hash
`cbc30311a6c9fdbe8cc7198e3a25f38d444c07c437e491528da5a924ecca408d`.
Inference guards record no unexpected forbidden reads. Original parent source
hashes and the frozen v11 CSV/weights and official scorer sources are preserved.
No claim of the requested BAR–ATH improvement follows from these external scores.

## Limits and further diagnostic

Weak SkillCorner possession groups do not establish dead-ball truth. A frozen
transfer diagnostic on the older public Metrica Game 3 compares the parent,
adapted and continuity models on native and externally degraded tracking.
Predictions are frozen before event-derived partial targets are parsed. This
fixture was opened by an earlier study, so this is exploratory, never a fresh
held-out test. Unknown event intervals remain unscored; no dense spell F1 is
computed from partial labels. Metrica source attribution and the pinned 2021
release are retained locally. No supplied video, manual labels or BAR–ATH
prediction inspection enters this diagnostic.

The completed diagnostic has 94,729 known event-derived frames and 49,032 unknown
frames, with median event/ball alignment error 0.102 m (95th percentile 0.748 m).
Known-frame H/A/D accuracy for parent / adapted-v3 / continuity-v4 is respectively
0.949804 / 0.972902 / 0.972712 native, 0.798319 / 0.917322 / 0.921291 masked,
and 0.758279 / 0.915021 / 0.916024 mixed. These are partial-label frame accuracies,
not spell F1. Continuity slightly reduces native partial-label accuracy versus v3;
therefore no universal improvement is claimed. The diagnostic does not reselect
the already frozen model. Predictions, checksums and results are private under
`data/research/metrica-transfer-20261114/`; no forbidden reads were recorded.

## Further fixed study

`configs/partial_had_memory_adaptation.json` declares a separate 30-epoch study
using Metrica games 1/2 partial HAD, strong/weak replay, unchanged normalization
and the fixed continuity penalty. Each 16-example batch has eight strong,
four conditional-weak and four partial-HAD crops. Partial examples carry no
boundary loss; unsupported intervals remain X. Native and mixed partial views
are prepared with the unchanged externally fitted visibility profile. Selection
uses existing strong/weak development guards only; no Game 3 diagnostic metric
enters selection or stopping. The frozen initial model remains the fallback.
All 30 epochs completed. The study retains the unchanged epoch-zero fallback;
the best trained eligible checkpoint (epoch 1, mean 0.669780) is below 0.678165,
and later checkpoints fail the missing-ball guard. Epoch 30 reaches 0.671256
mean but only 0.784998 missing-ball weak H/A agreement, versus the required
minimum 0.789851. No added-data checkpoint is promoted. Its 94,773 supported
native 25 Hz training frames are H 39,689, A 33,245 and D 21,839, with 191,389
unsupported frames remaining X. Augmented views reuse these targets. Receipts
remain private under `data/research/partial-had-adaptation-20261114/`.

## Rejected transition-head mixture

A separate grid uses the already trained v3 transition head only during
observation loss. Boundary mixtures are 0, 0.5 and 1; continuity penalties are
8 and 0. The head's change probability spans two full-rate transitions, so
equal per-frame hazards `1-sqrt(1-q)` are aligned to the bracketed interval.
Unbracketed endpoints retain the parent. Tests verify probability conservation,
clock endpoints and unchanged complete observations/state probabilities.

Both unchanged controls reproduce v3/v4. The best nonzero boundary mixture has
mean strong spell F1 0.675607, below the frozen 0.678165 fallback. Every candidate
passes weak guards but the existing model remains selected. A separate process
reproduces all six grid results, source/probability/target hashes and eligibility
guards. No weights or final BAR–ATH output change. Private receipts are under
`data/research/observation-boundary-20261114/`.

## D-odds calibration and uncertainty

A five-point grid (0, -1, -0.5, 0.5, 1) changes D log odds only during
observation loss, preserving conditional H/A odds and parent boundaries. The
continuity penalty stays 8. Native spell F1 must be unchanged, each strong-view
macro F1 may decline by at most 0.005, and the existing weak H/A guards remain.
Every candidate and guard reproduces in a separate execution.

The numerical winner is offset -0.5, mean 0.678273 versus 0.678165: gain
**0.000109**. This is not promoted. A post-selection paired bootstrap over the
421 GT possessions uses the same sampled indices for both models and all three
correlated views. Its 1,000 draws give a 5th–95th percentile interval of
**[-0.001535, 0.001871]** for the mean difference, spanning zero. Three tests
verify exact agreement with the official marginal bootstrap, identical-model
zero differences and pairing validation; three calibration tests cover H/A
odds, complete observations, symmetry and invalid inputs.

This is a post-selection, within-fixture uncertainty audit, without adjustment
for searching the grid or evidence about match-to-match generalization. The
conservative handoff rule was declared after the exploratory grid and before
resampling; it also requires a gain at least the official descriptive 0.04
spell noise band. No preregistered or independent test is claimed. The
verified v4 handoff remains unchanged. Receipts are under
`data/research/dead-odds-calibration-20261114/`.

The subsequent [expanded geometry study](FULL_GEOMETRY_EXTENSION.md) selected
an externally verified mean 0.681791, but its paired difference interval spans
zero, so it does not replace v4. A separate [direct input-alias weight
repair](INPUT_ALIAS_REPAIR.md) reproduces its mathematical projection change
but leaves the primary spell-F1 criterion unchanged. Neither study changes the
BAR–ATH CSV or establishes the numerical competition target.

Trajectory memory and all-direction receiver/lane geometry address some passing
context. They do not measure head movements, gaze or peripheral vision. The
external pose audit accepted no reliable head directions and is not deployed.

Full-match BAR–ATH performance can only be confirmed by the organisers' permitted
evaluation. Frozen v11's separate historical calibration eligibility caveat remains.
