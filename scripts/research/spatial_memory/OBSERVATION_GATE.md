# Observation loss and real broadcast transfer

This continues the external-only study in `NATURAL_VISIBILITY.md`. The requested
BAR–ATH spell F1 above 0.740 remains unverified. No new BAR–ATH possession score
or public-label-driven candidate choice is part of this work.

## Frozen observation gate

The previous long-ball-gap gate missed frames where the ball remained visible
but the controlling player left the camera. `possession/observation_gate.py`
uses input availability only: activate when the ball is absent or either team
has fewer than eleven finite player observations. Counts are the established
memory feature's per-team count divided by eleven. The gate follows the
expert's 12.5 Hz input clock, repeated onto the parent's 25 Hz clock. It does not
certify coordinate accuracy, infer gaze, or identify a missing player's true
position. Tracking duplicates and actual reduced team sizes remain limitations.

Two previously frozen checkpoints and four mixture strengths were compared on
the existing J03WPY native/masked/mixed development views. The unchanged prior
candidate remained eligible, ties retained it, and the native guard remained
0.01. All previous supplied-match inference files were blocked: the new guard
permits only an explicit read-only list of externally generated caches from
the preceding study. Input hashes and the incumbent's scores reproduced.

The selected expert is fixed epoch 50 at strength 1 in observation-loss frames.
Parent state probabilities remain elsewhere; parent boundary probabilities and
decoder remain everywhere. External spell F1 is **0.857005 native, 0.531959
masked, 0.616749 mixed**, mean **0.668571**, versus natural-v1 mean 0.568972.
The parent mean on these same views is 0.545693. These are exploratory results
on heavily reused development data, not independent or competition performance.
The gain above 0.20 in the single mixed view is not the requested BAR–ATH gain.

Commands:

```powershell
.\.venv\Scripts\python.exe scripts/research/spatial_memory/select_observation_gate.py --out data/research/observation-gate-20261114
.\.venv\Scripts\python.exe scripts/research/spatial_memory/export_observation_candidate.py --model models/coreorigin-observation-gate-v2 --ball data/research/dual-feed-20261113-v2/ball_hypotheses.npy --out data/research/observation-gate-20261114/candidate-packaged --device cuda
.\.venv\Scripts\python.exe evaluation/validate_submission.py data/research/observation-gate-20261114/candidate-packaged/CoreOrigin.csv
```

The new CSV has 199,319 rows and expected SHA-256
`0fb15e05701276e2659ded08ce07a6198a64b4a58210f7b25af1088d56cb44f5`.
Prior bundles and CSVs, including frozen v11 and natural-v1, are preserved.
Three CUDA exports, including two from the packaged model, have identical raw
and final bytes. The official format validator passed. Inference guards recorded
zero unexpected forbidden reads; source/model/input hashes remained unchanged.
The parent's original source checksums and frozen v11's weights/CSV and official
scorer code also match their receipts. Verification is retained privately under
`data/research/observation-gate-20261114/verification.json`.

## Real broadcast diagnostic, not candidate selection

Before opening its metric summary, `audit_broadcast_transfer.py` froze the
parent, natural-v1 and observation-gate models. It prepared native and
detected-only views of SkillCorner fixtures 2016236 and 2017461, then saved all
eight sets of predictions before computing metrics. The matches were excluded
from the eighteen weak training/profile-fit matches but had been opened during
earlier encoder validation. This is not a new independent test cohort.

Coordinates are resampled from 10 Hz to 25 Hz only between adjacent finite
observations, with the pre-existing 15 m/s player and 80 m/s ball limits. Scores
use the exact shared 5 Hz ticks, avoiding duplicated interpolated labels.
Native includes provider-inferred positions; detected-only retains actual
detection flags. Unknown possession groups remain unscored X. No D labels or
spell scores are inferred. Conditional H/A metrics are kept separate from
HAD predictions compared against the provider's weak H/A labels.

There are 31,013 known H/A ticks, including 2,693 with the ball not detected.
In the detected-only view, weak H/A frame accuracy is 0.736594 for the parent,
0.752136 for natural-v1, and 0.735982 for the observation gate. On missing-ball
ticks those values are 0.566654, 0.603045 and 0.610100. Conditional H/A accuracy
on missing-ball ticks is 0.629038, 0.665800 and 0.628296, respectively. The gate
mainly reduces D outputs where the provider supplies an H/A group (15.893% to
3.231% versus parent); this is not independently measured dead-ball accuracy.
The result is mixed and does not establish a general transfer improvement.
No model or mixture was reselected using this diagnostic.

A source-only semantics audit found all 17,065 known carrier IDs agreed with
their provider group/team mapping. Many known group frames have no carrier ID:
44,957 of 62,022 native 10 Hz group frames across these two fixtures. Group
labels remain weak supervision rather than independent manual truth.

Receipts, frozen probabilities and summaries remain ignored under
`data/research/broadcast-transfer-20261114/`.

## Completed external training study

`configs/real_broadcast_memory_adaptation.json` declares a separate study before
training: warm start the frozen epoch-50 expert, mix six strong Sportec fixtures
with detected-only observations from eighteen SkillCorner training fixtures,
retain initial normalization, and use only conditional H/A cross entropy for
weak examples. The D logit is excluded from that loss; unknown labels are
ignored. Dense D supervision and change-point supervision come only from
strong external data. No carrier-visible filter excludes missing-ball examples.
At fractional source ticks, both adjacent provider labels must agree.

Feature preparation is:

```powershell
.\.venv\Scripts\python.exe scripts/research/spatial_memory/prepare_real_broadcast_memory.py --out data/research/real-broadcast-adaptation-20261114
```

It writes 792-feature half-rate matrices and separate H/A/X weak targets.
Development fixtures are excluded. The trainer implements the
configuration's joint external criterion: strong three-view spell F1 and
detected-only conditional H/A macro F1, with the native guard, strong-score
guard, and natural-v1 weak H/A / missing-ball conditional H/A guards. Initial
epoch zero remains the fallback. Weak development metrics come from the opened
two-fixture diagnostic and must not be represented as independent confirmation.
No BAR–ATH data can drive training or candidate selection.

The completed 40-epoch run selected epoch 30, with every declared guard passing.
Mean strong development spell F1 is 0.668976; weak H/A agreement is 0.822591,
or 0.798366 with the ball missing. The subsequent externally selected continuity
adjustment raises strong mean to 0.678165. Separate executions reproduce both
selections; three exports reproduce each final CSV byte for byte and pass the
official format validator. See [full study](REAL_BROADCAST_ADAPTATION.md) and
the frozen `models/coreorigin-real-broadcast-v3` / `models/coreorigin-continuity-v4`
bundles. The numerical BAR–ATH target remains unverified.

Three observation-gate tests, three weak-metric tests, the explicit read-only
isolation test and three conservative-label/conditional-loss tests passed.
The loss tests verify zero direct D-logit and unknown-target gradients. They
do not imply that shared representation updates cannot indirectly affect D
predictions; strong replay and external guards are still required.
