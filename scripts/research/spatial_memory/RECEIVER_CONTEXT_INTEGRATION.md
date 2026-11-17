# Passer-marginalized receiver context for possession

The conditional receiver learner cannot receive the true actor at deployment.
This integration instead considers every currently observed player as a
possible actor. A fixed coordinate-only prior uses distance to the last observed
ball anchor (scale 2 m, maximum causal history 8 s). Each actor's receiver
distribution is computed from at most 2 s of past/present player observations,
then averaged under the actor prior. No event, passer ID or possession label
enters this API. Last-seen candidate positions stay masked when unavailable.
These are geometric proxies, not head direction, intent or measured vision.

Ten context features summarize actor/receiver team mass, unavailable mass,
entropy, expected relative receiver position, ball-anchor age and availability.
They are computed every ten 25 Hz frames and held causally on the 12.5 Hz
expert clock, with an explicit snapshot-age feature. Tests compare vectorized
actor features against the receiver learner, check probability conservation,
team exchange and candidate permutation, and handle missing ball/players.

Preparation reconstructs the existing external native/masked/mixed views with
the frozen TRAIN-only visibility profile and original seeds. Binary ball
availability must match the corresponding frozen cache exactly. The two real
weak-development fixtures use the same detected-only resampling as before.
The shared Sportec loader parses external possession fields, but they are not
used as receiver features. No provided-match data or public labels are read.

The first launch stopped at an allowlist omission before preparing features.
The second was deliberately interrupted to correct a misleading audit field
about parsed versus used external labels. Both remain preserved. Only the
new frozen `run-03` preparation can support fitting; partial outputs are not
promotion evidence.

The adaptation appends 11 zero-weight inputs to the 792-input expert,
preserves inherited normalization and all inherited tensors, and trains only
128 x 11 projection coefficients. New normalization uses six strong TRAIN
fixtures, with native/masked/mixed group weights 0.5/0.25/0.25. The inherited
eighteen weak training fixtures are not replayed. Existing weak development
guards still apply. The unchanged decoder, observation gate, 30-epoch schedule
and primary external spell-F1 criterion retain an epoch-zero fallback.

No receiver-task metric can establish possession improvement. Selected
possession scores and tensor constraints must reproduce, followed by the paired
possession uncertainty check, before any new BAR-ATH export. All external
development fixtures are reused; the full-match target remains unverified.

```powershell
.\.venv\Scripts\python.exe scripts/research/spatial_memory/prepare_receiver_context.py --out data/research/receiver-context-20261114/run-03
.\.venv\Scripts\python.exe scripts/research/spatial_memory/train_receiver_context.py --out data/research/receiver-context-adaptation-20261114/run-03 --device cuda
.\.venv\Scripts\python.exe scripts/research/spatial_memory/verify_receiver_context.py --run data/research/receiver-context-adaptation-20261114/run-03 --device cuda
```

All 30 epochs completed. Epoch 1 is selected at native/masked/mixed spell F1
0.857005 / 0.547165 / 0.632938, mean 0.679036 versus v4's 0.678165. Weak H/A
agreement is 0.826492 overall and 0.809135 with the ball missing. A separate
process reproduced these scores, six-match adaptation membership, TRAIN-only
normalization and frozen inherited tensors/optimizer moments. The paired
1,000-draw possession interval for the mean difference is
[-0.000524, 0.002296], spanning zero. No new BAR-ATH export is justified.
The cohort is reused development, and the audit is conditional on epoch
selection without match-level generalization evidence.

A synthetic probe also found that the prototype entropy normalizer depends
on never-observed roster slots. Padding four visible tracks with seven empty
slots changed entropy from 0.543624 to 0.352097 while other features stayed
unchanged. The historical module and all executed protocols are preserved.
`receiver_context_v2.py` instead normalizes over recently observed candidates
plus the unavailable class. Two regression tests preserve nonzero entropy,
ignore never-observed padding and retain recently observed hidden candidates.
That correction has not yet been fitted or possession-evaluated. It needs new
feature receipts and a fresh adaptation; the current candidate is not promoted.
