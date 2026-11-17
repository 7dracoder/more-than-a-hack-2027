# Receiver-context normalization correction and matched possession rerun

The v1 receiver-context audit found that never-observed roster slots changed
normalized entropy despite identical observations. V2 normalizes over recently
observed candidates plus the unavailable class. Historical sources and artifacts
remain unchanged. Padding and stale-candidate regression tests pass.

The versioned preparation reconstructs causal visibility counts and migrates
only the entropy column. Every other feature column must be exactly equal to
v1. Twenty fresh neural snapshots per half/view verify the corrected result,
for 920 direct checks across 46 feature files. No normalization parameters are
fitted during migration. Source, raw-coordinate and prior-feature hashes are
recorded. External possession fields are parsed by the shared loader but are
not used as receiver inputs; no supplied-match data enter this preparation.

The matched adaptation retains the same initial checkpoint, seed, 30 epochs,
losses, masks, 1,408 trainable projection coefficients, strong-only training,
weak-development guards and fixed decoder as v1. The GPU preflight reproduces
the v4 starting mean exactly and preserves inherited tensors after an update.

The completed run selects epoch 1, external development mean spell F1 0.679036
against v4's 0.678165. A separate verifier reproduces scores, TRAIN-only new
normalization, six-match adaptation membership and unchanged inherited tensors
and optimizer moments. The paired possession difference is 0.000871 with a
1,000-draw 5th-95th percentile interval [-0.000524, 0.002296]. The interval
spans zero. The corrected candidate is not promoted and no BAR-ATH export is
generated. This is reused external development, conditional on epoch selection,
not independent possession evidence or verification of the BAR-ATH target.

Receipts are local under `data/research/receiver-context-v2-20261114/run-01/`
and `data/research/receiver-context-v2-adaptation-20261114/run-02/`.

```powershell
.\.venv\Scripts\python.exe scripts/research/spatial_memory/prepare_receiver_context_v2.py --out data/research/receiver-context-v2-20261114/run-01
.\.venv\Scripts\python.exe scripts/research/spatial_memory/train_receiver_context_v2.py --out data/research/receiver-context-v2-adaptation-20261114/run-02 --device cuda
.\.venv\Scripts\python.exe scripts/research/spatial_memory/verify_receiver_context_v2.py --run data/research/receiver-context-v2-adaptation-20261114/run-02 --device cuda
.\.venv\Scripts\python.exe scripts/research/spatial_memory/audit_receiver_context_v2_uncertainty.py --run data/research/receiver-context-v2-adaptation-20261114/run-02 --device cuda
```
