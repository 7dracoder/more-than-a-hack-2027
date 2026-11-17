# Direct availability-input weight repair

This external-only experiment investigates a specific training/inference
mismatch. It is inspired by the general idea of directly repairing model weights,
not a reproduction of the Qwen experiment or a claim of using Astra to inspect
our checkpoint. No supplied match, possession truth or video inspection enters
the repair or its selection. The verified v4 handoff remains unchanged.

## Diagnosed mismatch

The base and memory banks reuse `ball_seen` and `ball_seen_known`. The historical
team-swap helper resolves duplicate names to their last occurrence. Training
normalizes before applying that permutation; symmetric inference swaps raw
inputs before normalization. Thus half the training crops can replace the base
normalized availability input with its memory-normalized copy, while inference
uses the base normalizer in both branches.

The two raw availability values are identical on every pinned external
development input. Their frozen normalizers differ: the base availability mean
is 0.786060 and scale 0.410085; the memory mean is 0.733700 and scale 0.442023.
The corresponding known-mask normalizers are identical. These are metadata from
the externally trained checkpoint; no BAR–ATH statistics determine the repair.

## Closed-form edit

For a shared raw value, normalized inputs satisfy
`z_memory = a*z_base + b`, with `a = scale_base/scale_memory` and
`b = (mean_base-mean_memory)/scale_memory`. Availability is binary and neither
normalizer clips its domain. Interpolating the first-layer preactivation at
strength lambda is implemented by scaling the base projection column by
`1-lambda+lambda*a` and adding `lambda*b*original_column` to the projection bias.
Other parameters and all normalization values remain unchanged.

The fixed strengths are 0, 0.5 and 1. The middle value corresponds to the existing
training swap probability; the last is a full-alias ablation. No coefficient is
fitted to match data. The nonzero edits change 128 projection weights and 128
bias values. Three tests verify the intended projection equivalence, zero-edit
identity, unchanged other tensors/statistics, non-mutation and invalid domains.
First-layer equivalence does not mean equality to a mixture of complete
nonlinear-network outputs, and does not guarantee improved possession inference.

## External result and verification

All three checkpoints are written before candidate scoring. Selection maximizes
mean strong native/masked/mixed spell F1 with native preservation, strong macro
decline at most 0.005 and the existing weak decoded H/A guards. Ties keep zero.
Every strength produces **exactly the same three spell-F1 values**: 0.857005,
0.544552 and 0.632938 (mean 0.678165). The full edit slightly increases weak
overall H/A agreement, but does not improve the declared primary criterion.
The unchanged model is retained. No new BAR–ATH inference/export is performed.

`verify_input_alias_repair.py` independently rebuilds every edited tensor,
checks unchanged normalization and other weights, reproduces all three candidate
metrics and guards, and rehashes sources and external inputs. Private receipts
are under `data/research/input-alias-repair-20261114/`.

```powershell
.\.venv\Scripts\python.exe scripts/research/spatial_memory/select_input_alias_repair.py --out data/research/input-alias-repair-20261114 --device cuda
.\.venv\Scripts\python.exe scripts/research/spatial_memory/verify_input_alias_repair.py --study data/research/input-alias-repair-20261114 --device cuda
```

These are already opened development fixtures, not an independent test. The
full-match BAR–ATH target remains unverified; the organisers' instructions give
no live accuracy feedback. A successful mathematical weight edit is not itself
evidence of a score gain.
