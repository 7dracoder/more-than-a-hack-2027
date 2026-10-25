# BlindSpot v0.7: verified local external-only pipeline

## Current result

The pipeline has trained and run end to end on the user's RTX 5070 Ti Laptop GPU (12 GB, compute capability 12.0). The latest gap-trained GRU exports a complete 199,319-row tactical CSV that passes the unchanged official format validator. No BAR-ATH reference labels enter training, selection or inference.

External development spell F1: fixed reference 0.62100955; first TCN 0.75526992; short-run-weighted TCN 0.74418605; bidirectional GRU 0.76617647; clean ensemble 0.77124842; gap-trained GRU 0.78675159. The newest clean development gain is 0.16574204, or 16.57 points.

Artificial-gap development spell F1: fixed reference 0.33360768; preceding clean ensemble 0.47369759; gap-trained GRU 0.63867403. This is +16.50 points over the preceding ensemble and +30.51 over the reference, on unchanged official external targets with simulated coordinate gaps. It does not prove natural-gap accuracy.

The first independent two-match test belongs to the earlier frozen clean ensemble: 0.73173885 versus reference 0.64296931, +8.88 points. Macro F1 is 0.94642018 versus 0.92749130; alive/dead F1 is slightly lower. The new gap-trained model has NOT been evaluated on an independent match. The requested +15 independent gain remains unachieved/unverified. No official competition score is known.

## Recent pull and rules

Pull 0215d48 incorporates organizer f8f2629 and changes eight documentation files, not model code. Current RULES.md and FAQ.md prohibit training/tuning on the public BAR-ATH labels and any manual annotation of that game for validation. Historical v0.3-v0.6 trained candidates are not current competition candidates. Historical supervised runners and the audit require --historical-reproduction; the same-match annotation script is disabled. Old results are preserved.

## Permitted data and protocol

Use Bassek, Rein, Weber and Memmert (2025), [Sportec Open DFL Dataset](https://doi.org/10.1038/s41597-025-04505-y), [official CC BY 4.0 files](https://doi.org/10.6084/m9.figshare.28196177). Credit the authors and DFL. BallPossession 1/2 and BallStatus 0/1 supply separate official targets; unknown active ownership stays X. Those fields never enter input features.

The split was fixed before fitting:

- Train: J03WMX, J03WN1, J03WOH, J03WOY.
- Develop/select: J03WPY.
- First independent test: J03WQQ, J03WR9, now opened and recorded in data/local-v07/test_registry.json. Neither the full cohort nor its subsets may be presented as a new independent test for another candidate.

Features accept only coordinates and team membership. Displacement-derived speeds avoid provider speed-unit differences. There are no possession/status inputs, event logs, ball-height inputs, match IDs, time-of-match features or identity codes. Each half is extracted/decoded separately. Team-swap augmentation and exact probability symmetrization enforce team symmetry. Normalization uses training matches only.

The fixed reference is the v0.3.1 classifier family retrained on four external matches and a common coordinates-only bank: 80 small histogram-boosted iterations, equal state-run weights and fixed 6/4 transition costs. It differs from the historical public-trained 68-feature model, so the old scores are not directly comparable. Its exact model bytes are reused across all experiments.

The TCN uses six residual dilated convolutions; the GRU uses two bidirectional layers. Both use 96 hidden channels. The gap-trained GRU combines native and artificial-gap external training views, with targets checked byte-for-byte against the originals. Alternating visible/missing intervals use exponential 9s/7s means, at least five frames, seeded by match/half only. No targets influence the mask. Native development labels select its checkpoint; the artificial-gap development evaluation is a separately recorded stress test. Repeated development selection remains exploratory.

## Local reproduction

The existing system torch was CPU-only. The isolated .venv now contains official torch 2.11.0+cu128; CUDA matrix multiplication and actual training/inference succeeded. Exact top-level versions are pinned in requirements-local-gpu.txt. CPU analysis uses two threads. Default sandbox temporary storage caused permission failures, resolved by data/test-tmp.

```powershell
New-Item -ItemType Directory -Force data/test-tmp | Out-Null
$env:TEMP = Join-Path (Get-Location) 'data\test-tmp'
$env:TMP = $env:TEMP
python -m venv .venv --system-site-packages
.\.venv\Scripts\python.exe -m pip install -r requirements-local-gpu.txt
# Reproduce native and missing-ball preparation, reference, GPU training,
# stress diagnostic, complete tactical inference and official CSV validation:
.\scripts\run_local.ps1 -Download -GapRobust
```

The default reproduction directory is data/local-v07/reproduction; choose a new -OutputDirectory after a substantive source/config change. Unchanged interrupted runs can resume. Omitting -GapRobust reproduces the earlier TCN/GRU ensemble. -OpenTest intentionally refuses already-opened matches in this workspace. The existing independent receipt must be preserved.

To use the already-trained latest model:

```powershell
.\.venv\Scripts\python.exe scripts/run_local.py --infer --config configs/local_v07_gaps.json --out data/local-v07/run-07-gaps
.\.venv\Scripts\python.exe evaluation/validate_submission.py data/local-v07/run-07-gaps/candidate/CoreOrigin.csv
```

If a system-site higher CPU-only torch interferes, install the pinned CUDA wheel with --ignore-installed --no-deps inside .venv. No cloud run, GPU purchase or competition submission was performed.

## Evidence and remaining work

All 80 tests pass. Checks cover exact decoder parity, team symmetry, slot permutation, feature/target isolation, half separation, recurrent state continuity, overlap parity, ensemble probability parity, label-independent masks, historical-training guards and cross-run test-cohort reuse prevention. Publisher file sizes/MD5 and recorded SHA256 verify external downloads; tactical files match the old notebook's pinned SHA256 values. Cache source/input hashes are checked. Checkpoints preserve optimizer/RNG state; configuration/source/input fingerprints and selection receipts identify the runs.

Actual artifacts live under ignored data/local-v07: run-02 TCN, run-03 decoder exploration, run-04 short-run regression, run-05 GRU, run-06 first-tested ensemble, run-07 gap robustness. The initial run-01 failed before fitting and is retained. Full match assets and predictions are excluded from Git. The technical draft is CoreOrigin.md.

Two complete inference passes reproduce the CSV byte-for-byte (SHA256 625445f4cf96d7799a94401b8066940b9e677809c068038f5b1c440cdf30f1b0); final source/evidence manifests are preserved. Remaining: obtain new independent labelled matches before claiming a +15 unseen-match gain. The new model's independent performance, private BAR-ATH score and natural-gap accuracy remain unknown. Full-match CSV format validation is not accuracy validation.
