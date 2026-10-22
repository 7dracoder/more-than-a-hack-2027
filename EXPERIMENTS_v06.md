# BlindSpot v0.6: fixed architecture comparisons

## Design

Five user-added variants extend team-only contacts plus the v0.5 out-of-play prior. Review added held-out-label exclusion and dead-mask tests, reused cached individual offset models in ensembles, and corrected unique feature counts (acceleration plus contacts uses 94 distinct inputs, not 84).

- v06_gated: switching to a team costs 3 instead of 6 near that team's observed contact, within one neighbouring 5-frame bin. The decoder integrates timing evidence during optimization rather than moving boundaries afterward.
- v06_liveteam: binary ownership classifier trained on H/A frames only. Baseline dead probability is retained.
- v06_bagged: average five classifiers using distinct within-run sampling offsets. These are correlated models; averaging is not guaranteed to improve accuracy.
- v06_accel: sixteen collective speed-change/context features supplied only to the dead-ball model. Contact ownership is unchanged.
- v06_combo: all four changes together, evaluated as an exploratory combination.

All constants and model settings were fixed before this run. Baseline and team_only_outplay are controls, evaluated under the same four whole-run and four fixed-time folds. Every variant uses the existing out-of-play prior. No adaptive threshold sweep was performed.

## Verification plan

65 unit tests pass, including exact reference-classifier reproduction at offset zero, directional gating, disjoint sampling offsets, team symmetry, held-out label perturbation for both model types across all offsets, and preservation of D/X masks. Saved fold predictions are independently rescored by scripts/verify_v06_results.py; controls must reproduce v0.5 byte-for-byte.

## Reproduce

```powershell
$env:LOKY_MAX_CPU_COUNT='2'
python -m unittest discover -s tests
python scripts/run_v04_experiments.py baseline team_only_outplay v06_gated v06_liveteam v06_bagged v06_accel v06_combo --out data/experiments/blindspot-v06
python scripts/report_v04_results.py --experiment data/experiments/blindspot-v06
python scripts/verify_v06_results.py
python scripts/build_v04_colab.py --advanced
python scripts/prepare_blind_validation.py
```

## Limits and next evidence

Both development protocols reuse the same inspected public five minutes. Trying five more variants increases selection bias; this does not establish private-match performance or a world ranking. The >0.06 gain in both protocols remains a development heuristic, not statistical significance. The new Colab pair is locally prepared; no v0.6 cloud execution is implied.

A separate deterministic sampling plan selects twelve 30-second windows outside the public labels, with a 300-frame margin and no model predictions. See data/validation/blind-v1. No labels have been fabricated. A blinded reviewer must annotate these before validation is possible; this remains same-match evidence, not a different match. Freeze models before annotation. Local RULES.md permits small hand-labelled training/validation samples but forbids submitting manual corrections and using private full-match truth.

## Results

| Method | Whole-run spell F1 | Fixed-time spell F1 | Whole-run dead F1 | Fixed-time dead F1 |
|---|---:|---:|---:|---:|
| baseline | 0.4457 | 0.6200 | 0.6619 | 0.5945 |
| team_only_outplay | 0.5111 | 0.6341 | 0.7066 | 0.6458 |
| v06_gated | 0.4947 | 0.6341 | 0.7066 | 0.6458 |
| v06_liveteam | 0.5100 | 0.6244 | 0.7066 | 0.6458 |
| v06_bagged | 0.4737 | 0.6400 | 0.7041 | 0.6433 |
| v06_accel | 0.5765 | 0.6200 | 0.7231 | 0.6273 |
| v06_combo | 0.5231 | 0.6000 | 0.5319 | 0.6304 |

The acceleration-only change produces the highest whole-run spell F1 measured in this project, 0.5765, a gain of 0.1308 over baseline. Fixed-time spell F1 remains 0.6200. Five-offset averaging scores 0.4737 / 0.6400; the combination scores 0.5231 / 0.6000. No group passes the pre-existing >0.06 gain in both protocols. The selected v0.3.1 CSV remains unchanged; the acceleration candidate is experimental. A whole-run record is not a verified improvement on unseen footage.

The first long batch was interrupted after its progress log appeared stalled. The ensemble checkpoint had completed and was verified on restart. Acceleration and combination were then run in isolated processes, followed by seven-group checkpoint consolidation. No model settings were changed after scoring.


Final verification: all 56 fold CSVs independently pass format validation and reproduce official scores/confusion matrices; all 16 controls reproduce v0.5 byte-for-byte. Source/output hashes match. Gated and live-only ownership variants preserve full-match D/X masks. The acceleration full refit passes the official validator with 199,319 rows and SHA256 `198b2adb1e422db36abb4299cca8e38903e5626312a9303a92069e74313a2fa8`. Its public labels were used for fitting, so direct public scoring would be training fit. The selected v0.3.1 hash remains `c572620d15f0c60d9cf14df1b073131e91cb0153dd1c892ede0bd0fd5c8fc46e`.
