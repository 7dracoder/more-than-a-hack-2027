# BlindSpot v0.4.1: contacts for ownership only

## Fixed design

The new `contacts_team_only` group takes live/dead probability from the 68-feature baseline classifier and conditional H/A probabilities from the 78-feature contacts classifier. Both fit the identical training mask with unchanged parameters, symmetrization, transition costs, five-frame bins, halftime handling and inferred-stoppage owner resets. Contact-model dead probability is excluded from decoding. There are 78 unique input features across two classifiers, not a single 146-feature model.

Evaluate exactly this one new approach against baseline and contacts, using the same four whole-run and four equal-duration folds, with the same 300-frame training-label embargo. Baseline must reproduce the frozen 0.4457142857 / 0.6200 scores and all eight reference CSVs. Every hybrid fold must preserve the baseline's full-match dead-ball and ungraded masks and use identical training masks in both classifiers. No dead-ball detector or boundary snapping is added in this experiment.

The suggestion was chosen after inspecting v0.4 public diagnostics; this is an additional same-sample development experiment, not independent validation. Raw tracking context can span whole halves. The conservative release heuristic remains a spell-F1 improvement greater than 0.06 in both protocols; that threshold is not a significance test. Keep the frozen v0.3.1 release unless the rule is met.

## Reproduce

```powershell
python -m unittest discover -s tests -v
python scripts/run_v04_experiments.py baseline contacts contacts_team_only --out data/experiments/blindspot-v041
python scripts/report_v04_results.py --experiment data/experiments/blindspot-v041
python scripts/run_v04_experiments.py --candidate contacts_team_only --out data/experiments/blindspot-v041
python evaluation/validate_submission.py data/experiments/blindspot-v041/contacts_team_only/candidate/CoreOrigin.csv
python scripts/build_v04_colab.py --team-only
```

The candidate command creates a labelled experimental export, not an automatic replacement of the frozen submission. Its model uses all public labels for fitting; its public score would be training fit. Use the separate v041 ZIP and notebook together for Colab.

## Source history

The verified v0.4 code was committed locally as `6ea1036` before implementation. Previous v0.4 outputs remain in their original experiment folder; new runs write to `data/experiments/blindspot-v041`.

## Results

| Method | Whole-run spell F1 | Fixed-time spell F1 | Whole-run dead F1 | Fixed-time dead F1 |
|---|---:|---:|---:|---:|
| baseline | 0.4457 | 0.6200 | 0.6619 | 0.5945 |
| contacts | 0.5600 | 0.6244 | 0.6906 | 0.5850 |
| contacts_team_only | 0.5000 | 0.6341 | 0.6619 | 0.5945 |

The hybrid improves baseline spell F1 by 0.0543 / 0.0141, below the pre-existing >0.06 requirement in both protocols. It scores lower than joint contacts on whole-run folds (0.5000 vs 0.5600) and slightly higher on fixed-time folds (0.6341 vs 0.6244). The frozen v0.3.1 candidate remains the provisional submission.

Wrong-live-team frames fall from 782 to 512 and 681 to 447. Live-called-dead remains exactly 310 / 238, and dead-called-live remains exactly 589 / 747, as required by preserved dead-ball masks. Role isolation works, but it cannot improve dead-ball detection by design.

All 46 tests pass, including both-head exclusion of held-out labels, independence from contact dead probability, team-swap symmetry, preservation of decoded and sampled dead states, and regression coverage for relative-path exports. The first experimental export produced a valid CSV before a log-writing path error; that logging bug was repaired and the unchanged numerical experiment was rerun for final source provenance.

The experimental full-public-trained CSV is kept separately under contacts_team_only/candidate and is not selected for submission. Colab v0.4.1 is packaged separately; local verification does not imply a new cloud run.

