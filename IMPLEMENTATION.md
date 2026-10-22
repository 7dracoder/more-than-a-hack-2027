# BlindSpot: runnable possession research system

## v0.3 candidate workflow

```powershell
pip install -r requirements-v3.txt
python -m unittest discover -s tests -v
python scripts/run_champion.py --resume --archive
python evaluation/validate_submission.py data/experiments/blindspot-v3/candidate/CoreOrigin.csv
```

Optional local video demo:

```powershell
pip install -r requirements-v3-video.txt
python scripts/build_candidate_review.py
```

The v0.3 suite tests three fixed approaches: a trajectory-feature reference, an unlabelled movement-regime model, and a small symmetric histogram gradient boosting classifier. Movement is derived from six-frame position displacement at 25 fps and known pitch dimensions; short windows containing missing positions or implausible slot jumps are excluded. The metadata says kph, but displacement comparisons show that raw speed values empirically resemble m/s. Blindly dividing by 3.6 was tested as a pilot and is **not** presented as a proven correction. That pilot and its source snapshot are retained under ignored `data/experiments/blindspot-v3-metadata-pilot/`.

The classifier has 59 features, 80 boosting iterations, at most seven leaves per tree, L2 regularization, team-swap augmentation and exact probability symmetrization. It is tested across four contiguous folds of complete labelled runs. A 300-frame embargo removes whole neighbouring training runs; the maximum feature context radius is conservatively 130 frames. Decoding uses full unlabelled halves without clamping test boundaries to reference labels. Matching counts are pooled across folds rather than scoring a stitched prediction.

Measured pooled development spell F1 is **0.4457**, compared with **0.3889** for the strongest reference on those same folds; macro F1 is 0.7484 and dead-ball F1 is 0.6619. The classifier improves three folds and is slightly worse on one. This is previously examined, single-match public development data, not independent validation. Full-public label-free motion-model spell F1 is 0.3610 versus smooth125's 0.4242. The separately exported provisional classifier fits all public labels; its training-window score is explicitly excluded from validation claims.

The candidate file, model/input/source hashes, fold sensitivity, model-disagreement review risk, coverage audit, artificial-gap stress test and evidence dashboard are under ignored `data/experiments/blindspot-v3/`. `candidate/CoreOrigin.csv` is a provisional artifact name, not the confirmed registered team name. Source code has not been pushed and no competition form submitted. See [SUBMISSION_TECHNICAL.md](SUBMISSION_TECHNICAL.md) for the current technical draft.

## v0.2 follow-up workflow

The follow-up adds a whole-state-run development split, stoppage error analysis and a fixed temporal stoppage candidate, public-window query error-enrichment controls (including 30 seeded random runs), tracking/tag/video-container audits, and a synchronized local video/pitch review lab. It retains the original label-free candidate as the main prediction.

```powershell
python -m unittest discover -s tests -v
python scripts/run_followup.py --resume --packets --review-video --archive
```

Open `data/experiments/blindspot-v2/index.html` for comparisons and `review/index.html` beneath that folder for synchronized video, 5 Hz pitch snapshots, model support and review notes. The twelve private clips are each up to 201 frames at 25 fps; their encoded frame counts are checked by decoding. Notes stay in browser storage and can be exported; no review note is ingested into a submission. This prepares a review experiment; it does not claim completed human annotations or measured time saved.

Each stage writes an atomic completion marker only after producing artifacts. `--resume` checks source/config/input provenance plus SHA256 of each stage artifact. An interrupted or corrupted stage is recomputed; source or input changes invalidate prior results. Video and optional asset fingerprints include file size and modification time; tactical inference inputs also have content hashes. The packet audit checks all compressed packets, not all decoded pixels. Broadcast is not fused without verified cross-camera synchronization.

`scripts/build_followup_colab.py` produces a separate source bundle and `notebooks/BlindSpot_v02_Colab.ipynb`, preserving the executed v0.1 snapshot. Default cloud execution downloads only four non-video tactical inputs. For durable Colab checkpoints, enable `PERSIST_TO_DRIVE` in the notebook's optional storage cell and authorize the mount; otherwise results are runtime-local and must be downloaded. The notebook never silently claims that runtime-local storage survives a reset.

The complete H/A/D run containing the target split is excluded or assigned wholly to one side; the minimum 250-frame embargo is rounded outward. The public data has already been examined during earlier experiments, so all new comparisons are explicitly **development** results. A new independent review set and final submission freeze remain future work.

Verified v0.2 Colab run: [BlindSpot_v02_Colab.ipynb](https://colab.research.google.com/drive/1jwU7f82AWzfRVmFEnpZ3-HgsdSOvayWo). All nine cells executed, including sixteen tests, the complete tracking-only follow-up, CSV validation, the rendered dashboard and result download. Both label-free CSV hashes match the local run; grouped scores agree. The cloud default omits video and broadcast. Archive-relative dashboard links are intended for the extracted result folder, not navigation inside Colab's inline HTML preview. The local results archive additionally contains twelve verified video clips. Direct browser automation of the local file was blocked by URL policy; local video interaction has not been browser-tested.

The full local command was rerun after checkpoint hardening and then rerun with `--resume`: all five stages verified and reused. No Drive mount was performed; durable storage remains an optional user-authorized notebook setting. See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for the measured results and remaining submission work.

This version implements an end-to-end experiment, not a claim of a winning competition entry.

## Run locally

```powershell
pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/run_experiment.py --ablations --train-public
python evaluation/validate_submission.py data/experiments/blindspot-v1/blindspot.csv
```

Optional video diagnostics require PyAV and OpenCV, and the tactical video:

```powershell
python scripts/run_experiment.py --ablations --train-public --visual-budget 4
```

## Run on Colab

Use `notebooks/BlindSpot_Colab.ipynb`, generated by `scripts/build_colab.py`. It downloads a checksum-pinned source bundle and the four tactical non-video assets from the user's organized public Drive copy. It verifies checksums, runs the tests and experiment, shows results, validates the CSV, and offers a result ZIP download. It does not require a GPU or grant access to the rest of the user's Drive. A notebook's runtime-local files are temporary; save the result ZIP before disconnecting.

The direct notebook bootstrap uses ordinary Python execution, not Colab's Gemini agent. No paid compute is provisioned. A GPU training stage should be added only when there is an actual model and adequate supervision that justify it.

Executed notebook: [BlindSpot_Colab.ipynb](https://colab.research.google.com/drive/1w6y-eYSDECPtd_Qe-aRMkNW_A6-_80oK). All eight code cells ran successfully on CPU, including the training comparison and tests. Cloud inference took 59.568 seconds. Every prediction CSV hash matches the local run exactly. The source-only execution snapshot is saved in the dataset's [project folder](https://drive.google.com/drive/folders/1odV1-SQAg2OcbVBi6A0TS5G_zyG5IEtf).

## Implemented components

- Strict streamed ATD indexing and content-hash keyed feature caching.
- Features aggregated over team sets rather than assumed persistent player IDs.
- Interpretable geometry, movement, phase/restart, missingness, and weak camera-pan evidence.
- Explicit code-by-code offensive/defensive Smart Tagging mapping; N/A is not assigned a team.
- Per-half Viterbi decoding at 5 Hz, expanded to every required video frame; X outside published blocks.
- Exact forward/backward marginals under the score model and seeded posterior state-history sampling.
- Fixed tag/camera/switch-cost ablations and source-sensitivity records.
- Gap histories showing alternative compressed state sequences and model-relative agreement.
- Boundary evidence records with local ambiguity bands, not statistical coverage guarantees.
- Query ranking without reference labels and random/uniform/entropy control policies.
- Optional camera-compensated scene-motion measurements at selected video queries. These are diagnostic; they do not identify the possessor or automatically change labels.
- A regularized two-head logistic calibration experiment with explicit temporal training provenance.
- Self-contained interactive HTML report, valid CSVs, cached diagnostics, and run manifests.

Sampling alternative H/A/D histories is not yet the full physical-history/ball-region formulation in RESEARCH.md. The system does not certify true possession, infer persistent player identities, reconstruct 3D ball paths, or recognize controlled contact visually.

## Training/evaluation separation

The label-free model is produced before reading any reference labels for evaluation. The opt-in calibration fits on frames **11185–14184** (120 seconds), leaves a **250-frame/10-second buffer**, and evaluates frames **14435–18684** (170 seconds). Feature context is at most two seconds, so the buffer exceeds it.

The calibration model's five-minute score includes its training block and is descriptive only. Its held-out score is still a very small, same-match experiment, not independent generalization evidence. No parameter sweep is performed. Calibration remains a separate candidate; the main `blindspot.csv` uses the label-free model.

## First measured results

| Comparison | Label-free v1 | Calibrated candidate | Organizer smooth50 | Organizer smooth125 |
|---|---:|---:|---:|---:|
| Public five-minute spellF1 | 0.3351 | 0.3097, includes training | 0.3256 | 0.4242 |
| Later 170-second spellF1 | 0.4000 | 0.3556 | 0.2923 | 0.3579 |

The calibration candidate did not improve the held-out interval metric and is not promoted. The label-free model beats smooth50 in this public slice but does not beat the strongest public baseline across all five minutes. These observations do not establish private full-match ranking. Camera ablation made no difference to the initial public score; phase tags also did not improve that metric in the first fixed comparison. They remain research features, not demonstrated improvements.

## Output files

All match-derived outputs default to ignored `data/experiments/blindspot-v1/`:

| File | Purpose |
|---|---|
| `blindspot.csv` | Main validated label-free submission candidate |
| `evidence_v1.csv`, ablation CSVs | Reproducible comparisons |
| `calibrated_v1.csv` | Separate supervised development candidate, when requested |
| `calibration_model.json` | Weights, feature names, training interval, label-file hash |
| `scores.json` | Explicit evaluation scopes and aggregate metrics |
| `data_audit.json` | Missing-ball gaps and tracking coverage |
| `gap_histories.json` | Alternative model state sequences across missing-ball intervals |
| `boundary_receipts.json` | Transition evidence and local ambiguity |
| `queries.json`, `query_controls.json` | Label-independent visual-query plan and control policies |
| `visual_measurements.json` | Optional local motion diagnostics |
| `inference.npz` | Model marginals, sampled histories, source disagreement |
| `manifest.json` | Source/input/output hashes, configuration, environment, runtime |
| `report.html` | Interactive timeline, alternative histories, query navigation, scores |

## Next research steps

Focus on the dead-ball estimator and boundary timing, then measure whether better uncertainty/query selection earns its compute. Additional legitimate labels or compatible outside data are needed before making a large supervised model the central approach. Compare fixed budgets and retain failures in the work log. See RESEARCH.md for the frontier experiments.

### CoreOrigin artifact organization

The registered team name is CoreOrigin. The model/project remains BlindSpot. `CoreOrigin.md` is the team-named technical draft; `data/experiments/blindspot-v3/candidate/CoreOrigin.csv` is the full-match development candidate. A portable private handoff folder is `data/submissions/CoreOrigin-v03/`. The earlier filename is retained under the ignored experiment naming-history directory. Changing the filename did not change any prediction bytes.

The immutable v0.3 source ZIP contains 58 source/public-benchmark files, SHA256 `8d73f162950880b8fae4e0270826d5ac15ec8386401f8e274e919751dd3231ec`. It contains no tactical assets, predictions, clips or fitted models. Later verification logs/documentation are maintained separately from that frozen snapshot. Current notebook: https://colab.research.google.com/drive/1U_fNZ-9AK-B-Bzq_-tx_HzurAMgThFNz .


### Current audit revision: v0.3.1

Read AUDIT.md for fixes, scope and two evaluation protocols. Run `python scripts/run_champion.py --config configs/champion_v031.json --out data/experiments/blindspot-v031 --resume`. Build the latest viewer with `python scripts/build_candidate_review.py --experiment data/experiments/blindspot-v031`. The experiment landing page is `data/experiments/index.html`.

`python scripts/audit_revision.py` compares the frozen v3 outputs/models with the revised run and retrains both methods using fixed-time folds. It requires both experiment directories; the source bundle intentionally omits match-derived models and outputs. `python scripts/build_champion_colab.py --revision --bundle-id ID` creates a separate v031 source ZIP/notebook. Use `--reuse-bundle` when inserting an uploaded ZIP ID so the uploaded source checksum remains unchanged.

The Python suite has 33 tests. The generated HTML is tested through a Node DOM test double when Node is installed; video files are independently decoded and counted. Real-browser interaction remains unverified because local browser file access is blocked.


### v0.4 experiment reproduction

Run `python scripts/run_v04_experiments.py` to evaluate all groups. Baseline is checked first and must reproduce 0.4457142857 / 0.6200. Each completed group saves hash-verified fold CSVs, detailed scores and a source/input fingerprint; matching completed groups can be reused. Generate the comparison/error report with `python scripts/report_v04_results.py`. Export the retained candidate with `python scripts/run_v04_experiments.py --candidate baseline`, then run the official validator.

`python scripts/build_v04_colab.py` rebuilds the source/public-benchmark ZIP and matching notebook. Upload that notebook and ZIP together to Colab; source/input checksums are pinned. The notebook now selects a group by the documented margin rule instead of blindly exporting `all`, and downloads the fold evidence as well as the candidate. The 39-test local run and bundle syntax/checksum checks are verified; v0.4 has not been executed again in Colab.


### v0.4.1: team-only contact fusion

New group: `contacts_team_only`. Run `python scripts/run_v04_experiments.py baseline contacts contacts_team_only --out data/experiments/blindspot-v041`, then `python scripts/report_v04_results.py --experiment data/experiments/blindspot-v041`. Both model heads fit the same training mask; the decoder receives baseline dead probability and normalized contact H/A scores separately. Runtime assertions and per-fold model audits enforce unchanged baseline dead/ungraded masks. There are two models (68 and 78 inputs), with 78 unique features.

An experimental export uses `python scripts/run_v04_experiments.py --candidate contacts_team_only --out data/experiments/blindspot-v041`. It is not the selected submission: the measured gains do not meet the existing release heuristic. The frozen v0.3.1 file stays selected. Tests total 46. The relative-output logging error found during export is repaired and regression-tested.

`python scripts/build_v04_colab.py --team-only` produces a separate BlindSpot_v041_project.zip and BlindSpot_v041_Colab.ipynb. Use that matching pair; no new v0.4.1 cloud execution is implied by local tests or packaging.


### Structural-prior verification v0.5

See EXPERIMENTS_v05.md. possession/structure.py adds a fixed observed out-of-pitch probability prior and optional 25-frame contact snapping. The experiment runner exposes baseline_outplay, team_only_outplay and team_only_outplay_snap alongside their controls. Corrected snapping tracks moved neighbouring boundaries and enforces ordered-spell/D/X invariants. No classifier parameters or structural constants were tuned during review.

Run the six-group command in EXPERIMENTS_v05.md with --out data/experiments/blindspot-v05, then scripts/report_v04_results.py --experiment data/experiments/blindspot-v05. This preserves previous runs. All 57 tests and 48 independent fold rescores pass; the 24 baseline/contact/team-only controls reproduce v0.4.1 exactly. The best unsnapped structural variant scores 0.5111 / 0.6341 spell F1 and stays experimental under the existing release rule.

scripts/build_v04_colab.py --structural creates data/drive-transfer/BlindSpot_v05_project.zip and notebooks/BlindSpot_v05_Colab.ipynb with matching source checksum, separate paths, fixed six-group comparison and conservative export selection. CPU is sufficient. Local notebook syntax and package checks do not establish cloud execution; v0.5 has not been cloud-run. Match-derived result CSVs and logs remain in the separate local evidence/handoff archives. The numerical source's original label-independence wording is qualified by the pre-score public-label inspection documented in EXPERIMENTS_v05.md.


### v0.6 fixed architecture comparison

The runner now includes v06_gated, v06_liveteam, v06_bagged, v06_accel and v06_combo. Their implementation is possession/v06.py; tests/test_v06.py verifies classifier parity, gates, symmetry and held-out-label exclusion. Individual sampling-offset probabilities are cached and averaged. Acceleration and contacts together use 94 unique inputs across their separate heads. EXPERIMENTS_v06.md lists all commands and measured outcomes.

Build the matching v06 Colab source/notebook with scripts/build_v04_colab.py --advanced. Validate saved fold evidence with scripts/verify_v06_results.py. scripts/prepare_blind_validation.py creates a prediction-blind six-minute annotation plan; it creates no labels. The acceleration candidate remains experimental because fixed-time spell F1 does not improve over baseline. Cloud execution and pretrained PathCRF transfer remain unperformed.
