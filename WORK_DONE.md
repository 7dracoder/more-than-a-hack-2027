# Work done and current project state

Last updated: October 24, 2026 (America/New_York). The current rules correction below supersedes historical permission/candidate statements.

This is the factual work log for the project. [RESEARCH.md](RESEARCH.md) contains researched options and the proposed implementation roadmap. An item is complete only when its output or result has been checked.

## Maintenance instructions

For every subsequent substantive project task:

1. Read this log and the relevant part of RESEARCH.md before making changes.
2. Add what changed, why, the files/artifacts involved, and the verification actually performed.
3. Separate **completed**, **in progress**, **planned**, and **blocked** work. Never describe a proposal as implemented or an attempted upload as complete.
4. Record experiment configuration and actual measured results, including regressions and limitations.
5. Update remaining tasks and unresolved issues. Preserve earlier results and useful decisions rather than replacing them with a success-only summary.
6. Keep restricted match assets, reconstructable extracts, caches, generated predictions, and credentials out of tracked documentation and source control. Aggregate observations and public-source links are sufficient here.

These instructions define how to maintain the log during work; this file is not a background automation.

## Current state at a glance

**October 24 correction:** the organizer clarification pulled in `0215d48` prohibits using BAR-ATH public labels for training/tuning and forbids hand labels on this match even for validation. Historical v0.3–v0.6 supervised exports are not current eligible candidates. The six-minute hand-annotation plan is disabled. Local external-supervision work is described in LOCAL_V07.md; its completion and target improvement must be established by actual saved evidence.

| Area | Status | Evidence / limitation |
|---|---|---|
| Project understanding | Complete | Documentation, scripts, metrics, and all three starter notebooks reviewed |
| Local dataset organization | Complete | All nine expected assets present under the canonical source folders |
| Tracking and homography alignment | Complete for existing checks | Tactical 199,319 records; broadcast 164,488 records |
| Smart Tagging checks | Complete | 638 intervals; both kickoff checks pass |
| Video checks | Sampled validation complete | Both decode at four sampled positions; full decode not performed |
| Baseline reproduction | Complete | All four regenerated CSVs match committed originals byte for byte |
| Local scoring | Complete | Public scores reproduced with unchanged official scorer |
| Drive upload to requested account | File-presence readback complete | Nine canonical asset names visible with expected rounded sizes; no cloud content-hash comparison |
| Public folder sharing | Complete | Anyone with the link, Viewer; owner is requested account |
| Initial research and roadmap | Complete | Primary-source findings and prioritized proposals in RESEARCH.md |
| Colab execution | Verified complete | All eight cells executed; tests and inference succeeded; every prediction CSV hash matches local output |
| Frontier research extension | Research brief and small probe complete | Competing-history possession proposal; camera-framing signal measured; full method not implemented |
| Custom possession algorithm | First implementation complete locally | Label-free model, exact sequence inference, alternative state histories, evidence/query diagnostics; see IMPLEMENTATION.md |
| New model training | Small calibration experiment executed | Early 120-second training block; later 170-second evaluation with 10-second buffer; candidate not promoted |
| Final competition submission | Not selected or submitted | No hidden-match score or winning-performance claim |
| v0.2 follow-up | Implemented and locally executed | Whole-run split, quality audit, query benchmark, resumable stages, twelve decoded review clips and technical-report draft |
| v0.3 candidate | Locally trained and verified | Trajectory measurement audit, symmetric classifier, four purged development folds, provisional CSV and updated twelve-clip review |

## Completed work: October 20, 2026

### 1. Reviewed the project and submission contract

Read README.md, OVERVIEW.md, CHALLENGE.md, DATA.md, EVALUATION.md, FAQ.md, RULES.md, SUBMISSION.md, LICENSE, CITATION.cff, dependencies, source helpers, the evaluation implementation, and all three starter notebooks.

Established that Phase 1 predicts Barcelona possession (H), Athletic possession (A), dead ball (D), or abstention (X) for each of 199,319 tactical frames. Ranking uses spellF1; macroF1 is the first tie-break. Only five minutes of truth are public. The private full-match evaluation cannot be reproduced locally.

Important interpretation recorded: delivered ATD uses `frame_id - 1`; the large per-half counters in games.py are not a correction to apply to those files. Controlled possession is different from simple proximity, and missing ball is not evidence that play has stopped.

### 2. Inspected and organized the downloads

The first 1,378-byte ZIP contained only a README. Later downloads supplied eight completed ZIPs containing the seven non-video assets and another README; the video files initially appeared as `.crdownload` downloads.

Extracted the actual assets into the canonical structure expected by `games.asset_paths()`. Preserved all nine original ZIPs in `data/archives/`; preserved the bundle README separately. Did not overwrite the repository's data/README.md. No completed dataset asset was intentionally deleted.

```text
data/
  tactical/
    BAR-ATH_tactical.mp4
    BAR-ATH_tactical_atd.xml
    BAR-ATH_tactical_atd.txt
    BAR-ATH_tactical_homography.csv
    BAR-ATH_tactical_smart_tagging.xml
  broadcast/
    BAR-ATH_broadcast.mp4
    BAR-ATH_broadcast_atd.xml
    BAR-ATH_broadcast_atd.txt
    BAR-ATH_broadcast_homography.csv
  archives/                    original downloads and bundle README
  dataset-status.json          local validation report
```

Moved the initial evaluation outputs from `data/project-review-evaluation/` to `evaluation/reports/project-review/`. Confirmed dataset assets and those generated reports are ignored by Git.

### 3. Validated local metadata and alignment

| Check | Tactical | Broadcast |
|---|---:|---:|
| Tracking records | 199,319 | 164,488 |
| Tracking frame range | 1..199319 | 1..164488 |
| Existing tracking alignment check | Pass | Pass |
| Track slots in metadata | 46 | 51 |
| Teams in metadata | 2 | 2 |
| Homography rows | 199,319 | 164,488 |
| Usable homographies | 199,319 | 108,248 |
| Existing homography index check | Pass | Pass |

Both metadata XML files parse successfully. Smart Tagging contains 638 intervals; the helper confirms that both known kickoffs lie within their tagged windows.

These checks establish the properties tested by the existing helpers; they do not certify tracking accuracy, identity continuity, or every numerical observation.

### 4. Organized and sampled both videos

After downloads completed, renamed/moved:

- `data/BAR-ATH_tactical-001.mp4` to `data/tactical/BAR-ATH_tactical.mp4`.
- `data/BAR-ATH_broadcast-002.mp4` to `data/broadcast/BAR-ATH_broadcast.mp4`.

Used installed PyAV to inspect metadata and decode at four sampled positions per video, including points near the beginning and end. No new video package was installed for this check.

| Property | Tactical | Broadcast |
|---|---:|---:|
| Local bytes | 5,222,307,334 | 13,962,360,320 |
| Codec | H.264 | H.264 |
| Resolution | 1920 x 1080 | 1920 x 1080 |
| FPS | 25 | 25 |
| Container-reported frames | 199,319 | 164,487 |
| Duration, seconds | 7,972.76 | 6,579.48 |
| Sampled decode checks | 4 successful | 4 successful |

**Unresolved:** broadcast reports one fewer video frame than its tracking and homography. No arbitrary padding, trimming, or frame remapping was applied. Full video decoding remains unperformed.

### 5. Ran evaluation and reproduced organizer baselines

The installed Python 3.11 environment runs the scorer successfully. Initially scored the four supplied CSVs and validated baseline_smooth50.csv. Subsequently regenerated all four from the actual downloaded tracking using:

```powershell
python starter/baseline.py --out data/baseline-reproduction
python evaluation/run_evaluation.py --results data/baseline-reproduction --out data/baseline-reproduction/evaluation --no-detail
```

All regenerated files pass `possession_metrics.validate_submission(..., 199319)`, match the supplied rows, and match the supplied bytes. Original committed predictions were not overwritten.

| Baseline | Public macroF1 | Public spellF1 | Public predicted possessions |
|---|---:|---:|---:|
| baseline_raw | 0.5972 | 0.0378 | 131 |
| baseline_smooth25 | 0.6200 | 0.2868 | 36 |
| baseline_smooth50 | 0.6207 | 0.3256 | 26 |
| baseline_smooth125 | 0.5812 | 0.4242 | 16 |

Public truth has 7,500 graded frames and 17 possessions. These results reproduce the repository, not a new algorithmic improvement. The full-match floor of 0.2433 spellF1 is organizer-reported, not independently verified here.

Reproduction output hashes (SHA-256):

```text
baseline_raw        2a639a6b3c7816ab619d543cc870b021dca6945980eae8978325be0cc311fd63
baseline_smooth25   57330f8bc13846751ceaf745f300f8bdb33cb92849b708d5350724eb7f843532
baseline_smooth50   d444caa1aebb6fb58386033a5967c83302b8dd984ea91c86381bf208332702ff
baseline_smooth125  9b81f2027bd19224d900bdd7529de3fac078ce21160b5228201bbe86f94b7991
```

The baseline produced usable nearest-player answers on 113,488 frames (56.9%). This is consistent with the documented overall ball coverage, but the baseline's count also requires a usable player observation.

### 6. Organized the Drive copy

The initial connector was authenticated to a different account. Created a partial organized copy there before the user requested the other account. Connector transfers exposed a 100 MiB per-file limit; compressed copies of the tracking files were uploaded to that earlier partial copy. It has not been deleted.

The user stated that organizer permission for sharing was available. Then used the browser, signed in to the requested account, to upload the canonical tactical and broadcast folders directly, including both videos and uncompressed tracking files. Uploaded a setup README and local validation report at the folder root.

**Current destination:** [More Than A Hack 2027 - Organized Dataset](https://drive.google.com/drive/folders/1yUucdtdZNGKQWwF9rq8j5v56MlxzOju7).

Sharing was saved and visibly confirmed as **Anyone with the link - Viewer**, with ownership by the account requested by the user. The local sharing screenshot is `data/drive-sharing-proof.png`.

During this documentation task, reopened Drive and verified all five tactical asset names and all four broadcast asset names, including both videos. Drive displayed their expected rounded sizes and ownership. Root README and dataset-status.json were also visible. This supersedes the earlier status saying the videos were still uploading.

Cloud file contents were not downloaded again or compared by checksum. The earlier partial copy on the first account remains separate; no ownership transfer or deletion was performed.

### 7. Researched candidate approaches and documented a roadmap

Checked primary sources on control-aware possession, player-only live/dead detection, rule-based event pipelines, set-based ball inference, temporal convolution, and structured possession-path decoding. Read the applicable publisher/arXiv material and the PathCRF authors' repository entry point. The TCN review was limited to its accessible publisher abstract; no external implementation or weights have been installed or validated.

RESEARCH.md now documents project-specific hypotheses, a prioritized component matrix, a first architecture, feature candidates, missingness handling, Smart Tagging interpretation, experiments, evaluation safeguards, possible artifacts, and open questions.

The initial recommendation is to begin with interpretable control and dead-ball evidence plus temporal decoding. No custom possession model was implemented or trained during that initial research/documentation task.

### 8. Developed a more ambitious research direction

Following the request for a more original idea, expanded the literature search to ball-free possession, multiple hypotheses, physical uncertainty, camera framing, and boundary-aware evidence acquisition. Checked PathCRF, the primary arXiv abstracts for boundary-centric active learning and boundary voting, a primary football availability paper, and the accessible primary proceedings excerpt on camera-assisted ball tracking. The search is not exhaustive and does not establish a first-ever invention.

Added RESEARCH.md section 11: a proposal to retain competing explanations of missing-ball stretches, identify model-relative agreement about possession, and acquire short algorithmic video measurements where disagreement affects boundaries. It includes closest-prior-work comparisons, limitations, a prototype scope, equal-compute controls, falsification criteria, and a cautious originality statement.

The earlier interpretable state model remains a reference implementation to compare against. The frontier system, visual-query policy, uncertainty calibration, and hypothesis search are **not implemented**.

### 9. Ran an exploratory camera-framing probe

Added `scripts/research_camera_probe.py` and executed it successfully:

```powershell
python scripts/research_camera_probe.py
```

The script inverse-projects the image center through tactical homography and fits a small ordinary least-squares mapping to provided ball coordinates. It fits on 47,106 finite bounded first-half observations and evaluates 46,942 second-half observations. It reads no possession truth and installs no dependency.

Measured second-half x association: correlation **0.8379**; y association: **0.1454**. Mean 2D ball-location error was **24.25 m**, compared with **31.28 m** for a constant training-median location and **38.89 m** with the camera signal shifted by 60 seconds. The latter is a diagnostic control, not a rigorous independence test.

Saved aggregate output to ignored `data/research/camera-probe.json`. These results support exploring a weak framing cue, not accurate ball reconstruction or improved possession prediction. Evaluation was on noisy observed-ball frames, not natural gaps or independent matches. No new possession score or full-match improvement is claimed.

### 10. Researched a practical cloud compute plan

Checked official Colab overview/FAQ and Kaggle notebook documentation after the user expressed concern about local hardware. Added RESEARCH.md section 12 recommending Colab plus the existing Drive dataset, beginning with CPU/no-training inference and using a GPU only when the selected experiment needs one. Recorded checkpointing, local runtime caching, variable resource availability, and the shortage of independent possession supervision.

No paid service was purchased, cloud training job launched, or Colab notebook created. The approximately 16 GB GPU suggestion is a starting engineering target for a compact model, not a profiled requirement. The local baseline and research probe already run successfully without a GPU.

### 11. Implemented the first complete research pipeline

Added `possession/` with streamed, fingerprinted feature extraction; reviewed tag evidence; interpretable H/A/D potentials; independent per-half Viterbi and forward/backward inference; deterministic posterior history sampling; source ablations; gap-history records; boundary ambiguity records; and label-independent visual-query selection with control policies. Added `configs/evidence_v1.json` and `scripts/run_experiment.py` to reproduce the run.

Added optional camera-compensated optical-flow scene-motion diagnostics in `possession/visual.py`. Four selected queries were measured successfully. These are not a trained ball/contact recognizer, do not establish team ownership, and are not injected into the final prediction.

Added a small regularized logistic calibration experiment. It fits only frames 11185–14184; frames 14185–14434 form a buffer; frames 14435–18684 are the later same-match evaluation block. The label-free model is produced before ground-truth evaluation. A regression test confirms that changing labels outside the training interval cannot change the fitted model.

Initial measured public spellF1: label-free 0.3351, without tags 0.3487, without camera 0.3351, lower switch cost 0.2927, higher switch cost 0.3086. These are fixed ablations, not a tuned winner search. Organizer smooth50 scores 0.3256 and smooth125 scores 0.4242 on the full public window.

On the later 170-second block, label-free spellF1 is 0.4000, calibration 0.3556, organizer smooth50 0.2923, and organizer smooth125 0.3579. Calibration's five-minute score is 0.3097 and includes training frames; it is not held-out evidence. Kept calibration as a separate candidate, with the main CSV remaining label-free. Public-only differences are exploratory; no hidden-match performance claim is supported.

Added eight mathematical/invariance tests. They pass locally, including exhaustive decoder/posterior comparison, repeated sampling, slot-order invariance, team symmetry, halftime reset, zero query budgets, missingness handling, and calibration training-label isolation. The first complete local run wrote validated full-length CSVs, metrics, model weights, query plans, visual measurements, and an interactive HTML report under ignored `data/experiments/blindspot-v1/`.

Added IMPLEMENTATION.md with commands, component scope, actual results, limitations, and output descriptions. The physical ball-region hypothesis search, calibrated correctness guarantees, trained visual contact recognition, and a demonstrated query-efficiency advantage remain unimplemented research goals.

### 12. Prepared Colab execution artifacts

Inspected the user's open Colab page and confirmed the requested Google account. Created a dedicated notebook shell. The official Colab CLI documents Windows as unsupported, so used the supported browser workflow instead of installing it.

Added `scripts/build_colab.py` and generated `notebooks/BlindSpot_Colab.ipynb` plus an ignored source/public-benchmark ZIP. The bundle excludes match inputs, private ground truth, credentials, and generated match predictions. The notebook verifies source and input hashes, downloads only four tactical non-video assets, runs tests and the experiment, validates the CSV, displays the report, and offers result archive download. Default execution is CPU; no paid GPU is requested.

Colab execution verification was initially pending; the completion/readback below supersedes that state.

### 13. Verified the implementation in Colab and the browser

Uploaded the checksum-pinned source-only ZIP to the requested Drive account's `project/` folder and opened [BlindSpot_Colab.ipynb](https://colab.research.google.com/drive/1w6y-eYSDECPtd_Qe-aRMkNW_A6-_80oK). Ran all eight code cells on an ordinary CPU runtime. Source and all four tactical non-video asset checksums passed; the tests returned success; the full ablation/calibration run and CSV validator completed with return code zero; the interactive report rendered; and the result-archive download cell executed.

The source bundle is a versioned execution snapshot at [project/](https://drive.google.com/drive/folders/1odV1-SQAg2OcbVBi6A0TS5G_zyG5IEtf), and contains code plus public benchmark files, not the raw match dataset or credentials. The running notebook is owned by the requested account; no notebook-publication action was performed. Its bootstrap uses existing public file access and does not mount or grant new access to unrelated Drive files.

Read the cloud manifest from the rendered report and saved it locally as `data/experiments/blindspot-v1/colab-manifest.json`. The cloud run used Python 3.13.16, NumPy 2.2.6, and pandas 2.2.3 and took 59.568 seconds. The local run used Python 3.11.1, NumPy 2.4.3, and pandas 3.0.1. All seven prediction-output hashes, including the main CSV, match exactly across environments. Main CSV SHA-256: `0128b44306c2c4d51fd7afaafb943d82ed88159558da2552b21ef262e114605b`.

Verified report navigation in a browser: Next ambiguous moment moved the timeline to the selected query interval, and the score tables and provenance rendered correctly. Saved `report-preview.png` and `colab-proof.png` with the local outputs. The localhost preview required sandbox-approved localhost network access; no public preview server was deployed. No purchase or GPU allocation was made.

The optional trained calibration is a real, completed CPU training experiment; it did not improve the held-out interval score. The functioning infrastructure and reproducibility are verified. Exceeding the best five-minute baseline or winning the private evaluation remains unproven.

Verified the Drive project-folder handoff: the source ZIP, source-only notebook copy, IMPLEMENTATION.md, RESEARCH.md, and WORK_DONE.md showed completed uploads. Packaged the local run, screenshots, and cloud-manifest evidence into ignored `data/experiments/BlindSpot_results_local.zip`. Stopped both temporary localhost preview servers after browser verification.

## Decisions and constraints to carry forward

- Use tactical video frames as the reference throughout.
- Do not make broadcast fusion a prerequisite for the first improved solution.
- Preserve uncertainty and missingness instead of converting absent observations into D or zero coordinates.
- Use team-level features that tolerate track fragmentation; do not assume stable player identities.
- Evaluate possession structure and boundary timing, not just frame-level accuracy.
- Do not tune a large search against 17 public possessions or claim hidden-match performance.
- Keep the official scorer and public reference labels unchanged.
- Record permission scope and follow competition data-use terms; the user's sharing authorization is not a general license for unrelated redistribution.

### 14. Implemented the v0.2 audit/evaluation/review workflow

Added `possession/quality.py`, `validation.py`, `stoppage.py`, `review.py`, `scripts/run_followup.py`, eight additional tests, a separate Colab notebook builder and `TECHNICAL_REPORT.md`. The original v0.1 prediction remains byte-identical. Fixed query budgeting to distribute a budget over the actual number of eligible blocks, including a single public evaluation window.

**Quality audit:** scanned all 199,319 tactical and 164,488 broadcast tracking rows and homography rows. No malformed/nonsequential raw rows or singular/nonfinite matrices flagged valid were found. Flagged 31 tactical and 623 broadcast slot discontinuities above 3 m per frame, and three tactical observed-ball jumps above 5 m. These are diagnostic flags, not verified tracking errors. Found 16,033 tactical frames with simultaneous opposing phase support, plus 36 tag instances without team attribution. No automatic track repairs were applied.

Scanned all compressed video packets: tactical has 199,319 and broadcast 164,487, matching their advertised video counts. Broadcast therefore genuinely has one fewer timestamped packet than tracking rows; the cause/cross-camera mapping is still unresolved. Full pixel decoding was not performed and broadcast fusion remains disabled.

**Leakage safeguards:** training/evaluation sets use entire H/A/D runs. The development split contains 2,600 training frames and 3,839 evaluation frames, separated by 1,061 excluded frames (at least the requested 250-frame embargo). This already-inspected public footage is development evidence, not an untouched test. Model fitting ignores all labels excluded by the training mask; tests verify this.

| Grouped development comparison | Spell F1 | Dead-ball F1 |
|---|---:|---:|
| Original evidence v1 | 0.3200 | 0.0000 |
| Fixed multisecond stoppage candidate | 0.3200 | 0.0000 |
| Separate grouped calibration | 0.2750 | 0.3458 |
| Organizer smooth25 | 0.3111 | 0.3307 |
| Organizer smooth125 | 0.2667 | 0.1603 |

The stoppage candidate changes full-match outputs but offers no public-window improvement; it is not promoted. The trained candidate improves dead-ball detection while reducing the ranking metric; it is not promoted either. Full-public evidence v1 remains 0.3351 versus smooth125's 0.4242. Scores and development scopes are shown together rather than reusing the earlier 170-second result as if it were the new grouped split.

**Query benchmark:** twelve 51-frame clips per policy, 24.48 seconds total, selected without labels. Boundary/source ranking finds clips with 47.22% erroneous frames versus entropy 44.61%, uniform 18.79%, and 33.17% mean over 30 random seeds. Random 5th–95th percentile range is 16.13%–49.29%; superiority is not established. Boundary ranking covers 11.90% of public errors and 4/20 true transitions. This measures error discovery, not correction, improved predictions, independent generalization or human time saved.

**Review demo:** generated twelve private eight-second tactical snippets, checked all 2,412 output frames by decoding, and paired them with 492 pitch snapshots at 5 Hz. `data/experiments/blindspot-v2/review/index.html` synchronizes playback, pitch slots, observed ball and model support. Browser-local notes can be exported; no notes feed back into predictions. Independent human review has not been completed. JavaScript syntax and payload/frame alignment checks pass. Direct browser navigation to the local file was blocked by the browser URL policy; no workaround was attempted. The cloud experiment dashboard rendered successfully in Colab; local video interaction remains unverified in the browser.

**Recovery:** each stage gets a completion marker only after its artifacts are hashed. Resume verifies source/config/input provenance and artifact hashes, recomputing corrupt or incomplete stages. Tests include interrupted stages, corrupt artifacts, truncated markers, leftover partial markers and changed source. The final source also avoids using fresh download modification times for tactical non-video inputs, allowing identical content to resume across Colab runtime resets when results are persisted. Default notebook storage remains temporary; optional Drive mounting is explicit and has not been authorized/executed in this run.

Sixteen tests pass locally and in the final [Colab notebook](https://colab.research.google.com/drive/1jwU7f82AWzfRVmFEnpZ3-HgsdSOvayWo). All nine cells completed: verified source/input downloads, tests, the tracking-only pipeline, CSV validation, grouped score/hash readback, dashboard rendering and archive download. Both label-free CSV hashes match local output exactly. The final local pipeline also completed, followed by a successful resume reusing all five verified stages. Source ZIP SHA256 is `bffffe793ec3b4a609fd0cc4c08ec6fafbeaa06b076e5dfdf14325125c636216`; the original v0.1 bundle remains unchanged. The final source ZIP contains 47 allowlisted source/public-reference files and no match inputs, videos or generated outputs.

Saved cloud readback in ignored `colab-verification.json` and screenshot `colab-v02-proof.png`; packaged local artifacts in `data/experiments/BlindSpot_v02_results.zip`. Match-derived outputs and clips remain ignored under `data/`; only the allowlisted source/public-benchmark bundle and documentation are handed to the existing project folder. No GPU was purchased, private truth accessed, or final submission made. Archive-relative links in Colab's HTML preview are for the downloaded/extracted result folder; the default cloud package has no video. The original intermediate v0.2 notebook was retained, and the final URL above is the verified handoff.

### 15. Built and evaluated the v0.3 submission candidate

Added `possession/metric_features.py`, `possession/champion.py`, `configs/champion_v3.json`, `scripts/run_champion.py`, a separate v0.3 Colab builder, pinned CPU/video requirements and `scripts/build_candidate_review.py`. The current submission-oriented technical draft is `SUBMISSION_TECHNICAL.md`; earlier documents/results are retained. Official scoring and reference labels remain unchanged. No broad parameter search was performed.

**Important correction to the initial speed interpretation:** metadata declares the speed channel as kph, so a pilot divided by 3.6 before filtering. That pilot improved the public score but the physical interpretation was wrong: displacement checks show that the values empirically resemble m/s. Three gameplay windows have median raw/displacement ratios about 0.995 with correlations 0.939–0.964. The final sampled audit uses 100,880 pairs: median ratio 0.99698, correlation 0.96253, median absolute error 0.113 m/s treating raw values as m/s versus 1.145 after the declared kph conversion. Do not repeat the earlier claim that v1 definitely confused kph and m/s. The confirmed issue is inconsistent metadata and the need to validate measurement semantics.

The metadata-scaling pilot (including its source, predictions and scores) was preserved under ignored `data/experiments/blindspot-v3-metadata-pilot/`. Its pooled classifier score 0.4914 is **not** the trajectory-based result and is not used as the current headline. The final pipeline derives speed from six-frame position displacement, rejects missing/jumping slot windows, resets across halves and keeps physical units explicit. The conservative maximum feature context radius is 130 frames; training embargo is 300 frames. Increasing the earlier 250-frame guard to 300 changed none of the four training masks because whole neighbouring runs were already excluded.

Also fixed an unreachable restart-ownership branch by using a separate new event parser: team restart ownership stays weak and separate from whether play is live. Added tests for unit conversion order, unknown units, trajectory continuity, restart ownership/N/A, complete-run purging, excluded-label isolation, probability team symmetry, hierarchical live/team decoding and conservative kickoff metadata. All 26 tests pass locally.

**Fixed methods evaluated:** trajectory-feature legacy architecture; two-component unlabelled movement-regime inference; and an 80-iteration symmetric histogram gradient boosting classifier with factorized live/dead and team decoding. Four contiguous folds retain entire H/A/D runs and exclude neighbouring training runs. Models decode unlabelled halves without reference-label clamping at fold boundaries. Per-IoU match counts are pooled across folds; a stitched cross-fitted sequence is not scored.

| Same four-fold development scope | Spell F1 | Macro F1 | Dead-ball F1 |
|---|---:|---:|---:|
| Cross-fitted symmetric classifier | 0.4457 | 0.7484 | 0.6619 |
| Unlabelled movement-regime model | 0.3524 | 0.7343 | 0.6733 |
| Historical v1 | 0.3200 | 0.4986 | 0.0000 |
| Organizer smooth125 | 0.3889 | 0.5812 | 0.1464 |
| Organizer smooth50 | 0.3043 | 0.6207 | 0.3225 |

Classifier fold spell F1 is 0.4000, 1.0000, 0.2400 and 0.3556, versus smooth125's 0.3636, 0.8667, 0.1333 and 0.3600. Omitting any one evaluation block retains a positive pooled difference (+0.0377 to +0.0769). This is sensitivity analysis, not a statistical guarantee: the sample is tiny, previously inspected and from one match, and training sets overlap. Unsegmented full-public label-free motion spell F1 is 0.3610 versus smooth125 0.4242; trajectory-feature v1 alone remains 0.3351. Do not compare pooled fold scores with unsegmented sample scores as if they shared a scope.

The provisional exported classifier fits **all public labels** after cross-fitting. Its training-window performance is stored separately as descriptive only. Candidate `data/experiments/blindspot-v3/candidate/CoreOrigin.csv` passes the official 199,319-row validator, SHA256 `17721e1c3c58f0d211cb46ed0e2893d7aaa589b452ac0da00df909e7f758d16d`. The user confirmed the registered team name **CoreOrigin**. BlindSpot remains the project/model name; the candidate CSV is CoreOrigin.csv. Candidate selection/final submission remains provisional; no private truth was accessed or competition form submitted.

Added coverage audit, 1%–99% training-feature envelope flags, related-model disagreement, fixed artificial-gap stress test and leave-one-evaluation-block sensitivity. These are diagnostics, not calibrated confidence or natural-gap accuracy. A conservative public kickoff-anchor prior marks 2,908 second-half pre-kickoff frames dead, stopping one second short of the observed anchor. Independent video control/contact recognition is not claimed.

Built an updated private review lab for the new candidate: twelve clips, 2,412 decoded/verified frames and 492 synchronized pitch snapshots. Queries use model disagreement/uncertainty rather than known error labels. The local candidate viewer has not been browser-interaction tested because the earlier file-URL policy block remains in force. The local pipeline rerun reused all five hash-verified stages. Final Colab verification and the source-package handoff are complete; see section 16. Source has not been committed/pushed by this run.

## Remaining work

### Next implementation steps

- [x] Extend the implemented gap/coverage audit with suspicious jumps, slot discontinuities, and tag conflicts.
- [x] Implement initial competing H/A/D state histories and model-relative ambiguity; full physical ball-region histories remain planned.
- [ ] Evaluate camera/visibility evidence on separated clips and naturally missing-ball intervals before treating it as useful to possession.
- [x] Build a canonical cached frame representation with missing-value masks and metadata.
- [x] Implement the first interpretable geometry/control-evidence model with sequence decoding.
- [x] Implement live/dead evidence and the separately trained calibration comparison; improvement remains needed.
- [x] Map offensive/defensive/restart Smart Tagging codes explicitly and preserve N/A.
- [ ] Compare hysteresis with a small temporal decoder and run ablations.
- [x] Add transition/query plans and local scene-motion diagnostics; full video overlays/contact recognition remain planned.
- [ ] Decide whether further annotations, learned models, or local video recovery are justified.
- [x] Compare label-independent review queries against fixed-budget controls; retain uncertainty about significance.
- [x] Build the synchronized local video/pitch review lab and exportable development-review notes.
- [ ] Complete an independent reviewer study/additional permitted validation labels; measure correction or time saved only after actual reviews.
- [x] Add a trajectory measurement consistency audit and symmetric supervised candidate with purged cross-fitting.
- [x] Prepare a reproducible provisional candidate CSV, fold/coverage diagnostics and a current submission technical draft.

### Unresolved issues and optional verification

- [ ] Explain broadcast's confirmed one-packet/row discrepancy and establish camera synchronization before fusion.
- [ ] Full video decode/integrity scan, only if needed; sampled checks are the current evidence.
- [x] Verify Colab source/input checksums and exact prediction CSV hash parity with the local run.
- [ ] Decide whether the earlier partial Drive copy should be retained; no deletion has been requested.

### Submission release work

- [ ] Select and freeze a defensible algorithm/configuration.
- [x] Generate and validate the 199,319-row CoreOrigin.csv development candidate; final release selection remains pending.
- [x] Pin Colab dependencies and document one-command reproduction for v0.1; freeze the final submission configuration later.
- [x] Prepare a technical report draft; finalize it only after model selection and reproduction freeze.
- [ ] Verify reproduction and final artifact hashes before submission.

The repository lists November 17, 2026 as the deadline. Confirm the organizer's current submission instructions before final submission.

### 16. CoreOrigin naming and clean-runtime verification

Registered team name confirmed by the user: **CoreOrigin**. Updated the candidate export, review loader, notebook validator and reproduction commands to CoreOrigin.csv. The earlier filename is retained locally under naming-history; the prediction bytes are unchanged. The initial clean Colab test run exposed a test fixture that depended on an existing data/test-tmp directory. Tests now create their own temporary parent independently, and notebook subprocess cells print captured diagnostics before raising. All 26 local tests pass after the repair. The corrected source package was then run in a fresh Colab CPU session: all nine code cells completed, all 26 tests passed, and all 11 prediction/risk CSV hashes exactly match the local run. Source and all four tactical input checksums were verified. No GPU purchase or Drive mount was required.

The verified notebook is https://colab.research.google.com/drive/1U_fNZ-9AK-B-Bzq_-tx_HzurAMgThFNz . The frozen 58-file source/public-benchmark ZIP has SHA256 `8d73f162950880b8fae4e0270826d5ac15ec8386401f8e274e919751dd3231ec`; later verification logs/documentation are updated separately, leaving that snapshot intact. The source package, notebook and current technical/reproduction documents were handed off in the existing Drive project folder under tanmaysahu392001@gmail.com. Restricted predictions, models and review clips remain local.

Organized `data/submissions/CoreOrigin-v03/` with CoreOrigin.csv, CoreOrigin.md, the frozen source ZIP, notebook, handoff README and hash manifest. Refreshed the twelve-clipped candidate review; all clips decode to the expected 201 frames, totaling 2,412 verified frames and 492 pitch snapshots. Embedded viewer JavaScript passes Node syntax checking; browser interaction with local file URLs remains unverified because of the existing policy block. Checkpoint resume again reused all five stages after hash verification. Independent validation, final release selection, source publication and organizer submission remain unfinished.


### 17. Audit revision v0.3.1 (21 October 2026)

Resumed the audit and completed the fixes documented in AUDIT.md: isolate live ownership across dead-ball intervals; add motion observation masks; validate feature caches against source and content hashes; select review queries from the exported sequence; autosave and isolate review notes by candidate; exact-frame labels/stepping, sampled-pitch timing disclosure and stale callback protection; correct sampled overview and unsupported ranking language.

All 33 local tests passed, including Node checks of generated viewer logic. The whole-run classifier spell F1 is unchanged at 0.4457. A separately retrained equal-duration fold audit scores v3 0.6100, revision 0.6200, and smooth125 0.4513. Dead F1 slightly decreases from 0.5974 to 0.5945; this is a mixed development result, not a blanket improvement. Both protocols reuse the same inspected public example. The four frozen v3 fold outputs reproduce byte-for-byte. The revised full-match CSV changes 915 frames; unknown private labels prevent assessing those changes.

v0.3.1 has its own experiment directory, source bundle and notebook; original v0.3 artifacts remain immutable. Main pipeline was rerun after source fixes. Current viewer is data/experiments/blindspot-v031/review/index.html; the v2 viewer is historical. Browser interaction with local files remains unverified under the existing policy block. Cloud verification and final handoff are recorded separately after the new source snapshot is frozen; the old v0.3 cloud success does not verify v0.3.1.


The separate v0.3.1 source snapshot contains 64 source/public benchmark files, SHA256 `eba9fefb39422fa66b2a6e209ccdf6ebfc2982f9153a73b4c449a1a998ef49d0`. It was uploaded to the existing project folder under tanmaysahu392001@gmail.com. A fresh Colab CPU runtime completed all nine cells and all 33 tests; source and all four asset checksums passed, and all 11 prediction/risk CSV hashes exactly match local outputs. Verified notebook: https://colab.research.google.com/drive/1wDWukgnGF0N7u7ElmLmnSP4dolMgRKks . The additional fixed-time audit and video review were verified locally, not rerun in that notebook.

All five local pipeline stages passed checkpoint reuse verification. The candidate has 199,319 rows and SHA256 `c572620d15f0c60d9cf14df1b073131e91cb0153dd1c892ede0bd0fd5c8fc46e`. The twelve new clips decode to 2,412 expected frames; 492 pitch snapshots and all per-frame model labels/support are packaged. Local report links and manifest source/output hashes were checked. A version landing page, detailed audit HTML and private CoreOrigin-v031 handoff were prepared. Updated notebook and documentation were uploaded; restricted derived outputs remain local. The source ZIP remains immutable; later verification records are maintained separately. Independent validation, final model selection, reviewed repository publication and organizer submission remain open.


### 18. User-added v0.4 features: full experiment run (21 October 2026)

Inspected the user changes: custom swap callbacks in the classifier, event_features.py, the seven-group experiment runner, and the Colab builder/notebook. Added six invariant tests (39 total passing), automatic baseline failure checks, metadata/pitch-aware feature cache checks, per-fold score/prediction/mask records, source/input fingerprints, verified completed-group reuse, and a score/error report. Fixed the notebook's unconditional `all` export; it now applies the documented conservative selection rule. The notebook result archive now includes fold evidence and provenance.

Ran all seven groups with the existing fixed classifier parameters, both four-fold protocols and the 300-frame training-label embargo (56 fold fits). All eight baseline fold CSVs match the frozen v0.3.1 reference byte-for-byte.

| Group | Features | Whole-run spell F1 | Fixed-time spell F1 |
|---|---:|---:|---:|
| baseline | 68 | 0.4457 | 0.6200 |
| contacts | 78 | 0.5600 | 0.6244 |
| asym | 80 | 0.4667 | 0.6564 |
| gap_out | 77 | 0.4324 | 0.6462 |
| ballmotion | 73 | 0.4324 | 0.6200 |
| shape | 88 | 0.3172 | 0.5486 |
| all | 124 | 0.3515 | 0.6286 |

No group clears the pre-existing conservative rule of a gain greater than 0.06 in BOTH protocols; this rule is a heuristic, not statistical significance. The provisional export therefore retains baseline. Contacts improves whole-run spell F1 by 0.1143 but fixed-time spell F1 by only 0.0044. Past/future windows improve both slightly; shape and combining everything degrade whole-run performance. Failed groups are retained in the report.

Regenerated and officially validated the 199,319-row baseline candidate at data/experiments/blindspot-v04/baseline/candidate/CoreOrigin.csv. Its SHA256 remains `c572620d15f0c60d9cf14df1b073131e91cb0153dd1c892ede0bd0fd5c8fc46e`. The official direct public scorer gives spell F1 0.9529 / macro F1 0.9827, explicitly recorded under training_fit_only: the exported classifier fits all public labels, so these are NOT validation results.

The seven-group comparison reuses previously inspected public labels for feature-group selection. New offline features may share unlabelled raw context across folds or use entire-half statistics; the embargo excludes training labels, not all raw observations. No private truth or independent match was evaluated. Local report: data/experiments/blindspot-v04/index.html. All source and output hashes are recorded. The v0.4 source ZIP and matching Colab notebook are rebuilt and syntax/checksum verified locally; a new full Colab run has NOT been performed for v0.4. Prior v0.3.1 cloud verification remains specific to that release. Frozen v0.3/v0.3.1 candidate/source artifacts remain unchanged.


Final v0.4 verification independently rescored and format-validated all 56 saved fold CSVs; every recorded score/confusion matrix and source/output hash matched. The 66-file source/public-benchmark ZIP has SHA256 `e048c2351a5ad6b2e51e4e2971ca9013db22f4388716f244bd4d74aeb58a87d5`, its CRC passed, and all eight notebook code cells compile. Prepared CRC-verified private evidence and CoreOrigin-v04-experiments handoff archives. This verification record follows the frozen source snapshot; the snapshot remains unchanged.


### 19. Fixed team-only contacts experiment v0.4.1 (21 October 2026)

Committed the verified v0.4 code locally as `6ea1036` before implementation. Implemented `contacts_team_only` as a new runner group: the baseline classifier supplies P(D), the contacts classifier supplies conditional H/A scores, and both use identical training masks and unchanged classifier/decoder parameters. The input heads stay separate to avoid flattening team odds when baseline live mass is tiny. In-memory probability reuse ensures comparisons use the same underlying fitted probabilities.

Evaluated baseline, contacts and the hybrid on both existing four-fold protocols under a separate data/experiments/blindspot-v041 directory. Baseline and contacts controls reproduce v0.4. Runtime checks require every hybrid full-match decoded D/X mask to match its baseline. A report records mask/model-training audits and error counts.

| Method | Whole-run spell F1 | Fixed-time spell F1 | Whole-run dead F1 | Fixed-time dead F1 |
|---|---:|---:|---:|---:|
| baseline | 0.4457 | 0.6200 | 0.6619 | 0.5945 |
| contacts | 0.5600 | 0.6244 | 0.6906 | 0.5850 |
| contacts_team_only | 0.5000 | 0.6341 | 0.6619 | 0.5945 |

The hybrid's 0.0543 / 0.0141 spell-F1 gains over baseline do not meet the existing >0.06 margin in both protocols. Wrong-team frames improve from 782 / 681 to 512 / 447 while the two dead-ball error counts remain identical to baseline. This is the intended role isolation, not improved stoppage detection. The verified v0.3.1 submission CSV remains unchanged and selected.

Added seven tests (46 total passing): head-source isolation, invalid/tiny-mass probability handling, team swapping, decoded/posterior/sampled dead-state invariance, both-head held-out-label exclusion, and relative-path candidate export. During export the CSV passed validation before a logging path error; fixed that error, then reran the unchanged experiment to align source fingerprints. Saved a separate experimental candidate rather than replacing the frozen release. Updated the report CLI with --experiment and Colab builder with --team-only, preserving the v0.4 ZIP/notebook and creating separate v041 artifacts. No private labels, independent-match validation, dead-ball detector, boundary snapping or new cloud execution are claimed.


Final v0.4.1 verification: all 24 fold CSVs independently format-validated and rescored; all 16 baseline/contact control CSVs match v0.4 byte-for-byte. Each of the eight hybrid full-match D/X masks matches baseline and both head training masks match exactly. All 46 tests pass. The experimental full refit passes the official 199,319-row validator, reproduces SHA256 `97f8b81ee3977110236f53718b01b86ab186d5081bf57043d5d0292f31ee1141`, and preserves frozen v0.3.1 full-match D/X masks. The selected submission hash remains `c572620d15f0c60d9cf14df1b073131e91cb0153dd1c892ede0bd0fd5c8fc46e`. Separate v041 source/notebook and private handoff artifacts are prepared; local compilation/checksum verification is separate from cloud execution.


### 20. v0.5 structural rules (22 October 2026)

Reran the already-verified `contacts_team_only` fusion control (baseline P(D), contacts P(H/A | live)); its earlier run is recorded in section 19: 0.5000 whole-run / 0.6341 fixed-time spellF1, versus baseline 0.4457 / 0.6200. Wrong-team frames fall from 782/681 to 512/447; the dead-ball decision is unchanged by construction.

Added `possession/structure.py` with two rules whose constants were fixed before scoring: an out-of-play stoppage prior (observed ball more than 1 m beyond a line is dead; a soft hold continues until the ball is next seen inside and kicked above 5 m/s, capped at 60 s; missing ball never starts a hold) and contact snapping of H/A changes (1 s window, never across a stoppage or another change). Added `tests/test_structure.py` (7 tests). Full suite passes. Basis: IFAB Law 9 and the possession-zone/contact rules of Vidal-Codina et al. 2022 (arXiv 2202.00804). The pre-scoring label check found 92 of 92 public frames more than 1 m out of the pitch are D.

| Group | Whole-run spellF1 | Fixed-time spellF1 | Total wrong frames (W / F) |
|---|---:|---:|---:|
| baseline | 0.4457 | 0.6200 | 1,681 / 1,666 |
| contacts_team_only | 0.5000 | 0.6341 | 1,411 / 1,432 |
| baseline_outplay | 0.4571 | 0.6200 | 1,553 / 1,543 |
| team_only_outplay | 0.5111 | 0.6341 | 1,320 / 1,341 |
| team_only_outplay_snap | 0.5111 | 0.6244 | 1,314 / 1,349 |

`team_only_outplay` is the best variant: about 20% fewer wrong frames under both protocols, transition matches within 1 s up from 4/7 to 5/9. It does not meet the 0.06-in-both-protocols spellF1 rule (+0.065 / +0.014), so the automatic selection keeps baseline. Snapping does not help. Exported and validated (not selected) `data/experiments/blindspot-v04/team_only_outplay/candidate/CoreOrigin.csv`, SHA256 `f1c6d022e84a6df92f514923b50e03f7f4065e72f4093650814cee28e4c5f9d2`, 252 possessions, 5,325 frames different from v0.3.1.

Remaining public errors are short possessions under 3 s that are missed entirely and long in-pitch stoppages (fouls, a restart after the ball was carried back). These were seen while inspecting fold errors, so any rule written for them would be tuned on the public sample. Same-sample results only; no private truth or independent match was used. The Colab bundle was rebuilt for this source; it has not been run in Colab.


### 21. Reviewed v0.5 rerun and structural regression fix (22 October 2026)

Preserved the received changes under data/experiments/blindspot-v05/received-source before editing. Fixed adjacent contact-boundary moves crossing and deleting a short possession; the algorithm now respects already-moved neighbours and asserts ordered-spell and D/X-mask preservation. Numerical constants were unchanged. Added four structural regression/invariant tests; all 57 tests pass, including 1,000 randomized sequences and team-swap checks.

Reran six approaches in the separate blindspot-v05 directory: 48 fold outputs across both protocols, with underlying model probabilities reused. All 48 CSVs independently pass the official format validator and reproduce their metrics/confusion matrices. All 24 control CSVs match frozen v0.4.1 byte-for-byte. All eight snapped folds preserve ordered spells and D/X masks. The fix changes five full-match frames in three snapped fold CSVs, without changing public headline scores.

Team-only contacts plus the out-of-play prior scores 0.5111 / 0.6341 spell F1 and 0.7066 / 0.6458 dead F1. Total wrong frames fall from 1,681 / 1,666 to 1,320 / 1,341 (21.5% / 19.5%). Dead-called-live falls by 93 frames in each protocol; false-dead increases by two. Corrected snapping scores 0.5111 / 0.6244, so it remains unhelpful. No group clears the >0.06 spell gain in both protocols; selected v0.3.1 remains unchanged.

Exported the unsnapped experimental candidate separately, officially validated 199,319 rows and reproduced f1c6d022e84a6df92f514923b50e03f7f4065e72f4093650814cee28e4c5f9d2 exactly. EXPERIMENTS_v05.md records design, bug, scores, error counts, verification and commands. README, RESEARCH and IMPLEMENTATION now describe this result. Added a separate --structural Colab build mode, avoiding replacement of older ZIP/notebook pairs.

Scope correction to the received section 20: the out-of-pitch rule is a strong probability prior rather than an unconditional decoded D constraint. Its design included checking 92 public outside-margin frames against labels before aggregate scoring. Results are therefore same-sample, diagnostic-informed development, not independent evidence. No private truth, new cloud execution or competition submission was used.


### 22. Fixed v0.6 architecture study (22 October 2026)

Reviewed the newly added possession/v06.py, runner groups and tests. Implemented individual-offset probability caching for ensemble reuse and corrected the acceleration/contact union to 94 unique features. Added two tests checking held-out label exclusion for both model families over all five offsets, and decoder parity plus D/X-mask invariance with gating. All 65 tests pass.

Ran five fixed new variants and two controls: 56 fold predictions under the existing two protocols. Acceleration features improve whole-run spell F1 to 0.5765 (previous project high 0.5600), but fixed-time stays 0.6200. Contact gating scores 0.4947 / 0.6341; live-only ownership 0.5100 / 0.6244; bagging 0.4737 / 0.6400; combined 0.5231 / 0.6000. No release-rule switch is justified. The initial batch was interrupted after apparent stalled logging; its completed bagging checkpoint was recovered, and the remaining heavy groups ran separately before consolidation.

Added scripts/verify_v06_results.py for independent official rescoring, control-byte parity, source/output hashes and dead-mask invariants. Added a separate --advanced Colab package mode and v06 report title. Added scripts/prepare_blind_validation.py and verified twelve disjoint 30-second windows outside public labels and embargo, selected without predictions. No annotations were generated; a blinded reviewer must supply them before validation is possible. Researched PathCRF's published pretrained models and external Metrica samples; documented the proposed transfer experiment in RESEARCH.md without claiming it was integrated or evaluated.

See EXPERIMENTS_v06.md and the local verification.json for final results and artifact checks. No private-match labels, new cloud execution or competition submission were used.


Final verification: all 56 fold CSVs independently pass format validation and reproduce official scores/confusion matrices; all 16 controls reproduce v0.5 byte-for-byte. Source/output hashes match. Gated and live-only ownership variants preserve full-match D/X masks. The acceleration full refit passes the official validator with 199,319 rows and SHA256 `198b2adb1e422db36abb4299cca8e38903e5626312a9303a92069e74313a2fa8`. Its public labels were used for fitting, so direct public scoring would be training fit. The selected v0.3.1 hash remains `c572620d15f0c60d9cf14df1b073131e91cb0153dd1c892ede0bd0fd5c8fc46e`.


### 23. Local external-supervision migration (24 October 2026; in progress)

Read the recent pull in reflog (23 October 23:03 America/New_York) and compared all eight upstream documentation changes. Confirmed current upstream HEAD remains f8f2629. The clarified rules invalidate competition use of prior public-label-trained candidates and same-match manual validation proposals. Added explicit historical-only CLI guards to supervised legacy runners, disabled scripts/prepare_blind_validation.py, and marked historical documentation accordingly. No past results or frozen artifacts were deleted.

This Windows checkout initially contained no match assets. Retrieved four tactical inputs from the previously documented Drive IDs and verified every existing pinned SHA256. Downloaded all 21 official external Sportec Open DFL files (seven separate matches), verifying publisher size and MD5 and recording SHA256. No restricted data was uploaded to a hosted model, no videos were needed, and all derived artifacts stay under ignored data/.

Detected RTX 5070 Ti Laptop GPU with 12 GB and compute capability 12.0. System torch 2.14.1 was CPU-only. Created ignored .venv with system-site-packages and installed the official torch 2.11.0+cu128 wheel there. GPU matrix multiplication succeeded. Fixed the local execution environment by putting temporary files under data/test-tmp: the original 65 tests pass; seven new adapter/temporal tests also pass (72 total at that checkpoint). Two additional policy tests were subsequently added and are being checked.

Added possession/external.py, possession/temporal.py, configs/local_v07.json, local download/preparation/training/inference scripts and a PowerShell launcher. External label flags are targets only; feature construction accepts coordinates/team membership only. Tests verify target perturbation cannot affect features, swap/permutation invariance, unknown-label handling, exact decoder parity and chunked inference parity. Split fixed before fitting: four train matches, J03WPY for development, J03WQQ/J03WR9 reserved for one final external test. Training uses a fixed external-only boosted-tree reference and a temporal candidate with train-only normalization, resumable RNG/checkpoints, validation-only selection and test receipts.

Completed: pull/rules analysis, data retrieval, hardware verification, implementation and initial tests. In progress: external feature caches and first local end-to-end model run. Not yet established: any new score, +0.15 absolute improvement, verified BAR-ATH candidate from this workflow, independent test result or competition submission. Historical public development scores are not comparable to the new external benchmark. See LOCAL_V07.md for reproduction and actual saved run evidence.


### 24. Completed local clean-data experiments and first independent test (24 October 2026)

All seven native external feature caches completed and passed source/input checks. The first training launch failed before fitting because NumPy treated a literal X separator as an indexing directive. Corrected input pooling, added a regression test, retained the failed run separately and restarted in run-02. The fixed external reference scores 0.62100955 development spell F1. A 60-epoch local GPU TCN reaches 0.75526992 (epoch 35). A recorded 15-pair lower-penalty check in run-03 does not improve it. The 120-epoch short-run-weighted TCN in run-04 reaches 0.74418605, a regression. The 100-epoch bidirectional GRU in run-05 reaches 0.76617647 (epoch 30).

A pre-recorded 12-combination development ensemble study in run-06 selects 25% TCN and 75% GRU with 2/3 switching costs: 0.77124842 spell F1 versus the exact retained baseline 0.62100955, +0.15023887 (15.02 points). Normalization equality, label exclusion, recurrent half continuity and ensemble probability parity are checked. All choices use the same development match, so this is selection evidence only.

Froze the selected ensemble and evaluated J03WQQ/J03WR9 once. Independent pooled spell F1 is 0.73173885 versus baseline 0.64296931: +0.08876954 (8.88 points), NOT the requested +0.15. Macro F1 is 0.94642018 versus 0.92749130; alive/dead F1 is slightly lower, 0.99295607 versus 0.99562140. Preserved test_results.json and test_receipt.json. These matches are now opened, not fresh holdouts. Added a global registry and subset-overlap guard so different candidate directories cannot reopen them as independent tests. No target success is claimed.

Both the TCN and frozen ensemble completed full local tactical inference without reading BAR-ATH truth. Their 199,319-row CSVs pass format validation. Frozen ensemble CSV SHA256: 79b8b47c73a06e06925f725fc0b3675648777cb5af11285c27016aab047ccfc5; checkpoint SHA256: ede4a1896ab0aaad58343a7a4ab196a1cd127b4156ebe5e66d66f508bd6ab49a. Source snapshots, environments, input hashes, loss histories and selections remain under ignored data/local-v07. The output is a provisional research candidate, not a submitted entry or known private-match score. All 80 tests pass at this checkpoint.

Continuing work: external ball coverage is much more complete than the challenge input, so direct transfer through natural gaps remains uncertain. Prepared label-independent alternating missing-ball views for the four training matches and the development match, preserving official targets. run-07 trains a GRU on native plus augmented external views. A clean/artificial-gap development diagnostic compares robustness without reopening the used independent test cohort. Additional independent labelled matches were requested while continuing useful work. See progress.json, LOCAL_V07.md and CoreOrigin.md for current evidence. The +15 independent target remains active and unachieved.


### 25. Missing-ball robustness and final local export (24 October 2026)

Completed native-plus-artificial-gap external training in run-07-gaps (100 epochs, selected epoch 15). Native development spell F1 is 0.78675159 versus the unchanged reference 0.62100955, +0.16574204 (16.57 points). On the same artificial-gap development view, the prior ensemble is 0.47369759 and the new model is 0.63867403, +0.16497645 (16.50 points). The reference on that view is 0.33360768. Artificial-gap macro F1 improves to 0.90271376 and alive/dead F1 to 0.97363285. Masks are label-independent; targets are identical. These are development/stress findings, not a new independent test or proof on natural gaps.

Generated the new complete tactical CSV on CUDA; the official validator accepts all 199,319 rows. Two complete inference passes produce identical CSV bytes: SHA256 625445f4cf96d7799a94401b8066940b9e677809c068038f5b1c440cdf30f1b0. Model checkpoint SHA256 ee12760f9c90e15863270406d41652bf9491d8d30bc2e23ad947a2a8529beb27. Reproducibility receipt, source ZIP/manifest, configuration, environment and output manifest are under ignored data/local-v07/run-07-gaps. All 80 tests pass; Python syntax, PowerShell parser and git diff whitespace checks pass. No BAR-ATH truth was read for these experiments or outputs.

Independent evidence remains the earlier frozen ensemble's +8.88-point DFL result. The newer model has NOT been tested on independent matches, and the independent +15 target is not achieved/verified. Original test matches cannot be recycled as fresh tests. Asked for another permitted labelled match while continuing work. Also inspected official SkillCorner open data and phase specifications: its phase labels are derived from tracking/ML, preserve disruption phases and apply a 30-second event-gap rule. These are useful potential provider/pseudo-label data but not interchangeable with independent corrected H/A/D reference truth; no score was manufactured from them. Driblab's documented input data does not establish suitable dense possession/in-play reference labels.

Updated CoreOrigin.md to the current external-only technical draft and marked older implementation/report documents historical. Added a one-command PowerShell reproduction path with -GapRobust. No public repository push, competition upload or final competitive-accuracy claim was made. Development gains exceed 15 points; the unseen-match requirement remains open pending suitable new reference data.


### 26. Frozen-run completion audit (24 October 2026)

Rechecked the actual v0.7 configuration, source/input hashes, training/development separation and frozen artifacts during goal continuation. Recomputed normalization from native plus augmented training matches only and matched the saved tensors exactly. Confirmed that synthetic views leave target labels unchanged and the retained baseline model bytes still match run-02.

Independently decoded the actual frozen model probabilities with possession/decode.py's reference infer routine rather than the export decoder, then used the unchanged official scorer. The development results reproduce exactly: baseline 0.6210095497953615, gap-trained model 0.7867515923566878, gain 0.1657420425613263. Revalidated all 199,319 CSV rows and both full inference hashes. Source/config/input checks pass. Evidence: ignored data/local-v07/run-07-gaps/completion_audit.json.

This strengthens the development and reproducibility evidence but is not a new independent match. The independent +15 requirement remains unverified: the earlier ensemble's test gain is +8.88 points, the original test cohort was opened, and no suitable new corrected H/A/D reference match has been supplied or verified. Further same-development-set tuning cannot close that evidence gap. The previously requested additional permitted match remains the next required input for independent confirmation. No test labels were reopened, no model was retrained and no success claim was substituted for the missing evidence.

### 27. Continued local development toward 0.90 (24 October 2026)

The user confirmed no additional independent labelled match and requested continued GPU development until 0.90. The two previously opened external test matches now explicitly join training; only J03WPY remains excluded for exploratory development. Configurations declare development_only and refuse independent test evaluation. The official BAR-ATH scorer and human labels remain excluded from model development.

Added coordinate-only contact, relative-motion, acceleration and interception features, auxiliary transition supervision, several temporal architectures and fixed development comparisons. Completed run 13 reaches 0.8374845869297165 spell F1, with corrupted-coordinate stress score 0.6870588235294118. Two complete tactical inference passes produce the same validated 199,319-row CSV (SHA256 0d0c3f60b61b9c186ad3f294c2c9071b36776c2ebad2cd33c1fe96810b6585fb).

Full-25-Hz training keeps the frozen reference's original 12.5-Hz grid. Exact sampled-feature and reference-score parity checks pass. Repaired cuDNN's long-sequence limitation using carried-state directional GRU chunks, checked against direct CPU and CUDA inference. Run 17 completes 60 epochs and reaches 0.8490291262135923 at epoch 24; a fresh audit reproduces it. Its full tactical export validates with SHA256 61eef72996f895d2ce94faf964483e2dd4242cb9a8377c0caed4d768f3c0c65d. Its corrupted score is 0.6605504587155963, below run 13, so robustness and clean development improvements are reported separately.

The spell-overlap auxiliary loss completes 60 epochs with 0.8415327564894932, below the control. Further auxiliary ball-height training is in progress; height supplies training targets only and is not an inference input. Every run preserves configuration, source snapshots, input hashes and histories under ignored data/local-v08. The target remains unmet; no new independent score, official score or public submission is claimed.

Ball-height auxiliary training then completes at 0.8444444444444444, below the full-rate control. A frozen cross-rate blend also regresses to 0.8449877750611247. The control's second complete CUDA tactical inference reproduces the same 199,319-row CSV bytes and hash. Added exact compiled decoder acceleration, verified against the frozen control's development paths. The integrated suite passes 131 tests; the unchanged official scorer still calculates results.

Prepared and audited 14 expanded feature caches with 116 additional fine-motion features; their original 664-feature prefixes, labels and frames are exactly unchanged. A finite local batch now tests fine motion, deeper recurrent context, soft boundary targets and native-only context sequentially. Improvements require a fresh scoring audit and byte-identical full exports before target acceptance. Its current ledger is data/local-v08/campaign_status.json; it is an active subprocess rather than a scheduled Codex automation. All pending configurations remain experiments, and 0.90 is not yet achieved.

The verified best then becomes run-26-feature-ensemble at 0.851770451770452. It combines 25% of run 17 and 75% of the completed fine-motion run 22, with fixed 4/6 switching costs. A fresh audit reproduces the score; two complete CUDA exports give identical validated 199,319-row CSVs, SHA256 82ca7340c54344d0eb3420ac8c91f2c2c56a54b06166d6588c71c506487445a8. Its artificial-gap score is 0.6588235294117647, below the retained run-13 robustness fallback at 0.6870588235294118. No independent or official score is inferred from this small development gain.

Preserved the oversized run 23 as a deliberately interrupted three-epoch trial (0.82915), after observing 124–139 seconds per epoch. Soft transition targets (run 24: 0.84296), native-only longer context (run 25: 0.84293), long boundary TCN (run 27: 0.82110), independent boundary mixing and lower decoder penalties do not beat run 26. Cosine learning-rate tests continue in a revised sequential local batch. All 152 tests pass at this checkpoint. Exact structured chain loss and short-spell sampling prototypes are tested separately but not yet trained; they are not counted as score gains. The 0.90 target remains unmet.


### Completed cosine and short-spell studies; contact geometry next

Run 29 reaches 0.8533007334963326 at raw epoch 36. Run 33, initialized from it and trained with complete short-spell crop sampling, completes 40 epochs and reaches 0.8533333333333333 at raw epoch 1. This is a negligible numerical increase. Fresh scoring and two complete CUDA tactical exports agree: 199,319 rows, CSV SHA256 87615ce4d005465e2a1ce2081f6f30d60d4a536908d4a839a06bfc3efe076b5e, checkpoint b9d7d2f7116e811d8b28349e9969d9a08d78677727d7bb08f7ac9d29dd1456ae. Its artificial-gap score is 0.6740157480314961, below run 13 at 0.6870588235294118.

The earlier structured-loss and sampling prototypes are now integrated. Cold structured training (run 32) completes 60 epochs at 0.832258064516129; structured warm fine-tuning (run 34) retains its epoch-zero teacher at 0.8533007334963326. The combined structured/sampling experiment (run 35) is being completed. Failed alternatives and source archives remain preserved. Archives are verified against their frozen protocol rather than overwritten by later working-copy changes.

The next experiment adds 56 coordinate-only contact geometry inputs to the existing 780-feature prefix. Added-input coefficients start at zero and reuse the parent training-only prefix normalization. Fourteen native/corrupted caches are being audited for exact prefix, label, frame and checksum lineage. All 169 tests pass at this integration checkpoint. The unchanged official scorer still reports spell F1; 0.90 remains unmet and there is no current independent or official score.


### Verified frozen-teacher correction result

Run 46 completes 40 epochs and selects EMA epoch 24 at 0.8559036144578313 spell F1, macro F1 0.974063290142703 and alive/dead F1 0.9994104497776903, with 421 true/409 predicted spells. A fresh audit agrees, and two complete CUDA tactical inference passes give identical validated 199,319-row CSVs. CSV SHA256 b605437dff7edd60441b7b61f6da1cf6a12dcef3ca98b2a1d662136c65493c21; checkpoint 0d8209cd3e1c99ccf48b2e765bef286e43efd78820dfd8c1e6a59865f10fb151; source ZIP 83c62ac4d2586b7074368c9494d75e27da0ce07bcdcfc0f6a17d9343a7182d97. Its artificial-gap score is 0.66282722513089, below both the previous run 33 (0.6740157480314961) and retained robustness fallback run 13 (0.6870588235294118).

The clean gain is 0.26 points. At IoU 0.5, sub-second matches remain 25/57; matches improve by one in each of the 1�2 s, 2�5 s and 5�10 s groups. This does not establish a new independent or official score. The 0.90 development target remains unmet. The local-width candidate is now running, followed by four predeclared loss/data-emphasis candidates. Higher-rate and teacher-context correctors are prepared but untrained; all 193 current tests pass.


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


### 28. Artifact recovery, exact reproduction and decoder-ceiling audit (29 October 2026)

Recovered the working tree from the user's archived bundle `more-than-a-hack-2027.zip` (34,687,979,399 bytes) with HTTP range requests. Parsed the archive's central directory and fetched only the four tactical non-video assets and the run-58 artifacts, not the 19 GB of video. The archive's stored CRC fields do not match the decompressed bytes, so extraction was verified by exact uncompressed size and by byte agreement with the repository for tracked files. No match asset left this machine.

Rebuilt a CPU-only macOS arm64 environment (`.venv` with system site-packages, torch 2.15.0) and ran the suite: **266 tests pass**. Decoding the saved run-58 development probabilities with the recorded costs [0, 2, 1] reproduces the recorded selection exactly (spell F1 0.8570048309178743, macro F1 0.9742746076990892, 421 true / 407 predicted spells). Nothing was retrained, and the user's earlier two-attempt training cap was not reopened.

Added `scripts/audit_decoder_ceiling.py`, an evidence-only audit of the frozen development probabilities. It reproduces the recorded score, sweeps a declared 144-setting decoder grid (range 0.724145 to 0.858041; best gain 0.001), and substitutes two oracles. With ground-truth spell boundaries and the model's own team decisions the score is **0.900635**; with the model's boundaries and ground-truth teams it is 0.890076. Both halves of the problem must improve to pass 0.95, which rules out further decoder work.

Diagnosed the remaining error: at IoU 0.5, 359/421 possessions match; 79/124 spells shorter than 2 s match; 113 of those 124 are brief changes inside the other team's possession. The retained run-58 history shows training loss falling 0.06361 to 0.04563 while development spell F1 falls 0.8570 to 0.8432, so frame-level fitting beyond the warm start is anti-correlated with the ranking metric.

Found and fixed one certain submission defect: the exported candidate assigned possession to 1,609 home and 1,200 away frames inside the pre-second-half window that FAQ.md publishes as all dead ball. Added `possession/anchors.py`, `scripts/apply_published_anchors.py` and `tests/test_anchors.py` (5 tests). The correction rewrites frames 110808-113717 only, with a one-second guard before the published 113742 kickoff anchor; 2,784 frames change and predicted possessions fall 413 to 399, removing 14 possessions that cannot exist in the ground truth. No label and no tuning set is read. New candidate `data/submissions/CoreOrigin-v08/CoreOrigin.csv` (199,319 rows, official validator pass), SHA256 42b0a75bf954a887ea84615bdcd312c78ab0be315ff58d2c03d2f232b1e919d3; the public-window score is unchanged because the window is in the first half.

Public-window pipeline check: 0.505263 spell F1 / 0.651007 macro F1. Two long in-pitch stoppages (20.4 s free-kick, 14.5 s goal-kick setup) inside that window are called alive, accounting for 1,198 of 1,469 dead frames. Smart Tagging brackets both, but adopting a set-piece-to-dead rule because the public window rewards it would be tuning on the labels the rules reserve for pipeline checking; it was not adopted and is recorded as an open question instead.

**Verdict on the 0.95 request:** not reachable with the current 960-feature bank. The measured segmentation oracle is 0.900635, the decoder grid is saturated, and the training history is anti-correlated with the metric. The remaining routes require evidence the geometry bank does not hold: ball recovery from the tactical video on the 41% of graded frames where tracking loses it, frame-resolution contact detection, or substantially more labelled matches. No new independent or official score is claimed and no training was launched.

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
(±2 frames) of each team's nearest-player distance under 2 m: 95% of all 2,027 training H<->A
changes have a candidate within 5 frames. With perfect candidate labels the chain decoder reaches
**0.9714** mean spell F1 on train (ceiling of the representation). A two-stage LightGBM
(65 geometry + 27 ball-kinematics + 65 ±12-frame sequence features, then neighbour-candidate
context), candidate-level 2-state Viterbi:

| system (J03WPY unless noted) | spell F1 |
|---|---:|
| frame-level LightGBM (132 feats) + Viterbi, best of grid | 0.6895 |
| touch chain, stage 2, λ=0.5 (chosen on train OOF 0.7745), truth dead mask | 0.8204 |
| run 48 + chain posterior fusion (any weight tried) | 0.8385–0.8498 |
| run 48 + chain short-spell insertion (thr 0.95 from train precision 0.771) | 0.8564 |
| run 48 + OOB dead-ball rule (prior) | 0.8580 |

The decisive sub-task is "an opponent comes within 2 m: did it win the ball?" (1,140 such cases on
J03WPY, 25% positive). Best tracking-only classifier: AUC 0.931, accuracy 0.88 (train OOF 0.916 /
0.851). Spell F1 > 0.95 needs roughly 98%+. Run 48 missed 62/421 J03WPY possessions at IoU 0.5;
32 are under 1 s, 28 have <10% of their frames predicted for the right team, and the touching
player was within 0.65 m (median) of the ball — the touch is in the tracking geometry but not
separable from a near-miss with 25 fps centre-of-mass positions.

**Verdict.** No allowed CPU-side change lifts J03WPY above 0.858; run 48 stays selected. >0.95 on
J03WPY is not reachable with tracking-only evidence. Remaining routes (none runnable here): (1) much
more labelled tracking (e.g. PFF FC World Cup 2022 open tracking+events, ~64 matches) to train the
contact classifier; (2) video-based contact/touch detection — impossible on DFL (no video), so it
cannot be validated on J03WPY at all; (3) ball height/foot-level pose. The Mac has no tactical video
and no CUDA host is registered, so no video work was attempted. Nothing committed or pushed.
