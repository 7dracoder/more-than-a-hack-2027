# Work done and current project state

Last updated: October 20, 2026 (America/New_York).

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
| Frontier research extension | Research brief and small probe complete | Competing-history possession proposal; camera-framing signal measured; full method not implemented |
| Custom possession algorithm | Not implemented | Only the organizer's starter algorithms exist |
| New model training / final submission | Not started | No claimed improvement over the organizer baseline |

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

## Decisions and constraints to carry forward

- Use tactical video frames as the reference throughout.
- Do not make broadcast fusion a prerequisite for the first improved solution.
- Preserve uncertainty and missingness instead of converting absent observations into D or zero coordinates.
- Use team-level features that tolerate track fragmentation; do not assume stable player identities.
- Evaluate possession structure and boundary timing, not just frame-level accuracy.
- Do not tune a large search against 17 public possessions or claim hidden-match performance.
- Keep the official scorer and public reference labels unchanged.
- Record permission scope and follow competition data-use terms; the user's sharing authorization is not a general license for unrelated redistribution.

## Remaining work

### Next implementation steps

- [ ] Produce a detailed quality audit: missing-ball gap distributions, suspicious jumps, slot discontinuities, and tag conflicts.
- [ ] Prototype and compare competing possession histories, as specified in RESEARCH.md section 11, after establishing the interpretable reference.
- [ ] Evaluate camera/visibility evidence on separated clips and naturally missing-ball intervals before treating it as useful to possession.
- [ ] Build a canonical cached frame representation with explicit masks and metadata.
- [ ] Implement the first control-aware rule/state-machine baseline.
- [ ] Add and compare separate live/dead evidence.
- [ ] Review Smart Tagging code-to-possession mappings before using them.
- [ ] Compare hysteresis with a small temporal decoder and run ablations.
- [ ] Add synchronized transition-focused video diagnostics.
- [ ] Decide whether further annotations, learned models, or local video recovery are justified.

### Unresolved issues and optional verification

- [ ] Investigate broadcast's one-frame count discrepancy before using broadcast fusion.
- [ ] Full video decode/integrity scan, only if needed; sampled checks are the current evidence.
- [ ] Stronger cloud integrity verification if exact checksum parity is needed.
- [ ] Decide whether the earlier partial Drive copy should be retained; no deletion has been requested.

### Submission work, not started

- [ ] Select and freeze a defensible algorithm/configuration.
- [ ] Generate and validate the final 199,319-row team-named CSV.
- [ ] Pin dependencies and document one-command reproduction.
- [ ] Write the short technical submission document.
- [ ] Verify reproduction and final artifact hashes before submission.

The repository lists November 17, 2026 as the deadline. Confirm the organizer's current submission instructions before final submission.
