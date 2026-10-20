# BlindSpot: technical report draft

The current v0.3 submission-oriented draft is [SUBMISSION_TECHNICAL.md](SUBMISSION_TECHNICAL.md). The v0.2 findings below are retained as the historical development record.

Status: development prototype, October 20, 2026. No final submission or private-match result is claimed.

## Problem and method

Infer home possession (H), away possession (A), or dead ball (D) for each tactical video frame. X marks frames outside the published match blocks. Inputs are the supplied tactical ATD tracks and metadata, homographies and Smart Tagging. Broadcast remains excluded pending verified camera synchronization.

Team features reduce over sets of tracking slots; slots are not assumed to identify persistent players. Nearest-team distance and phase evidence supply team odds. Player movement, stationarity, ball position/stillness and bounded camera activity supply weak live/dead evidence. Missing ball supplies no invented team observation. Fixed transition costs discourage implausibly rapid changes. Mean log scores in five-frame bins feed per-half Viterbi decoding; output is expanded to every required frame. Inference resets across halftime.

Forward/backward inference yields model-relative marginals and seeded posterior histories. These reveal ambiguity within this model; neither high support nor sampled agreement certifies correctness. Query ranking combines uncertainty, source sensitivity and boundary proximity without reading reference labels. A local review lab synchronizes tactical clips with pitch tracks and model support. Reviews cannot overwrite algorithmic submission labels.

## Training and development evaluation

The main label-free model does not fit public labels. An optional two-head regularized logistic experiment fits dead/alive and symmetric home/away evidence. Whole H/A/D runs remain intact across the v0.2 temporal split: 2,600 training frames, 1,061 excluded frames and 3,839 evaluation frames. The actual gap exceeds the 250-frame minimum embargo and feature context. This footage was already examined in earlier work, so these are development results, not an untouched test or independent-match validation.

| Method | Full public spell F1 | Grouped development spell F1 | Grouped dead-ball F1 |
|---|---:|---:|---:|
| Label-free evidence v1 | 0.3351 | 0.3200 | 0.0000 |
| Fixed temporal stoppage v2 | 0.3351 | 0.3200 | 0.0000 |
| Grouped logistic candidate | Not reported as validation; includes fitting | 0.2750 | 0.3458 |
| Organizer smooth125 | 0.4242 | 0.2667 | 0.1603 |
| Organizer smooth25 | 0.2868 | 0.3111 | 0.3307 |

The proposed stoppage rule changes predictions elsewhere but provides no public-window gain. The learned candidate improves dead-ball F1 while reducing possession spell F1. Neither is promoted. The strongest full-public reference still beats the main model. All reference scores use the unchanged organizer scorer; no private truth is accessed.

## Review-query experiment

Twelve queries of 51 frames each yield 24.48 seconds of review at 25 fps. Error density is measured only after queries are selected. Boundary/source ranking selects clips containing 47.22% erroneous frames; entropy selects 44.61%; uniform selects 18.79%. Thirty fixed random seeds average 33.17%, with a broad 5th–95th percentile range of 16.13%–49.29%. Boundary ranking covers 11.90% of all public errors and four of twenty true transitions. This is encouraging error enrichment, not statistical superiority, corrected predictions or measured human time saved. No independent reviewer study has been completed.

## Data quality findings

All tactical and broadcast raw rows and homography indices were scanned. No malformed/nonsequential rows or invalid matrices marked usable were found. Tactical has 31 slot jumps above 3 m per frame and three observed ball jumps above 5 m; broadcast has 623 slot jumps. These are diagnostic flags, not verified errors or player identity claims. Simultaneous opposed phase support occurs in 16,033 tactical frames, motivating careful treatment of tags rather than treating them as possession truth.

All compressed video packets were scanned. Tactical has 199,319 timestamped packets, matching its tracking and advertised count. Broadcast has 164,487, versus 164,488 tracking rows. The discrepancy is confirmed at packet level, but its cause/alignment is unresolved. No arbitrary padding or frame shift was applied. A full pixel decode of both complete videos remains unperformed.

## Reproduction and artifacts

```powershell
pip install -r requirements.txt
# Install optional av and opencv-python-headless for local visual workflows.
python -m unittest discover -s tests -v
python scripts/run_followup.py --resume --packets --review-video --archive
python evaluation/validate_submission.py data/experiments/blindspot-v2/predictions/evidence_v1.csv
```

Source, config, tactical input hashes and stage artifact hashes are recorded in the run manifest/checkpoints. Completed stages are reused only after verification; incomplete/corrupted stages are recomputed. Source-only Colab bundles exclude match assets and predictions. Colab's optional Drive output path supports durable checkpoints after the user authorizes mounting; default runtime files remain temporary.

The original main CSV remains byte-identical, SHA256 `0128b44306c2c4d51fd7afaafb943d82ed88159558da2552b21ef262e114605b`. Private local artifacts include full-length candidate CSVs, aggregate evaluation, error diagnostics, query plans, twelve verified short clips and 492 sampled pitch snapshots. Match-derived clips/tracks/predictions remain under ignored `data/`; this document contains aggregate observations only.

## Before submission

Improve dead-ball discrimination without overfitting the public sample; collect permitted independent training/validation examples; validate any video recovery method; select and freeze a final algorithm; regenerate the final team-named CSV with exact hash parity; finalize team details and organizer submission requirements. Manual review is for training/validation and diagnostics, never frame-by-frame correction of the submitted prediction file.
