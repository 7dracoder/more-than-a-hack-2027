# BlindSpot: possession reconstruction with auditable evidence

**Provisional v0.3.1 audit candidate — not a submitted entry.** Registered team: **CoreOrigin**. The project/model is named BlindSpot; the submission file is CoreOrigin.csv. No private-match ranking is known.

## Approach

We infer home possession (H), away possession (A), or dead ball (D) on the tactical video clock. The system combines physical movement and ball-control features with a small supervised model, then decodes persistent possession intervals. It also records model disagreement and evidence coverage so that ambiguous decisions can be inspected in synchronized video/pitch views.

Inputs are tactical tracking/metadata, Smart Tagging, and the supplied pitch/frame constants. Homography is retained in the audit and historical features but is not a direct feature of the final classifier. Tactical video is used only for optional local review. Broadcast was investigated and excluded: its 164,487 compressed video packets do not match its 164,488 tracking rows, and no verified tactical/broadcast synchronization exists.

## Measurement consistency and features

The metadata declares speed in kph. However, a sampled displacement audit finds reported values approximately equal to trajectory-derived m/s (median ratio approximately 0.997; correlation approximately 0.963). Dividing by 3.6 was therefore rejected as a proven physical correction. We derive movement from normalized positions, pitch dimensions and frame timing instead.

Speed is a centered six-frame displacement (0.24 s at 25 fps). Windows containing missing points or adjacent jumps above 15 m/s are excluded. Slots are assumed continuous only over this short window; no persistent player identities are inferred. Team-set reductions give player-speed quantiles, stationary/running fractions and team movement differences. Further features describe ball visibility, motion, pitch exit, nearest-team distances, control margin, player coverage, opposed phase support and restart type/ownership. Three centered context widths (12, 50, 125 frames) yield 59 physical/context features plus nine observation-mask features (68 total). Missing ball observations remain explicitly missing; no trajectory or ownership is fabricated from absence alone.

## Learned model and possession rules

A CPU histogram gradient boosting classifier uses 80 iterations, learning rate 0.05, at most seven leaves per tree, L2 regularization 10 and minimum leaf size 20. Training samples every five frames and gives each retained H/A/D run equal total weight. Team-swapped augmentation and exact probability symmetrization prevent the learner from relying on a fixed home/away preference. Absolute pitch location and presumed player identities are not classifier features.

Live/dead and conditional home/away scores are decoded separately with two-state dynamic programs, independently within each half. Five-frame bins use mean log evidence. Fixed change costs are 4 for live/dead and 6 for team changes. The state persists through weak or missing control evidence until opposing evidence supports a transition; a touch alone is not explicitly classified as controlled possession. Dead-ball decisions use motion, visibility and restart evidence rather than equating every tracking gap with D. Published kickoff metadata supplies a conservative dead-ball prior, stopping one second before each observed kickoff anchor.

Ownership is reset at each algorithmically inferred stoppage and decoded independently within each live run. Forward/backward live/dead marginals and conditional owner support expose ambiguity. Owner support is conditional on the MAP live/dead segmentation; the product is not an exact joint marginal. Seeded alternative histories sample owners within their own live segments. Related fold-model disagreement and training-feature envelope flags rank review moments. These are not calibrated confidence, correctness certificates or proven out-of-distribution probabilities. Review notes never overwrite the algorithmic candidate CSV.

## Development evaluation

The official scorer is unchanged. The public example has 7,500 labelled frames and 17 possessions and has already been inspected in previous experiments. We use four contiguous evaluation folds containing complete H/A/D runs. A 300-frame embargo excludes neighbouring training runs; the conservative feature context radius is 130 frames. Each held-out prediction is produced by a model fitted without that fold's labels. Decoders run on unlabelled half inputs without being reset/clamped at ground-truth fold boundaries.

We pool per-IoU matching counts across the four folds, applying the same clipping/scopes to all methods. We do not score a stitched cross-fitted sequence or compare pooled fold scores with unsegmented full-public scores.

| Same four-fold development comparison | Spell F1 | Macro F1 | Dead-ball F1 |
|---|---:|---:|---:|
| Cross-fitted symmetric classifier | **0.4457** | **0.7484** | 0.6619 |
| Unlabelled movement-regime model | 0.3524 | 0.7343 | 0.6733 |
| Historical evidence v1 | 0.3200 | 0.4986 | 0.0000 |
| Organizer smooth125 | 0.3889 | 0.5812 | 0.1464 |
| Organizer smooth50 | 0.3043 | 0.6207 | 0.3225 |

Classifier fold spell F1 values are 0.4000, 1.0000, 0.2400 and 0.3556. The smooth125 reference scores 0.3636, 0.8667, 0.1333 and 0.3600 on the same folds. The classifier improves three of four folds; omitting any one evaluation block retains a positive pooled difference. This is descriptive sensitivity, not a statistical guarantee: only four folds are available, their training sets overlap, and all footage is from one previously inspected match.

The unlabelled motion candidate scores 0.3610 on the unsegmented full-public sample versus smooth125's 0.4242; it is not the selected provisional classifier. The exported classifier is refitted on **all permitted public labels**. Its public-window score includes training and is not presented as validation. No private ground truth or other corrected labels from this fixture are used.

## Audit sensitivity comparison

The v0.3.1 audit preserves the above whole-run scores. An additional equal-duration four-fold protocol (300-frame embargo, boundaries chosen without state labels) gives spell F1 0.6200 versus v3 0.6100 and smooth125 0.4513. Dead F1 decreases slightly, from 0.5974 to 0.5945. This is the same public sample, not independent data; do not compare scores across protocols as a gain. All four historical fold outputs reproduce byte-for-byte. See [AUDIT.md](AUDIT.md) for findings, tests and limitations.

## Reproduction and deliverables

```powershell
pip install -r requirements-v3.txt
python -m unittest discover -s tests -v
python scripts/run_champion.py --config configs/champion_v031.json --out data/experiments/blindspot-v031 --resume --archive
python evaluation/validate_submission.py data/experiments/blindspot-v031/candidate/CoreOrigin.csv
```

Place the permitted tactical assets under `data/tactical/` as documented in DATA.md. The pipeline writes the full 199,319-row candidate, trained models, fold provenance, all scores, unit/coverage audits, fixed artificial-gap stress test, review queries and self-contained HTML reports. Optional video review uses `requirements-v3-video.txt` and `python scripts/build_candidate_review.py --experiment data/experiments/blindspot-v031`.

Run manifests record input, source and CSV hashes, configuration and environment. Completed stages resume only after provenance/artifact verification. Dependencies, seeds and source snapshots are pinned. Match assets, clips, tracks, predictions and model artifacts remain ignored under `data/`; public bundles contain source and public benchmark files only. The team repository must contain the reviewed final code snapshot before submission; working-tree changes have not been published by this run.

## Limitations and next work

Seventeen public possessions cannot establish generalization or winning full-match performance. Stoppage and short-possession errors remain, tracking-derived speed inherits tracking noise, and restart/phase tags can be wrong. Artificial gaps do not reproduce every natural occlusion. Video control/contact recognition and broadcast fusion are not implemented. An independent permitted validation set and an actual reviewer study are needed before claiming improved natural-gap reconstruction or time saved. Final model selection, publication and form submission remain explicit release steps.
