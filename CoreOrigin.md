# CoreOrigin / BlindSpot: external-only possession reconstruction

Provisional local research candidate, October 25, 2026. The best completed external development result is 0.85700 spell F1. The two capped improvement attempts and the subsequently authorized cross-fitted transition-repair study are complete without beating this result. The latter scored 0.85617 and was not promoted. All training and verification have stopped. The 0.90 target remains unmet. No competition submission or private-match score is claimed. The old public-label-trained candidates are historical and are superseded by the organizer's October 23 rules clarification.

## Task and inputs

Produce H (home possession), A (away possession), D (ball inactive), or X (no answer) for every one of the 199,319 tactical video frames. The current workflow uses player/ball coordinates and team membership from the automatic tactical tracking. It does not read the BAR-ATH public possession labels, human event records about this game, or private ground truth. Both halves are processed independently; the shared game metadata identifies the active half windows.

Coordinates are converted to metres. Speeds and relative motion are derived from coordinates with finite-observation and speed checks. The 664-feature physical bank includes nearest-team geometry, collective motion, past/future geometric contacts, ball acceleration and direction changes, projected interception distances and team-relative movement. Another 116 fine-motion inputs describe two temporal scales of ball motion and motion among each team's nearest few players, giving the selected model 780 inputs. Finite-observation masks are explicit. Temporal summaries and future context remain within a half. Input features never include possession/status targets, provider speed fields, ball height, absolute match time, or player identity codes.

## Training data and separation

Training uses the public Sportec Open DFL dataset by Bassek, Rein, Weber and Memmert (2025), doi:10.1038/s41597-025-04505-y; official files: doi:10.6084/m9.figshare.28196177, CC BY 4.0. Credit the authors and DFL. Its documented BallPossession and BallStatus fields supply frame targets: inactive is D, active possession 1 is H, active possession 2 is A, and unknown is X. These fields are kept out of feature extraction. No labels were manually added by this project.

The original split was fixed before fitting: J03WMX/J03WN1/J03WOH/J03WOY for training; J03WPY for development; J03WQQ/J03WR9 for the earlier independent external test. After that test was opened and retired, the latter two matches explicitly joined training for the current six-match development experiments. J03WPY remains excluded from training. Current candidates have no independent test cohort; the runner refuses independent evaluation for these development-only configurations. Earlier independent results apply only to the earlier frozen ensemble.

## Models and possession decoding

Two frozen external references are distinguished: the v0.7 reference scores 0.62101; the v0.8 four-match reference scores 0.63465 using its 484-feature bank at 12.5 Hz. The latter uses 80 small histogram-boosting iterations and fixed 6/4 switching penalties. Its exact bytes and original temporal grid are preserved even when a candidate uses additional features or finer sampling. Neither is the old public-trained model.

The selected model freezes the full preceding run 46: a 128-channel convolution/BiGRU teacher and its 64-channel correction TCN. A second trainable 64-channel TCN with six dilated blocks adds state and transition-logit corrections. Its final heads start at zero; the entire preceding model stays in evaluation mode without gradients. Native and corrupted training views, team-swap augmentation, exact probability symmetrization and training-only normalization are retained. Corruptions include ball gaps, player-track gaps and coordinate noise; targets are unchanged.

The selected checkpoint uses raw weights at epoch 12 of a completed 40-epoch fit. The new corrector learning rate starts at 0.001 with cosine scheduling. Training oversamples complete sub-second possessions; the frozen parent scores 0.8559036. Predictions stay at 25 Hz. Decoder team/stoppage costs are 0/2 with boundary log-odds weight 1, after removing the weighted-BCE prior. Whole-half recurrent inference carries hidden states through exact directional GRU chunks for long sequences. Raw and averaged weights were compared on development; raw weights won.

Checkpoint selection, blend weights and decoder settings used only the development match. All tested alternatives, including regressions, are retained. Short-run weighting and lower penalties alone did not improve the first model. All model inference and training ran locally on an RTX 5070 Ti Laptop GPU with 12 GB, torch 2.11.0+cu128 and Python 3.13.7.

## Measured results

Fixed development match: original gap-trained GRU 0.78675; current correction model 0.85700, a gain of 7.03 points. Its gain over verified run 46 is 0.11 points. Sub-second spell matches at IoU 0.5 remain 25/57. At IoU 0.7 one additional spell matches, while at IoU 0.5 one fewer matches; the mean score improves across five thresholds. Artificial-gap score is 0.66404. The retained 12.5-Hz fallback scores 0.83748 native / 0.68706 corrupted and remains stronger on the stress test. Repeated development selection introduces selection bias. The 0.90 target remains unmet.

First independent two-match test of the EARLIER frozen ensemble: reference spell F1 0.64297; ensemble 0.73174, gain 0.08877 (8.88 points). Macro F1 increases from 0.92749 to 0.94642; alive/dead F1 decreases from 0.99562 to 0.99296. The newer gap-trained model has not been independently tested. The minimum 15-point gain is NOT independently established. No BAR-ATH official score or ranking is known.

The current export passes the official format validator with 199,319 rows. Two complete CUDA inference passes produce identical CSV bytes. CSV SHA256: 602b7f1beefc538eac04ed881028ec609de4dacb95f6db33b425b38a599aca19. Checkpoint SHA256: 52bc76488d046f0d87347f09dd1d4d4cafe77a29b82238509baa92a233c0de03. A fresh audit reproduces the selected development score. Exact source, inputs, selections and repeat receipt are recorded under ignored data/local-v08/run-48-correction-rate. Earlier candidates' repeat receipts and the original independent test receipt remain preserved.

## Limitations and reproduction

The external tracking is much more complete than the challenge tracking. Natural missing-ball gaps and camera/tracking differences can impair transfer, especially stoppage detection; format validation does not prove prediction quality. There is no visual control recognizer, independent accuracy measurement on BAR-ATH, or certificate that a nearby touch represents control. Provider target conventions may differ. The dense-supervision pool contains seven Sportec matches. Additional public weak/event-derived training data did not improve the selected checkpoint; repeated development selection introduces selection bias.

The external-only missing-ball augmentation experiment has completed locally. Its artificial gaps are generated without labels, and official targets remain unchanged. Any stress-test result must be distinguished from natural-gap or independent-match accuracy.

See LOCAL_V08.md and the local run evidence. For the best completed candidate:

```powershell
.\.venv\Scripts\python.exe scripts/run_local.py --infer --config configs/local_v24_residual_fast.json --out data/local-v08/run-48-correction-rate
.\.venv\Scripts\python.exe evaluation/validate_submission.py data/local-v08/run-48-correction-rate/candidate/CoreOrigin.csv
```

Rebuild full-rate physical features with scripts/prepare_external_v10.py (native and --corrupt), then append fine motion with scripts/prepare_external_v13.py --both. Train with configs/local_v24_residual_fast.json; its frozen reference is built by the v0.8 contact workflow. Preserve every run's archived source, its exact configuration, the six-match development split and historical independent receipt. Match assets, models and frame-level derived outputs remain private/ignored, while the code and external data provenance are reviewable.


Additional-data study (October 25): Metrica sample Games 1/2 were integrated as partial event-derived training supervision (94773 known frames). Game 3 stayed out of training and was evaluated once after the matched control/treatment checkpoint selections were frozen. The two fixed 40-epoch fits both retained the 0.8570048309178743 epoch-zero parent control. No new Metrica-trained weights were promoted; the selected model still uses the preceding six-match strong-supervision result. Game 3 agreement is reported only on labelled intervals and is not a dense possession-spell or official score. Data source: https://github.com/metrica-sports/sample-data at e706dd506b360d69d9d123d5b8026e7294b13996; credit Metrica Sports.
