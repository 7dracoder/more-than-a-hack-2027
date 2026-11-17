# Frozen observation-loss continuity candidate

**BAR–ATH spell F1 above 0.740 remains unverified.** This experimental candidate
uses externally trained v3 weights unchanged and an externally selected
continuity penalty. No supplied-match possession labels or video drive training,
calibration or candidate selection. The official format check measures file
correctness only. Full-match accuracy requires the organisers' evaluation.

On the reused strong external development views, spell F1 is 0.857005 native,
0.544552 masked and 0.632938 mixed: mean **0.678165**, versus v3's 0.668976.
Detected-only weak H/A agreement on two already opened SkillCorner fixtures is
0.825815 overall and 0.799851 with the ball missing. These weak group labels
are not dense HAD truth, and these development results are not independent tests.

The frozen grid was 0, 2, 4, 8. The selected penalty is 8, applied only when the
existing observation-loss gate activates and scaled by state uncertainty.
Complete observations retain the original decoder exactly. Parent change-point
probabilities, model weights and normalization remain unchanged. Every candidate
was independently reproduced from checksum-verified frozen probabilities and
external targets. Native performance and the declared weak-label guards passed.

Inference requires the parent `models/coreorigin-video-v1` and the pinned CUDA
environment in `requirements-video-ball.txt`. With the verified coordinate artifact:

```powershell
.\.venv\Scripts\python.exe scripts/research/spatial_memory/export_continuity_candidate.py --model models/coreorigin-continuity-v4 --ball data/research/dual-feed-20261113-v2/ball_hypotheses.npy --out data/research/continuity-calibration-20261114/candidate-packaged --device cuda
.\.venv\Scripts\python.exe evaluation/validate_submission.py data/research/continuity-calibration-20261114/candidate-packaged/CoreOrigin.csv
```

Expected 199,319-row CSV SHA-256:
`cbc30311a6c9fdbe8cc7198e3a25f38d444c07c437e491528da5a924ecca408d`.
Inference blocks training data and possession truth, preserves original ball
observations, and verifies model/source/input hashes. Coordinate reconstruction
commands are retained in `models/coreorigin-natural-visibility-v1/README.md`.
The local CSV and supplied-match data are excluded from this model bundle.

Training provenance and current technical details are in `TECHNICAL.md`; the
preceding v3 study is also preserved. The unchanged v3 weights use six Sportec
fixtures (CC BY 4.0) and eighteen SkillCorner fixtures (MIT notice retained).
Trajectory memory and receiver/lane geometry do not measure gaze, peripheral
vision or head direction. No new BAR–ATH public possession score was computed.
Frozen v11 and preceding bundles remain preserved.
