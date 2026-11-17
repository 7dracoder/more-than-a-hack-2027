# Frozen observation-loss expert

This is an external-only experimental candidate. BAR–ATH spell F1 above 0.740
is **not verified**. No supplied match labels or video were used for training,
calibration or model selection. No supplied-match data or predictions are
included in this bundle.

The model is the fixed final epoch of the preceding 50-epoch camera-loss
adaptation. It uses 664 geometry and 128 spatial-history/passing-option
features. It replaces parent state probabilities when the input ball or any
nominal player observation is missing, at the expert's half-rate input clock.
The parent boundary head and decoder remain unchanged. This is input-availability
gating, not tracking confidence, gaze, head orientation or cognitive memory.

External J03WPY development spell F1 is 0.857005 native, 0.531959 masked and
0.616749 mixed, mean 0.668571. The preceding natural-v1 mean was 0.568972.
J03WPY is heavily reused development data, not an independent benchmark.
The separate real-broadcast weak H/A audit is mixed: missing-ball frame accuracy
improves, but overall weak accuracy does not improve and conditional team
ownership still needs work. No candidate was reselected from that diagnostic.

Use the parent bundle `models/coreorigin-video-v1`, the pinned environment in
`requirements-video-ball.txt` and the previously verified coordinate artifact:

```powershell
.\.venv\Scripts\python.exe scripts/research/spatial_memory/export_observation_candidate.py --model models/coreorigin-observation-gate-v2 --ball data/research/dual-feed-20261113-v2/ball_hypotheses.npy --out data/research/observation-gate-20261114/candidate-packaged --device cuda
.\.venv\Scripts\python.exe evaluation/validate_submission.py data/research/observation-gate-20261114/candidate-packaged/CoreOrigin.csv
```

The exporter blocks training-data and possession-truth reads and checks its
source/model/input hashes before and after inference. Expected CSV SHA-256:
`0fb15e05701276e2659ded08ce07a6198a64b4a58210f7b25af1088d56cb44f5`.
Format validation establishes file correctness, not possession accuracy.
No new public possession scoring is part of this workflow.

Coordinate reconstruction commands from the supplied bundles are unchanged
and documented in `models/coreorigin-natural-visibility-v1/README.md`.
Research, limitations, weak-label interpretation and further-training plans
are documented in `scripts/research/spatial_memory/OBSERVATION_GATE.md`.
Sportec: Bassek et al. 2025, CC BY 4.0. SkillCorner open data: MIT; the notice
is retained here. Full-match target confirmation requires the organisers'
permitted evaluation. Frozen v11 and preceding experimental bundles remain
unchanged.
