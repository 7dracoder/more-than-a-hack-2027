# Frozen real-broadcast adaptation

This is an externally trained experimental candidate. **BAR–ATH spell F1 above
0.740 is not verified.** The joint selection criterion's value of 0.74449 is
not a BAR–ATH score or a spell-F1 result. No supplied-match possession labels or
video were used for training, calibration or model selection.

The selected checkpoint is epoch 30 of a fixed 40-epoch mixed-domain run,
initialized from the preceding external camera-loss expert. It uses six strong
Sportec fixtures and eighteen real detected-only SkillCorner fixtures. Weak
examples supervise conditional H/A only; unknown targets are ignored and the
D logit is excluded. Strong external data retain HAD/change-point supervision.
Normalization, observation-loss gate, parent boundaries and decoder stay fixed.

On the reused external strong development views, spell F1 is 0.857005 native,
0.532105 masked and 0.617817 mixed, mean **0.668976** versus initial 0.668571.
Detected-only weak H/A agreement on the two development fixtures is **0.822591**
versus 0.735982; on missing-ball ticks, **0.798366** versus 0.610100. These are
provider weak labels, not independent dense HAD truth. The fixtures had been
opened previously. All declared cross-domain guards passed and an independent
execution reproduced the selected metrics and every input checksum.

Use the parent bundle `models/coreorigin-video-v1` and the pinned CUDA environment
in `requirements-video-ball.txt`. External training data are not needed for
inference. With the verified coordinate artifact:

```powershell
.\.venv\Scripts\python.exe scripts/research/spatial_memory/export_observation_candidate.py --model models/coreorigin-real-broadcast-v3 --ball data/research/dual-feed-20261113-v2/ball_hypotheses.npy --out data/research/real-broadcast-adaptation-20261114/run-02/candidate-packaged --device cuda
.\.venv\Scripts\python.exe evaluation/validate_submission.py data/research/real-broadcast-adaptation-20261114/run-02/candidate-packaged/CoreOrigin.csv
```

Expected 199,319-row CSV SHA-256:
`ed4c84360e265330d2f1a2c277d1446477c167a907cb562b58e29d810361e068`.
The exporter blocks training and possession-truth reads and verifies frozen
sources, weights and inputs throughout inference. Format validation establishes
file correctness, not possession accuracy. No new public possession scoring is
part of this workflow. Supplied-match data/predictions are not in this bundle.

Coordinate reconstruction from the supplied bundles uses the unchanged commands
in `models/coreorigin-natural-visibility-v1/README.md`. The technical description
is in `TECHNICAL.md`; research and preceding experiments are in
`scripts/research/spatial_memory/OBSERVATION_GATE.md`. Sportec is CC BY 4.0;
SkillCorner open data is MIT, with its notice retained here. Frozen v11 and all
preceding experimental bundles remain unchanged. Full-match confirmation requires
the organisers' permitted evaluation.
