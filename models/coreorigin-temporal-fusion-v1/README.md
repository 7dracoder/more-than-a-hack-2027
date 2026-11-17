# Frozen external temporal-fusion experiment

`selection.json` is the exact external-only selection receipt. It references
the unchanged neural checkpoints in `models/coreorigin-video-v1` and
`models/coreorigin-robust-v13`. `model_manifest.json` pins their weights/protocols
and this selection. No BAR-ATH data, predictions, labels or pose crops are
included here.

The selected mixture uses 75% robustness-model state probability in ball-missing
runs lasting at least two seconds, retaining the parent elsewhere and retaining
its boundary head/decoder everywhere. It uses track-derived missingness and
temporal context; it does not measure players' cognitive memory, head orientation,
gaze or peripheral vision.

External J03WPY development spell F1: native **0.85700** (unchanged), synthetic
gaps **0.68057** versus parent **0.66404**. This is exploratory development,
not independent-test evidence or a BAR-ATH improvement claim. The static
spatial-memory learner ablations were rejected.

Source, negative evidence, external data attribution, pose investigation and
reproduction commands are in `scripts/research/spatial_memory/README.md`.
Use `requirements-memory.txt` and the recorded CUDA execution protocol for the
same-environment reproduction. Competition outputs remain under ignored `data/`.

After selection froze, one unchanged official public-window pipeline check
reported **0.54000 spell F1 / 0.69035 macro F1**, with spell-F1 5th–95th bootstrap
range **0.39–0.68** over 7,500 frames / 17 possessions. No model/parameter choice
followed that check. The result does not establish an improvement over historical
v11 (0.54054 spell F1 / 0.74187 macro F1), private full-match accuracy or the 0.80
target. Three same-environment CUDA exports are byte-identical and format-valid.
