# Causal conditional receiver-choice auxiliary model

This stage uses only the pinned ELASTIC annotations for existing external
TRAIN fixtures WMX/WN1. Its 1,468 distinct pass/cross releases produce 2,936
examples: native coordinates and a label-independent recent-history mask.
Only the two seconds ending at release enter candidate features. Inputs are
relative positions, distance, team relationship, observation age, backward
velocity and alignment with actor movement. These are coordinate proxies,
not head direction, peripheral vision or inferred mental intent.

The task is actual receiving-player choice conditional on a known passer at
release. Actor identity supplies a conditional origin; it is not an available
ground-truth possession signal during deployment. Future integration must infer
or marginalize the actor. The future receiver/time are targets and eligibility
metadata, never input columns. Missing candidates retain only causal history;
an unavailable receiver maps to an explicit extra class. The mask does not
select a replacement visible player. Ambiguous receiver assignments, self
receivers, nonplayer targets and invalid flight intervals are excluded under
the frozen preparation policy. IDs/frame counters are stored outside features.

Three feature tests check stale history, future-coordinate exclusion and
candidate permutation equivariance. Input/source hashes and raw clocks are
frozen in the preparation receipt. The existing provider possession labels
remain unchanged and are not read by the feature extractor.

A small shared candidate MLP trains on WMX, selects among 100 fixed epochs on
WN1 and retains epoch 13. Native top-1/top-3 accuracy is 0.3750/0.5921; masked
accuracy is 0.3701/0.5987. A nearest-teammate baseline gives top-1
0.2319/0.2352. A separate functional forward reconstruction exactly reproduces
checkpoint logits and scores. A 1,000-draw paired bootstrap groups both views
by release: mean top-1 difference 0.13898, 5th-95th percentile interval
[0.10280, 0.17270]. This is conditional on epoch selection on one auxiliary
development fixture, without search or match-level uncertainty adjustment.
It is not independent possession evaluation and is not BAR-ATH spell F1.

The model is not integrated into possession inference. The selected v4 model,
CSV and PDF remain unchanged. The next required experiment must measure whether
receiver-derived signals improve external possession spell F1 before a new
submission is exported. The full-match target still needs organiser scoring.

```powershell
.\.venv\Scripts\python.exe scripts/research/spatial_memory/prepare_receiver_choices.py --out data/research/receiver-choices-20261114
.\.venv\Scripts\python.exe scripts/research/spatial_memory/train_receiver_choices.py --data data/research/receiver-choices-20261114 --out data/research/receiver-choice-model-20261114/run-01
.\.venv\Scripts\python.exe scripts/research/spatial_memory/verify_receiver_choices.py --run data/research/receiver-choice-model-20261114/run-01
```

All raw data, training examples, weights and receipts remain local under ignored
`data/` paths. New annotations carry CC BY 4.0 attribution documented in
[the acquisition audit](ELASTIC_TRAINING_ANNOTATIONS.md).
