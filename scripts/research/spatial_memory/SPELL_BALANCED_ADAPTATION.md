# External short-possession diagnostic and equal-run adaptation

The frozen v4 error audit reproduces its native/masked/mixed spell scores
exactly, without changing predictions or reading supplied-match assets.
On the reused external development fixture there are 421 labelled possessions,
including 124 shorter than two seconds. At IoU 0.5, v4 matches 79 of those short
possessions with native observations, 26 with the masked view and 29 with the
mixed view. For possessions lasting at least five seconds, the corresponding
counts are 165/168, 126/168 and 143/168.

The masked view has 218 unmatched possessions at IoU 0.5. Of those, 84 overlap
a same-team prediction that also spans another same-team ground-truth spell;
99 have no overlapping same-team prediction, and 102 are mostly predicted as
the opponent. These descriptions overlap. They do not establish additive
causes or prove that changing the continuity penalty will recover the spells.
Only five missed spells overlap multiple same-team prediction fragments.
The prediction count is 304 against 421 labelled possessions, so indiscriminate
additional smoothing is not motivated by this diagnostic.

The audit's sources, input checksums, per-spell rows and summary are under
ignored `data/research/continuity-error-audit-20261115/`. It is exploratory
development analysis on one previously opened external fixture, not an
independent test or BAR-ATH score.

## Controlled experiment

The prior strong loss uses inverse-square-root state-run length weighting.
Test inverse-duration weighting instead: each fully labelled H/A/D run of at
least five training frames receives equal total frame-loss weight within its
half. A five-frame floor limits very short annotation runs; unknown labels
retain zero weight. Recompute these weights on strong TRAIN halves only.
This balances labelled state runs, including stoppages, not just H/A spells.

Everything else matches the occurrence-safe expanded-geometry control:
initial v3 weights, zero-initialized added projection columns, unchanged old
normalization, externally fitted new-column normalization, seed, 30 epochs,
80 batches per epoch, six strong and eighteen weak training fixtures,
occurrence-preserving team exchange, weak conditional H/A loss, boundary
auxiliary loss, observation gate, continuity decoder and development guards.
No new parameters are fit to competition data. The initial v4 fallback stays
eligible. A higher development point score alone is insufficient for promotion;
selected-model reproduction and a paired spell uncertainty check are required.
That check remains conditional on reused development data.

Config: `configs/spell_balanced_geometry_adaptation.json`.
Runner: `train_spell_balanced_geometry.py`.
Independent verifier: `verify_spell_balanced_geometry.py`.
Run: `data/research/spell-balanced-adaptation-20261115/run-01/`.
The verifier reconstructs weighting directly from run boundaries, checks the
unknown mask, recomputes normalization from TRAIN features and reproduces the
selected strong/weak scores. Training status and final conclusions belong in
the run receipts; this document does not assert an improvement in advance.

## Completed result

All 30 epochs completed. The best trained checkpoint was epoch 25, with mean
development spell F1 0.673425; it failed the weak missing-ball guard. The best
eligible trained checkpoint was epoch 15, mean 0.671172. Both trail v4's
0.678165, so selection keeps the initial fallback. Independent verification
reconstructed the training weights, confirmed unknown masks and old
normalization, recomputed added normalization from TRAIN inputs and reproduced
all selected strong/weak scores. It also confirmed the fallback weights exactly
equal the zero-extended initial model. No uncertainty resampling or competition
export is needed to reject trained candidates that already lose on the point
objective. Eight focused geometry/augmentation tests passed.

The exact retained CSV, frozen v11 weights/CSV and official scoring sources
retain their recorded hashes. No supplied-match scoring was performed. This
negative result argues against this particular stronger run-weighting change;
it does not establish that short possessions can be recovered without further
information, or that the BAR-ATH target is achieved.

The audio detector remains separate. Available SoccerNet-Echoes files are ASR
text. Public metadata for
[WhissleAI/speech-soccer-broadcast](https://huggingface.co/datasets/WhissleAI/speech-soccer-broadcast)
was then pinned to `c686459b425888ce65c8db9cec038b443784d831` and inspected
without downloading audio. Its 327 JSONL rows each contain only
`audio_filepath`, `duration` and `text`; every duration is 30 seconds, for
9,810 seconds total. There are no explicit whistle timestamps or possession
states in that schema. The 31-byte card only declares Apache-2.0, with no
descriptive documentation. This is insufficient for the proposed validation.
Metadata and hashes are under ignored
`data/research/football-audio-source-review-20261115/`.
No audio integration is justified by the cross-sport whistle result.
