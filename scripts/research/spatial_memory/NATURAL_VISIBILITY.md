# External natural visibility study

The user's target is BAR–ATH spell F1 **above 0.740**. All development must use
external data. Neither an external result nor another public-sample score proves
that full-match target. The previous temporal-fusion candidate remains frozen.

The visibility profile is fitted exclusively to the 18 existing SkillCorner
training matches; the two weak-label validation matches are excluded. It uses
native and actually detected coordinates, with no possession labels. The pinned
raw source is SkillCorner's MIT-licensed open data. Complete bracketed ball gaps,
player fragments and context-conditioned loss rates are measured in seconds.
The fit covers 1,063,756 native 10 Hz frames and 3,361 complete ball gaps; the
missing-ball fraction is 0.400768 and detected players average about 4.72 per
team. These are **external source statistics**, not BAR–ATH measurements.

`possession/external_visibility.py` applies this profile to external DFL
coordinates. A second augmentation hides or moves the parked training-source
ball along a nearby player's observed trajectory, without possession labels.
It prevents the learner from depending exclusively on the source's exact
stationary-ball encoding. These augmented coordinates never replace supplied
match observations. They are synthetic views, not naturally occluded truth.

`train_natural_visibility.py` declares its protocol before preparing any new
development view. It uses the same six external training fixtures and the
previously used development fixture J03WPY. A temporal model starts from frozen
external run 13, adds spatial history and receiver/lane features, and adapts on
native and externally degraded views. A native-score guard and the unchanged
comparison at zero mixture strength protect against a degraded candidate being
selected merely for its synthetic-view result. Development selection is
exploratory; J03WPY is not an independent test. An I/O audit hook denies access
to supplied assets, predictions and public ground truth throughout preparation,
training and selection. Cache/model/source hashes are checked on reuse.

Commands (local outputs are ignored):

```powershell
.\.venv\Scripts\python.exe scripts/research/spatial_memory/calibrate_external_visibility.py --out data/research/natural-visibility-20261114
.\.venv\Scripts\python.exe scripts/research/spatial_memory/train_natural_visibility.py --out data/research/natural-visibility-20261114 --stage prepare
.\.venv\Scripts\python.exe scripts/research/spatial_memory/train_natural_visibility.py --out data/research/natural-visibility-20261114 --stage train
```

## PathCRF audit

The official PathCRF repository is pinned at
`1b7c6f85a6b78c92e4543fde72ca2b86a933f997` (MPL-2.0). The dynamic masked
checkpoint and adapter source were downloaded locally; Git blob hashes and
SHA-256 hashes were verified. Loading must use `weights_only=True`.

Its fold-6 checkpoint trains on J03WPY; it therefore **cannot** be selected or
claimed independently validated using our current development fixture. Its
training dataset also excludes episode zero, short episodes and incomplete
player-coordinate windows. Its player/control/transit edge targets provide no
direct dead-ball classifier. Any adapter needs explicit coverage/target checks
and a valid external split before integration. Downloading an open model alone
does not make it suitable or establish an improvement.

The first frozen transfer audit used J03WR9, the checkpoint's source-declared
fold-6 test fixture (sorted seven-match file order). It fed player coordinates
only and decoded the published dynamic masked CRF, mapping previous-controller
nodes to teams. It used complete 10-second windows at 5 Hz, five-second stride,
and abstained on incomplete rosters, outside sources or tied overlapping votes.
All predictions were generated before the external metric summary was computed.
The checkpoint loaded strictly with `weights_only=True`.

On 85,316 external H/A truth frames, it covered 79,333 (92.987%) and achieved
93.511% team accuracy on those covered frames; counting abstentions as errors
gives 86.953%. The H/A-only spell F1 was 0.655294. The generic metric helper's
macro F1 includes a dead class that this audit never predicts, so that field
must not be presented as a fair H/A macro F1. No dead-ball performance was
measured, no mode or parameters were selected from these results, and the
adapter is not an exact reproduction of the paper's event-edge benchmark.
J03WR9 is also in our other models' training split; this is a transfer audit of
PathCRF alone, not an independent comparison or validation of a fused model.
No BAR–ATH inference or public scoring was performed for this audit. Receipts
and arrays remain ignored under `data/research/pathcrf-heldout-audit-20261114/`.

A separately frozen input-only coverage audit then applied the external camera
model to J03WR9 coordinates. Native input coverage was 97.469% / 96.068% across
the two halves; both masked and mixed views had **zero complete windows**.
Detected player counts in those views averaged 4.65–4.88 per team. This audit
used no possession labels or model predictions. The unmodified complete-roster
PathCRF adapter therefore cannot be deployed as a general camera-loss fallback.
Relaxing its gate to fill missing players would require a separately developed
and externally validated reconstruction/robust training method. No such gate
change was made. Receipts are under the ignored
`data/research/pathcrf-missingness-audit-20261114/` directory.

Focused tests cover missing-roster abstention, physical velocity units, gap
preservation, team mapping under pitch reflection, external-only visibility
fitting, reproducible degradation and rejection of target-derived calibration,
and the native-performance selection guard. The expanded temporal model also
passed a direct frozen-weight check: appending zero-initialized memory columns
preserves the original robustness model's state and boundary logits.

## Frozen candidate comparison

`compare_natural_candidates.py --freeze-only` records the final comparison
protocol before adaptation training results are available. It compares the
frozen parent, the prior temporal-fusion candidate and the new externally
selected adaptation on identical native/masked/mixed development views. It
does not retune any of their weights, mixtures, gates or decoder costs. The
prior candidate's full-rate memory gate is reconstructed exactly; repeating a
half-rate gate would be an approximation and is deliberately avoided. Cached
memory and selected development scores must reproduce exactly. Ties retain
the prior candidate; the parent native-performance guard still applies.

The SkillCorner metadata overlap audit confirms all twenty downloaded matches
are 2024/25 A-League fixtures; none is Barcelona–Athletic. The profile still
uses only the eighteen designated training fixtures.

## Additional external data review

[SoccerTrack v2](https://huggingface.co/datasets/atomscott/soccertrack-v2)
is a possible separate-domain resource: ten university matches, player
positions and player-linked action events, with CC BY 4.0 data rights. The
current canonical release requires the user's acceptance of Hugging Face
access conditions. No conditions were accepted and no gated files were
downloaded. Its auxiliary ball track is reconstructed by interpolating between
annotated event anchors. It must not be treated as independent measured ball
tracking in a benchmark whose possession targets are derived from those same
events. The action labels are also not dense HAD truth. A valid use would need
a declared partial-label or ball-free protocol and verified time/team mapping.
The publisher's older GSR format notes document schema and second-half timing
discrepancies, so the actual pinned release must be audited before ingestion.
This review did not change model parameters or open any supplied-match labels.

## Completed adaptation result and reproduction

All 42 half/view feature files were prepared with input/source checksums. The
predeclared 50-epoch run completed. It selected epoch 20 at a 25% mixture with
the parent. On the new native/masked/mixed development views, spell F1 was
0.847343 / 0.415979 / 0.443593, mean **0.568972**. The same-view parent mean was
0.545693 and prior temporal-fusion mean was 0.546466. Native F1 declined from
0.857005 by 0.009662, within the predeclared 0.01 guard. This is a modest
exploratory development gain, not the requested BAR–ATH improvement.

A second protocol was frozen before final training results: retain the prior
two-second long-gap gate, 75% expert strength, parent boundary predictions and
decoder, replacing only the expert. The gate uses the new expert's declared
half-rate input clock. Both the development-selected and fixed final-epoch
checkpoints were compared. Their means were 0.563398 and 0.568677, respectively;
neither beat the preceding frozen winner, so the global 25% adaptation remained
selected. No candidate or parameter was selected from BAR–ATH data or scores.

The first exploratory export is excluded from the reproduction evidence
because additional exporter checks were edited during that run's lifetime.
After those checks were frozen, two fresh exports and a third export from
`models/coreorigin-natural-visibility-v1` produced identical raw and final CSV
bytes. The final file has 199,319 rows and SHA-256
`e5bd12c22fbeecf85ceefe5de13161e00a154c25e12711e1ef4b29368f893e86`.
All final runs verified their inference guards and recorded zero unexpected
forbidden reads. The unchanged official format validator passed. The inference
source/model freeze, manifests, validator output and complete verification are
under ignored `data/research/natural-visibility-20261114/`. Frozen v11's model,
CSV and the official scoring code match their existing checksum receipts.
No new public possession scoring was performed and no provided video was used
for training or human prediction checks.

Seventeen focused tests passed across visibility fitting, spatial memory,
coordinate input adaptation, the native-performance guard and expert
replacement. Direct checks also verified zero-extension initialization
equivalence and full-size CUDA forward/backward/optimizer execution. The model
bundle carries external source attribution and the SkillCorner MIT notice;
source match data and final predictions remain ignored.

The main candidate is
`data/research/natural-visibility-20261114/candidate-packaged/CoreOrigin.csv`.
It is an externally selected experiment. BAR–ATH spell F1 above 0.740 is **not
verified**. `SUBMISSION.md` states there is no pre-deadline full-match feedback;
only the organisers' permitted full-match evaluation can establish that result.

Primary sources: [PathCRF](https://github.com/hyunsungkim-ds/pathcrf),
[paper](https://arxiv.org/abs/2602.12080),
[SkillCorner](https://github.com/SkillCorner/opendata),
[Sportec dataset](https://www.nature.com/articles/s41597-025-04505-y).
