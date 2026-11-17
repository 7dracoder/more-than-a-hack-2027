# External event/reception annotation acquisition

[ELASTIC](https://github.com/hyunsungkim-ds/elastic#license-and-attribution)
provides corrected Sportec events and manually re-annotated occurrence/reception
times. Its code is MPL-2.0 and benchmark data CC BY 4.0; credit Kim et al.,
Bassek et al. and Deutsche Fussball Liga. Only existing TRAIN fixtures J03WMX
and J03WN1 were acquired at commit
`e245224aad7450961eca40954ae2bc85e1e6e696`, with original notices and hashes.
No J03WPY benchmark annotations were downloaded. This adds no independent match.

The local audit finds 2,359 events and 1,936 non-null receive-frame fields.
Those include 109 out/goal destinations, leaving 1,827 player receptions.
Only `home_<integer>` and `away_<integer>` identifiers supply team labels;
provider person IDs do not directly match these annotation identifiers.
Do not assume a per-player identity mapping.

All event and accepted reception frames map to the existing 25 Hz TRAIN clocks.
Seventeen frames contain conflicting teams among intentional-action/reception
anchors. Intentional actors disagree with dense provider labels at 136 of
1,789 points; receptions disagree at 272 of 1,827 points. This does not establish
which annotation is correct or define dense HAD truth. Timing, action semantics
and conflicts need resolution before any training adapter. No labels were
overwritten, no targets generated, and these annotations do not enter the
current trained model or its selection.

The follow-up TRAIN-only timing audit uses fixed radii of 1, 5 and 25 frames,
without selecting an offset. Of 136 mismatched intentional-action anchors,
85 agree within one frame and another 30 within five. Of 272 mismatched
receptions, 144 agree within one frame and another 71 within five. Thus 330
of 408 disagreements have nearby agreement within 0.2 seconds. Forty-eight
have no agreement within one second. These counts describe label neighborhoods;
they do not establish a global lag, which source is correct, or secure control.
No labels were changed. Direct dense-label replacement is not supported by
this diagnostic. A receiver-selection auxiliary task would first require
verification of the annotation player namespace against the source adapter.

The completed timing receipt is at
`data/research/elastic-timing-audit-20261114/`. Its fixed-radius logic is checked
at half boundaries to prevent wrapping into the opposite end of a period.

The subsequent identity audit inspected the pinned source's
`SportecData.load_lineup_data`: identifiers combine home/away role and integer
shirt number. Metadata provides the corresponding provider PersonId. General
home/guest IDs and team roles are cross-checked, and ambiguous shirt numbers
are rejected. No nearest-player heuristic assigns identities.

All 1,827 receiver identifiers and 1,789 intentional actors resolve uniquely.
At their event frames, all actors and 1,826 receivers have finite tracking
coordinates. Receiver-to-ball median distances are 0.425 m (WMX) and 0.489 m
(WN1), with 90th percentiles 0.912 m and 0.959 m. These are consistency
diagnostics, not fitted acceptance thresholds or proof of secure possession.
Receipts, exact raw/source hashes and summaries are at
`data/research/elastic-identity-audit-20261114/`. Only the two TRAIN matches were
read. This resolves the namespace prerequisite for a receiver-choice dataset;
no new supervision has yet been fitted and no selected prediction changed.

Acquisition and audit receipts are local/ignored at
`data/external/elastic-benchmark-e245224aad74/` and
`data/research/elastic-training-audit-20261114/`. Exact input allowlists block
other data during the audit. Both completed with zero unexpected accesses.

```powershell
.\.venv\Scripts\python.exe scripts/research/spatial_memory/prepare_elastic_training.py --out data/external/elastic-benchmark-e245224aad74
.\.venv\Scripts\python.exe scripts/research/spatial_memory/audit_elastic_training.py --out data/research/elastic-training-audit-20261114
```
