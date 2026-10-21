# CoreOrigin / BlindSpot audit revision v0.3.1

Audited development revision, 21 October 2026. Not a submitted entry and not evidence of private-match rank. The frozen v0.3 source, candidate and cloud proof remain intact.

## Findings and repairs

| Finding | Repair and evidence |
|---|---|
| Team evidence during dead-ball intervals could influence the next live run. | Infer ownership independently in each algorithmically decoded live run; regression test varies dead-ball evidence. |
| Imputed zero motion could conceal missing measurements. | Add nine observation features (68 total); unavailable motion gives neutral movement-regime evidence. |
| Base feature caches lacked source/content validation. | Fingerprint extraction code and inputs; verify cache SHA256 before reuse, rebuild corrupted/stale caches. |
| Review queries used posterior argmax boundaries, which can differ from the exported sequence. | Select boundary candidates from the actual decoded CSV sequence. |
| Viewer notes could be lost on navigation or mixed between candidates. | Autosave on input, candidate-hash storage keys, malformed-storage recovery and portable JSON export. |
| Rounded playback time and five-Hz labels could display the wrong frame. | Per-frame labels/support, floor-based timing, frame slider and stepping, stale callback guard. Pitch positions remain explicitly sampled at five Hz. |
| The overview displayed sampled values as if they were exact-frame values. | Label the selected frame and preceding one-second sample separately. |
| Report wording implied knowledge of the full-match winner. | Replace it with qualified language; no private truth or ranking is known. |

Owner support in the revised decoder is conditional on the MAP live/dead segmentation. Combined support is a model-relative review signal, not an exact joint posterior or calibrated confidence. Alternative histories sample their own live/dead segmentation before sampling live-run owners.

## Measured comparison

Four whole-run folds (pooled counts):

| Method | Spell F1 | Macro F1 | Dead F1 |
|---|---:|---:|---:|
| frozen_v3 | 0.4457 | 0.7484 | 0.6619 |
| owner_reset_only | 0.4457 | 0.7484 | 0.6619 |
| audit_revision | 0.4457 | 0.7484 | 0.6619 |
| smooth125 | 0.3889 | 0.5812 | 0.1464 |

Four equal-duration folds, chosen without possession-state boundaries, with a 300-frame training embargo:

| Method | Spell F1 | Macro F1 | Dead F1 |
|---|---:|---:|---:|
| frozen_v3_method | 0.6100 | 0.7392 | 0.5974 |
| audit_revision | 0.6200 | 0.7401 | 0.5945 |
| smooth125 | 0.4513 | 0.5812 | 0.1464 |

The revision improves fixed-time spell F1 by 0.0100; dead-ball F1 decreases slightly (0.5974 to 0.5945). Original whole-run scores are unchanged. This is a mixed, small development result, not a universal accuracy improvement. Both methods were retrained separately for each fixed-time fold. The four original v3 fold outputs were reproduced byte-for-byte before the comparison.

Both protocols reuse the same previously inspected 7,500-frame / 17-possession public example. Their values must not be compared across protocols as a gain. No parameter sweep was performed. The exported model is fitted on all public labels; its training-window score is not validation. Compared with v3, 915 full-match labels change; private correctness of those changes is unknown.

## Reproduce

```bash
pip install -r requirements-v3.txt
python -m unittest discover -s tests -v
python scripts/run_champion.py --config configs/champion_v031.json --out data/experiments/blindspot-v031 --resume
python evaluation/validate_submission.py data/experiments/blindspot-v031/candidate/CoreOrigin.csv
python scripts/audit_revision.py
pip install -r requirements-v3-video.txt
python scripts/build_candidate_review.py --experiment data/experiments/blindspot-v031
```

The audit comparison requires the frozen v3 outputs/models at data/experiments/blindspot-v3. Those match-derived files are deliberately absent from the source bundle. A fresh reproduction can first produce that reference with the v0.3 source ZIP and configs/champion_v3.json. Audit artifacts are local under data/experiments/blindspot-v031/audit.

## Verification and release scope

33 local tests passed, including the generated viewer's JavaScript logic under a Node DOM test double. Real browser interaction with local file URLs remains unverified because the existing browser policy blocks access; no alternate hosting route was used. Video decoding and source checks are separate from browser interaction.

v0.3 remains the frozen, previously cloud-verified reference. v0.3.1 is the current audited development revision, with its own output directory and source bundle. Cloud reproduction status belongs to each release's verification JSON and must not be inherited from v0.3. Final model selection remains provisional.

The next meaningful model experiment is an independent permitted validation set, followed by a fixed-budget reviewer study and a boundary-error breakdown. More tuning on these same five minutes would not establish generalization. Video contact/control recognition and broadcast fusion remain unimplemented research directions.
