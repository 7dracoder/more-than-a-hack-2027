# External-only spatial-memory and orientation study

This work follows `RULES_CLARIFICATION.md`: train/select on external matches
only; never use BAR-ATH possession truth/video for fitting or model selection,
never inspect prediction overlays, and keep supplied data and predictions local.
The user's request motivated longer spatial context, no-look/through-pass
options and an investigation of head orientation. It did not supply labels.

## Current status

The retained handoff is `models/coreorigin-continuity-v4`, documented in
the root `CoreOrigin.md`. BAR-ATH full-match spell F1 above 0.740 remains
unverified; the official CSV validator checks format, not label accuracy.
The studies below include historical experiments and are not interchangeable
with the selected handoff or evidence for competition performance.

Recent external investigations:

- [Whole-half frozen boundary classifier](FROZEN_BOUNDARY_TRANSFER.md).
- [Equal-run weighting and short-possession errors](SPELL_BALANCED_ADAPTATION.md).
- [Adapted whistle classifier and fresh external test](WHISTLE_TRANSFER.md).
- [Audio feasibility and source review](AUDIO_FEASIBILITY.md).
- [Receiver context with roster-invariant entropy](RECEIVER_CONTEXT_V2.md).

Further training and candidate selection must stay external. Historical public
scorer commands at the end of this document record completed checks; they must
not be repeated to select or tune alternatives.

## What was implemented

`possession/spatial_memory.py` builds 128 coordinate-only features with explicit
missing flags: recent team-set positions over 1/2/4/8 seconds, past/future ball
anchors for offline reconstruction, receiver clearance and passing lanes in all
directions. It never equates velocity with gaze or excludes receivers using an
assumed facing cone. It uses team sets rather than persistent slot identities;
each call is one half, so memory cannot cross halftime.

Three matched XGBoost studies were completed using six external training
matches and the existing J03WPY development match, in native and synthetic-gap
views: parent-probability correction with/without memory; independent geometry
versus memory learners; and full-kinematics learners with/without memory. All
six nonzero learned corrections failed the declared mean-development objective
and native guard. They remain experimental and are not deployed. We do not
claim the new static memory features improved possession performance.

The subsequent temporal comparison reuses two unchanged externally trained
models. The half-rate robustness model's published external scores reproduced
exactly, and its 664-feature prefix was checked bit-for-bit against the
full-rate parent's corresponding native/corrupt features. Its state emissions
are repeated onto the full-rate clock; the parent's boundary head and decoder
are preserved.

A small external grid selected weight **0.75** for the robustness model inside
missing-ball runs lasting at least **2 seconds**, with the current parent used
outside those gaps. On J03WPY, clean spell F1 stays **0.8570048309** and
synthetic-gap spell F1 changes **0.6640419948 -> 0.6805736636**. This is a modest
development point estimate, with no independent-match confirmation. It is not
a BAR-ATH score or proof of an improvement on the competition match.

The external I/O guard blocks competition assets, old competition predictions
and public ground truth during preparation/training/selection. Caches are
content-hashed; both supplied-match inputs and labelled targets are absent from
the new feature APIs. Selection settings are frozen before competition export.

## Head/body orientation investigation

The pinned public SkillCorner body-pose sample and the complete 1925299 archive
were downloaded and checked against publisher SHA256. The latter contains
153,252 frames and 1,540,713 detected player-frames with pose. The fixed
uncertainty gate accepted 1,430,552 unsigned shoulder-axis observations and zero
head directions. In the short sample, predicted propagated head-joint error is
of the same order as the nose-to-ear-midpoint vector. The gate was not relaxed
to manufacture a usable signal.

`possession/pose_orientation.py` and `audit_pose.py` retain this investigation
separately. These are provider-predicted joints/errors, not human head-angle
truth or orientation-accuracy validation. An unsigned shoulder axis cannot
resolve forward/backward facing. Head orientation itself would still not measure
eye gaze, attention, peripheral vision or intent. No head/body/gaze feature is
deployed in BAR-ATH inference.

SkillCorner pose is 25 fps and its tracking is 10 fps; frame numbers cannot be
joined without conversion. Credit SkillCorner, MIT-licensed data, pinned commit
`4340d274572876239c154c90bc507a9b3250a656`.
Sources: [provider documentation](https://github.com/SkillCorner/opendata/blob/4340d274572876239c154c90bc507a9b3250a656/data/bodypose/README.md),
[dataset card](https://huggingface.co/datasets/SkillCorner/opendata-bodypose).
The official [6DRepNet implementation](https://github.com/thohemp/6DRepNet) was
investigated as a face-image head-pose option, but was not deployed without a
validated football-video head-orientation pipeline.

## Reproduction

Use the existing CUDA environment; `requirements-memory.txt` records its
dependencies. All generated data below is ignored. The frozen inference
selection is also copied to `models/coreorigin-temporal-fusion-v1/selection.json`
so inference does not depend on a private training-run path.

```powershell
.\.venv\Scripts\python.exe scripts/research/spatial_memory/run_external.py --out data/research/spatial-memory-20261113-v2 --stage prepare --device cuda
.\.venv\Scripts\python.exe scripts/research/spatial_memory/run_external.py --out data/research/spatial-memory-20261113-v2 --stage fit --device cuda
.\.venv\Scripts\python.exe scripts/research/spatial_memory/fit_independent.py --study data/research/spatial-memory-20261113-v2 --device cuda
.\.venv\Scripts\python.exe scripts/research/spatial_memory/fit_full_context.py --study data/research/spatial-memory-20261113-v2 --device cuda
.\.venv\Scripts\python.exe scripts/research/spatial_memory/select_temporal_fusion.py --study data/research/spatial-memory-20261113-v2 --device cuda
.\.venv\Scripts\python.exe scripts/research/spatial_memory/export_candidate.py --study models/coreorigin-temporal-fusion-v1 --ball data/research/dual-feed-20261113-v2/ball_hypotheses.npy --out data/research/spatial-memory-20261113-v2/temporal_fusion/candidate --device cuda
.\.venv\Scripts\python.exe scripts/research/spatial_memory/audit_pose.py --match 1925299 --out data/research/spatial-memory-pose-full.json
```

Study commands refuse to silently refit an already completed selection. Use a
fresh study directory for a deliberate external-only reproduction. The original
six-run XGBoost studies are negative evidence; reproducing them is optional for
inference. The two neural checkpoints, fixed external selection, dual-feed
coordinate artifact and pinned environment suffice to regenerate the CSV.

25 focused geometry, memory, pose, fusion and possession-rule tests passed.
No existing v11 weights or candidate were modified.

## Frozen BAR-ATH pipeline check

The final CUDA CSV has 199,319 rows and passes the official format validator.
Three exports (including one from the packaged selection with access to private
training folders and possession truth denied) give identical raw/final CSVs.
Final SHA256:
`ce9c8ba2c2fa07ac4c7f8e22302cd9de64f3a54d4ee6338f4dc83745163bb9da`.
Verification is retained under
`data/research/spatial-memory-20261113-v2/temporal_fusion/verification.json`.

After external selection and byte-level reproduction were complete, the user
authorized an official check. Exactly one frozen final CSV was staged in an
isolated scoring folder, with no raw/control alternatives. The freeze receipt
pins the CSV, external selection and unchanged official scorer before public
labels are opened. The scorer ran once with 1,000 bootstrap draws and exited 0.

On the public 7,500-frame / 17-possession sample it reports **spell F1 0.54000**
and **macro F1 0.6903488685**. The 5th–95th percentile bootstrap ranges are
**0.39–0.68** for spell F1 and **0.518093–0.806314** for macro F1. The prior
two-feed candidate's public spell F1 was 0.52000. Historical frozen v11's
documented public result is 0.54054 spell F1 / 0.74187 macro F1; this update does
not establish an improvement over that result. The wide uncertainty and small
public window do not establish private full-match performance or attainment of
0.80.

No per-transition reports or video/prediction overlays were inspected. No
training, threshold/mixture change or candidate selection followed the public
check. Both the algorithm and external selection stayed frozen. The exact
freeze, scorer outputs and summary are private under
`data/research/spatial-memory-20261113-v2/temporal_fusion/public_pipeline_check/`.
The scorer command was:

```powershell
.\.venv\Scripts\python.exe evaluation/run_evaluation.py --gt public --results data/research/spatial-memory-20261113-v2/temporal_fusion/public_pipeline_check/submission --out data/research/spatial-memory-20261113-v2/temporal_fusion/public_pipeline_check/official --bootstrap 1000 --no-detail
```

This command records the completed pipeline check, not an invitation to rescore
alternatives for selection. Further development must use external data only.
