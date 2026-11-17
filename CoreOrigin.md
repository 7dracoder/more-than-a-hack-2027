# CoreOrigin: external supervision with observation-aware temporal memory

## Scope and current evidence

The algorithm recovers H/A/D labels offline from the supplied match inputs.
The target is BAR-ATH spell F1 above 0.740. That result has not been established;
the organisers' full-match ground truth is unavailable. The current file passes
the official format validator. No new BAR-ATH possession scoring was performed
during this study, and no supplied video/prediction overlay was inspected.

The frozen mixed-domain study selected epoch 30 of 40; the selected weights are
unchanged here. A separate external continuity grid selected penalty 8. Strong
J03WPY spell F1 is 0.857005 native, 0.544552 with synthetic camera loss, and
0.632938 with camera loss and removal of the source's parked-ball cue: mean
0.678165. Weak H/A broadcast agreement is 82.581% overall and 79.985% when the
ball is missing. These are different metrics with different target quality.
They establish neither BAR-ATH F1 nor independent-test performance. All external
development fixtures were opened previously. No public BAR-ATH score is used
to select this model or to decide that the numerical goal has been achieved.

## Supplied inputs

Original tactical tracking supplies player/ball coordinates and team membership.
Its original finite ball observations remain exact. Existing COCO-pretrained
YOLO11m inference supplies tactical-video hypotheses for missing observations;
detector confidence and motion filters are fitted externally. Provided tactical
homography projects those hypotheses onto the pitch. This is local inference,
with no detector training on the supplied match.

Broadcast tracking is synchronized from team-aware player geometry with fixed
acceptance rules. Accepted proposals precede tactical-video fallback hypotheses.
Ambiguous or inconsistent points remain missing. Broadcast pixels are not used:
the supplied video reports one fewer frame than its tracking/homography records.
No implicit one-frame pixel correction is assumed.

Provided automatic Smart Tagging contributes only the established pre-kickoff
dead-ball rule. Phase intervals are not copied into possession labels. Neither
provided data nor public/private possession truth calibrate the model, decoder,
fusion weights or acceptance settings. All raw inputs, coordinate artifacts and
predictions remain under ignored local `data/` directories.

## Label sources and rights

Strong supervision uses six fixtures from the [Sportec Open DFL dataset](https://www.nature.com/articles/s41597-025-04505-y)
(Bassek et al., 2025, CC BY 4.0): J03WMX, J03WN1, J03WOH, J03WOY, J03WQQ and
J03WR9. The adapter maps vendor BallStatus/BallPossession fields to HAD. The
team creates no manual match labels. These fixtures contain 856,433 unique
25 Hz labelled frames: H 242,739; A 228,906; D 384,788. Native/masked/mixed views
share labels; augmentation does not create independent annotations. J03WPY is
the repeatedly used strong development fixture.

Weak supervision uses eighteen [SkillCorner open-data](https://github.com/SkillCorner/opendata)
A-League 2024/25 fixtures, under MIT. Their exact identifiers and pinned source
hashes are in the configuration/receipts. Detected-only coordinates retain
`is_detected=True`; interpolation uses adjacent finite observations only.
The training representation contains 1,329,677 ticks at 12.5 Hz, with 715,315
H/A weak targets and 614,362 ignored X targets. These are resampled provider
targets, not that many independent manual annotations. The provider's group
field supplies H/A; null/unknown never becomes D. At fractional source ticks,
adjacent labels must agree. Carrier visibility is not required, so missing-ball
examples remain eligible. The separate fixtures 2016236 and 2017461 are used
for weak development and excluded from training/profile fitting.

The inherited detector is COCO-trained and licensed AGPL-3.0; confidence evidence
uses the existing externally calibrated, attributed COCO subset. Its weight hash,
image-license attribution and calibration JSONs are in the parent bundle.
Confidence calibration used 112 COCO 2017 images, with 28 reserved for external
validation; the licensed selection contains 35 sports-ball and 105 negative
images. COCO annotations are CC BY 4.0, and the image-specific CC BY/CC BY-SA
notices are retained. This calibrates still-image detection, not possession.
The inherited representation also used provider H/A and carrier-rank weak
supervision from the same eighteen SkillCorner training fixtures, with unknown
ownership kept unknown; this adds no independent labelled matches. The
audited MPL-2.0 PathCRF model is not deployed because its complete-roster adapter
accepted no externally occluded windows. SkillCorner body-pose data were audited,
but no sufficiently reliable head direction was accepted. No head/gaze estimate
is used. No human-produced data about the supplied game are introduced.

## Temporal recovery and possession rules

A temporal convolution/recurrent expert consumes 664 established geometry
features and 128 track-derived memory/receiver/lane features. Past/future ball
anchors, age/availability masks and sets of player positions provide context
through gaps without assuming stable tracking identities. Receiver options are
considered in every direction. These are geometric proxies, not measurements
of gaze, peripheral vision, intent or cognitive memory. Future coordinates may
be used because inference is offline.

The expert runs at 12.5 Hz; state probabilities and the input-availability gate
are repeated onto the 25 Hz output clock. When the ball is absent or either
team has fewer than eleven observed players, its state probabilities replace
the parent's. The parent remains elsewhere. Its calibrated change probabilities
retain their original 25 Hz clock. During observation loss, the three-state MAP
decoder adds transition cost `8 * clip(1.5 * (1-max(H,A,D)), 0, 1)`. This suppresses
uncertain fragmentation; complete observations use the original decoder exactly.
The availability gate does not certify
coordinate accuracy; duplicated tracks and genuine reduced team sizes remain
limitations. All half-level state is reset between periods.

H/A/D semantics are learned from the external provider fields. Loose balls and
challenges use temporal/spatial evidence and the three-state MAP decoder rather
than a new nearest-player touch flipping ownership. No rule assumes that an
unknown ball is dead. The established pre-kickoff rule applies to automatic
kickoff-start tag intervals; the existing externally specified out-of-bounds
rule uses original ball observations in graded windows. No new manual restart
or possession anchors are added for this match.

## Adaptation and selection

The expert starts from the externally trained camera-loss checkpoint. Each
batch mixes strong HAD examples and real broadcast weak H/A examples. Weak
loss is conditional two-class cross entropy: it has zero direct gradient on
the D logit and ignores unknown targets. Shared representations can still
affect D indirectly, so strong replay and strong-only change supervision are
retained. Input normalization remains frozen. No fitting is performed at export.

The 40-epoch protocol was declared before training. Selection maximizes the
joint external criterion subject to native preservation, a 0.01 strong-mean
decline limit and weak-domain reference guards. The unchanged initial model is
always eligible as fallback. Epoch zero reproduced all initial strong/weak
metrics. Epoch 30 passed every guard. A separate verification execution rehashed
all training/development caches and reference receipts, verified frozen
normalization, and reproduced the selected scores.

A separate frozen grid of 0, 2, 4 and 8 selects the continuity cost using mean
strong three-view spell F1. Native performance must be unchanged; weak decoded
H/A agreement may decline by at most 0.005 overall and 0.01 on missing-ball ticks.
Ties retain zero. Penalty 8 passes every guard. A separate process verifies all
frozen probability/target/source checksums and reproduces every grid candidate.
Three CUDA exports, including original and packaged weights, reproduce the raw
and final CSV bytes. The final file has 199,319 rows and SHA-256
`cbc30311a6c9fdbe8cc7198e3a25f38d444c07c437e491528da5a924ecca408d`.
Inference denies training-data and possession-truth access. Official format
validation passes; no full-match possession score is available.

Additional external experiments were not deployed: a transition-head mixture
did not improve the criterion, and partial-HAD training on public Metrica games
1/2 failed the guards. The latter used 94,773 supported event-derived frames
(H 39,689; A 33,245; D 21,839), with unsupported spans unknown. Source: Metrica
Sports' public 2021-04-15 sample release, published for community analytics with
source acknowledgement requested. No Game 3 labels train the model. Its frozen,
previously opened partial-label diagnostic supports transfer on degraded tracking
but does not establish dense spell F1 or an independent result. Detailed rejected
experiments and source receipts are retained in the repository.

## Reproduction and limitations

Install `requirements-video-ball.txt` in the recorded CUDA environment and retain
`models/coreorigin-video-v1`. The README gives exact export and official-format
validation commands. It also links the existing local coordinate-reconstruction
commands from the supplied bundles. Final inference needs weights and supplied
inputs, not external training datasets. Model/source hashes guard the copied
checkpoint, selection chain and input artifacts. Outputs stay local/ignored.

The main limitations are noisy or wrong observed ball coordinates, incomplete
player coverage, track fragmentation, weak-label/provider differences, the small
and heavily reused strong development cohort, and camera-domain shift. A gain
in extra ball coverage or weak H/A agreement does not establish a spell-F1 gain
on BAR-ATH. The full-match numerical goal remains pending permitted organiser
evaluation. Frozen v11's historical CSV/weights are preserved, including the
separate eligibility caveat concerning its old target-derived gap calibration.


## Additional completed checks

A 30-epoch expanded-geometry experiment reproduced mean external spell F1
0.681791, but its paired difference interval [-0.003650, 0.010709] spans zero.
It does not replace the selected model. A separate metadata-derived projection
repair changed 128 weights and 128 biases without gradient training; its three
fixed variants leave spell F1 unchanged, so zero repair is retained. Both studies
use external data only. Neither result is evidence of BAR-ATH full-match F1.

## Exact regeneration commands

Use the provided bundles in the repository's documented `data/` layout, the
committed detector `yolo11m.pt`, and both `models/coreorigin-video-v1` and
`models/coreorigin-continuity-v4`. Run from the repository root in the recorded
CUDA environment. No external training datasets are needed for inference.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-video-ball.txt
.\.venv\Scripts\python.exe scripts/research/video_ball/detect.py --device 0 --frames missing --imgsz 1920 --conf 0.05 --checkpoint-every 1000 --out data/research/reproduction-inputs/ball_missing_gpu.npz
.\.venv\Scripts\python.exe scripts/research/video_ball/fuse_detections.py --detections data/research/reproduction-inputs/ball_missing_gpu.npz --out data/research/reproduction-inputs/tactical_ball.npy
.\.venv\Scripts\python.exe scripts/research/broadcast/dual_feed.py --out data/research/reproduction-inputs/dual --tactical-hypotheses data/research/reproduction-inputs/tactical_ball.npy
.\.venv\Scripts\python.exe scripts/research/spatial_memory/export_continuity_candidate.py --model models/coreorigin-continuity-v4 --ball data/research/reproduction-inputs/dual/ball_hypotheses.npy --out data/research/reproduction-output --device cuda
.\.venv\Scripts\python.exe evaluation/validate_submission.py data/research/reproduction-output/CoreOrigin.csv
```

The three current CSV repetitions use the verified coordinate artifact and a
checksum-verified feature cache. They establish deterministic inference from
that artifact; a fresh cold-video extraction is not claimed by those three
repetitions. Earlier extraction/fusion and dual-feed verification receipts are
preserved separately. The detector, input and source hashes must match the
recorded protocols; the detector's numerical environment is part of reproduction.
