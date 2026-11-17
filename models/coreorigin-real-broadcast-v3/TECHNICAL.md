# CoreOrigin: external supervision with observation-aware temporal memory

## Scope and current evidence

The algorithm recovers H/A/D labels offline from the supplied match inputs.
The target is BAR–ATH spell F1 above 0.740. That result has not been established;
the organisers' full-match ground truth is unavailable. The current file passes
the official format validator. No new BAR–ATH possession scoring was performed
during this study, and no supplied video/prediction overlay was inspected.

The fixed external study selected epoch 30 of 40. Strong J03WPY spell F1 is
0.857005 native, 0.532105 with synthetic camera loss, and 0.617817 with both
camera loss and removal of the training source's parked-ball cue. Mean 0.668976
is essentially unchanged from the preceding model's 0.668571. On two real
detected-only broadcast development fixtures, agreement with provider H/A weak
labels rises from 73.598% to 82.259%; on missing-ball ticks, 61.010% to 79.837%.
Conditional H/A accuracy on missing-ball ticks rises from 62.830% to 79.799%.
These are different metrics with different target quality. They do not establish
BAR–ATH F1, dead-ball accuracy, or independent-test performance. All development
fixtures have been opened previously. The joint selection objective 0.74449 is
an average of external spell and weak conditional-H/A metrics, not the goal score.

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
image-license attribution and calibration JSONs are in the parent bundle. The
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
the parent's. The parent remains elsewhere, and its calibrated change
probabilities/decoder remain throughout. The availability gate does not certify
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
on BAR–ATH. The full-match numerical goal remains pending permitted organiser
evaluation. Frozen v11's historical CSV/weights are preserved, including the
separate eligibility caveat concerning its old target-derived gap calibration.
