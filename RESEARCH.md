# Research and implementation roadmap

Last researched: October 20, 2026 (America/New_York).

This document records research directions, their evidence, and the order in which to investigate them. **A first implementation now runs locally and on Colab:** see [IMPLEMENTATION.md](IMPLEMENTATION.md) and [WORK_DONE.md](WORK_DONE.md). Roadmap items are proposals unless those implementation records explicitly mark them complete; the full physical-history/visual-contact system remains a research goal.

## 1. Recommendation

**Updated research ambition:** investigate possession inference that retains competing explanations of missing-ball sequences, identifies where those explanations agree on possession, and automatically requests visual evidence only where disagreement could change a possession boundary. Section 11 develops this proposal, the closest prior work, a small measured camera-framing probe, and explicit ways to reject the idea if it fails. This is a proposed contribution, not a claim of worldwide originality.

The engineering reference remains a tactical-tracking-first possession segmentation pipeline. Combine evidence about **whether play is live**, **which team controls the ball**, and **whether a state change is credible**. Start with interpretable features and a small state model; add learned models or video recovery only after error analysis justifies them. This reference gives the ambitious method a meaningful baseline to beat.

The first substantial improvement to investigate is replacing the baseline's missing-ball/dead-ball assumption with explicit missingness handling and stoppage evidence. The second is retaining possession through passes, deflections, and occlusion until the opposing team has convincing control. The third is decoding coherent intervals rather than classifying every frame independently.

These are project-specific hypotheses, not measured improvements. No score gain is promised.

## 2. What the project actually requires

Authoritative local references: [CHALLENGE.md](CHALLENGE.md), [DATA.md](DATA.md), [EVALUATION.md](EVALUATION.md), [SUBMISSION.md](SUBMISSION.md), [RULES.md](RULES.md), and the [metric implementation](evaluation/possession_metrics.py).

| Requirement | Consequence for implementation |
|---|---|
| Exactly 199,319 tactical video frames at 25 fps | Every stage must preserve the reference frame index. |
| Output labels H / A / D / X | Barcelona / Athletic / dead ball / abstention. Internal states may be richer but must map back to this alphabet. |
| Ranking by spellF1 | Recover whole same-team possession intervals, not just good frame accuracy. |
| macroF1 is the first tie-break | Avoid solving fragmentation by making persistently incorrect predictions. |
| Public truth covers 7,500 frames and 17 possessions | This is a pipeline sanity check, not a reliable model-selection benchmark. |
| All submitted labels must be generated algorithmically | No frame-by-frame hand corrections to the final prediction. Small training/validation annotations are permitted by RULES.md. |
| Exact reproducibility | Preserve configuration, dependencies, seeds, asset checksums, and output hashes. |

The published full-match baseline floor is spellF1 **0.2433** for `baseline_smooth50`. That number is supplied by the organizers; we cannot independently compute it without private truth.

Delivered ATD uses `video_frame = frame_id - 1`. Smart Tagging uses `round(seconds * 25)`. Homography already uses zero-based video frames. The large per-half offsets in `games.py` concern a separate ground-truth counter and must not be applied to delivered ATD.

## 3. Evidence from this repository and the local data

- All nine expected local match assets are present in `data/tactical/` and `data/broadcast/`.
- Tactical tracking contains 199,319 records; broadcast tracking contains 164,488. Both passed the existing alignment checks.
- Tactical homography has 199,319 valid matrices. Broadcast has 108,248 valid matrices out of 164,488 rows.
- Metadata has 46 tactical and 51 broadcast track slots. Slots are not persistent player identities.
- Smart Tagging has 638 intervals, and both kickoff containment checks pass.
- Running the baseline on the actual tactical data produced 113,488 frames with a ball-and-nearest-player answer (56.9%). The documentation separately reports that the ball is absent on approximately 41% of graded frames.
- Tactical video metadata reports 199,319 frames. Broadcast video metadata reports 164,487, one fewer than its tracking and homography. The discrepancy is unresolved; a full frame decode has not been performed.
- Four-point sampled decoding succeeded for both videos. This does not establish full-file integrity.

### What the baseline leaves open

`starter/baseline.py` chooses the team of the closest tracked player using distances in metres. With no usable ball/player answer, its raw prediction becomes D. Smoothing counts only H and A observations in a centered window: D is returned only if neither team has any observations in that window. Thus it is not a three-class majority vote with a learned stoppage model.

It has no control threshold, explicit restart detector, missingness uncertainty, or learned transition model. It ignores Smart Tagging, video, and homography. Missing ball observations can become invented stoppages; nearby opponents can become invented turnovers; long smoothing windows can erase genuine short possessions.

All four baselines were regenerated locally and matched the supplied CSVs **byte for byte**. Public-window results were reproduced:

| Baseline | macroF1 | spellF1 | Predicted possessions in public window |
|---|---:|---:|---:|
| raw | 0.5972 | 0.0378 | 131 |
| smooth25 | 0.6200 | 0.2868 | 36 |
| smooth50 | 0.6207 | 0.3256 | 26 |
| smooth125 | 0.5812 | 0.4242 | 16 |

Public truth contains 17 possessions. Matching that count alone is not evidence of correct possession timing or ownership. The documentation shows that the public winner is not the full-match winner.

## 4. Research findings and their limits

The following primary sources were checked on October 20, 2026. Their datasets, labels, and metrics differ from this competition. Published results are not expected scores for our match.

### R1. Possession requires motion and control evidence

[Link and Hoernig, Individual ball possession in soccer (2017)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0179953) distinguish proximity, action, and control. Their method uses trajectory preparation and motion-related features as well as distances. It also assumes an available in-play flag, which our solution must infer.

**Application proposed here:** compare ball proximity with relative motion, persistence, and opponent pressure. Retain the last controlling team through ambiguous phases. Do not adopt the paper's possession definitions as the answer key: this challenge's public labels and scorer are authoritative.

### R2. Player movement can inform dead-ball detection

[Lang et al., Predicting the in-game status in soccer with machine learning using spatiotemporal player tracking data (2022)](https://www.nature.com/articles/s41598-022-19948-1) study in-play prediction from player positions, speeds, and temporal context. They compare frame and stoppage evaluations and describe errors caused by excessive smoothing.

**Application proposed here:** test a separate live/dead estimator using collective movement and restart context, even when the ball disappears. Their models used many labeled matches and more stable player organization; our one-match, fragmented-slot setting requires different features and much more cautious validation.

### R3. Rules and staged inference are credible starting points

[Vidal-Codina et al., Automatic event detection in football using tracking data (2022)](https://arxiv.org/abs/2202.00804) describe a deterministic pipeline using possession and player configurations to derive events and set pieces.

**Application proposed here:** implement an auditable rule-based reference with explicit uncertainty and transition rules before adding a large neural model. Their event-detection results do not establish performance on our possession-spell metric.

### R4. Set-based models can use team context during ball gaps

[Kim et al., Ball Trajectory Inference from Multi-Agent Sports Contexts Using Set Transformer and Hierarchical Bi-LSTM (2023)](https://arxiv.org/abs/2306.08206) infer ball trajectories from player trajectories, using possession as an intermediate prediction and set-based representations.

**Application proposed here:** consider permutation-invariant team representations and a missing-ball auxiliary task. First verify code, weights, training requirements, and licensing before adopting any implementation. A plausible inferred ball location is not proof of possession.

### R5. Temporal neural models are an alternative, not the initial dependency

The publisher abstract of [Using Temporal Convolutional Networks to estimate ball possession in soccer games (2023)](https://www.sciencedirect.com/science/article/pii/S0957417423002816) describes trajectory embeddings and permutation-invariant aggregation for possession estimation.

**Application proposed here:** a compact temporal convolution over pooled team features is a later comparison if we obtain enough legitimate training supervision. Only the abstract/search-accessible publisher material was available in this review; implementation details were not verified.

### R6. Structured sequence decoding is particularly relevant

[PathCRF (2026)](https://arxiv.org/html/2602.12080v1) models ball-free event inference as possession paths, using dynamic graph representations, constrained transitions, and Viterbi decoding. The [authors' repository](https://github.com/hyunsungkim-ds/pathcrf/blob/main/README.md) provides an implementation entry point.

**Application proposed here:** borrow the principle of globally consistent sequences; start with a much smaller H/A/D decoder. The full player-edge formulation assumes trajectory continuity and supervision that we do not currently have. Repository availability does not mean pretrained weights or data access have been verified locally.

## 5. Candidate implementations, ranked by current project fit

Priorities describe engineering judgment, not measured benefit. No new runtime dependency has been installed for these proposals.

| Priority | Component | What to implement | Why it fits | Main risk |
|---|---|---|---|---|
| P0 | Data audit and cache | One canonical frame-aligned cache, quality masks, metadata checks | Prevents alignment and missingness bugs in every experiment | Accidentally treating absent coordinates as real zeros |
| P0 | Experiment runner | Configured inference, validation, scoring, run manifest, hashes | Makes improvements reproducible and reviewable | Mistaking public tuning gains for generalization |
| P1 | Control-aware rule model | Team distances, margins, confidence, hysteresis, possession memory | Directly addresses nearest-player flicker | Excess persistence delays real turnovers |
| P1 | Dead-ball estimator | Movement, ball dynamics, boundary context, tagging evidence | Addresses the baseline's largest conceptual gap | Slow buildup can resemble a stoppage |
| P1 | Smart Tagging evidence | Map codes and team tags into soft possession/restart evidence | Provided signal unused by the baseline | Tag intervals are noisy and do not define exact boundaries |
| P1 | Temporal decoder | H/A/D sequence with evidence-dependent switching costs | Aligns inference with whole-interval scoring | Excess duration bias merges real short intervals |
| P1 | Error inspection | Synchronized tactical video, pitch overlay, label timeline, uncertainty | Makes transition failures explainable | Growing a dashboard before solving core inference |
| P2 | Small learned classifier | Compare logistic/tree-based emissions if supervision is adequate | Captures feature interactions with manageable complexity | Five minutes do not provide diverse independent examples |
| P2 | Bounded gap recovery | Short-gap interpolation/filtering with uncertainty and reset rules | Can restore local geometry for brief gaps | Hallucinated trajectories through turnovers or stoppages |
| P3 | Player-only context model | Set pooling plus compact TCN or recurrent encoder | Potential fallback in long ball gaps | Training-data availability and domain shift |
| P3 | Targeted video ball recovery | Process selected low-confidence tactical segments locally | Adds visual evidence where ATD fails | Annotation, compute, tiny-ball ambiguity |
| P3 | Broadcast fusion | Diagnose frame-count mismatch, align live segments, reject replays | Optional second-view evidence | Alignment errors can overwhelm any benefit |

## 6. Proposed first solution architecture

```text
Tactical ATD + metadata ----> validated arrays + masks ----> geometric/motion features --+
Smart Tagging -------------> normalized soft evidence ---------------------------------+
                                                                                     |
                                                   live/dead evidence + team control evidence
                                                                                     |
                                                    temporal decoder, separately per half
                                                                                     |
                                                       frame,label CSV + diagnostics
                                                                                     |
                                                       validator + unchanged official scorer

Tactical video + homography ---> local error viewer and optional later recovery branch
Broadcast --------------------> optional later branch after alignment is established
```

### Canonical representation

Preserve video frame, source, fps, normalized coordinates, metre coordinates, observation masks, and team membership. Keep original observations separate from interpolated or predicted values. Record pitch-size convention: `games.py` uses 105.3 x 68.0 m, while ATD metadata uses 105 x 68 m. Use one explicit convention for model distances.

Do not treat slot order as player semantics. For team context, aggregate observed players with counts, quantiles, trimmed means, spatial spread, and masks. If deriving motion from coordinate differences, require verified track continuity; prefer provided speed where suitable and inspect its quality. Separate halves and reset state across halftime.

### Candidate features

- Ball visible; time since last trusted observation; gap duration; coordinate jump flags.
- Closest and next-closest observed player distance per team, plus the team-distance margin.
- Ball speed and change in direction where observations support reliable derivatives.
- Player/ball relative motion only across trustworthy observations.
- Team speed quantiles, fraction nearly stationary, centroids, and spatial spread; use valid-player counts to indicate reliability.
- Distance to pitch boundaries, with outlier and airborne-ball caveats.
- Active phase evidence, restart windows, team availability, and conflicts among tags.
- Previous confident team and elapsed time since credible control or restart evidence.

There is no universal distance or speed threshold selected yet. Choose small, interpretable candidate ranges and record their rationale rather than searching hundreds of combinations on the public window.

### Missing ball and controlled turnovers

Missing ball is an observation condition, not a game state. Brief ambiguous gaps should reduce confidence without automatically becoming D. Long gaps may use collective motion and tags, or preserve uncertainty; they must not become long synthetic straight-line ball paths. Confidence should decrease with gap length.

Compare a simple hysteresis state machine with a three-state sequence decoder. A candidate score is the sum of observation support minus switching costs and optional soft duration costs. Allow H-to-A, A-to-H, and transitions to/from D; direct turnovers are legitimate. A semi-Markov extension can score durations explicitly if a simple decoder is insufficient. Do not encode a hard minimum duration that deletes valid short possessions.

Separate live/dead and team evidence conceptually, but fuse them during decoding or compare against joint three-class inference so a mistaken hard live/dead gate does not become irreversible. Ground-truth X gaps define evaluation runs; our inference must use published boundaries and input evidence, not hidden labels.

### Smart Tagging mapping

Normalize spelling and case, preserve N/A, and write a reviewed mapping per code. Offensive phases may support the tagged team; defensive phases generally describe the defending team, so team ownership evidence may need inversion. Defensive/attacking transitions require their own interpretation. Do not blindly assign possession to every `Team` tag. A set-piece interval surrounds a restart; its start is not the exact restart frame. Detect and expose conflicting intervals.

### Offline processing and abstentions

The current task is an offline full-match submission, so centered context and bidirectional decoding are viable. Label any future live/real-time version separately because it cannot use future observations.

Prefer meaningful H/A/D outputs where evidence supports them. X outside graded blocks is harmless. On graded frames, X is a miss for macroF1 and is forward-filled for temporal metrics, with leading gaps back-filled from the first known answer; an entirely abstained run stays unanswered. Avoid treating X as an unpenalized transition detector or optimizing around scorer behavior.

## 7. Experiment and validation plan

1. Keep the regenerated baseline results as the immutable reference.
2. Audit gap lengths and implausible observations before selecting interpolation limits.
3. Implement one interpretable control-aware model; retain raw evidence and reasons for transitions.
4. Add live/dead evidence and test whether short real intervals are preserved.
5. Add Smart Tagging as a separate switchable feature group.
6. Compare hysteresis with sequence decoding, then combine only justified components.
7. Use video inspection to classify failures: dead/alive confusion, wrong team, gap handling, false turnover, missed turnover, and boundary timing.
8. Consider learned or video-based models only after identifying the residual error they target.

Suggested experiment identifiers: E00 supplied baseline; E01 possession memory; E02 live/dead evidence; E03 tag evidence; E04 sequence decoding; E05 combined model; E06 targeted gap recovery. These are a planned matrix, not completed experiments beyond E00.

Each run should save configuration, input hashes, code version, seed, dependency versions, output hash, runtime, missingness summaries, label counts, possession counts/durations, macroF1, spellF1, per-class F1, transition errors, and ablation settings. Keep match-derived caches and restricted artifacts under ignored local directories.

Do not randomly split adjacent frames into train/test. If additional permitted annotations are created, use temporally separated clips and a buffer at least as large as the model context; report their limitations. Do not silently train on the public window and then present its score as held-out evidence. No method available here establishes full-match generalization from 17 labeled possessions.

Use the existing bootstrap option to express uncertainty, not to manufacture independent samples. On unlabeled portions, gap stress tests, symmetry checks, slot-order invariance, and video diagnostics assess robustness but do not establish accuracy. Artificial missingness on observed frames is easier than naturally occluded sequences; disclose that limitation.

Never force full-match possession counts, mean durations, or team share to match assumed targets. Never source corrected possession labels for this fixture beyond the permitted public truth.

### Acceptance gates

- Every generated CSV passes the existing validator with 199,319 ordered frames.
- Repeated inference with the same inputs/configuration produces the same file hash.
- No temporal operation carries state across halftime or crosses a disallowed gap.
- Slot permutations within teams do not change pooled features or final team predictions.
- Team-swap and pitch-reflection checks expose unintended home-side or orientation bias; these are engineering checks, not new labeled test data.
- A claimed improvement includes ablations and boundary/error diagnostics; public-only gains are labeled exploratory.
- The official scorer and reference labels remain unchanged.

## 8. Concrete proposed deliverables

These paths are suggestions, not files already created:

| Proposed artifact | Responsibility |
|---|---|
| `src/data.py` | Streaming ATD parser, common frame index, validated cache |
| `src/features.py` | Geometry, pooled team motion, quality indicators |
| `src/tagging.py` | Reviewed per-code soft evidence |
| `src/rules.py` | Interpretable state-machine reference |
| `src/decode.py` | Three-state temporal decoding |
| `src/predict.py` | Full-match inference and submission export |
| `configs/` | Versioned baseline and experimental settings |
| `scripts/run_experiment.py` | Validation, scoring, manifests, reproducibility |
| `reports/` or an ignored working directory | Human-readable aggregate research findings |
| `TECHNICAL_REPORT.md` | Submission method, limitations, and reproduction instructions |

A later local viewer could show synchronized video, pitch coordinates, tag windows, predicted states, and confidence, with jump-to-error controls. Its purpose is diagnosis; it should not edit final labels by hand. Downstream possession summaries and transition maps are possible extensions. Reliable player-specific pass networks or individual performance claims require identity reconstruction that this dataset does not directly provide.

## 9. Immediate next work

- [x] Reproduce the four organizer baselines from the downloaded ATD.
- [x] Verify generated files against the supplied CSVs and rerun public scoring.
- [ ] Build a read-only data-quality report with gap distributions and motion outlier rates.
- [ ] Specify and implement the first control-aware state-machine baseline.
- [ ] Compare explicit live/dead evidence with the baseline's missing-ball proxy.
- [ ] Add reviewed Smart Tagging mappings and a small temporal decoder.
- [ ] Build transition-focused video diagnostics.
- [ ] Decide whether further supervision or video recovery is worth its cost.

## 10. Open questions

- What compute, memory, and optional GPU are available on the machine that will run experiments?
- How many additional permitted validation/training clips, if any, should be annotated?
- Are provided speeds trustworthy during slot fragmentation, and which derivatives remain reliable?
- How do gap duration, field location, and nearby team configuration correlate with observed failure types?
- Which Smart Tagging codes improve possession evidence rather than merely correlating with it?
- Is broadcast's one-frame discrepancy a container count issue, an actual missing frame, or an export mismatch? Never repair it by arbitrary padding.
- Can outside code/weights be reproduced legally and practically without depending on unavailable training data?

The research references motivate experiments. They do not replace the competition's possession definition, prove a winning method, or establish that an advanced architecture is necessary.

## 11. Frontier proposal: possession under missing evidence

Added October 20, 2026 after the request for a more original direction.

### 11.1 The central research question

**Can team possession remain identifiable when the ball's exact location is not? And can the system determine which additional observation would resolve the cases where it is not?**

This changes what we try to recover during an occlusion. Instead of immediately inventing one ball trajectory and trusting the resulting team label, retain several plausible explanations and examine whether they imply the same possession. Position uncertainty and possession uncertainty need not be identical.

Example, explicitly hypothetical: the ball disappears during a Barcelona sequence. Some feasible continuations place it near one teammate and others near another teammate. If every retained explanation supports Barcelona, ball localization can remain uncertain while the team decision is stable. Conversely, seeing Barcelona before and after a gap does **not** rule out an Athletic turnover followed by a Barcelona recovery, or a stoppage and restart. Those must be represented as competing explanations rather than silently erased.

The possible contribution is **an observation-aware, multiple-explanation possession model with boundary ambiguity and automatic visual verification**, designed for fragmented, automatically generated football tracking. It is a combination whose benefit and novelty still need to be established.

### 11.2 What the literature search already rules out as a novelty claim

| Tempting claim | Closest checked work | Consequence |
|---|---|---|
| We infer possession without seeing the ball | [PathCRF](https://arxiv.org/html/2602.12080v1), plus ball inference work in R4 | This already exists; cannot be our originality claim. |
| We use legal sequence transitions and Viterbi | PathCRF explicitly does this | A different state decoder alone is insufficient novelty. |
| We focus computation or supervision near uncertain boundaries | [Boundary-Centric Clip-Budgeted Active Learning](https://arxiv.org/abs/2604.15173) | Boundary-focused annotation is established; our visual-query setting must be distinguished. |
| We represent ambiguous segment boundaries | [Boundary Voting Network](https://arxiv.org/abs/2609.08167) | Boundary uncertainty alone is not new. |
| We reason about physical availability and uncertainty | [Who can receive the pass?](https://link.springer.com/article/10.1007/s10618-022-00827-2) | Physics and uncertainty in football are established tools. |
| We use camera geometry to assist ball tracking | [Yu et al., trajectory tracking with camera motion recovery](https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICME-2007/pdfs/0001543.pdf) | Camera-assisted tracking is old; camera framing as an observation-model cue needs a narrower claim. |

The active-learning paper queries human supervision, whereas our proposed acquisition loop requests additional **algorithmic visual measurements** from already available video. That distinction creates a research question; it does not prove novelty. The camera-motion paper was available through its primary proceedings search excerpt; its full PDF could not be opened in this review, so its technical scope has not been exhaustively checked.

Searches covered combinations of soccer possession, missing observations, uncertainty, multiple hypotheses, camera framing, reachability, and temporal boundary acquisition. This is a targeted literature review, not a systematic proof of absence. Any later publication claim requires a broader related-work and implementation review.

### 11.3 Four connected ideas to test

#### A. Keep possible histories, not one invented ball path

For an ambiguous interval, maintain a bounded set of candidate histories containing team state, live/dead state, possible control changes, and coarse ball-location regions. Start with team-level histories to avoid making stable player IDs a prerequisite.

Generate hypotheses from reliable observations before and after the gap, candidate change points, team geometry, and restart evidence. Score physical plausibility softly. Include a residual unknown/model-mismatch option so bad assumptions do not force false certainty. Long unobserved intervals should allow growing uncertainty and additional turnovers; do not hard-code a single change per gap.

At a frame t, collect the labels implied by retained histories:

```text
S(t) = { possession label at t from each retained plausible history }

S(t) = {H}      -> agreement within the retained model
S(t) = {H,A}    -> team ambiguity
S(t) = {H,D}    -> live/dead ambiguity
S(t) = {H,A,D}  -> unresolved or weakly informative evidence
```

This is a **model-relative agreement set**, not a proof of the real label. A finite beam can miss the correct history. Do not call it a formal certificate unless exhaustive coverage under explicit assumptions is established. Even an exhaustive model can be wrong about reality.

Proposed output beyond the required CSV: a boundary ambiguity band, retained alternatives, and the observations that eliminated alternatives. The submitted labels still come from one deterministic algorithmic decision rule; uncertainty is not an extra ground-truth class.

#### B. Turn camera framing into a weak observation channel

The tactical homography is available on every frame. Invert it to map image center and selected viewport points onto the pitch. Study the temporal movement of this visible field footprint alongside the player sets.

The camera operator or automated crop may follow developing play even when a detector misses the ball. That suggests useful information about a coarse region of play, not direct knowledge of the possessor. Do not assume the camera is human-operated or intentionally centered on the ball; camera provenance is unverified.

Use the camera footprint to distinguish plausible causes of missingness: possible off-screen movement, crowded visible play, calibration issues, or an uninformative view. Tracking absence should be interpreted conditional on visibility and context. Camera, tracking, and tags can share the same underlying source, so they are not independent witnesses whose likelihoods may simply be multiplied.

Treat this as an empirical visibility/framing model. A pitch-plane homography does not reconstruct true 3D camera pose, gaze, or an airborne ball. Crops can anticipate play, lag behind it, or follow irrelevant action.

#### C. Spend visual computation where it changes the possession answer

When histories disagree, rank candidate tactical video clips by estimated reduction in **possession or boundary uncertainty per unit compute**. The best query may be the reception immediately after a gap, rather than the least visible frame inside it.

Example query: determine whether a short candidate reception contains controlled opposing-team contact. Another query may detect a restart rather than recover every ball coordinate. Keep detection failures and ambiguous contacts as uncertain measurements instead of forcing an answer.

Start with a heuristic score based on history disagreement, competing boundary locations, and estimated visual usefulness. A rigorous expected-value-of-information policy would require a calibrated observation model that does not yet exist. This is inference-time automatic video analysis, not asking a human to hand-correct the final CSV.

Compare against random clips, uniform sampling, largest ball-position uncertainty, and largest frame-class entropy at the same decoded-frame or runtime budget. If targeted queries do not beat those controls, the acquisition argument fails.

#### D. Expose sensitivity instead of providing a decorative explanation

For a predicted turnover, produce a local evidence record:

- Earliest and latest boundary locations supported by retained alternatives.
- Which observations favor retention, turnover, or a stoppage.
- Whether removing one source, widening coordinate uncertainty, or increasing hypothesis diversity changes the decision.
- Whether additional video resolved ambiguity or merely agreed with an already stable decision.

These are **perturbation and source-ablation checks**, not causal counterfactuals. Removing a team's players changes the input distribution and does not establish that the team caused an outcome. The purpose is to reveal brittle assumptions and give error analysis something testable.

### 11.4 A measured feasibility probe: camera framing contains some signal

Added the reproducible exploratory script [research_camera_probe.py](scripts/research_camera_probe.py). It uses the existing tactical ATD and homography and never reads possession ground truth.

```powershell
python scripts/research_camera_probe.py
```

Method: inverse-project pixel (960, 540) through each tactical homography; fit a linear mapping from that pitch-plane center to observed ball x/y using the first graded half; evaluate on observed ball coordinates in the second half. The report records finite-coordinate filters and a 60-second within-test-half circular-shift control. Output is local and ignored at `data/research/camera-probe.json`.

| Quantity | Result |
|---|---:|
| First-half observed samples used for fitting | 47,106 |
| Second-half observed samples evaluated | 46,942 |
| Second-half correlation: camera-center x vs ball x | 0.8379 |
| Second-half correlation: camera-center y vs ball y | 0.1454 |
| Linear proxy mean ball-location error | 24.25 m |
| Constant training-median ball-location baseline | 31.28 m |
| Same mapping with test camera signal shifted by 60 seconds | 38.89 m |

**Interpretation:** framing appears to contain synchronized information, especially along the pitch length. The large remaining location error and weak widthwise association rule out treating image center as an accurate ball estimate. These numbers justify a modest follow-up experiment, not a claim of possession improvement.

Limitations: these are noisy visible-ball observations, not corrected coordinates or naturally missing-ball truth; frames within a half are dependent; the half split is not cross-match validation; a fixed time shift is only a diagnostic control; no model-selection sweep or possession score was computed. The phrase camera attention remains a hypothesis about framing, not a measured psychological quantity.

### 11.5 Alternative ambitious directions and why they are secondary

| Idea | What makes it interesting | Why it is not the first research bet |
|---|---|---|
| Infer possession from collective player reaction dynamics | Possession changes may alter movement beyond the player touching the ball | Tactical anticipation and pressing can mimic reactions; causality and generalization are hard to establish from one match. |
| Train a full football world model and simulate alternatives | Jointly represents ball and player evolution | Public work already explores counterfactual pass/world models, e.g. [Monte Carlo Pass Search](https://arxiv.org/abs/2606.11120); compute and external supervision are major dependencies. |
| Fuse both videos into a replay-aware latent match clock | Could exploit complementary viewpoints while rejecting misleading replay frames | Strong engineering challenge, but synchronization and the existing frame-count mismatch become prerequisites. |
| Apply a large vision-language model to the full match | Could provide semantic descriptions | Does not itself establish accurate boundaries, reproducibility, or novelty; restricted-data permissions also require separate care. |

The proposal favors a narrow technical contribution that can be tested against the actual challenge rather than a large collection of unrelated features.

### 11.6 First prototype and credible stopping rules

Use the existing baseline and planned interpretable decoder as controls. Keep the first frontier prototype small: CPU inference, three output states, coarse ball regions, a limited configurable beam, and no new detector training. A starting beam size such as 8 is an engineering budget, not a validated optimum; compare several budgets for convergence and report pruning failures.

| Experiment | Question | Minimum comparison | Reason to reject or narrow the idea |
|---|---|---|---|
| F00 framing probe | Does camera geometry contain signal? | Observed-ball linear proxy vs constant and shifted signals | Existing result supports only a weak cue; no possession inference yet. |
| F01 competing histories | Does retaining alternatives avoid gap-induced false turnovers? | Same features, single best path vs multiple retained histories | Alternatives collapse to copies or add runtime without useful ambiguity detection. |
| F02 missingness/visibility model | Does framing distinguish detector failure from game-state evidence? | Same decoder with and without camera/visibility features | Improvement disappears on separated evaluation clips or causes false dead-ball decisions. |
| F03 adaptive visual verification | Does query selection recover better boundaries per video budget? | Equal-budget random, uniform, location-uncertainty, and state-entropy queries | No repeatable budget advantage. |
| F04 robustness records | Does disagreement or sensitivity identify actual errors? | Error enrichment and risk-versus-coverage curves on held-out labeled clips | Stable agreement is frequently confidently wrong or sensitivity is unrelated to errors. |
| F05 final segmentation | Does the full system improve possession intervals? | Same official scorer, ablations, short-possession and restart diagnostics | Only frame metrics improve, or public-window gains depend on narrow tuning. |

Only F00 has been executed. The other rows are proposed experiments. No public score for the new concept exists.

For early gap experiments, hide contiguous spans of observed ball coordinates while retaining underlying observations only for evaluation. Separate training clips from evaluation clips before constructing masks. Stratify spans by duration, nearby traffic, and proximity to state boundaries where legitimate labels exist. Do not use hidden ball observations inside the acquisition policy or gap model. Simulated missingness does not establish performance on natural occlusions.

For real-gap validation, use the permitted public labels and, if needed, a small separately annotated development/evaluation set under the competition's rules. Freeze held-out clips and context buffers before tuning. Annotation is for model development/validation; final labels remain algorithm-generated. Never fetch corrected possession data for the fixture outside the permitted truth.

Avoid early numerical score promises. Agree on an experiment budget and a minimum practically useful effect before running model selection; report every ablation rather than selecting one favorable public-window result.

### 11.7 What would make the result worth showing

A compelling demonstration would take an occluded passage and show:

1. The raw nearest-player baseline flickering or declaring a stoppage.
2. Several plausible histories with distinct possession boundaries.
3. Where those histories agree despite uncertain ball position.
4. A selected short visual query that resolves a genuine disagreement.
5. A deterministic final interval prediction with an evidence/sensitivity record.
6. Held-out and equal-compute comparisons showing whether the behavior improves the official metric.

Use an error-inclusive demo set selected before viewing outcomes, not one hand-picked success clip. Keep any shared match visuals within the user's actual permission scope. A polished animation or an impressive model name is not evidence.

### 11.8 Defensible originality statement

Working claim to investigate: **We study whether team-level possession can be partially identified across missing-ball intervals by retaining observation-consistent histories, and whether possession-specific ambiguity can guide targeted visual verification more efficiently than ball-location uncertainty.**

The candidate distinction is the joint treatment of label agreement, observation failure, fragmented identities, and possession-specific evidence acquisition. Existing work already contributes the individual building blocks. A literature search cannot guarantee that no one has considered the combination. Success requires a precise formulation, reproducible implementation, and evidence that the added machinery earns its cost.

## 12. Compute and training plan

Added October 20, 2026 after the user expressed concern about local compute capacity. **Recommended starting environment: Google Colab, with Drive storage and ordinary Python execution.** This plan has now been exercised: the CPU-first notebook, small calibration training experiment, tests, and inference run succeeded in Colab. No paid plan was purchased or GPU requested. PyTorch is not required by the current version.

[Google's Colab overview](https://developers.google.com/colab) confirms hosted notebooks, GPU access, and Drive integration. The user's PC can act as the browser client for cloud runs.

| Workload | Starting environment | Training requirement |
|---|---|---|
| Data checks, baseline, rules, small hypothesis search | Local CPU or standard Colab runtime | None; these are algorithmic components. Local baseline reproduction already succeeded. |
| Small feature-based classifier | CPU runtime initially | Needs legitimate labels or carefully documented weak supervision. |
| Compact TCN/recurrent encoder for team context | Colab GPU if required by measured runtime | A single approximately 16 GB GPU is an engineering starting target, not a demonstrated requirement or availability guarantee. |
| Targeted visual feature extraction or detector fine-tuning | Colab GPU, selected clips and small batches | Need suitable pretrained weights or labeled visual training examples. Avoid training a large detector from scratch as the first step. |
| Long, predictable GPU jobs | Reassess paid Colab or a dedicated GPU service later | Choose only after measuring memory/runtime and setting a budget. |

According to the [Colab FAQ](https://research.google.com/colaboratory/faq.html), GPU types and limits vary, sessions have finite lifetimes, and paid resources are also subject to availability. Copy active inputs to runtime-local disk rather than repeatedly reading many small items from mounted Drive. Save resumable checkpoints and final artifacts to persistent storage; runtime files are temporary. Check account benefits and current offers before paying; Drive storage capacity is not proof of a GPU entitlement. Exact regional pricing was not verified because the signup page requires sign-in.

Proposed workflow: keep code/configuration versioned; mount the existing organized dataset from the same Google account; copy only required files to runtime-local storage; prepare compact feature arrays; run inference or small-model training; periodically persist checkpoints; validate the exported CSV with the unchanged scorer. Track actual allocated hardware, peak memory, runtime, seeds, input hashes, and configuration. There is no need to load both full videos into RAM or decode every frame for tracking-first experiments.

**Supervision is the more immediate training bottleneck.** Approximately 199,000 tracking frames do not mean 199,000 labeled possession examples: only 7,500 public frames spanning 17 possessions have reference labels. Random adjacent-frame splitting is invalid evidence. Use the no-training reference first, self-supervised tasks with legitimate observed coordinates where useful, and separately held-out permitted annotations or compatible external data before scaling supervised models. Artificial gap recovery and weak-label performance do not establish natural-gap possession accuracy.

The repository distinguishes ordinary model use from disclosing restricted assets to hosted LLMs. Run data processing/training as code, and do not place restricted video, tracking extracts, or confidential outputs into Colab's Gemini/chat features unless separately authorized. The user's sharing permission is recorded; its scope should not be silently expanded to unrelated services.

[Kaggle Notebooks](https://www.kaggle.com/docs/notebooks) are a possible secondary environment with GPU access and resource quotas, but adding a second data copy is unnecessary at the start. Prefer Colab because the organized data is already in the user's Drive.

## 13. First implementation findings

The first pipeline and a temporally separated calibration experiment have run locally and in Colab. All prediction CSV hashes match between the two environments. Eight correctness/invariance tests pass in both. The cloud run completed in approximately 60 seconds without a GPU; four optional visual-motion queries were additionally measured locally.

The current implementation retains sampled H/A/D state histories and exposes model-relative marginals, source sensitivity, 137 missing-ball gap records, boundary evidence, and budgeted query/control plans. It does not yet implement exhaustive physical ball-region reasoning or a trained visual control/contact recognizer. Its visual measurements are diagnostic rather than label corrections.

Full public-window spellF1 is 0.3351 versus 0.3256 for organizer smooth50 and 0.4242 for organizer smooth125. In the later 170-second block, the label-free model scores 0.4000 versus 0.3579 for smooth125. The learned calibration candidate scores 0.3556 on that later block and remains a separate, unpromoted experiment. These are tiny, same-match comparisons; they do not establish hidden-match improvement.

The initial public ablations do not demonstrate benefit from camera evidence or Smart Tagging. Keep them visible in the research record rather than presenting them as successful innovations. The immediate empirical priorities are dead-ball discrimination, boundary timing, and whether disagreement-driven video queries outperform equal-budget controls. See IMPLEMENTATION.md for actual commands, evaluation scopes, and artifact definitions.

## 14. v0.2 follow-up: measured research decisions

Implemented whole-H/A/D-run temporal development splitting, 250-frame minimum embargo rounded to complete run boundaries, a fixed multisecond inactivity/outside/stillness stoppage candidate, detailed errors, query error-enrichment controls, full tracking/homography audits and video-packet scans, hash-verified stage recovery, and a synchronized private review lab. The new split contains 2,600 training frames, 1,061 excluded frames and 3,839 evaluation frames. Previously inspected public data remains development evidence, even after improving the split.

The original and fixed stoppage models both score 0.3200 spell F1 on this development slice; grouped supervised calibration scores 0.2750, versus reference smooth25 0.3111 and smooth125 0.2667. Calibration raises dead-ball F1 from zero to 0.3458 but worsens the ranking metric. Neither candidate is promoted. Full-public performance remains below the strongest reference. **The evidence currently supports improving supervision and live/dead representation, not claiming success from larger networks or additional heuristics.**

Twelve label-independent boundary/source queries cover 24.48 seconds of the public window and contain 47.22% erroneous frames. Thirty random seeds average 33.17%, with a wide 16.13%–49.29% 5th–95th percentile range. The policy finds 11.90% of errors and 4/20 transitions. Error enrichment is promising but is not yet superiority over chance variation, corrected possession, reviewer efficiency or generalization. The next useful study is a fixed-budget reviewer protocol recording resolution, evidence adequacy and actual interaction time; the implemented notes/export UI prepares that study but does not supply its results.

Tracking audits flag 31 tactical/623 broadcast slot jumps above 3 m per frame, three tactical ball jumps above 5 m and 16,033 frames of opposed phase support. These signals justify source-reliability research; they are not automatically errors or ground-truth corrections. Complete packet scans confirm 164,487 broadcast video packets against 164,488 tracking rows. Cross-camera fusion remains unjustified until synchronization and the discrepancy are explained.

The local review package has twelve decoded/verified tactical clips and 492 sampled pitch snapshots; no persistent player identities or missing ball trajectories are invented. Review outcomes remain separate from generated submission files. An independent labelled review set and controlled video recovery remain future work. See WORK_DONE.md and the technical report draft for actual verification and limitations.

The final v0.2 tracking workflow also completed in Colab: all nine cells, sixteen tests, grouped scores and both label-free CSV hashes were verified against the local run. Local resume reused all five stages after artifact-hash validation. Optional persistent Drive output is implemented but was not mounted during this run. This confirms reproducible execution, not improved hidden-match accuracy.

## 15. v0.3: measurement consistency before model complexity

The source metadata calls player speed kph, but independent displacement checks show supplied values approximately equal to m/s. An initial metadata-scaling pilot produced a score gain and was initially described as a unit correction; that interpretation was retracted after the kinematic check. The sampled final audit has median raw/derived ratio 0.99698, correlation 0.96253 and median absolute error 0.113 m/s for raw values versus 1.145 after dividing by 3.6. This demonstrates why a public score gain alone cannot validate an engineering assumption. The pilot remains preserved, and its 0.4914 pooled score is not used as the current result.

The final feature bank derives speed from short continuous trajectory windows, keeps missingness explicit and uses team-set features without persistent identities or absolute pitch location. A small symmetric classifier learns 59 physical/contextual features, with team-swap augmentation, exact probability symmetrization and equal run weighting. Four contiguous whole-run folds with 300-frame embargo and conservative 130-frame feature context provide development evidence. Decoder states are not reset or clamped at known fold-label boundaries. An unlabelled movement model is a separate fixed comparison rather than an assumed substitute for true in-play supervision.

The trajectory-based classifier scores 0.4457 pooled development spell F1, versus 0.3889 for smooth125 and 0.3043 for smooth50 on the same folds. Macro F1 is 0.7484 and dead-ball F1 0.6619. Three folds improve over smooth125; one is slightly worse. Leaving out any one evaluation block preserves a positive pooled difference, but only four related folds from an already examined match exist. Unsegmented label-free motion performance is 0.3610, still below smooth125's 0.4242. These facts justify further study of the supervised candidate, not a claim of winning or independent generalization.

The exported provisional model is refitted on all permitted public labels; its public score includes training and remains descriptive only. Related-model disagreement and training-feature envelope flags target a new twelve-clip review package. No correctness certificate, calibrated out-of-distribution detection or reviewer-time improvement is claimed. The next scientific bottleneck remains independent permitted validation, especially for stoppage boundaries and natural missing-ball intervals. See SUBMISSION_TECHNICAL.md for the current approach and WORK_DONE.md for the retained failures, artifacts and verification.

### v0.3 reproduction evidence

CoreOrigin is the confirmed registered team. The trajectory-based v0.3 pipeline completed all nine cells on a fresh Colab CPU runtime, passed all 26 tests, and reproduced all 11 local prediction/risk CSV hashes exactly. Source and four asset checksums were verified. This establishes implementation reproduction across the local/Colab environments; it does not turn the previously inspected 17-possession public example into independent validation. The fixed four-fold classifier spell F1 remains 0.4457 versus same-fold smooth125 0.3889.


## Audit revision and next evidence (21 October 2026)

See AUDIT.md for the implemented audit fixes and fixed-protocol comparisons. Ownership resets and explicit observation features preserve the original whole-run spell F1 (0.4457), improve equal-duration fold spell F1 from 0.6100 to 0.6200, and slightly reduce dead-ball F1 (0.5974 to 0.5945). Four historical fold outputs reproduce exactly. These results use the same 7,500-frame public sample; changing the split does not create independent validation. The current development revision is v0.3.1, with the cloud-verified v0.3 snapshot retained as a reference.

Prioritize an independent permitted validation set with fixed annotation rules and blinded scoring. Then assess boundary error and stoppage errors separately, and run a fixed-budget reviewer comparison to test whether queries save time without reducing review accuracy. Avoid another same-sample parameter search. The current review improvements are correctness/usability repairs, not a measured human-productivity gain. Video contact recognition and calibrated joint state uncertainty remain future work.


## v0.4 measured feature experiment (21 October 2026)

Seven predeclared groups were evaluated under both existing protocols. See EXPERIMENTS_v04.md and data/experiments/blindspot-v04/RESULTS.md for all outcomes. Contact features improve whole-run spell F1 from 0.4457 to 0.5600, but fixed-time improvement is only 0.6200 to 0.6244. Directional contact/control windows score 0.4667 / 0.6564. Team shape scores 0.3172 / 0.5486, and all features score 0.3515 / 0.6286. The baseline remains selected under the predeclared conservative two-protocol margin.

Contact features reduce wrong-live-team frames from 782 to 512 under whole-run folds and 681 to 457 under fixed-time folds. Stoppage and precise transition timing remain weak: whole-run transition F1 at one second only changes from 0.2222 to 0.2286. This supports investigating contact timing and a separate stoppage estimator rather than adding all features indiscriminately. These are same-sample diagnostics and group-selection results, not independent confirmation; future experiments need additional permitted labelled footage and a fixed evaluation protocol. The full-public refit score (0.9529 spell F1) includes training and cannot establish unseen accuracy.


## Fixed role-isolation experiment v0.4.1

The one proposed hybrid combines baseline P(D) with contacts P(H/A | live), using the same folds and model parameters. Whole-run / fixed-time spell F1 is 0.5000 / 0.6341, versus baseline 0.4457 / 0.6200 and joint contacts 0.5600 / 0.6244. Dead-ball F1 and full-match D/X masks are identical to baseline in every fold. Wrong-team frame counts improve from 782 / 681 to 512 / 447. This verifies that the contact ownership gain can be retained while isolating the stoppage decision; it does not improve stoppage detection and does not produce a robust interval-score gain under the existing release rule.

The model still misclassifies 589 / 747 dead frames as live and 310 / 238 live frames as dead under the two protocols. The next separately evaluated experiment should target P(D) directly rather than add more ownership features. Boundary refinement and additional permitted labelled footage remain subsequent work. This design was suggested after inspecting public diagnostics, so the results remain same-sample development rather than independent confirmation.
