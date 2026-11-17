# Broadcast feed as extra evidence (negative result, not used in any submission)

## Local two-feed experiment, 13 November 2026

The newly supplied broadcast bundle is present locally. `dual_feed.py` provides
a separate experimental preparation path with no `/tmp` dependency or
match-specific offset seeds. It hashes both bundles, checks sequential ATD and
homography timelines, automatically samples video decodes without rendering
images, and synchronizes each half using team-aware player geometry. Team names
are checked and slot/team ordering may differ between cameras. Every aligned
frame is screened separately so a constant offset does not authorize filling
from replays, close-ups or unmatched geometry.

The supplied broadcast video reports 164,487 frames at 25 fps; its ATD and
homography have 164,488 records. This discrepancy is recorded explicitly.
Broadcast pixels are **not** used for new detection or assumed frame-aligned to
the ATD. The broadcast ATD is synchronized independently to the tactical ATD;
the tactical video still supplies the previously frozen detector hypotheses.
No implicit one-frame correction is made.

Fusion preserves all original tactical ball observations. A broadcast proposal
requires the pre-existing geometric acceptance rule, reachability from original
anchors, and adjacent motion support under the externally fitted speed bound.
Existing verified tactical-video hypotheses are a lower-priority fallback and
must also pass motion checks at cross-source junctions. Missing or ambiguous
evidence stays missing. The settings are written before processing; no
possession truth is read and no provided-video validation or tuning occurs.

```powershell
.\.venv\Scripts\python.exe scripts/research/broadcast/dual_feed.py --out data/research/dual-feed-20261113-v2 --tactical-hypotheses data/research/target080/ball_hypotheses.npy
.\.venv\Scripts\python.exe scripts/research/video_ball/export_candidate.py --ball data/research/dual-feed-20261113-v2/ball_hypotheses.npy --out data/research/dual-feed-20261113-v2/candidate
# Repeat the CPU export into a separate folder before verification:
.\.venv\Scripts\python.exe scripts/research/video_ball/export_candidate.py --ball data/research/dual-feed-20261113-v2/ball_hypotheses.npy --out data/research/dual-feed-20261113-v2/candidate-repeat
.\.venv\Scripts\python.exe scripts/research/broadcast/verify_dual_feed.py --run data/research/dual-feed-20261113-v2 --receipt-name verification-final.json
.\.venv\Scripts\python.exe scripts/research/broadcast/benchmark_dual_feed.py --out data/research/dual-feed-external-20261113
.\.venv\Scripts\python.exe -m pytest tests/test_dual_feed.py tests/test_video_fusion.py tests/test_automatic_kickoff.py tests/test_dead_ball_rules.py -q --basetemp=data/test-tmp/dual-feed-complete
```

The external diagnostic is bounded to two fixed seeds on J03WPY, a development
match rather than an independent test. It compares synthetic independent
five-second block losses with and without the same fusion method. Sensor
settings do not use BAR-ATH data or statistics; an I/O audit guard denies access
to competition assets and public possession labels. This deliberately simplified
experiment cannot reproduce the correlated failures of real cameras and is not
an estimate of BAR-ATH accuracy. It does not select a competition candidate.

The completed two-seed diagnostic reports mean spell F1 **0.67152 -> 0.72780**
(+0.05627). Seed 713 gives 0.67003 -> 0.72084; seed 714 gives
0.67302 -> 0.73475. These are synthetic-loss J03WPY development results only.
The earlier proxy study used a different degradation protocol and found harm;
the results must not be pooled or used to assert a BAR-ATH improvement.

On the supplied tracking, the fixed preparation retains 94,214 original ball
observations in the graded windows, adds 5,477 broadcast proposals and 16,705
tactical-video fallback hypotheses, and leaves 42,461 frames missing. These
counts are coverage diagnostics, not coordinate or possession accuracy.
Automatic geometric synchronization estimates tactical = broadcast + 4,013
in the first half and +26,920 in the second. These are data-derived geometric
matches, not manual video or possession-transition anchors; they are not a
verification of broadcast pixel timing.

The experimental final CSV has 199,319 rows, passes the unchanged official
format validator and matches a second CPU export byte for byte (both raw and
final outputs). Final SHA256:
`9838ecbd8ef686eb3edbb80f702ae4e1fab25b50b8983bb145418a3932264265`.
The input-integrity verifier also repeats coordinate fusion and source
assignments, preserves every original observation and verifies the unchanged
parent. Receipts are in the ignored run directory. No new BAR-ATH possession
score was computed during candidate generation and no replacement submission was selected. All 20 focused
tests in the command above passed.

### Subsequent frozen public-window pipeline check

After the user authorized checking the frozen candidate, exactly that final CSV
was copied into an isolated scoring directory containing no raw/control
alternatives. Its SHA256 was checked against `verification-final.json` before
the public labels were opened. `public_pipeline_check/freeze.json` records the
candidate hash, official scorer hashes, 1,000 bootstrap draws and the commitment
not to train, tune or select candidates from the result.

The unchanged official scorer exited successfully and reported **0.52000 spell
F1** and **0.630610 macro F1** on **7,500 graded frames / 17 possessions**. The
spell-F1 5th–95th percentile bootstrap range is **0.38–0.66**; the macro-F1 range
is **0.497916–0.706838**. Both staged and original CSV hashes remained unchanged.
No transition-level reports or video overlays were inspected. The official
reports, stdout/stderr, freeze receipt and summary are retained privately under
`data/research/dual-feed-20261113-v2/public_pipeline_check/`.

The 0.80 target was not met on this public sample. This check does not establish
an improvement on BAR-ATH or determine private full-match performance. No
parameter/model change or replacement selection followed the check. Further
training or tuning must use external data only; do not rescore alternatives to
turn this public sample into a selection set.

All match-derived coordinates, mappings, receipts and CSVs stay under ignored
`data/`. The existing submission and pretrained weights remain unchanged. The
earlier negative proxy findings below remain material evidence; neither extra
ball coverage nor a successful CSV validator establishes better possession F1.

## Historical proxy study

The broadcast tracking is aligned to the tactical timeline from player positions alone: a coarse
chamfer search (`coarse.py`) then per-frame refinement (`fine.py`) give a constant offset per half,
tactical = broadcast + 4014 (first half) and + 26920 (second half), with no drift. Replays and
close-ups simply have no matching tactical frame. Where both feeds see the ball they agree to a
median 0.75 m (90th percentile 5.4 m). `load.py` parses both sources; caches live under `/tmp`.

Two uses were tested; both lower J03WPY proxy spell F1, so neither is applied.

| Lever | Script | BAR-ATH effect | Proxy test | Proxy spell F1 change |
|---|---|---|---|---|
| View as stoppage evidence: P(D) times P(view \| stoppage) / P(view \| play) from a label-free latent-class model (`latent.py`) | `apply_view_prior.py` | D in live views 0.139 -> 0.101, in close-ups 0.733 -> 0.810; splits 3 -> 10 | `proxy_view.py`: views simulated per DFL state from BAR-ATH view Markov chains | -0.021 at weight 1, worse at every weight |
| Broadcast ball fills tactical ball gaps (live views only) | `apply_ball_fill.py` | 5,706 of 64,643 missing frames filled; 3.5% of labels change | `proxy_ball_fill.py`: true DFL ball revealed at the BAR-ATH fill rate with BAR-ATH-sized error | -0.008 mean, negative in all four runs |

Why they fail: the model already sees stoppages through ball loss and player slowdown, the
director cuts to and from close-ups seconds after the whistle and before the restart (boundary
errors cost spell F1), and the broadcast loses the ball at the same moments as the tactical feed.

Only label-free aggregates (view transition matrices, fill rate, error quantiles) were used on
the proxy host; no BAR-ATH tracking left the local machine.
