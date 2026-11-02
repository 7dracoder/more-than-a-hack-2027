# Broadcast feed as extra evidence (negative result, not used in any submission)

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
