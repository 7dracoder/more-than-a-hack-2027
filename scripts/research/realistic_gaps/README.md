# Realistic degradation of external training data (CoreOrigin v10)

The DFL training matches are cleaner than BAR-ATH in two ways that matter for stoppages:

- **The DFL ball is parked during every stoppage.** On J03WPY, 97% of dead-ball frames move
  less than 1 m in a second and 60% sit outside the pitch. On BAR-ATH only 0.08% of visible
  frames repeat the previous ball position. A model trained on DFL learns "frozen ball means
  dead", and that cue never fires on BAR-ATH.
- **BAR-ATH loses the ball mostly during stoppages.** Inside BAR-ATH's missing-ball gaps,
  players move at 0.81 / 0.70 / 0.56 / 0.39 of their in-play speed (5-10 / 10-20 / 20-60 / 60+ s
  gaps). If ball loss were independent of game state, the ratio would be about 0.97, which is
  what the old random corruption produced.

Everything here uses aggregate, label-free BAR-ATH statistics only. No BAR-ATH labels, public
window included, and no frame-level BAR-ATH data leave the local machine.

| file | role |
|---|---|
| `degradation_profile.py` | BAR-ATH ball-loss hazard by context, gap and fragment lengths -> `data/research/bar_ath_degradation_profile.json` |
| `simulate.py` | Degrades DFL tracking: unfreeze the parked ball, hide it over a share `q` of labelled DFL stoppages, fill the rest with capped in-play gaps |
| `calibrate.py` | `--bar-ath` writes BAR-ATH gap statistics; the sweep picks `q`, the hidden fraction and the gap cap by matching gap counts, shares and player-speed ratios |
| `evaluate_views.py` | Scores a checkpoint on clean, old-corruption and realistic J03WPY views |
| `finetune.py` | Fine-tunes the corrector of a residual checkpoint (teacher frozen) on the degraded views and selects on degraded J03WPY views |

## Result

The calibration sweep (`data/research/calibration_v2.json`) selected `q=0.85`, whole stoppages
hidden, and in-play gaps capped at 20 s. Starting from run 58, the corrector was fine-tuned for
9 epochs of 150 steps on native, carry and two state-tied views of the six training matches
(Colab L4, about 16 s per epoch).

On this proxy, unchanged run 58 predicts dead time inside each gap-length group about as often
as it does on BAR-ATH. Dead shares on proxy vs BAR-ATH: visible 0.02 vs 0.01, 5-10 s gaps 0.24
vs 0.12, 10-20 s 0.46 vs 0.55, 20-60 s 0.90 vs 0.87.

| | J03WPY proxy spell F1 | J03WPY clean spell F1 | BAR-ATH predicted dead share |
|---|---|---|---|
| run 58 | 0.688 | 0.847 | 0.299 |
| v10 (this) | 0.730 | 0.837 | 0.348 |

On BAR-ATH, v10 predicts dead time in more of the gaps: 5-10 s gaps 0.12 -> 0.32, 10-20 s
0.55 -> 0.74, 20-60 s 0.87 -> 0.94. It inserts only 4 short dead gaps that split a possession,
against 16 for the v09-oob rule.

The proxy is still easier than BAR-ATH: run 58 scores 0.688 on it against 0.505 public. Treat
the gain as a ranking signal, not a score estimate. BAR-ATH predicted dead time (0.35) also
remains below the expected 0.44, and the simulator produces fewer 5-10 s gaps than BAR-ATH
(12 vs 26 per half).

## Reproduce

Weights are committed in `models/coreorigin-v10/`. CPU inference is byte-deterministic:

```bash
mkdir -p data/local-v10/state2 data/submissions/CoreOrigin-v10
cp models/coreorigin-v10/selected.pt models/coreorigin-v10/fixed_protocol.json data/local-v10/state2/
python scripts/run_local.py --config configs/local_v26_pretrained_attention.json \
    --out data/local-v10/state2 --infer --device cpu
python scripts/apply_published_anchors.py --candidate data/local-v10/state2/candidate/CoreOrigin.csv \
    --out data/submissions/CoreOrigin-v10/CoreOrigin.csv
python evaluation/validate_submission.py data/submissions/CoreOrigin-v10/CoreOrigin.csv
```

Expected SHA256 values:

- checkpoint `d2449c5a47534c702a17853a093dee8fc60a5e90ef571e848bd9bd70d4bcfd20`
- model CSV `0aa6c922eaf001d6d4b7851c74d377f79e0a6b1fecf848d9f85fab5e0fe14bdf`
- submission `3e09f45559c52988ca3c2cd14f83c4066e033f63c4988b6cd571a2bcfeafb903`

These were verified with torch 2.15.0 CPU, numpy 2.4.6, scipy 1.17.1 and Python 3.11.

To retrain (GPU; not bit-deterministic, which is why the weights are committed):

```bash
python scripts/research/realistic_gaps/degradation_profile.py          # BAR-ATH profile (local)
python scripts/research/realistic_gaps/calibrate.py --bar-ath           # BAR-ATH gap reference (local)
python scripts/research/realistic_gaps/calibrate.py --q 0.7 0.85 1.0 --fraction 1 0.5 0.25 --cap 0 20 \
    --out data/research/calibration_v2.json
python scripts/research/realistic_gaps/finetune.py --checkpoint <run-58 selected.pt> \
    --profile data/research/bar_ath_degradation_profile.json --out <dir> \
    --train-views native carry stateA stateB --dev-views clean stateA stateB \
    --stoppage-q 0.85 --stoppage-fraction 1 --max-alive-gap 20
```
