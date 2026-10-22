# BlindSpot v0.5: structural-prior verification

## Change and review

The user added three fixed structural variants: baseline ownership plus an out-of-play prior, team-only contacts plus that prior, and optional contact-aligned H/A boundaries. They were rerun in a separate `data/experiments/blindspot-v05` directory after review. The received files are preserved under `received-source/`; older experiment outputs remain available.

The prior uses the observed ball more than 1 m outside the tracked pitch, raises its dead probability to at least 0.999, and adds 2 log-odds during a hold. The hold ends on an observed inside-pitch movement above 5 m/s and below 60 m/s, or after 1,500 frames. It resets between halves; missing observations never initiate it. This is probabilistic evidence passed through the decoder, not a guarantee that every flagged frame decodes D. Tracking geometry alone cannot prove a referee's stoppage or restart decision.

Contact snapping moves a decoded H/A boundary towards a new-team 1 m contact within 25 frames. Review found that adjacent moves could cross and erase a short possession. For example, `HHHHHAAAAAHHHHH` with an away contact at frame 9 and home contact at frame 6 became all H in the received implementation. The corrected algorithm tracks already-moved neighbouring boundaries, rejects crossing, and checks that ordered spell labels and all D/X frames are preserved. The regression fixture now retains H/A/H. All numerical constants and classifier parameters remain unchanged.

The received design checked public labels before scoring: all 92 observed public frames outside the 1 m margin were D. Consequently this is diagnostic-informed development on previously inspected labels, even though constants were fixed before the aggregate score run. The numerical source's original description of label independence must be read with this qualification.

## Results

| Method | Whole-run spell F1 | Fixed-time spell F1 | Whole-run dead F1 | Fixed-time dead F1 |
|---|---:|---:|---:|---:|
| baseline | 0.4457 | 0.6200 | 0.6619 | 0.5945 |
| contacts | 0.5600 | 0.6244 | 0.6906 | 0.5850 |
| contacts_team_only | 0.5000 | 0.6341 | 0.6619 | 0.5945 |
| baseline_outplay | 0.4571 | 0.6200 | 0.7066 | 0.6458 |
| team_only_outplay | 0.5111 | 0.6341 | 0.7066 | 0.6458 |
| team_only_outplay_snap | 0.5111 | 0.6244 | 0.7066 | 0.6458 |

The unsnapped team-only out-of-play variant improves baseline spell F1 by 0.0654 / 0.0141. No approach clears the existing >0.06 requirement in both protocols, so v0.3.1 remains selected. This threshold is a conservative selection heuristic, not a significance test. Joint contacts still has the highest whole-run spell score among these approaches. No single new method wins every comparison.

| Pooled error count (whole-run / fixed-time) | Baseline | Team-only out-of-play |
|---|---:|---:|
| Wrong live team | 782 / 681 | 512 / 447 |
| Dead called live | 589 / 747 | 496 / 654 |
| Live called dead | 310 / 238 | 312 / 240 |
| Total wrong frames | 1,681 / 1,666 | 1,320 / 1,341 |
| Matched transitions within 1 s | 4 / 7 | 5 / 9 |

Total frame errors decline 21.5% / 19.5%. The prior fixes some dead frames while adding two false dead frames per protocol. In-pitch stoppages and short possessions remain unresolved. Snapping lowers fixed-time spell F1; the crossing correction changes five full-match frames in three of eight fold outputs, but leaves the public aggregate scores unchanged.

## Verification and export

All 57 tests pass, including four additional structural tests covering half resets, hold expiry, adjacent crossing, and 1,000 randomized sequences with team-swap checks. All 48 fold CSVs were independently format-validated and rescored using the unchanged official metric; scores, confusion matrices, source and output hashes match. All 24 baseline/contact/team-only control CSVs reproduce v0.4.1 byte-for-byte. Ordered spells and D/X masks are preserved in all eight snapped outputs.

The experimental full refit validates with 199,319 rows and SHA256 `f1c6d022e84a6df92f514923b50e03f7f4065e72f4093650814cee28e4c5f9d2`. It reproduces the received unsnapped export exactly and stays separate from the selected v0.3.1 CSV (`c572620d15f0c60d9cf14df1b073131e91cb0153dd1c892ede0bd0fd5c8fc46e`). It fits all public labels; direct public scoring would measure training fit.

Both protocols reuse the same 7,500 public frames. The 300-frame embargo excludes training labels, while offline raw tracking context can span whole halves. No private-match labels, independent validation, new video recognition, or v0.5 cloud execution were used. The separate v05 source ZIP and notebook are prepared for Colab CPU; prior cloud parity applies only to its original release.

## Reproduce

```powershell
$env:LOKY_MAX_CPU_COUNT='2'
python -m unittest discover -s tests
python scripts/run_v04_experiments.py baseline contacts contacts_team_only baseline_outplay team_only_outplay team_only_outplay_snap --out data/experiments/blindspot-v05
python scripts/report_v04_results.py --experiment data/experiments/blindspot-v05
python scripts/run_v04_experiments.py --candidate team_only_outplay --out data/experiments/blindspot-v05
python evaluation/validate_submission.py data/experiments/blindspot-v05/team_only_outplay/candidate/CoreOrigin.csv
python scripts/build_v04_colab.py --structural
```

Use the matching `BlindSpot_v05_project.zip` and `notebooks/BlindSpot_v05_Colab.ipynb`. The notebook conservatively exports baseline unless the same two-protocol rule is met. Match-derived fold evidence and experimental predictions remain in local private result archives. No competition submission was made.
