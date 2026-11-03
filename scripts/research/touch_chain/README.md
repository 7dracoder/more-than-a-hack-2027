# Touch-chain study (29 Oct 2026) — CPU-only, tracking-only, DFL external matches

Research scripts, not wired into the submission pipeline. Paths assume the box layout
`/workspace/mtah/{repo,cache,dfl,pt.npz}`; `parse.py <MATCH>` builds `cache/<MATCH>.npz` from the
Bassek et al. (2025) Sportec XMLs. `pt.npz` = saved run-58/48 development probabilities for J03WPY.
Train = J03WMX, J03WN1, J03WOH, J03WOY, J03WQQ, J03WR9; dev = J03WPY. No BAR-ATH data read.

Order: feats.py -> gbm.py (frame GBM); touch.py -> kin.py -> seqf.py -> cv3.py (candidate GBM, 2-stage,
leave-one-match-out OOF); fuse.py / ins.py (fusion with run 48); oracle2.py, diag.py, miss.py, crit.py (diagnostics).
Deps: numpy scipy lightgbm scikit-learn numba (no torch needed; r48.py imports possession.fast_decode).
