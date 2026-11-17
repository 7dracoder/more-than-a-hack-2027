# External whistle-detection pilot

The public [Volleylitics dataset](https://huggingface.co/datasets/GYdevy/volleyball-whistles)
declares MIT licensing and provides manually timed volleyball whistles.
Revision `5f23e58f6057f82d40fa8a132da09ba70ccf02ce` is pinned. Only recordings
1, 2 and 3 were acquired. Their LFS SHA256 hashes match. The annotation source
is `whistles_all.json`, using `t_raw`; DSP-anchored annotations are not used.
This is cross-sport acoustic validation, not football possession truth.

The frozen [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) source is
commit `d2f4b8c18eab` (full revision in the acquisition receipt), MIT code.
The [DecisionLevelMax checkpoint](https://zenodo.org/records/3987831) carries
CC BY 4.0 metadata and matches its published MD5. Class indices 402/40 are
verified against the downloaded label index. Their maximum forms the neural
whistle score. A fixed spectral peak-energy baseline provides a comparison.
The three source files, checkpoint and notices total about 1.04 GB locally.

The audio dependencies were added without changing Torch 2.11.0+cu128 or
NumPy 2.5.3. GPU preflight passes. Inference uses frozen weights, 32 kHz mono,
two seconds of context and 10.24-second central chunks aligned to the CNN
stride. Full-recording scores are frozen before threshold fitting. Inputs are
external only; no supplied audio is decoded or uploaded. Source/input hashes,
package versions and isolation receipts are preserved locally.

The initial protocol calibrates thresholds on match1, chooses between detectors
on match2 and freezes the selection before scoring match3. Event matching uses
0.5-second tolerance and one-to-one maximum cardinality, independently checked
with bipartite matching. Three focused matcher/extraction tests pass.

The first grid (0.05-0.9) selects the spectral detector at threshold 0.8. Its
development F1 is 0.160883. On the frozen final recording it has 34 TP, 82 FP
and 294 FN: precision 0.293103, recall 0.103659, event F1 0.153153. This does
not justify using it as a possession-state anchor.

The neural detector returns no events at the first grid's chosen threshold.
A subsequent calibration-only diagnostic finds its maximum match1 confidence
is 0.027628, below the original grid. That grid was inadequate for calibration.
A wider threshold follow-up uses only match1 and match2, preserving the
original trial and never reusing match3. Threshold 1e-8 gives calibration F1
0.280242 and development F1 0.231474, with development precision 0.183862.
This is a provisional development result, not a new held-out result. Any new
detector needs a fresh final recording. A plausible next experiment is to
adapt a classifier on frozen audio representations, rather than rely on poorly
calibrated raw AudioSet class outputs.

FSD50K was reviewed but not acquired: its primary release asks users interested
in ML competitions to contact the authors. No messages were sent. SoccerTrack
access-status clarification remains pending; no access conditions were accepted.

All evidence is under ignored `data/external/volleyball-whistles-5f23e58f6057/`
and `data/research/whistle-pilot-20261114/`. These are whistle-event metrics,
not possession spell F1. No detector is integrated or promoted; the selected
v4 model, exact CSV and technical PDF remain unchanged. The BAR-ATH target is
still unverified and requires permitted organiser scoring.

The follow-up [frozen-feature classifier](WHISTLE_TRANSFER.md) has completed
its fresh recording-4 test, reaching 0.779964 whistle-event F1. It does not
reuse the original recording-3 final test or establish a possession gain.
