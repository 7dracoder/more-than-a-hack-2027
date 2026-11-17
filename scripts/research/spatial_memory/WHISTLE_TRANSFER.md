# External whistle classifier adaptation

The adapted audio head achieves whistle-event F1 **0.779964** on the predeclared
fresh volleyball recording `match4`: 218 TP, 34 FP, 89 FN, precision 0.865079,
recall 0.710098 at a 0.5-second matching tolerance. This is a component result,
not BAR-ATH possession spell F1. It does not justify changing the submission.

## Data and selection

The public [GYdevy volleyball corpus](https://huggingface.co/datasets/GYdevy/volleyball-whistles)
is pinned to `5f23e58f6057f82d40fa8a132da09ba70ccf02ce`; its dataset card declares
MIT. Only raw manual `t_raw` timestamps are used, with no feature-anchored labels.
Recording 1 supplies training labels, recording 2 supplies threshold selection,
and recording 4 is the fresh final test. Recording 3 was evaluated in the earlier
pilot and is not reused here. Recordings may share venues or competition
conditions; one fresh file does not establish independence across environments.

The backbone is frozen PANNs Cnn14 DecisionLevelMax, with MIT source and
[CC BY 4.0 weights](https://zenodo.org/records/3987831), attributed to Kong et al.
The checkpoint has published MD5 `70539c43c18b6a289b3199c503a82c5a`.
AudioSet pretraining overlap is not independently certified.

## Approach

Extract the 2048-dimensional post-ReLU features immediately before
`fc_audioset`, using the unchanged pilot frontend: mono 32 kHz, 10.24-second
chunks, two seconds of context on each side. Keep the native stride of 0.32
seconds. Reconstructed pretrained class outputs match the original extraction
within 2.8e-8; chunk phase and native-to-dense frame mapping are checked.

Concatenate previous/current/next native embeddings (6144 dimensions), fit
StandardScaler on included recording-1 samples only, then fit balanced logistic
regression with C=0.01. Positive samples lie within 0.32 seconds of a raw whistle
point; negatives are at least two seconds away; ambiguous samples are excluded.
The head converges in 85 iterations on 11,175 included samples, 574 positive.
Features use future context, so this is an offline detector.

Choose the threshold from the fixed grid using recording-2 event F1, breaking
ties toward the higher threshold. The selected threshold is 0.9, with development
F1 0.741656 (300 TP, 64 FP, 145 FN). Freeze the model and threshold before
acquiring and evaluating recording 4. No choice is made using its score.
The earlier recalibrated raw-class output had development F1 0.231474 on
recording 2; this comparison is on reused development data, not an independent
estimate of the improvement.

Event extraction joins gaps up to 0.1 seconds, chooses the strongest peak, and
applies a 0.35-second refractory interval. Final point matches are independently
checked by maximum bipartite matching. Independent normalization, label
membership, functional probabilities, and development metrics are verified.
Float32 functional probability differences are at most 3.58e-7 and leave event
metrics unchanged. The head SHA256 is
`212c51b7115e063009a7deec7be32ea6e014b2c7bb77bab4e00d7fdac848edac`.

## Reproduction and limitations

Use `requirements-audio.txt` with the existing CUDA PyTorch environment.
The config is `configs/audio_whistle_transfer.json`; scripts are
`extract_whistle_embeddings.py`, `train_whistle_transfer.py`,
`verify_whistle_transfer.py`, `extract_whistle_test_embeddings.py`, and
`evaluate_whistle_transfer.py` in this directory. All scripts offer CLI help.
The final evaluator refuses to overwrite a completed final-test result.
Receipts, scores and hashes are under ignored
`data/research/whistle-transfer-20261115/`. Six event/mapping/target tests pass.
Isolation guards record no access to competition assets, predictions or truth.

No supplied audio was decoded or used for fitting. No audio predictions are
integrated into possession; v4, its CSV and technical PDF remain unchanged.
Whistles can indicate restarts as well as stoppages. Football audio transfer,
clock alignment and a learned relationship to possession states need external
evidence before any integration or promotion.

An additional primary-source search found that
[SoccerNet-Echoes](https://github.com/SoccerNet/sn-echoes) publishes ASR text,
which cannot validate acoustic whistle detection. The
[RoboCup whistle resources](https://spl.robocup.org/datasets/) concern robot
soccer, not professional broadcast possession. Neither is treated as evidence
for a football possession gain. SoccerTrack access remains optional and pending;
no gated access conditions have been accepted.
