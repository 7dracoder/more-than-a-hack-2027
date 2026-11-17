# Audio feasibility and external model review

A local MP4 container-metadata check finds one audio track in each supplied
feed. It seeks past media payloads and reads handler metadata; it does not
decode audio or video, inspect predictions or fit parameters. Track presence
does not establish audible content, referee whistles or utility for possession.
The receipt is local at `data/research/video-audio-metadata-20261114/`.

[PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) supplies pretrained
sound-event detection models. Its code is MIT. The primary
[Zenodo archive](https://zenodo.org/records/3987831) API specifies CC BY 4.0
for the weights. The candidate DecisionLevelMax file has MD5
`70539c43c18b6a289b3199c503a82c5a`, size 327,428,481 bytes. The published
label index includes Whistle at index 402. These are provenance facts,
not football-specific performance evidence.

The external [whistle pilot](WHISTLE_PILOT.md) has now downloaded and tested
the candidate on volleyball recordings. It has not run on supplied audio.
The subsequent [adapted classifier](WHISTLE_TRANSFER.md) reaches 0.779964
whistle-event F1 on a fresh fourth recording. Football possession utility
remains unverified and no audio component has been integrated.
Training-source overlap,
audio alignment, licensed external validation and the relationship between
whistle events and possession states need assessment before integration.
Generic whistle detection alone cannot establish a spell-F1 gain or reliably
distinguish stoppage whistles from restarts or crowd sounds. No supplied data
were sent to an audio service.

SoccerTrack v2 is an optional external-video resource already reviewed in
[the visibility study](NATURAL_VISIBILITY.md). Its canonical host requires
access-condition acceptance that shares account information. A text-only
access-status question is pending; no conditions were accepted and no gated
files were requested. Current coordinate experiments continue independently.
