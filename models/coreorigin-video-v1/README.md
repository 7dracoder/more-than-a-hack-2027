# Experimental external-only video-input protocol

This is **not a selected submission**. Its frozen final candidate scores 0.5200
spell F1 on one official public five-minute pipeline check, with no subsequent
tuning or selection. Private full-match F1 and the 0.80 target remain unverified.
Frozen v11 is preserved separately.

- `selected.pt` and `fixed_protocol.json` are an exact copy of the existing
  external run-58 parent. The weights were not fitted on BAR-ATH. The protocol
  preserves the full strong/weak external-data declarations and warm-start
  history; this is not a new independently tested model.
- `confidence_evidence.json` contains the fixed class-balanced scalar fit on
  112 development images from a licensed COCO 2017 subset. Twenty-eight images
  are reserved for external validation. The small sample does not establish
  football-video accuracy, and the evidence is not a BAR-ATH posterior.
- `motion.json` contains a 99.9th-percentile live-ball speed bound fitted on the
  six existing Sportec training matches. Credit Bassek et al. (2025),
  DOI 10.1038/s41597-025-04505-y; dataset CC BY 4.0.
- `coco_attribution.json` retains the image references/licenses used by the
  external fit. COCO Consortium annotations are CC BY 4.0. Only images explicitly
  marked CC BY or CC BY-SA were used. No image bytes are included.
- The existing root `yolo11m.pt` is the COCO-trained Ultralytics detector,
  AGPL-3.0. Its exact hash is in `model_manifest.json`. It is used for local
  inference only, with no training on the provided video.

No supplied video, tracking, frame-level predictions or detections are in this
bundle. Source, commands and the controlling participant clarification are
documented in `scripts/research/video_ball/README.md` and `RULES_CLARIFICATION.md`.
All match-derived outputs stay private under ignored `data/`.

[Technical draft](TECHNICAL.md) describes the supplied inputs, complete declared
external training history, possession rules, current verification and limits.
