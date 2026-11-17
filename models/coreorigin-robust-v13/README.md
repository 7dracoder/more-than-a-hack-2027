# External robustness-model copy

`selected.pt` and `fixed_protocol.json` are an unchanged copy of the historical
`data/local-v08/run-13-physical-boundary` checkpoint and protocol. They are
retained here so the external temporal-fusion experiment has no hidden dependency
on an old local training directory. This copy does not include any BAR-ATH input,
prediction or label.

Training uses the six declared external Sportec/DFL matches; J03WPY is development
only. The model consumes the first 664 v09 coordinate-only features at sample
step 2. The corresponding full-rate v25 prefix was checked for exact equality
on both native/corrupted development halves before fusion. Published external
native spell F1 0.8374845869297165 and synthetic-gap spell F1
0.6870588235294118 are reproduced before any mixture is selected.

Checkpoint SHA256:
`ec72a2bb8ef14f3fee68daed4b4a0944de5f18fc2b86ac07f46b1aa771a0f03c`.

Attribute the external data to Bassek et al. (2025), Sportec Open DFL dataset,
CC BY 4.0, DOI `10.1038/s41597-025-04505-y`. No claim of independent test or
BAR-ATH accuracy is made. See `scripts/research/spatial_memory/README.md` for
the external selection and inference commands.
