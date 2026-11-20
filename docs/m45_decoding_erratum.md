# M4/M5 effective decoder-width correction

Correction version: `m45-effective-decoding-v1`, owner decision D41.
The frozen protocol, model weights, predictions, scores and reported metrics are
unchanged. This corrects the interpretation of a generic configuration field.

| Family | Actual decoding used in the completed study | Candidate evidence |
| --- | --- | --- |
| Whisper parent, LoRA and full, including joint RDL | 5 beams, 5 requested hypotheses | `score_details.num_beams = 5` |
| CTC parent and adapted, including joint RDL | Prefix beam width 10, 5 requested hypotheses | `score_details.beam_size = 10` |

The frozen adapter forwards `beam_size` to CTC only. Its Whisper path calls
`whisper_nbest(requested_n=5)`, which invokes core GenerationMixin with
`num_beams=requested_n` and `num_return_sequences=requested_n`. The shared
protocol's `decoder.beam_size=10` was therefore not an effective Whisper setting.
Earlier prose describing every system as beam 10 was incorrect. The per-candidate
score details already recorded the actual values correctly.

[The source-and-export audit](../results/final/m45_decoding_provenance_correction.json)
checks all 28 native/control/timing N-best exports and the original frozen source
hashes. Every observed Whisper candidate records 5 beams; every CTC candidate
records prefix width 10. No reference transcript is read by this audit.

All corresponding parent/adapted systems used the same actual family-specific
decoder. Validation selection and final matched comparisons remain valid; this
correction does not require retraining, new test inference or changed metrics.
Original manifests/hashes are preserved as historical bytes. Use this versioned
sidecar and the frozen code/candidate evidence when interpreting their generic
beam field. A different decoding experiment needs its own new protocol.

For M2: beam-5 Whisper settings agree with the measured M4/M5 Whisper width.
The earlier warning that M2 must switch Whisper to beam 10 was wrong. Preserve
the full remaining checkpoint, precision, Turkish prefix, scoring and decoding
policy when checking compatibility; matching beam width alone is not sufficient.

The independent 2026-11-17 M6 smoke run used different settings, including
Whisper beam 10 with requested N=10. Its historical records are not changed by
this correction to the later completed M4/M5 N=5 study.
