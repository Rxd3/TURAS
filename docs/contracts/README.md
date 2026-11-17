# Shared written contracts — version 0.1

These documents specify future interfaces. They are not implemented validators,
executable configuration, sample data, or proof of compliance. Required values
must be supplied from real sources/runs; no synthetic transcript or score records
are included in this scaffold.

| Contract | Authority |
| --- | --- |
| [Data, provenance, IDs, verification and frozen splits](data.md) | M1, reviewed by M6 |
| [Inference input and prediction JSONL](predictions.md) | M6, reviewed by model owners |
| [N-best scores and joint rescoring](nbest.md) | M4/M5, reviewed by M3/M6 |
| [Turkish normalization](normalization.md) | M6, reviewed by M1/model owners |
| [Relational CSV and heterogeneous graph](rdl.md) | M1 schema; M5 graph; M6 leakage review |
| [Checkpoints, experiments and artifact provenance](artifacts.md) | M6, reviewed by producing owner |

WER/CER, terminology, timing and aggregation are normative in
[evaluation](../evaluation.md). A future machine schema must preserve these
semantics; changing them requires a contract version and coordinated migration.
Unknown required provenance blocks a reproducible release instead of being filled
with a guessed path/hash. Optional unknown values use JSON `null`, never NaN,
Infinity, an invented number, or a misleading empty object.
