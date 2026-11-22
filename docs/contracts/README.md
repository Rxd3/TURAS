# Shared written contracts

These documents specify future interfaces. They are not implemented validators,
executable configuration, sample data, or proof of compliance. Required values
must be supplied from real sources/runs; no synthetic transcript or score records
are included in this scaffold. Every contract started at version 0.1 on 2026-11-17;
the version column lists the current text and any proposed amendment, each with
its decision entry in [decisions](../decisions.md) (D17–D21) and
[owner decisions](../owner_decisions.md) (D24, D42).

| Contract | Authority | Version |
| --- | --- | --- |
| [Data, provenance, IDs, verification and frozen splits](data.md) | M1, reviewed by M6 | 0.3 (owner adopted 2026-11-21, see [D42](../owner_decisions.md); independent M6 release approval pending); 0.2 owner adoption for M4/M5 integration: 2026-11-19, see [D24](../owner_decisions.md) |
| [Inference input and prediction JSONL](predictions.md) | M6, reviewed by model owners | 0.2; owner adoption for M4/M5 integration: 2026-11-19, see [D24](../owner_decisions.md); project approval, not independent peer-review sign-off |
| [N-best scores and joint rescoring](nbest.md) | M4/M5, reviewed by M3/M6 | 0.2; owner adoption for M4/M5 integration: 2026-11-19, see [D24](../owner_decisions.md); project approval, not independent peer-review sign-off |
| [Turkish normalization](normalization.md) | M6, reviewed by M1/model owners | 0.2; owner adoption for M4/M5 integration: 2026-11-19, see [D24](../owner_decisions.md); project approval, not independent peer-review sign-off |
| [Relational CSV and heterogeneous graph](rdl.md) | M1 schema; M5 graph; M6 leakage review | 0.1 |
| [Checkpoints, experiments and artifact provenance](artifacts.md) | M6, reviewed by producing owner | 0.1 |
| [Evaluation and timing policy](../evaluation.md) | M6 | 0.2 (proposed by M6, pending review by M1 and model owners M2–M5); owner adoption for M4/M5 integration: 2026-11-19, see [D24](../owner_decisions.md); project approval, not independent peer-review sign-off |

WER/CER, terminology, timing and aggregation are normative in
[evaluation](../evaluation.md). A future machine schema must preserve these
semantics; changing them requires a contract version and coordinated migration.
Unknown required provenance blocks a reproducible release instead of being filled
with a guessed path/hash. Optional unknown values use JSON `null`, never NaN,
Infinity, an invented number, or a misleading empty object.
