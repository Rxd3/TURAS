# Data area — metadata only at initialization

There is no approved dataset release and no audio or transcript has been added.
WorldSpeech `tr_tr` and the legislative/financial domain are working decisions;
read the [dataset plan](../docs/dataset.md) and [data contract](../docs/contracts/data.md).

| Area | Future reviewed contents |
| --- | --- |
| `metadata/` | Candidate/source/rights inventory, release cards and actual statistics |
| `manifests/` | Small release manifests/hashes; protected utterance/reference manifests distributed separately |
| `splits/` | Immutable train/validation/test ID lists and grouped-split manifests |
| `verification/` | Annotation policy and aggregate verification/audit evidence, without personal data |
| `relational/` | Documented relational table schema and approved small inventories |

Raw/processed audio, downloads, private references, bulky tables and local
databases belong in ignored locations or external hashed bundles. Permission to
use a source does not imply permission to redistribute it. Do not add fabricated
rows, transcripts or split lists. `.gitkeep` retains otherwise empty metadata areas.
