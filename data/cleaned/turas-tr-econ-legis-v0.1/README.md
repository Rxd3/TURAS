# Cleaned dataset `turas-tr-econ-legis-v0.1`

The cleaned TURAS benchmark used for every reported result: 5,450 Turkish
parliamentary (TBMM) clips on the economy and legislation, 25 h in total. It is
derived from [`disco-eth/WorldSpeech`](https://huggingface.co/datasets/disco-eth/WorldSpeech)
`tr_tr` (revision `b76b8539`) by
[`scripts/m1_build_release.py`](../../../scripts/m1_build_release.py); the cleaning
steps and counts are in the [main README](../../../README.md#dataset) and the
[dataset card](../../../docs/dataset.md).

License: CC BY-NC 4.0 (WorldSpeech), source recordings from the Turkish Grand
National Assembly (<https://www.tbmm.gov.tr/>). Non-commercial use with
attribution only.

| Path | Contents |
| --- | --- |
| `audio/<utterance_id>.flac` | 5,450 clips, 16 kHz mono FLAC, 10–20 s each |
| `turas-tr-econ-legis-v0.1.inference.{train,validation,test}.jsonl` | Model inputs in split-file order: IDs, audio path and hash, duration; no transcript text |
| `turas-tr-econ-legis-v0.1.references.train.jsonl` | 4,376 train references (official TBMM minutes) with source, timing, session, verification and topic fields |
| `turas-tr-econ-legis-v0.1.references.validation.jsonl` | 438 validation references: the working variant used for every reported validation number (`owner-recovered-validation-minutes-v1`, SHA-256 `d0c8519c…`, described in `references.validation.VARIANT.json`). All are the raw minutes, unreviewed |
| `turas-tr-econ-legis-v0.1.references.test.jsonl` | 636 test references, same fields. Use for final scoring only |
| `relational/train/`, `relational/validation/`, `relational/test/` | Relational tables extracted from the references (utterances, quantities, relationships, provenance). Only `train/` may enter the RDL training graph |
| `exclusion_ledger.jsonl.gz` | Removal log: one line for each of the 239,654 upstream rows with its TURAS split or the first reason it was removed, plus matched domain terms and issue codes. Rebuilt from the [raw data](../../raw/worldspeech-tr_tr-b76b8539/README.md) by `scripts/m1_export_raw.py`; its SHA-256 (uncompressed) equals the frozen `exclusion_ledger_sha256` |
| `SHA256SUMS` | SHA-256 of every file in this folder (`sha256sum -c SHA256SUMS`) |

Split membership is fixed by [`data/splits/turas-tr-econ-legis-v0.1/`](../../splits/turas-tr-econ-legis-v0.1/)
(split manifest SHA-256 `a8fff65b…095b8`). Splits are grouped by recording
(session); there are no speaker IDs.

Checks done when this folder was added (2026-11-22): the three release bundles
matched the SHA-256 values published at the freeze (2026-11-18); all 5,457 bundle
checksums passed; all 5,012 train and test reference rows match the hash of their
audio file; the bundle split files are byte-identical to `data/splits/`.

The validation references and `relational/validation/` come from the working
validation bundles of the 2026-11-22 handoff (their SHA-256 values matched the
handoff checksums). The handoff's evaluator bundle holds byte-identical train and
test references and test relational tables.

References are the upstream minutes (edited, not verbatim), except 13 clips (3
train, 10 test) corrected after listening; 35 reference rows are `verified`. The listening review
records are in [`data/verification/`](../../verification/) and
[`docs/m1_deadline_handoff.md`](../../../docs/m1_deadline_handoff.md).
