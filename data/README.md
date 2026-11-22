# Data area — metadata, splits, and relational schema

This directory contains the public metadata, split ID lists, relational schema
tables, and verification/audit records for the working release
`turas-tr-econ-legis-v0.1` (5,450 Turkish legislative/economic speech clips,
~25 h total, temporal train/validation/test split derived from
`disco-eth/WorldSpeech` `tr_tr`, CC BY-NC 4.0).

See [docs/dataset.md](../docs/dataset.md) and
[docs/contracts/data.md](../docs/contracts/data.md) for full provenance,
inclusion/exclusion rules, and known limitations (611 of 636 test clips use
unreviewed edited TBMM minutes; 25 clips carry M1-reported single-reviewer
tags whose original payloads are unavailable). See [M1 closeout](../docs/m1_closeout.md)
for the new pinned 224-clip verification plan and truthful working-bundle status.

| Area | Committed contents |
| --- | --- |
| `metadata/` | Source audit (`worldspeech-tr_tr-source-audit.md`), release card (`turas-tr-econ-legis-v0.1.json`), source registry, and test recording groups |
| `manifests/` | Public release manifest (`turas-tr-econ-legis-v0.1.json`) and hashed handoff manifest (`turas-tr-econ-legis-v0.1-handoff.json`) |
| `splits/turas-tr-econ-legis-v0.1/` | Frozen `train.txt`, `validation.txt`, `test.txt`, `m45-timing-30min.txt`, and `split_manifest.json` |
| `verification/` | Pilot verification IDs (`turas-tr-econ-legis-v0.1-pilot-ids.txt`), audit summary, and verbatim verification policy |
| `relational/turas-tr-econ-legis-v0.1/` | Relational CSV tables (`entities.csv`, `entity_aliases.csv`, `topics.csv`, `relation_types.csv`, `relationships_static.csv`, `provenance.csv`) and `relational_manifest.json` |
| `cleaned/turas-tr-econ-legis-v0.1/` | **The cleaned dataset:** 5,450 FLAC clips (25 h), train and test references, inference inputs, relational tables and checksums. [Details](cleaned/turas-tr-econ-legis-v0.1/README.md) |

The cleaned data is built from upstream WorldSpeech `tr_tr` (CC BY-NC 4.0) by
`scripts/m1_build_release.py`.
