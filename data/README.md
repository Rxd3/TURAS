# Data area — metadata, splits, and relational schema

This directory contains the public metadata, split ID lists, relational schema
tables, and verification/audit records for the working release
`turas-tr-econ-legis-v0.1` (5,450 Turkish legislative/economic speech clips,
~25 h total, temporal train/validation/test split derived from
`disco-eth/WorldSpeech` `tr_tr`, CC BY-NC 4.0).

See [docs/dataset.md](../docs/dataset.md) and
[docs/contracts/data.md](../docs/contracts/data.md) for full provenance,
inclusion/exclusion rules, and known limitations (611 of 636 test clips use
unreviewed edited TBMM minutes; 25 clips received single-reviewer verification).

| Area | Committed contents |
| --- | --- |
| `metadata/` | Source audit (`worldspeech-tr_tr-source-audit.md`), release card (`turas-tr-econ-legis-v0.1.json`), source registry, and test recording groups |
| `manifests/` | Public release manifest (`turas-tr-econ-legis-v0.1.json`) and hashed handoff manifest (`turas-tr-econ-legis-v0.1-handoff.json`) |
| `splits/turas-tr-econ-legis-v0.1/` | Frozen `train.txt`, `validation.txt`, `test.txt`, `m45-timing-30min.txt`, and `split_manifest.json` |
| `verification/` | Pilot verification IDs (`turas-tr-econ-legis-v0.1-pilot-ids.txt`), audit summary, and verbatim verification policy |
| `relational/turas-tr-econ-legis-v0.1/` | Relational CSV tables (`entities.csv`, `entity_aliases.csv`, `topics.csv`, `relation_types.csv`, `relationships_static.csv`, `provenance.csv`) and `relational_manifest.json` |

Raw/processed audio, private test references, and large intermediate archives
are not committed to Git (upstream WorldSpeech is CC BY-NC 4.0). To rebuild
locally from upstream WorldSpeech, see `scripts/m1_build_release.py`.
