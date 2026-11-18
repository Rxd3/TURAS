# Relational data handoff

Schema: [RDL contract 0.1](../../docs/contracts/rdl.md). M1 owns these source
tables; M5 owns graph construction, learning and rescoring.

## `turas-tr-econ-legis-v0.1/` (proposed, 2026-11-18)

Static tables from the domain lexicon (`origin_kind = external_knowledge`,
usable by every split): `provenance.csv`, `topics.csv` (7), `entities.csv` (90),
`entity_aliases.csv`, `relation_types.csv`, `relationships_static.csv`
(entity `belongs_to` topic). Hashes: `relational_manifest.json`.

Per-split utterance evidence lives outside Git under
`<artifact-root>/turas-tr-econ-legis-v0.1/relational/<split>/` with
`utterances.csv`, `quantifiers.csv`, `relationships.csv`, `provenance.csv`
(rebuild: `python scripts/m1_build_relational.py --artifact-root <dir>`):

| Split | Utterances | Quantifiers | Relationships | Access |
| --- | ---: | ---: | ---: | --- |
| train | 4,376 | 1,435 | 31,640 | graph training allowed |
| validation | 438 | 274 | 3,784 | evaluator/selection only, never in training snapshots |
| test | 636 | 370 | 6,248 | evaluator-only |

Relations: `utterance -mentions-> entity`, `utterance -has_topic-> topic`,
`utterance -expresses-> quantifier` (amounts with TRY/USD/EUR and percentages,
decimal-string values), `entity -related_to:co_mentioned_in_utterance-> entity`
(undirected, count in `annotation_status`). All utterance evidence is
`auto_extracted` by lexicon matching on upstream references, not human
annotation. Quantifier IDs are shared value keys
(`q__amount__TRY__5000000000`), so M5 must build the training graph from
`train/` only and treat validation/test quantifier nodes as unseen.
