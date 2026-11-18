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
| train | 4,376 | 1,459 | 31,640 | graph training allowed |
| validation | 438 | 274 | 3,784 | evaluator/selection only, never in training snapshots |
| test | 636 | 373 | 6,248 | evaluator-only |

Relations: `utterance -mentions-> entity`, `utterance -has_topic-> topic`,
`utterance -expresses-> quantifier` (amounts with TRY/USD/EUR and percentages,
decimal-string values), `entity -related_to:co_mentioned_in_utterance-> entity`
(undirected, count in `annotation_status`). All utterance evidence is
`auto_extracted` by lexicon matching on upstream references, not human
annotation. Quantifier IDs are shared value keys
(`q__amount__TRY__5000000000`), so M5 must build the training graph from
`train/` only and treat validation/test quantifier nodes as unseen.

### Corrections after code review (2026-11-18, regenerated)

Two extractor bugs were fixed in `src/turkish_asr_rdl/data/relational.py`
and the per-split tables were regenerated from the local rebuild (counts and
hashes above; quantifiers went from 1,435 to 1,459 in train and 370 to 373 in
test). In train, 144 compound amounts such as `1 trilyon 30 milyar liraya`
now carry their full value. They are regenerated again at freeze, after the
listening corrections.

- Amount parsing: compound amounts were cut to their last part
  (`5 milyon 300 bin lira` gave 300000, now 5300000), a dot decimal before a
  scale word was misread (`1.5 milyar` gave 5e9, now 1.5e9), and `yüzde 3.5`
  and a trailing `3%` were missed or truncated. Covered by
  `tests/test_data_relational.py`.
- Co-mention relationship IDs now include the split (`c__<split>__A__B`), so
  train, validation and test primary keys can no longer collide if M5 loads
  the split tables together. Quantifier IDs stay shared value keys on purpose
  (a hypothesis amount must be able to link to a train value node); their
  `provenance_id` and `access_split` still come from the split's own table.
- `annotation_status` for co-mentions is `auto_extracted:count=<n>`: the
  contract defines no count column, so the evidence count is carried there.
