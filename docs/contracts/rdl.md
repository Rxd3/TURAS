# Relational CSV and trainable heterogeneous graph contract — 0.1

M1 owns the relational source schema; M5 owns graph construction, learning, and
rescoring. M1's relational CSV tables and lexicon extractor exist (see
"Populated by M1" below); no database or graph model exists yet. CSVs use UTF-8, a header, RFC 4180 quoting, stable
string keys, and empty fields for declared nullable values. All foreign keys,
enum values, duplicates and split access must be validated before graph building.

## Proposed normalized tables

| Table | Primary key | Required columns in addition to key |
| --- | --- | --- |
| `provenance.csv` | `provenance_id` | `source_uri`, `source_revision`, `content_sha256`, `rights_record_id`, `origin_kind`, `available_at_utc`, `allowed_split`, `review_status` |
| `topics.csv` | `topic_id` | `canonical_label`, nullable `parent_topic_id`, `provenance_id` |
| `entities.csv` | `entity_id` | `canonical_label`, `entity_type`, `provenance_id` |
| `entity_aliases.csv` | `alias_id` | `entity_id`, `alias_text`, `normalization_version`, `provenance_id` |
| `quantifiers.csv` | `quantifier_id` | `surface_form`, `quantifier_type`, nullable `numeric_value`, nullable `unit`, `provenance_id` |
| `utterances.csv` | `utterance_id` | `dataset_release`, `split`, `leakage_group_id`, `provenance_id` |
| `relation_types.csv` | `relation_type_id` | `name`, `head_type`, `tail_type`, `directed`, `provenance_id` |
| `relationships.csv` | `relationship_id` | `head_type`, `head_id`, `relation_type_id`, `tail_type`, `tail_id`, `evidence_provenance_id`, nullable `evidence_utterance_id`, `annotation_status`, `access_split` |

`origin_kind` distinguishes `train_annotation`, `validation_annotation`,
`test_annotation`, and `external_knowledge`. `allowed_split`/`access_split`
declare usage, not automatic trust. Test-derived entries are evaluator-only;
validation annotations may evaluate/select models but never enter a training
snapshot. Quantifier types include amount, percentage, count and date, with
versioned unit/value parsing. Numeric values use decimal strings, not binary
floating-point approximations; do not fill unknown values by guesswork.

Typed polymorphic head/tail keys are checked against the named node table;
`relation_types` constrains allowed pairs. M1/M5 must approve any new type.
Seed relation vocabulary (proposed, not populated):

| Canonical edge | Meaning |
| --- | --- |
| `utterance -has_topic-> topic` | Audited training utterance topic annotation |
| `utterance -mentions-> entity` | Training mention evidence |
| `utterance -expresses-> quantifier` | Training quantity mention evidence |
| `entity -belongs_to-> topic` | Evidence-based domain association |
| `entity -related_to-> entity` | Typed relation whose subtype is explicitly registered |
| `entity -has_quantifier-> quantifier` | Evidence-based entity/quantity relation |
| `topic -parent_of-> topic` | Taxonomy relation with cycle checks |

### Populated by M1 for `turas-tr-econ-legis-v0.1` (proposed, 2026-11-18)

The relational source tables now exist; inventory, counts and hashes are in
[`data/relational/README.md`](../../data/relational/README.md). Choices made
within this schema (M5 to review):

- Registered relation types: `rel_has_topic`, `rel_mentions`, `rel_expresses`,
  `rel_belongs_to`, and one `related_to` subtype,
  `related_to:co_mentioned_in_utterance` (undirected). `has_quantifier` and
  `parent_of` are not populated: no audited entity–quantity evidence exists,
  and the 7 topics are flat (`parent_topic_id` empty).
- Static lexicon rows (`entity -belongs_to-> topic`, aliases, topics) have
  `origin_kind = external_knowledge` and may be used by every split. They are
  stored as `relationships_static.csv`, the lexicon part of `relationships.csv`.
- Utterance evidence is written per split with `origin_kind =
  <split>_annotation`, `access_split = <split>`,
  `annotation_status = auto_extracted` (co-mentions: `auto_extracted:count=<n>`).
  It is automatic lexicon matching on upstream references, not human annotation.
- Quantifier types populated: `amount` (units `TRY`, `USD`, `EUR`) and
  `percentage` (unit `percent`); `count` and `date` are not extracted. IDs are
  value keys (`q__amount__TRY__5300000`) shared across splits by design; a
  split's table still carries that split's provenance and access.

Automatically generated reverse message-passing edges receive distinct type IDs
and retain source edge provenance. No blanket reversal of directed fact meaning.
Aliases map text to entities; they are not independent relational facts.

## Trainable RDL specification

Required model: heterogeneous message passing with per-node-type projections,
per-edge-type GraphSAGE-style aggregation, learned node representations and a
trainable typed edge decoder. M5 will select dimensions/layers/optimizer/dropout
on validation and document the exact choice. Binary cross-entropy on verified
training positives and type-compatible sampled negatives is the initial objective.
Unknown edges are not guaranteed false; document this negative-sampling
assumption, relation class balance, sampling seed and exclusion checks.

Hold out relation edges for validation/test by provenance/utterance group before
message passing. Exclude held-out positive edges and their reverse/equivalent
duplicates from the training adjacency, all training labels and features. Do
not mine negatives using protected test annotations. Disclose closed-world
sampling limitations and collisions; freeze training-only sampling policy.
Validate with held-out relation PR-AUC and ranking metrics (MRR/Hits@k) under a
declared typed candidate/filter policy, then evaluate downstream ASR separately.
Unknown node types/entities require a documented inductive feature or unknown
embedding policy; test-only vocabulary cannot silently create learned embeddings.

## Reference-free hypothesis scoring

Use a versioned extractor/linker on candidate hypothesis text to obtain typed
relations. Match against a frozen graph snapshot built from training annotations
and audited external knowledge available independently of test audio/references.
Held-out source transcripts, gold topics, entities and quantities are forbidden.
An external document duplicating a held-out reference is excluded from the primary
condition; a declared contextual condition requires separate leakage analysis.

For k extracted supported relations, let each decoder probability be p_j. Use
`R(h) = sum(log(max(epsilon, p_j))) / k` when k > 0, with documented epsilon.
When k = 0, use a fixed validation-selected floor probability in (0,1), record
`relation_count = 0`, and report extraction coverage. Freeze this missing-relation
policy to avoid rewarding candidates merely for having no extractable relations.
Report linked/unlinked mentions and relation counts; confidence calibration, if
used, is fit without test references. Do not label a heuristic lookup as learned
RDL. Compare learned, disabled, and nonlearned/shuffled-graph controls on the same
N-best candidates. Keep acoustic and graph score scales explicit.

Snapshot manifests record table hashes, relation vocabulary/hash, access filters,
feature construction, split/edge audit, training seed, graph code, learned
checkpoint, extractor/linker hash and external provenance cutoff. Reference-only
relation evaluation tables must remain outside inference permissions.
