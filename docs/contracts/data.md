# Dataset, provenance, IDs and split contract — 0.3

Owner adoption for M4/M5 integration: 2026-11-19; see [D24](../owner_decisions.md).
This records project approval, not independent peer-review sign-off.

## Serialization and release boundary

UTF-8 JSONL contains one object per utterance, no comments/NaN/Infinity. File bytes
are SHA-256 hashed as stored. Each manifest declares `schema_version: "0.1"`.
Full curated manifests contain protected references; inference exports use the
strict field allowlist in [predictions](predictions.md). Access separation is
required, even if a reference manifest is stored outside Git. Manifests keep `schema_version: "0.1"`; contract 0.3 changes the working
verification/publication policy, not the existing inference schema.

## Release audio storage

Release audio is stored as 16 kHz mono FLAC files under one artifact root kept
outside Git. `audio_uri` is the file's path relative to that root: non-empty,
with `/` as the only separator, no leading `/`, no drive letter, no backslash,
and no `..` that leads outside the root. Readers refuse any other form, so every
system reads the same files the same way. `audio_sha256` hashes the stored FLAC
bytes, and the measured `sample_rate_hz` and `channels` are 16000 and 1.

## Curated utterance manifest

| Field | Type | Meaning/constraint |
| --- | --- | --- |
| `schema_version` | string | Contract version |
| `dataset_release` | string | Immutable TURAS release identifier |
| `utterance_id` | string | Stable unique TURAS ID |
| `source_id` | string | Stable source recording ID; FK to source registry |
| `source_segment_id` | string | Original upstream segment identity; never overwritten |
| `upstream_dataset`, `upstream_config`, `upstream_revision`, `upstream_split` | strings | Exact origin, `tr_tr`, immutable revision, original split |
| `source_url`, `source_recording_key` | strings | Canonical source location and original recording identity |
| `start_ms`, `end_ms` | integers | Nonnegative original recording offsets, end > start |
| `audio_uri` | string | Portable relative path under the release audio root, `/` separators; no embedded credentials |
| `audio_sha256` | string | 64 lowercase hex characters for exact stored audio bytes |
| `sample_rate_hz`, `channels` | positive integers | Measured decoded properties |
| `duration_s` | positive number | Measured decoded duration; boundary discrepancy audit required |
| `language` | string | `tr_tr`; individual language issues retained in exclusions |
| `domain_topics` | string array | IDs from approved topic taxonomy; no guessed labels |
| `speaker_ids` | string array or null | Pseudonymous IDs if known; null means unknown |
| `session_id` | string | Stable source session/recording group; fallback and uncertainty documented |
| `duplicate_cluster_id`, `leakage_group_id` | strings | Duplicate grouping and leakage group; singleton placeholders must be labelled unaudited |
| `reference_text` | string | Original verified transcription, evaluator/training access only |
| `reference_origin` | string | `upstream_human`, `local_human`, or explicitly approved alternative |
| `verification_status` | enum | `unreviewed`, `in_review`, `verified`, `adjudicated`, `rejected` |
| `verification_record_id` | string or null | FK to review evidence; required for verified/adjudicated/rejected |
| `annotation_policy_version` | string | Verbatim transcription policy version |
| `split` | enum | `train`, `validation`, `test` after freeze |
| `rights_record_id` | string | FK to reviewed source-specific rights/attribution entry |

Do not include normalized references as substitutes for originals. If cached,
they require the normalizer version/hash and remain protected gold fields.
Source registry entries include source ID/key/URL, upstream revision, source
terms URL and reviewed text hash, access date, attribution, allowed uses/sharing,
review status and limitations. Approval cannot be inferred from a URL alone.

## Stable identity policy

Use a source registry to map canonical upstream recording keys to opaque stable
`src_` IDs. Assign opaque `utt_` IDs once in an append-only identity registry keyed
by source ID, original segment ID and original offsets. Use ASCII alphanumerics,
underscore and hyphen; IDs are case-sensitive. Never derive IDs from transcripts,
row numbers, local filenames or split membership. Do not use hashes of protected
reference text as public IDs.

Transcript corrections preserve utterance IDs and increment the release. A changed
segmentation creates new IDs with `supersedes_utterance_ids` recorded in a separate
lineage registry. Source URL aliases map to the same canonical recording. Preserve
identity mappings across all model runs; reject duplicate/missing IDs at joins.

## Verification evidence and exclusions

Review records require record/utterance IDs, pseudonymous reviewer ID, review UTC
timestamp, audio hash, annotation policy version, decision, corrections/change
log, issue codes and adjudicator ID if applicable. Do not expose personal reviewer
details in public metadata. Source transcript claims and local listening status
are distinct. Rejected rows go to an exclusion ledger with IDs, reason and policy;
never silently disappear. The default policy requires accepted local evidence
for every validation/test reference. The D42 working exception below permits
explicitly unreviewed minutes outside a pinned sample. A post-freeze correction
creates a new reference variant and requires newly labelled shared evaluation
for every comparison; no old artifact is overwritten.

## Frozen split files

Future `data/splits/<release>/{train,validation,test}.txt` contain unique IDs, one
per line, UTF-8/LF, sorted lexicographically. Their sets are disjoint and their
union equals the accepted release. A split manifest records release, source and
curation manifest hashes, each split file hash, actual counts/durations, ordered
ID digest, group map hash, algorithm/version, seed, ratios, upstream split policy,
duplicate thresholds, exclusions, speaker uncertainty, creation commit and review.

Freeze audio, references, grouping and membership together. No resplitting per
model, tuning on test, or post-hoc deletion of hard examples. Reference files
have separate access control; inference ID/audio exports must match the frozen
ID set without reference text, review corrections, or gold domain annotations.
M1/M6 approve a release only after documented overlap/duplicate checks pass.

## Working-release exception and immutable publication (0.3, D42)

Owner adopted 2026-11-21 following the M6 PR #2 Option C request; independent
M6 release approval remains pending. See [closeout](../m1_closeout.md).

- Preserve `turas-tr-econ-legis-v0.1` split/audio bytes. Commit the recording-
  stratified 224-test-ID selection (8 per 28 recordings, seed 2026, original 25
  test pilot IDs included) and SHA before new listening. This plan was prepared
  **after working test exposure**, not blindly before all evaluation.
- Validate genuine primary evidence for the sample plus all 50 pilot IDs
  (249 distinct clips). Existing status tags alone do not satisfy the gate.
  Validation remains labelled upstream minutes outside its reviewed pilot.
  The recovered working validation file currently has no local review evidence.
- Require a different second listener for all corrections/rejections/conflicts,
  known pilot corrections, and a fixed-seed 10% accept sample. Pin exact primary
  record IDs and the shown reference snapshot. New primary decisions invalidate
  stale adjudication. Refuse pending, conflicting, rejected or missing evidence.
- Write UTF-8/LF atomically into a fresh **reference variant** directory, never
  into frozen references/splits. `--sampled-verification` is required when any
  held-out rows remain unreviewed. Reference status/origin reflect actual records;
  a second listener accepting a proposed correction must preserve that correction.
- Record rejection IDs, issues and review evidence in the private check report;
  do not silently exclude them or regenerate splits. Resolve rejected audio under
  a separately reviewed future release policy, or keep publication blocked.
- Export separately hashed train and validation references. Keep test references,
  held-out relational evidence and the full exclusion ledger evaluator-only,
  outside the reference-free inference/audio root. A separately labelled working
  reference export need not pretend to be the unavailable full curated manifest.
- Report full-test and reviewed-subset results separately with actual coverage,
  uncertainty and edited-minutes/output-style bias. No universal WER adjustment
  can be inferred from the pilot. Rebuild relational evidence if its source text
  changes, version/hash it separately, and revalidate its access boundary.

Reference review completion does not establish rights, cloud ACLs, peer review,
consumer receipt or speaker/near-duplicate disjointness. These need real evidence.

## Version history

Deadline note (D43, 2026-11-21): owner acceptance permits continued use of the
existing, honestly labelled working reference snapshots. It changes no verified-
variant evidence requirement above, supplies no human decision or external
approval, and authorizes no overwritten references, test tuning or re-splitting.

- 0.1 (2026-11-17): initial contract. `audio_uri` could also be an
  access-controlled artifact URI, and the audio format was not fixed.
- 0.2 (proposed by M6 on 2026-11-17, pending review by M1;
  [D20](../decisions.md)): release audio is 16 kHz mono FLAC under an artifact
  root, and `audio_uri` is a relative path with `/` separators.

- 0.3 (owner adopted 2026-11-21, D42): pinned stratified working verification,
  independent adjudication and non-overwriting reference variants; no inference
  schema or frozen split/audio change. Independent M6 release approval pending.
