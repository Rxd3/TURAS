# Working dataset and curation plan

Candidate: [disco-eth/WorldSpeech](https://huggingface.co/datasets/disco-eth/WorldSpeech),
configuration `tr_tr`. Target contribution: curated Turkish legislative/financial
speech. This is a working decision, not a released dataset or approval to publish
audio/references. No data was downloaded during initialization.

The [upstream card](https://huggingface.co/datasets/disco-eth/WorldSpeech/blob/main/README.md)
was inspected on 2026-11-17. It lists `tr_tr`, upstream train/test files,
`human_transcript`, `asr_transcript`, and source/segment metadata. It describes
CC BY-NC 4.0 with per-source terms and lists Turkish parliamentary proceedings.
This card review does not establish redistribution rights or local transcript
verification. Do not treat upstream ASR-alignment scores as quality guarantees.

## Required audit before release

Member 1 records the exact upstream revision, source URLs, source-specific
rights/attribution, permitted access/sharing, transcript origin, audio encoding,
sampling rate, segment boundaries, availability and any privacy restrictions.
Check whether financial speech is actually present; narrow or supplement the
working domain only through a documented team decision with equivalent audits.
Do not infer domain membership from the dataset name alone.

Define inclusion/exclusion criteria before inspecting model test errors:
Turkish language, relevant legislative/financial topics, intelligible aligned
audio, verified reference fidelity, and permitted use. Log every rejection and
reason. Preserve natural numerals, currencies, names, acronyms, hesitation policy,
and verbatim transcription guidelines; changes require versioned annotation policy.

## Human transcription verification

An upstream `human_transcript` starts as `unreviewed` for TURAS. A verifier listens
to the full segment, checks timing and fidelity, and records an evidence-linked
decision. Disagreements or difficult speech go to another reviewer/adjudicator.
The default policy requires all validation/test references to be `verified` or
`adjudicated`. The versioned working-release exception below samples test and
retains unreviewed minutes explicitly. Ambiguous text is never guessed; a rejection
blocks publication under the frozen-membership policy.
Train reference status is reported and any weaker supervision is labeled in
training runs. Do not use ASR output as a human-approved reference by default.

## Leakage-resistant splits

Build connected groups from source recording/session identity, overlapping time
segments, audio duplicate/near-duplicate clusters, repeated transcript clusters,
and known speaker identity where available. Keep each connected component in
one split. Unknown speakers must be explicit: do not claim speaker-disjointness
without evidence. Audit lexical duplication separately so formulaic parliamentary
phrases do not masquerade as distinct independent test observations.

Keep upstream test records out of training/validation and preserve upstream
assignments in provenance. Prefer an audited upstream test as a candidate TURAS
test, then derive validation from upstream train at group level. If upstream
boundaries leak, quarantine conflicting groups or define a new versioned subset;
do not silently recycle upstream test as train. Freeze group algorithm, thresholds,
ratios, seed, and actual membership only after the audit; none are chosen here.
Report limitations from unknown model pretraining contamination.

M1 and M6 review counts, duration, topics, source/speaker overlap, verification,
exclusions, duplicate statistics, rights and split hashes before approving a
release. Split manifests and a release card are required; the actual working
release counts are reported below. See the [data contract](contracts/data.md) for required fields.

## Proposed release `turas-tr-econ-legis-v0.1` (M1, 2026-11-18)

Status: **frozen by M1; independent M6 release approval pending**. PR #2
merged on 2026-11-20, but its latest review requests changes; a merge does not
approve the 50-clip policy. See the [M6 review](https://github.com/Rxd3/TURAS/pull/2#pullrequestreview-5962363421)
and [closeout instructions](m1_closeout.md).

The original split manifest remains byte-identical at SHA-256
`a8fff65b99fcf11d9452685e438e250207032ecfe11152741562147126d095b8`.
Its embedded approval statement is superseded by
[`status-erratum-v1`](../data/metadata/turas-tr-econ-legis-v0.1-status-erratum-v1.json).
Never overwrite this manifest, audio, split membership, references or existing
experiment outputs. New listening corrections create a separately named and
hashed reference variant; comparisons against it require newly labelled shared
evaluation for all systems. Source audit:
[worldspeech-tr_tr-source-audit](../data/metadata/worldspeech-tr_tr-source-audit.md).
Configuration: [`configs/datasets/turas-tr-econ-legis-v0.1.json`](../configs/datasets/turas-tr-econ-legis-v0.1.json);
lexicon: [`turas-domain-lexicon-v0.1.json`](../configs/datasets/turas-domain-lexicon-v0.1.json).
Rebuild: `python scripts/m1_build_release.py --artifact-root <dir outside Git>`.

### Inclusion and exclusion (applied in this order; first reason is logged)

| Reason | Rows |
| --- | ---: |
| `source_not_selected` (Ankara municipal council) | 2,451 |
| `non_spoken_or_truncated_text` (speaker headers, stage directions, brackets, `...`) | 16,768 |
| `repeated_text` (identical normalized text elsewhere in the corpus) | 3,175 |
| `upstream_cer_above_max` (ASR alignment CER > 0.25) | 3,886 |
| `not_in_domain` (lexicon rule below) | 184,567 |
| `upstream_test_in_train_recording` (quarantined) | 1,069 |
| `eligible_not_sampled` (eligible, beyond the per-recording cap) | 22,288 |
| **selected** | **5,450** |

Domain rule: at least 2 distinct economic terms, or 1 economic term plus a
legislative term or an amount/percentage expression. Economic topics:
public finance/budget, tax, inflation/interest/currency, banking/credit,
wages/pensions/social security, trade/investment/growth; legislative topic:
laws, articles, motions, constitution, regulations. 126.6 h in 172
recordings are eligible; the release samples 25 h of it.

### Splits (grouped by recording; one recording = one session date)

| Split | Utterances | Hours | Recordings | Session dates | Upstream-test rows |
| --- | ---: | ---: | ---: | --- | ---: |
| train | 4,376 | 20.00 | 129 | 2017-01-10 – 2025-12-23 | 0 |
| validation | 438 | 2.00 | 15 | 2025-10-01 – 2025-12-24 | 18 |
| test | 636 | 3.00 | 28 | 2026-01-06 – 2026-04-22 | 23 |

- **Test is temporal**: every 2026 recording, nothing else. It postdates the
  training data and the planned baseline checkpoints.
- **Validation**: 15 recordings drawn with seed 2026 from Sep–Dec 2025 sessions
  (the window includes December budget debates).
- **Train**: every other recording up to 2025-12-31.
- One per-recording cap per split (water-filled: test 6.6 min, validation
  8.4 min, train 10.1 min) spreads the hours over as many sessions as possible.
  Inside a recording, segments are ranked by distinct economic terms +
  min(legislative terms, 2) + min(quantities, 1).
- Leakage checks (all 0): recordings or dates in two splits, train rows from
  upstream test, normalized texts shared across splits, duplicate IDs, rows
  without a split or reason.
- Speakers are unknown; speaker-disjointness is **not** claimed.

Share of utterances with each automatically detected topic (an utterance can
have several; labels are lexicon screening, `domain_topics_origin =
lexicon-screening-unverified`):

| Topic | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Inflation, interest, currency | 67% | 63% | 70% |
| Wages, pensions, social security | 44% | 39% | 57% |
| Trade, investment, growth | 41% | 47% | 46% |
| Public finance, budget | 29% | 32% | 28% |
| Legislation | 24% | 22% | 23% |
| Banking, credit | 18% | 16% | 30% |
| Tax | 13% | 20% | 26% |

### Outputs

Public (in Git): `data/splits/turas-tr-econ-legis-v0.1/{train,validation,test}.txt`
and `split_manifest.json`; `data/manifests/turas-tr-econ-legis-v0.1.json`
(hashes of the protected files and inference exports); source registry CSV.
Committed since D44 (2026-11-22) under data/cleaned/turas-tr-econ-legis-v0.1/ (before D44: protected, outside Git): `audio/<utterance_id>.flac` (16 kHz mono
FLAC), `<release>.references.jsonl` (full data-contract rows),
`<release>.inference.<split>.jsonl` (strict allowlist), `exclusion_ledger.jsonl`
(every upstream row with its split or reason), relational tables.

### Known limitations

- References are TBMM minutes, edited rather than verbatim; validation/test
  need listening verification ([guide](../data/verification/turas-verbatim-v0.1.md)).
- The upstream ASR-agreement CER ≤ 0.25 filter selects easier/aligned speech
  and may favour Whisper-family output; it is not a human quality guarantee.
- Validation includes recordings one day from train sessions in December 2025
  budget debates. Recording separation does not imply independent topics,
  speakers or recurring phrasing.
- Topic labels and relational facts are automatic lexicon matches.
- Unknown speakers; public YouTube/minutes origin means pretraining
  contamination cannot be excluded for pre-2026 data.

### Decisions, deviations and errata (M1 review, 2026-11-18)

These need M6 approval with the split policy (D07); none changes split membership.

- **Upstream-test rows in validation/test (deviation).** The scaffold rule above
  says to keep upstream test records out of training *and validation*.
  Validation holds 18 upstream-test rows and test holds 23. They are kept: upstream
  test shares every one of its 188 recordings with upstream train, so the
  upstream label marks no held-out boundary. The TURAS boundary is the
  recording. The rule that matters is kept: no upstream-test row is in TURAS train.
- **Contamination through WorldSpeech-trained checkpoints.** 613 of the 636 test
  rows (and most validation rows) are upstream *train* rows. Any checkpoint
  fine-tuned on WorldSpeech `tr_tr` train has therefore seen TURAS test audio
  and text. Do not use externally WorldSpeech-fine-tuned parents/baselines or the
  unfiltered WorldSpeech train split. Group-owned training on the frozen TURAS
  **train** IDs is permitted. No test-session text (2026-01-06 through 2026-04-22)
  may enter training, graph evidence, an LM or prompts. M2–M5 record each exact
  checkpoint's training-data evidence and revision; undisclosed pretraining is
  an uncertainty, not proof of no exposure.
- **Lexicon frozen by hash.** `turas-domain-lexicon-v0.1` is frozen as sha256
  `8ef903fe…` ([freeze record](../configs/datasets/turas-domain-lexicon-v0.1.freeze.json)).
  The file stays byte-identical, so its inner `status` text still says
  "proposed". Known matcher limitations are listed in the record.
- **Erratum, resampler (fixed).** The release config said `soxr VHQ`; the audio
  was always written with soxr `HQ` (`release.py`). The config text now says
  `HQ` (new config sha256 `d530be46…`).
- **Duplicate audit scope.** Only exact duplicates of screening-normalized text
  were audited and removed. No audio near-duplicate audit was done.
  `duplicate_cluster_id` is a singleton placeholder (equal to `utterance_id`),
  not the result of a clustering.
- **Rebuild reproducibility.** The split ID files must come out byte-identical
  on any machine. FLAC bytes depend on libsndfile/soxr versions, so audio
  hashes, the references hash, the split manifest hash and the inference
  export hashes can differ between machines. The build now records the code
  commit and library versions in `split_manifest.json`. The hashes that count
  are those of the freeze build, acknowledged by consumers.
- **Local rebuild (2026-11-18).** Rebuilt on M1's machine from the pinned
  revision. The exclusion ledger (`6077dc6f…`), the three split files and the
  source registry are byte-identical to the first (cloud) build. The FLAC bytes
  differ: same length, at most 1 LSB per sample, SNR ≥ 99.8 dB on the 50 pilot
  clips. So the audio, references, manifest and inference hashes changed. The
  local build is now the release; new split manifest sha256 `1af56ddb…`. The
  pilot page was made from the cloud audio, so the pilot clips' equivalence is
  recorded with its evidence in `<artifact-root>/<release>/audio_equivalence.jsonl`
  (`scripts/m1_audio_equivalence.py`), and ingestion accepts only those pairs.
- **Relational extractor fixes.** Amount parsing and co-mention IDs were
  corrected after the first build ([details](../data/relational/README.md)).
  Per-split relational tables were regenerated after the fix.
- **Review tooling.** `scripts/m1_apply_reviews.py` now refuses invalid
  decisions (all or nothing), ignores re-imports of the same export, resolves
  status per the guide (`verified`/`local_human`/`in_review`/`adjudicated`/`rejected`;
  `--role adjudication` for the second reviewer), and rewrites a cumulative
  audit.

### Current verification policy (D42; owner adopted 2026-11-21)

The prior M1 proposal to stop at the 50-clip pilot is superseded. The M6 review
requests **8 test clips per recording**, including the original 25 test pilot
IDs, seed 2026: 224 test clips across 28 recordings, 199 additional clips.
The [plan](../data/verification/turas-tr-econ-legis-v0.1-stratified-test-v1.json)
and sorted LF ID file are pinned in Git before new listening. This preparation
occurred **after the working test results were opened**; it is not blind before
all evaluation. Selection uses recording IDs and pilot IDs, never model errors.

The available original test/train references contain reported verification tags,
but the 50 actual review-record payloads are missing. The recovered validation
variant contains upstream minutes for all 438 rows, all unreviewed; it does not
recover M1's three corrections or 15 review records. The primary evidence scope
is therefore 224 test + 15 validation pilot + 10 train pilot = **249 clips**.
The prepared pages cover 199 new test clips and 50 pilot evidence-recovery clips.
If genuine compatible original records arrive, import them instead of inventing
or re-attributing old decisions. Audio/snapshot mismatches require real evidence
or listening to the current kit.

A second listener must adjudicate every correction/rejection/conflict, all known
pilot corrections, and a seed-2026 10% sample of accepts. Because the original
three validation correction identities are unavailable, all 15 validation pilot
clips are conservatively in mandatory adjudication scope. Adjudication is pinned
to exact primary record IDs; the second reviewer must differ from every primary
listener for that clip. Rejections block publication and enter the check report's
rejection ledger; no frozen ID disappears or changes split.

M1 reported a test pilot correction rate of **10/25 = 40%** (Wilson 95% interval
23.4–59.3%). The pooled 16/50 is not the test estimate. The historical 38 word
edits were 20 substitutions, 9 deletions and 9 insertions; substitutions were the
largest category. These are historical reported diagnostics, not newly reproduced
listening results. Edited minutes create output-style-dependent WER/CER bias;
there is no justified common 2–3 point correction or universal ranking threshold.
Report full-636 scores and separately labelled verified-subset scores with the
reference variant, coverage, review status and post-test chronology. A newly
reviewed subset cannot retroactively make the earlier 25 clips adjudicated.

### Commands and access boundaries

Use [M1 closeout](m1_closeout.md) for the exact prepare → ingest → adjudicate →
check → publish commands. `m1_freeze_release.py` now delegates to the safe
versioned publisher; its old in-place `--artifact-root` freeze interface is
retired. A sampled publication requires the committed plan and genuine accepted
primary/adjudication evidence. It never overwrites the old references or splits.

Separate private working bundles now exist for inference/audio, train references,
recovered validation references and evaluator-only references/test relational
evidence. The evaluator working bundle explicitly lists missing review records,
original validation corrections and exclusion ledger; it is not the unavailable
original M6 bundle. Whitelisted archives contain no AppleDouble entries.
See the [current inventory](../data/manifests/turas-tr-econ-legis-v0.1-m1-working-handoff-v1.json).
Delivery/consumer acknowledgments and restricted Drive permissions still require
actual evidence; archive creation is not proof of delivery or an ACL change.
