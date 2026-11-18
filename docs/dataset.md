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
Validation and test references must all be `verified` or `adjudicated` before
freeze; ambiguous/unintelligible segments are excluded with reasons, never guessed.
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
release. Split manifests and a release card are required; no counts or hours are
claimed now. See the [data contract](contracts/data.md) for required fields.

## Proposed release `turas-tr-econ-legis-v0.1` (M1, 2026-11-18)

Status: **proposed, not frozen.** Validation/test references are still
`unreviewed`; the release freezes only after listening verification and M6
review. Source audit: [worldspeech-tr_tr-source-audit](../data/metadata/worldspeech-tr_tr-source-audit.md).
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
Protected (artifact root, outside Git): `audio/<utterance_id>.flac` (16 kHz mono
FLAC), `<release>.references.jsonl` (full data-contract rows),
`<release>.inference.<split>.jsonl` (strict allowlist), `exclusion_ledger.jsonl`
(every upstream row with its split or reason), relational tables.

### Known limitations

- References are TBMM minutes, edited rather than verbatim; validation/test
  need listening verification ([guide](../data/verification/turas-verbatim-v0.1.md)).
- The CER ≤ 0.25 filter favours segments an upstream ASR system handled well.
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
  and text. No baseline or parent checkpoint may be trained on WorldSpeech.
  M2–M5 state each checkpoint's training data and release date.
- **Lexicon frozen by hash.** `turas-domain-lexicon-v0.1` is frozen as sha256
  `8ef903fe…` ([freeze record](../configs/datasets/turas-domain-lexicon-v0.1.freeze.json)).
  The file stays byte-identical, so its inner `status` text still says
  "proposed". Known matcher limitations are listed in the record.
- **Erratum, resampler.** The release config says `soxr VHQ`; the audio was
  written with soxr `HQ` (`release.py`). The config text is corrected at
  freeze, when the manifests are regenerated anyway.
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
- **Relational extractor fixes.** Amount parsing and co-mention IDs were
  corrected after the first build ([details](../data/relational/README.md)).
  Per-split relational tables are regenerated at freeze.
- **Review tooling.** `scripts/m1_apply_reviews.py` now refuses invalid
  decisions (all or nothing), ignores re-imports of the same export, resolves
  status per the guide (`verified`/`local_human`/`in_review`/`adjudicated`/`rejected`;
  `--role adjudication` for the second reviewer), and rewrites a cumulative
  audit.
