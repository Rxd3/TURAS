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

Status: **frozen on 2026-11-18** under the sampled-verification policy
approved by M6 (D08; approval reported by M1). Final split manifest sha256
`a8fff65b99fcf11d9452685e438e250207032ecfe11152741562147126d095b8`; all hashes in
`data/splits/…/split_manifest.json`, `data/manifests/turas-tr-econ-legis-v0.1.json`
and the handoff record `data/manifests/turas-tr-econ-legis-v0.1-handoff.json`.
A correction from now on creates a new release. Source audit: [worldspeech-tr_tr-source-audit](../data/metadata/worldspeech-tr_tr-source-audit.md).
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

### Commands (artifact root outside Git)

| Step | Command |
| --- | --- |
| Build release (≈1 h, streams ≈26 GB, keeps ≈2 GB) | `python scripts/m1_build_release.py --artifact-root ../turas-artifacts` |
| Relational evidence tables | `python scripts/m1_build_relational.py --artifact-root ../turas-artifacts` |
| Listening pages, 100 clips each, pilot IDs left out | `python scripts/m1_review_kit.py --artifact-root ../turas-artifacts --split test --exclude-ids-file data/verification/turas-tr-econ-legis-v0.1-pilot-ids.txt --part-size 100 --out <dir>/turas-test.html` (same for `validation`) |
| Pages built from another machine's audio | `python scripts/m1_audio_equivalence.py --artifact-root ../turas-artifacts <page.html>` |
| Ingest exported decisions | `python scripts/m1_apply_reviews.py --artifact-root ../turas-artifacts <decisions.json> ...` (`--role adjudication` for the second reviewer) |
| Freeze (refuses while validation/test rows are unreviewed) | `python scripts/m1_freeze_release.py --artifact-root ../turas-artifacts --status "frozen by M1 on <date>; M6 review: <PR>"`, then rerun the relational step |

### Verification policy: sampled verification (M1 decision 2026-11-18, needs M6 approval, D08)

M1 stops listening review after the 50-clip pilot. This **replaces** the rule
above that every validation/test reference must be `verified` or
`adjudicated` before freeze, and the matching item in M1's definition of done
in `TEAM_TASKS.md`. M6 must approve it (D08) before the release is frozen.

- **What the references are.** Validation/test references are the upstream
  TBMM minutes (`reference_origin = upstream_human`), not verbatim transcripts.
  The 50 pilot clips were checked by listening, and their status and
  corrections are merged at freeze.
- **The sample.** Drawn with seed 2026, one random clip from each of 25 random
  test recordings (of 28), each of the 15 validation recordings, and 10 train
  recordings; chosen before any model output existed. IDs:
  `data/verification/turas-tr-econ-legis-v0.1-pilot-ids.txt`.
- **Measured reference quality.** 0/50 rejected (95% Wilson CI 0–7.1%);
  16/50 needed a correction (32%, CI 20.8–45.8%). Upstream minutes vs
  verified text: WER 2.43% over 1,561 words (20 S, 9 D, 9 I). Most differences
  are spoken words the minutes omit.
- **Consequence for results.** Every WER/CER on the full test set includes
  reference noise of about this size. It is the same for all systems, but
  absolute WERs are overstated by roughly 2–3 points, and differences between
  systems smaller than that are not reliable. Reports state this.
- **Required reporting (M6).** Report full-test metrics on the frozen test
  (3.0 h, minutes references), plus the same metrics on the 25 verified test
  clips as a reference-quality check, labelled as a small sample.
- **Freeze.** `scripts/m1_freeze_release.py --sampled-verification` records
  this policy and the verified/unreviewed counts in the split manifest.
  Without the flag the script still refuses unreviewed validation/test rows.
