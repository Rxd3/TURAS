# Member 1 — Dataset and Relational Schema

Name: unassigned (member to add). GitHub username: unassigned (member to add).
Branch: `feat/m1-data-relational-schema`.
Ownership/tasks/review routing: [TEAM_TASKS.md](../TEAM_TASKS.md).

**Status (2026-11-21): frozen by M1; working technical closeout prepared.
M6 release approval, genuine review/adjudication evidence and consumer ACKs
remain pending. PR #2 merged with an outstanding changes-requested review.**

## Actual contributions

### 2026-11-18 — source audit, lexicon, curation code, proposed release

- Audited WorldSpeech `tr_tr` at revision `b76b85395039f50332fbc421f7c098ee2f2653d7`
  from the card and every non-audio column of all 389 shards:
  [source audit](../data/metadata/worldspeech-tr_tr-source-audit.md).
  Found that the upstream test leaks at recording level, 7.1% of references
  contain non-spoken stenographic text, and the audio is 24 kHz stereo.
- Wrote the domain lexicon v0.1 (7 topics, 90 terms, Turkish suffix-aware
  aliases with exclusions for ambiguous forms):
  `configs/datasets/turas-domain-lexicon-v0.1.json`.
- Implemented `src/turkish_asr_rdl/data/` (`screening`, `selection`, `release`,
  `relational`, `worldspeech`) and the thin scripts
  `scripts/m1_build_release.py`, `scripts/m1_build_relational.py`, `scripts/m1_apply_reviews.py`,
  `scripts/m1_review_kit.py`.
- Proposed release `turas-tr-econ-legis-v0.1`: 20 h train / 2 h validation /
  3 h test (5,450 utterances, 172 recordings), recording-grouped, temporal 2026
  test, with leakage checks at zero. Split files and manifests under
  `data/splits/`, `data/manifests/`, `data/metadata/`; details in
  [dataset](../docs/dataset.md#proposed-release-turas-tr-econ-legis-v01-m1-2026-11-18).
- Wrote the listening verification guide
  [`turas-verbatim-v0.1`](../data/verification/turas-verbatim-v0.1.md) and a
  self-contained review page generator for the 50-clip pilot.
- Relational CSV tables (train 31,639 relationships at freeze, evaluator-only validation/test) per [RDL contract](../docs/contracts/rdl.md): static
  lexicon tables in `data/relational/turas-tr-econ-legis-v0.1/`; per-split
  automatic evidence tables in the artifact root, with train-only access for
  graph training.

Checks run: `pytest tests/test_data_screening.py tests/test_data_selection.py
tests/test_data_relational.py` (offline, synthetic fixtures); full release build
against the pinned revision with leakage report all zero.

### 2026-11-18 — local verification, lexicon freeze, review fixes

- Re-verified the bundle locally (hashes, split disjointness, sorting) and
  pushed the branch.
- Froze the domain lexicon by hash for terminology metrics:
  `configs/datasets/turas-domain-lexicon-v0.1.freeze.json`.
- Added the release card `data/metadata/turas-tr-econ-legis-v0.1.json`.
- Rewrote listening-review ingestion (`src/turkish_asr_rdl/data/review.py`,
  `scripts/m1_apply_reviews.py`): strict validation, idempotent re-import,
  status resolution with adjudication, cumulative aggregate audit.
- Fixed amount/percentage parsing and cross-split key collisions in the
  relational extractor; made `--skip-audio` non-destructive; build records
  commit and library versions; review page escaping and state keying.
- Documented split-policy deviations and errata in `docs/dataset.md`.
- Reviewed M6's contract 0.2 proposals (comments sent separately; M6 files
  not edited).

Checks run: `pytest tests/test_data_*.py` — 49 passed (offline, synthetic
fixtures); end-to-end run of `m1_apply_reviews.py` on a synthetic artifact
root in the pilot page's export format (re-run adds 0 records).

### 2026-11-18 (evening) — local rebuild and freeze tooling

- Rebuilt the full release locally; split files and exclusion ledger
  byte-identical to the first build; audio equivalence (≤1 LSB) measured and
  recorded; regenerated relational tables with the fixed extractor.
- Fixed the resampler erratum in the release config; source registry now
  has the contract's rights/terms fields; RDL contract relational section
  documents the populated tables.
- Wrote `scripts/m1_freeze_release.py` (merge reviews, drop rejects to the
  ledger, regenerate splits/manifests/exports; refuses while validation/test
  rows are unreviewed) and `scripts/m1_audio_equivalence.py`; paged review
  kits (7 test + 5 validation pages).

Checks run: `pytest tests/test_data_*.py` — 51 passed; trial freeze on real
artifacts (split files identical) and on a synthetic copy (reject/correct paths).

### 2026-11-18 — pilot listening review

- Listened to and judged the 50 pilot clips myself (`rev_m1`): 34 accept,
  16 correct, 0 reject; measured 2.43% WER between the minutes and the speech.
- Decided on sampled verification (pending M6 approval D08) and documented
  its consequences for reported metrics.

### 2026-11-18 — release frozen and handed off

- Froze `turas-tr-econ-legis-v0.1` (split manifest `a8fff65b…`) under M1's
  sampled-verification decision; the claimed M6 approval was incorrect (see the
  2026-11-21 status erratum). M1 reported merging 16 listening corrections;
  regenerated relational evidence; 24 integrity checks passed.
- Built the three access-separated handoff bundles and recorded their hashes
  (`data/manifests/turas-tr-econ-legis-v0.1-handoff.json`).

## Experiments and artifacts

No model runs. Full audio build done 2026-11-18: 5,450 FLAC files (1.9 GB, 16 kHz mono), 200 sampled hashes re-verified, max duration drift 0.0001 s. Hashes: see
`data/splits/turas-tr-econ-legis-v0.1/split_manifest.json` and
`data/manifests/turas-tr-econ-legis-v0.1.json`.

## Reviews and handoffs

- M6 requested a larger stratified sample and adjudication in [PR #2](https://github.com/Rxd3/TURAS/pull/2#pullrequestreview-5962363421); the old approval statement is withdrawn.
- M1 reviewed M6's data/predictions/normalization 0.2 proposals: approve, with
  an audio-root clarification and a raw-text term-matching note (sent to M6).
- PR #2 merged; release re-audit against the closeout head and actual consumer hash acknowledgments remain pending.

## 2026-11-21 — owner-requested technical closeout

Repository tooling work performed with coding-agent assistance; this entry
does not attribute new human listening to Member 1 or any invented reviewer.

- Checked every remote branch and all PRs #1–#10, including reviews and files;
  rebased the focused `feat/m1-release-closeout` work on current merged `main`.
- Prepared a seed-2026, eight-per-recording 224-test-ID plan (25 original pilot
  IDs included), 199-new-test and 50-pilot evidence-recovery listening pages.
  Preparation is after working test exposure, selected without model errors.
- Added evidence-bound ingestion/adjudication, actual UTC ordering, immutable
  publication, rejection blocking and meaningful gate tests. Existing release
  implementation and frozen model-producing sources stay unchanged.
- Rechecked all 5,450 audio hashes and 16 kHz mono PCM16 properties; exact
  recording/audio/shared-normalized-text intersections across splits are zero.
- Built and independently checked four clean, separately hashed private working
  archives. No data, audio, references or review pages are committed to Git.
- Corrected approval and reference-bias claims, recorded D42, preserved frozen
  source bytes through a separately hashed status erratum.

Evidence/commands: [M1 closeout](../docs/m1_closeout.md),
[preparation audit](../data/verification/turas-tr-econ-legis-v0.1-preparation-audit-v1.json),
[working handoff](../data/manifests/turas-tr-econ-legis-v0.1-m1-working-handoff-v1.json).
These are technical checks; zero new human review decisions were supplied.

## Remaining work

- Deadline technical handoff is complete for the available inputs under D43;
  this list remains the genuine outstanding evidence/access work, not completed
  listening. See [deadline handoff](../docs/m1_deadline_handoff.md).
- Obtain compatible original 50 pilot decision payloads, or have real Turkish
  listeners review the recovery pages; review the 199 additional test clips.
- Complete independent adjudication, including corrections and the pinned
  accept sample. Do not mark reference tags as newly verified without records.
- Recover the original corrected validation bundle and full exclusion ledger
  if available. The current working validation minutes are explicitly different.
- Restrict evaluator Drive access, deliver the appropriate role bundles, and
  record actual M2–M6 receipt/hash ACKs (never prefilled).
- Have M6 re-audit the named closeout head/release variant, M5 check relational
  provenance, and a non-author verify retrieval. Regenerate/version relational
  evidence and rerun shared comparisons if new listening changes text.

## 2026-11-21 — deadline working handoff (owner-assisted)

- Generated four validation relational CSVs from the exact pinned recovered
  minutes (438 utterances, 274 quantifiers, 3,784 relationships); kept all 438
  references unreviewed and wrote a separate private, hashed evaluator supplement.
- Checked utterance/quantifier/relationship keys, relation types, provenance and
  access scope across train, recovered validation and test CSVs.
- Prepared a private page linking all 636 real test audio files, plus the existing
  evidence-bearing listening pages. No decisions, corrections or ACKs were added.
- Recorded D43 and machine-readable deadline status; owner accepts continued
  working-study use with explicit reference-quality/missing-artifact limitations.
  The strict verified-reference publication gate remains active.

These are technical contributions with coding-agent assistance. This entry does
not assign a human identity to Member 1 or claim new human listening/adjudication.

## 2026-11-21 ? partial supplied-review import

The owner (`rev_rasit`, Member 5) supplied three review exports. Imported 57
compatible primary records: 53 accept, four correct (49 test, four validation,
four train). One inconsistent text-changed flag was amended only after the
owner explicitly confirmed accepting the original reference; original export
bytes and decision timestamps are preserved. No transcript edits were inferred.

A private provisional page binds the four corrections to their primary record
IDs for a different listener. The full pinned adjudication/accept-sample plan
still awaits complete primary coverage. There are 192 missing primary records;
no assumed acceptances were added. Frozen references, splits and metrics remain
unchanged. See [aggregate receipt](../data/verification/turas-tr-econ-legis-v0.1-review-progress-v1.json).
These are supplied primary decisions, not independent confirmation of listening
or a completed verified reference release.

## 2026-11-21 ? completed primary scope from explicit owner attestation

The owner (`rev_rasit`) explicitly confirmed reviewing and accepting all 192
remaining items and retaining the four earlier corrections. Generated a primary
JSON export that declares its source as a user batch attestation. The shared
UTC timestamp is the recording time of that attestation; no historical per-clip
listening/playback timestamps are claimed. The earlier 57 records are preserved.

The canonical set now has 249 primary records: 245 accept and four correct
(224 test, 15 validation, 10 train). The reproducible final adjudication plan
pins 55 items, including 25 sampled accepts. Generated two private second-listener
pages with the primary reviewer blocked from adjudicating their own records.
No second-listener decisions, new reference variant or changed metric is claimed.
See [current receipt](../data/verification/turas-tr-econ-legis-v0.1-review-progress-v2.json).
