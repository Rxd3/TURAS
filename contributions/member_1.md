# Member 1 — Dataset and Relational Schema

Name: unassigned (member to add). GitHub username: unassigned (member to add).
Branch: `feat/m1-data-relational-schema`.
Ownership/tasks/review routing: [TEAM_TASKS.md](../TEAM_TASKS.md).

**Status (2026-11-18): release v0.1 rebuilt locally and fully tooled;
lexicon frozen; pilot review done (sampled verification, pending M6 approval), then freeze.**

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
- Relational CSV tables (train 31,640 relationships, evaluator-only validation/test) per [RDL contract](../docs/contracts/rdl.md): static
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

## Experiments and artifacts

No model runs. Full audio build done 2026-11-18: 5,450 FLAC files (1.9 GB, 16 kHz mono), 200 sampled hashes re-verified, max duration drift 0.0001 s. Hashes: see
`data/splits/turas-tr-econ-legis-v0.1/split_manifest.json` and
`data/manifests/turas-tr-econ-legis-v0.1.json`.

## Reviews and handoffs

Pending: M6 review of the source/schema policy and the data contract 0.2
amendment (16 kHz mono FLAC, which this build follows); M5 review of the
relational tables; consumer hash acknowledgments after freeze.

## Remaining work

- Pilot listening review (50 clips), then full validation/test verification.
- Freeze the release with M6 after verification; regenerate references with
  review records.
- Lexicon frozen by hash 2026-11-18; waiting for M6 acknowledgment.
- Regenerate per-split relational tables at freeze (extractor fixed).
