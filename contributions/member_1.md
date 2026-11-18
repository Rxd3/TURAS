# Member 1 — Dataset and Relational Schema

Name: unassigned (member to add). GitHub username: unassigned (member to add).
Branch: `feat/m1-data-relational-schema`.
Ownership/tasks/review routing: [TEAM_TASKS.md](../TEAM_TASKS.md).

**Status: proposed release v0.1 built; validation/test listening verification
pending; not frozen.**

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
- Freeze the lexicon for terminology metrics (M6 deadline 2026-11-19).
