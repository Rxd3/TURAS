# Member 1 data notes

## 2026-11-18 — WorldSpeech `tr_tr` audit and proposed release v0.1

- Read every non-audio column of all 389 `tr_tr` shards at revision
  `b76b8539…`: 239,654 rows, 1,007.7 h, 197 recordings (YouTube), 2017–2026.
- Finding 1: upstream test is not recording-disjoint from upstream train
  (188/188 test recordings also appear in train). Decision: new
  recording-grouped split; upstream test rows never go to TURAS train.
- Finding 2: 7.1% of parliament rows carry non-spoken stenographic text
  (speaker headers, `(Alkışlar)`, `(Devamla)`). Excluded, with issue codes.
- Finding 3: lexicon iteration. First pass showed false positives from
  ambiguous Turkish forms (`zaman`→zam, `yasak`→yasa, `ücretsiz`→ücret,
  `ihraç` = expulsion, `ödenen` = paid, `gelir` = comes). Fixed with exact forms
  or exclusions before any split existed; tests cover them.
- Finding 4: first ranking (quantities uncapped) made 78% of segments
  monetary and only 8% legislative in test. Capped quantity and legislative
  contributions; final shares are 63–70% monetary and 22–24% legislative.
  This was decided on topic shares only, never on model results.
- Release v0.1 proposed: 20 h train / 2 h validation / 3 h test (5,450 clips).
- Next: pilot listening review of 50 clips; then full validation/test review
  or a smaller verified test if the pilot reject rate is high.

## 2026-11-18 (later) — branch applied locally, review, lexicon freeze

- Applied the cloud bundle on a fresh clone of `Rxd3/TURAS`; pushed
  `feat/m1-data-relational-schema` (`2e67e8d`). Tests: 22 passed. Every
  committed hash re-computed and equal to the manifests; split files sorted,
  LF, disjoint (0 shared IDs); pilot = 25 test / 15 validation / 10 train.
- Lexicon `turas-domain-lexicon-v0.1` frozen by hash (`8ef903fe…`) for M6's
  terminology metric, without editing the file: its hash is already in the
  split manifest and drives selection. Matcher checks run:
  `TL'ye → tl` matches, `tlye` (tr-normalized-v1 form) does not, so the
  metric must match on raw text. Recorded limitations: `dolarak`→dolar,
  `madde` (substance), `liranın` blocked by the `liran` exclusion,
  `borçlanma` → BANK_BORC.
- Independent code review of the M1 package and scripts. Fixed:
  review ingestion (no dedup, weak validation, no status mapping or
  adjudication), amount parsing (compound amounts and dot decimals gave wrong
  values), co-mention key collisions across splits, `--skip-audio`
  overwriting references and manifests with empty hashes, review page
  escaping and stale saved decisions. Build now records code commit and
  library versions. Tests: 49 passed (22 before).
- Documented (not changed): 18/23 upstream-test rows in validation/test,
  613/636 test rows are upstream-train (no WorldSpeech-trained checkpoints
  allowed), exact-text-only duplicate audit, resampler erratum (config VHQ vs
  code HQ).
- Reviewed M6's data/predictions/normalization 0.2 proposals (read only):
  approve, with a clarification on the audio root and the raw-text term
  matching note. Comments handed to M1 to send.
- Not done yet: pilot listening decisions (none received), full
  validation/test review, freeze build, consumer acknowledgments.

## 2026-11-18 (evening) — local rebuild, relational regeneration, review pages

- Full rebuild on M1's machine at commit `ad66fa1` (clean tree): 239,654
  metadata rows; exclusion ledger sha256 `6077dc6f…` and all three split files
  byte-identical to the cloud build. Max |measured − upstream duration|
  0.0001 s. Hours 20.002 / 2.002 / 3.004.
- FLAC bytes differ from the cloud build (total 28 bytes smaller). Checked on
  the 50 pilot clips: same length, max 1 LSB per sample, SNR ≥ 99.8 dB, i.e.
  decoder/encoder rounding. New split manifest sha256 `1af56ddb…`; library
  versions now recorded in the manifest (libsndfile 1.2.2, soxr 1.1.0).
  Pilot equivalence recorded as evidence (`audio_equivalence.jsonl`, 50/50 pass)
  so pilot decisions made on the cloud audio can be ingested.
- Relational evidence regenerated with the fixed extractor: quantifiers
  train 1,435 → 1,459, test 370 → 373; relationships unchanged
  (31,640 / 3,784 / 6,248). 144 compound train amounts now have full values.
- Listening pages for full verification: test 611 clips in 7 pages,
  validation 423 clips in 5 pages, ≤100 clips each, pilot IDs left out.
- Freeze script written and tested on synthetic decisions in a temporary
  copy (one reject left its split and entered the ledger with reason
  `rejected_in_listening_review`; one correction became `local_human`). On the
  real artifacts with no reviews it reproduces the split files exactly and
  refuses to freeze (1,074 validation/test rows unreviewed).
- Source registry now carries the contract's rights/terms columns.

## 2026-11-18 — pilot listening results

- Pilot ingested: 34 accept / 16 correct / 0 reject (test 15/10/0,
  validation 12/3/0, train 7/3/0). Upstream minutes vs verified text: 2.43%
  WER over 1,561 words. Re-ingesting the same export added 0 records.
- Decision needed: full verification of 1,074 validation/test clips is
  ≈39 h at the measured pace; options are a seeded, recording-stratified
  verified subset or more reviewers. No reject-driven shrinking is needed.
- Decision (M1, 2026-11-18): stop listening review at the 50-clip pilot
  (sampled verification); documented in `docs/dataset.md`, sent to M6 for
  approval (D08). Freeze script gained `--sampled-verification`; trial run:
  40 validation/test clips verified (13 corrected), 1,034 unreviewed, 0 rejected.

## 2026-11-18 — release frozen

- M6 approved sampled verification (D08), as reported to M1. Froze with
  `m1_freeze_release.py --sampled-verification`: 5,450 rows kept, 0 rejected,
  16 references replaced by listening corrections (`local_human`), 50 rows
  `verified`, 1,034 validation/test rows upstream minutes (`unreviewed`).
  Split files unchanged; split manifest sha256 `a8fff65b…`.
- Relational evidence regenerated from frozen references (train 31,639,
  validation 3,784, test 6,249 relationships).
- 24 integrity checks passed: hashes of references, ledger, exports and split
  files; allowlist-only inference rows in split-file order; audio hashes;
  verified IDs equal the pilot IDs.
- Handoff bundles built (inference for all model owners; train references
  for M3/M4/M5; evaluator-only for M6), hashes in
  `data/manifests/turas-tr-econ-legis-v0.1-handoff.json`. Consumer
  acknowledgments pending.
