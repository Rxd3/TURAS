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
