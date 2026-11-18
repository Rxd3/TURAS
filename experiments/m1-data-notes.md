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
