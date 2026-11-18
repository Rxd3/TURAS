# TURAS listening verification guide — `turas-verbatim-v0.1`

Applies to every validation and test utterance of `turas-tr-econ-legis-v0.1`
and to the pilot sample. Train references stay `unreviewed` (upstream human
minutes) and are labelled as such in every training run.

## What the reviewer does

1. Listen to the **whole** clip (10–20 s) at normal speed; replay as needed.
2. Compare it with the upstream reference text, word by word.
3. Choose one decision:
   - `accept`: the text matches what is said (see the rules below).
   - `correct`: the speech is clear but the text differs; type the corrected
     text. The correction becomes the reference with
     `reference_origin = local_human`.
   - `reject`: unintelligible speech, overlapping speakers, words cut so badly
     at a boundary that they cannot be written, non-Turkish speech, audio that
     does not match the text at all, or anything you are unsure of. **Never
     guess.** Add an issue code.
4. Optionally note issue codes: `boundary_cut`, `overlap`, `noise`,
   `edited_minutes` (minutes omit or reword spoken words), `wrong_segment`,
   `non_speech`, `other`.

A second reviewer adjudicates every `correct` and `reject` decision and 10% of
`accept` decisions chosen at random (seed 2026). Disagreements are resolved by
listening together; the adjudicator ID is recorded.

## Transcription rules (verbatim)

- Write what is **spoken**, in standard Turkish orthography with Turkish
  letters (ç ğ ı İ ö ş ü, â î û where the minutes use them).
- Keep the minutes' punctuation and capitalization style; the evaluator
  reports raw and normalized WER, so do not spend time on commas.
- Numbers: keep digits as the minutes write them (`5 milyar lira`,
  `yüzde 42`, `2026 yılı`). Do not expand digits to words or the reverse.
  If the speaker says a number differently from the text (e.g. text `%3`,
  speech `yüzde üç`), write `yüzde 3`.
- Proper names, acronyms (TÜİK, KDV, SGK) and suffix apostrophes
  (`Merkez Bankası'nın`) as in the minutes.
- Remove anything not spoken: speaker labels, `(Alkışlar)`, `(Devamla)`,
  constituency names in parentheses, signatures.
- Add words the minutes left out if they are clearly spoken (restarts and
  repeated words count as spoken). Do not write fillers (`ııı`, `eee`).
- A word cut at the start or end of the clip: if fully understood, write it;
  otherwise `reject` with `boundary_cut`.

## Records (data contract 0.1)

Each decision produces a review record with: record ID, utterance ID,
pseudonymous reviewer ID (e.g. `rev_m1`), UTC timestamp, audio SHA-256,
policy version `turas-verbatim-v0.1`, decision, corrected text (if any),
issue codes and adjudicator ID. Records live with the protected references
outside Git; this folder holds only the policy and aggregate audit counts.

Status mapping: `accept` → `verified`; `correct` → `verified` with
`local_human` origin; adjudicated decisions → `adjudicated`; `reject` →
`rejected` (the row moves to the exclusion ledger, reason
`rejected_in_listening_review`). Validation/test freeze requires every row to
be `verified` or `adjudicated`. A correction after freeze creates a new release.

## Pilot (before full verification)

A pilot of 50 utterances (25 test, 15 validation, 10 train, spread over
recordings, seed 2026) is reviewed first. Its accept/correct/reject rates
decide whether full verification of validation (2 h) and test (3 h) is
feasible before the deadline, or whether test must shrink. Pilot results are
reported in `data/verification/turas-tr-econ-legis-v0.1-audit.md`.
