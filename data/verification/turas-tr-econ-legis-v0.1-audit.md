# turas-tr-econ-legis-v0.1 listening-verification audit (aggregate, no text)

Historical M1-reported audit (2026-11-18), policy `turas-verbatim-v0.1`.
The generating 50 review payloads are unavailable in the owner workspace and
supplied Drive as of 2026-11-21. The table is retained as reported history,
not independently reproduced listening/adjudication evidence. See
[current closeout](../../docs/m1_closeout.md) and the status erratum.

## Pilot sample

| Split | In scope | Reviewed | accept | correct | reject | conflict | adjudicated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 25 | 25 | 15 | 10 | 0 | 0 | 0 |
| validation | 15 | 15 | 12 | 3 | 0 | 0 | 0 |
| train | 10 | 10 | 7 | 3 | 0 | 0 | 0 |

## Whole release

| Split | In scope | Reviewed | accept | correct | reject | conflict | adjudicated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 636 | 25 | 15 | 10 | 0 | 0 | 0 |
| validation | 438 | 15 | 12 | 3 | 0 | 0 | 0 |
| train | 4376 | 10 | 7 | 3 | 0 | 0 | 0 |

Issue codes over all records: none

Validation/test fully reviewed (every row verified, adjudicated or rejected): **no** (validation 438 rows, test 636 rows).
## Pilot findings (M1, 2026-11-18)

- 50/50 pilot clips reviewed by `rev_m1`: 34 accept (68%), 16 correct (32%),
  0 reject. The pilot page held the first build's audio; all 50 clips were
  measured equivalent to the release audio (≤1 LSB) before ingestion.
- Word edits in the 16 corrections, measured on lowercased text without
  punctuation: 20 substitutions, 9 deletions, 9 insertions, over 1,561 upstream
  reference words. That is a word error rate of 2.43% for the upstream minutes
  against the verified text (one correction changed punctuation only).
  Insertions are spoken words the minutes left out.
- Test-only pilot correction rate: 10/25 = 40%, Wilson 95% interval
  23.4–59.3%. The pooled 16/50 is not the test estimate. Twenty of 38 edits
  were substitutions, so omitted words were not the dominant edit category.
- Edited minutes have output-style-dependent bias. These reported pilot
  diagnostics do not justify a common WER offset or ranking threshold.
  Zero reported rejects in 50 clips does not establish full-corpus fidelity.
- Pace: 50 clips in 112 minutes (≈2.3 min/clip). The remaining 1,034
  validation/test clips would take ≈39 h for one reviewer.
- Pending: second-reviewer adjudication of the 16 corrections and of 10% of
  the accepts (seed 2026), as the guide requires.

## Current evidence status (2026-11-21)

The Option C plan pins 224 test IDs (199 additional plus 25 original pilot
test IDs) and requires primary evidence for all 249 distinct sample/pilot clips.
Prepared pages are undecided. The recovered validation variant contains all
438 rows unreviewed; it is missing the original three corrections. All 15
validation pilot clips therefore require independent adjudication if records
remain unavailable. No new listening or M6 release approval has occurred.
The plan is prepared after working test exposure; its selection uses metadata
and prior pilot IDs only. Frozen split/audio/reference bytes remain unchanged.
