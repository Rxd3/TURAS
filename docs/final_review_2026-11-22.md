# Final technical review — 2026-11-22

The original working benchmark and initial submission materials are available
for review. PR #26 is open, awaiting non-author review. **The full
project definition of done is not yet satisfied.** This is an engineering audit,
not an independent team approval or a verified-reference release.

## Completed and checked

| Work | Evidence |
| --- | --- |
| Frozen data identity | 4,376 train, 438 validation, 636 test IDs; disjoint frozen files; manifest `a8fff65b99fcf11d9452685e438e250207032ecfe11152741562147126d095b8` |
| Runtime audio integrity | All 5,450 referenced audio files matched their frozen SHA-256 values; all three reference-free inference exports matched the release manifest and split order. No reference text was read by this audit. |
| M4/M5 study | 20 measured native/rescored conditions, 56 unchanged pinned code hashes, counts/metrics consistency, selected checkpoints, graph controls and repeated timing |
| B5 | Continued M3's epoch-13 weights with fresh AdamW at `3e-4`; epoch 22 selected on validation; early stop at epoch 28; complete validation/test and three-repeat timing |
| Supplied classical stage | Three Viterbi EM passes for a **single-diagonal-Gaussian character HMM**; 4,072 feasible training clips, 304 logged infeasible clips; full frozen evaluation |
| Output integrity | M3 predictions/N-best passed shared validators, raw/normalized metrics independently matched JiWER, archive CRC and model hashes passed |
| Presentation | A title slide and four content slides with native charts, and a timed English script (238 spoken words, about 1:52) |
| Reproduction delivery | Separate role packages, exact runtime source snapshots, restore and scoring commands, file hashes and upload/link instructions |
| Fresh Python-environment restoration | Archived B5 source/weights/audio matched 3/3 validation transcripts. M4/M5 restoration verified 226 indexed files and passed 3 acoustic, 3 online joint and 15 cached runs on two validation clips, with no references/test audio read. Same-machine owner-assisted check; independent review is not claimed. |
| Current offline suite | At this head: 1,256 passed and 43 skipped in a lean environment without torch (earlier: 1,200 passed, 19 skipped), four upstream deprecation warnings and 51 passing subtests; controller-lock and metadata-only weight-restoration regressions are covered. |

M3 normalized test results, against the same working test reference bytes:

| System | WER | CER | Terminology micro F1 |
| --- | ---: | ---: | ---: |
| B5, epoch 22 | 47.7763% | 13.5029% | 89.1200% |
| Single-Gaussian character HMM | 99.9951% | 88.7601% | 0.0000% |

The HMM result is a negative finding. The supplied method has one Gaussian per
character state and a DCT-of-normalized-log-Mel feature recipe. It does not
establish a conventional mixture-Gaussian phoneme HMM baseline.

## Required before claiming the whole project complete

| Remaining item | Owner / evidence needed |
| --- | --- |
| Publish this local closeout | Prepare and review the new baseline code, results, guides and final deck through the repository PR workflow. No push, merge or peer approval is claimed here. |
| Upload and test Drive retrieval | User uploads each role folder, sets actual named-account access, supplies the public submission URL, and tests downloads/checksums from a recipient account. |
| B2/B3 official results | Done by M6 in [`results/m6/`](../results/m6/README.md): B2 12.319% and B3 (Qwen3-ASR-1.7B) 16.094% normalized test WER, paired intervals, N-best oracle and same-session timing. M6 also scored M2's B2 handoff (`fd8471e9`) as an independent reproduction (12.392%). Remaining: B1 parity of the M6 scorer and paired intervals against F1/F2, both needing the private M4/M5 bundle. |
| Owner B2/B3 local rerun | A separate reproduction, not the reported result. It completed all 438 B2 validation IDs (12.847% WER, 5.679% CER) and 636 test exports with zero error rows; both Qwen candidates completed validation and its minimum-CER rule also selected 1.7B (5.423% versus 7.067% CER). Any timing or test scores from it should be reported only as agreement checks against `results/m6/`. |
| Expanded baseline closeout | Seda Turkish RNN-T (29.96% WER) and a four-mixture grapheme HMM (99.97% WER, 64.55% CER): aggregate results, edit counts, export hashes and timing are committed ([completion report](../results/final/BASELINE_COMPLETION_REPORT.md)); per-utterance exports are reported but not verified in the repository. The new HMM uses all 4,376 train clips, three states per grapheme, four diagonal Gaussians and five fixed EM passes; it has no phoneme lexicon or external LM. The supplied single-Gaussian result is preserved. |
| Refreshed delivery and presentation | Populate folders 10–13, refresh source/reports and the four-slide two-minute deck, render/inspect every slide, then verify checksums/restoration before declaring the new delivery ready. The staged immutable payloads alone are not the finished delivery. |
| Independent reference adjudication | 249 primary decisions are recorded for the separate review plan, but 55 independent adjudications/release acceptance remain pending. Current benchmark test labels retain 611 unreviewed / 25 historical single-reviewer tags. |
| Independent reproduction and review | A non-author retrieves the packages, records exact hashes/environment/commands, runs the appropriate restore or scoring check, and records a real review/receipt. The new isolated Python-environment checks above are owner-assisted same-machine checks, with no independent approval or fresh-OS claim. |
| Presentation and contribution acceptance | Actual owners review the claims and contributions. Do not infer a person's authorship or approval from file ownership. |

The original corrected validation bundle and full exclusion ledger remain
unavailable. The working study explicitly uses recovered validation minutes.
Recording-group disjointness does not prove speaker disjointness, and single-seed
runs do not establish training-seed robustness. These limitations stay in the
presentation and final report.

## Review scope and source history

The local checkout advanced without conflicts from `d2bdf5e` to merged GitHub
`1dade5b`. The M3 runtime source was preserved **before** that update. The only
BiLSTM model-file change in those commits adds exact repeated-target feasibility
checks; it does not change the model's forward pass. Frozen M4/M5 code and report
bytes remain unchanged. Recent unmerged M2 records are used as reported evidence,
not as proof that a missing artifact was received.

Machine-readable local check results and package checksums accompany the
delivery. Historical test counts in older reports describe those runs, not this
review's current test count. No new model or rescoring configuration is selected
from these final test results.

## Related evidence

- [M3 aggregate results](../results/final/m3_test_summary.json)
- [M4/M5 frozen report](../results/final/M4_M5_FINAL_REPORT.md)
- [Reproduction guide](final_reproduction.md)
- [M1 working-reference disclosure](m1_deadline_handoff.md)
- [HuBERT/transducer audit](asr_language_support_audit.md)
- [Follow-up protocol and validation selection](baseline_completion_followup.md)
- [Fresh-environment restoration evidence](../results/final/fresh_environment_restoration.json)
- [Machine-readable review checks](../results/final/final_review_checks.json)
