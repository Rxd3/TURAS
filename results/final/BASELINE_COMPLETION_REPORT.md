# Additional completed baseline results

These completed runs fill the B2/B3/B5 and classical/transducer result gaps.
They are imported as archived experiment evidence. The cleaned repository at
`1cf8129` remains the source of truth; no older implementation, configuration,
presentation or frozen M4/M5 report is restored by this update.

Every archived prediction/N-best export was hash-checked and validated with the
current shared schemas. The current shared evaluator reproduced raw and
normalized corpus edit counts and terminology micro metrics exactly on all
438 validation and 636 test clips for each of the six systems below. Inference
was not rerun on this checkout. The shared evaluation/decoding code, frozen
splits, reference bytes and lexicon are unchanged from the run sources.

## Frozen test results

| System | Normalized WER | Normalized CER | Term micro F1 | End-to-end RTF | Device |
| --- | ---: | ---: | ---: | ---: | --- |
| B2 Whisper-large-v3 | 12.387% | 5.461% | 97.735% | 0.15110 | RTX 5070 |
| B3 Qwen3-ASR-1.7B | 16.172% | 5.620% | 91.322% | 0.11824 | RTX 5070 |
| B5 BiLSTM-CTC, epoch 22 | 47.776% | 13.503% | 89.120% | 0.00932 | RTX 5070 |
| Seda Zipformer2 RNN-T | 29.958% | 12.812% | 93.445% | 0.02842 | CPU |
| Four-mixture grapheme HMM | 99.971% | 64.549% | 0.295% | 0.00167 | CPU |
| Supplied single-Gaussian character HMM | 99.995% | 88.760% | 0.000% | 0.00170 | CPU |

The [CSV](baseline_completion_metrics.csv) includes raw WER/CER. The
[summary](baseline_completion_summary.json) retains corpus counts, validation
metrics, recording-group bootstrap intervals, exact export/model/decoder hashes,
run configurations, timing environments and artifact-package checksums.

Headline RTF uses shared `run_timed`, batch one, three train warm-ups and three
repeats over the frozen 107-clip test timing subset. Audio loading/resampling,
inference and in-memory result serialization are included; model loading and
output-file writing are excluded. The newer four-model timing jobs ran serially
after their downloads/training ended. CPU and GPU rows use distinct devices and
do not establish a matched hardware-efficiency ranking. Available thread settings and
per-repeat RTF are preserved in the summary. RTF is not a streaming guarantee.

## Selection and method boundaries

- **B2:** a local GPU rerun, separate from M2's originally reported ZIP. No
  receipt of that earlier handoff is inferred.
- **B3:** both Qwen candidates completed full validation. The fixed minimum-CER
  rule selected 1.7B (5.423% CER) over 0.6B (7.067%) before selected-model test
  inference. Validation WER was 15.343% and 22.345%, respectively. The frozen
  selection timestamp and report hashes are retained. No hotwords or external
  context were supplied; Qwen scores remain speech-language joint scores.
- **B5:** continued the supplied epoch-13 weights with fresh AdamW at `3e-4`.
  Validation selected epoch 22; training stopped at epoch 28. The archived
  primary decoder is greedy CTC with scored N=1, not a claimed beam result.
- **Seda:** pinned `atasoglu/seda-v0.1` Turkish RNN-T, int8 encoder/joiner,
  fp32 decoder, one CPU thread and greedy decoding. Its optional external LM
  was not used. Output is raw spoken form, without inverse text normalization.
- **Mixture HMM:** all 4,376 train clips were feasible. MFCC/delta/acceleration
  features have 39 dimensions, with three states per grapheme and four diagonal
  Gaussian components per state. Five fixed Viterbi-EM passes were used.
  Distinct component means and positive normalized weights were verified.
  This has no pronunciation lexicon or external LM and is distinct from a
  conventional phoneme/lexicon GMM-HMM system. Its weak result is preserved.
- **Supplied HMM:** the earlier single-Gaussian/DCT feature baseline is a separate
  negative result. It is not renamed or counted as the new mixture model.

## Provenance and reproduction

All rows use the frozen `turas-tr-econ-legis-v0.1` 636 test IDs and working
reference payload SHA-256
`31e372f0e35f504de59b49f1e0e1f3ed6fcc2772138c591a2cf2b054a4e7ef3f`.
The split-manifest and lexicon hashes are recorded in the summary. The original
20-condition M4/M5 study and its paired adaptation/RDL findings remain unchanged.

The original run source is preserved in the separately prepared role delivery;
see [reproduction instructions](../../docs/reproducibility.md#additional-completed-baseline-runs).
The summary identifies the archived results commit, source-aggregate hashes and
exact source/model package hashes. Owner-assisted restoration in new isolated
Python environments matched three validation transcripts per new evaluated
model, and all four newer complete saved-test score reports reproduced exactly.
No independent person, cloud download or full-test inference replay is implied.

These are follow-up experiments after earlier working-test exposure. References
are edited minutes, with 611 unreviewed and 25 historical single-reviewer test
tags. Single-seed training, unproven speaker disjointness and incomplete
pretraining-overlap disclosures limit the conclusions. Independent reference
adjudication and technical approval remain pending. No team member's approval
or contribution is inferred from the imported artifacts.
