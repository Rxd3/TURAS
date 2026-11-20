# M4/M5 numerical and inference-linker audit

The owner requested a correctness review after working validation completed.
This audit distinguishes computational checks from label quality and final
generalization; it does not certify every transcript as spoken-word truth.

## Acoustic evidence checked

The five selected/baseline systems each contain exactly the 438 frozen validation
IDs, with no dropped/error rows. Prediction hashes, reference identity and the
three selected checkpoint/member checksums passed. Declared decoder hashes match
within each parent/adapted family; CTC vocabulary extension remains explicit.

An independent JiWER 4.0.0 pass exactly reproduced all ten reported raw/normalized
WER and CER pairs, including total edit distances. It reused the shared Turkish
normalizer, with word splitting matched to the contract and raw character
whitespace preserved rather than JiWER's default trimming. The authoritative
scorer was not replaced. See [JiWER usage](https://jitsi.github.io/jiwer/usage/)
and the professor's [ASR metric reference](https://huggingface.co/learn/audio-course/en/chapter5/evaluation).

Train/validation intersections were zero for utterance IDs, recording groups,
exact audio hashes and shared-normalized reference text. This does not establish
speaker disjointness or rule out near-duplicate audio under a different encoding.

Paired recording-group bootstrap used all 15 validation recording groups,
2,000 replicates, seed 2026, the same sampled groups for paired systems and the
shared corpus aggregator. Normalized WER difference intervals (contribution
minus its parent baseline, percentage points):

| Comparison | Difference | 95% group-percentile interval |
| --- | ---: | ---: |
| LoRA − Whisper parent | −12.667 | [−14.241, −11.162] |
| Full Whisper − parent | −14.932 | [−16.606, −13.292] |
| Extended CTC − parent | −11.576 | [−12.693, −10.500] |
| Full Whisper − LoRA | −2.264 | [−2.895, −1.653] |

The legacy no-co-mention CTC diagnostic's one-word change has interval
[−0.093, +0.078] percentage points, including zero. These intervals are
conditional on the selected systems/labels; they do not remove validation
selection/multiple-comparison optimism, training-seed variation, label noise or
unknown shared speakers. They are not final-test confidence intervals.

Full machine-readable evidence: [result audit](../results/evaluation/m45_result_audit.json).

## RDL defect found and corrected

The frozen M1 alias table contains suffix wildcards such as `banka*` and explicit
exclusions in the source lexicon. The legacy RDL linker passed aliases through
punctuation tokenization, dropping `*`, then required exact token equality. It
therefore missed inflected forms while training relations had used M1's prefix
matcher. Earlier numerical RDL results were real runs of this limited linker,
but cannot establish results for the intended frozen-alias semantics.

D37 corrects this by embedding the hash-verified frozen lexicon into new versioned
graph bundles and calling M1's existing raw-text matcher, preserving suffixes,
exclusions, longest spans, apostrophes and canonical-ID ties. No new text
normalizer or model-generated reference was introduced. Existing legacy bundles
retain their original behavior; no old artifact is overwritten.

The corrected bundles preserve learned weights, training graph snapshots and
training provenance byte-for-byte. Only linker/version metadata changes. The
full and both relation-removal graphs must repeat the same validation grids on
the same acoustic candidates, reference version and gamma search. Acoustic
models/predictions/metrics require no retraining or recomputation for this fix.
Regression tests cover inflection, exclusions, multiword aliases, apostrophes,
provenance and legacy compatibility; the full suite passed 1,090 tests with one
skip and two subtests (four existing Transformers deprecation warnings).

## Corrected-linker rerun completed

All three revised graph bundles passed safe-load/member checks, with learned
weights and training snapshots byte-identical to their parents. All nine fusion
grids / 36 trial artifacts completed and passed shared validators/hash checks.
Their gamma-0 acoustic controls exactly reproduce the previous controls, confirming
the acoustic candidate/scoring path was unchanged.

Suffix-aware linking substantially increases known-entity coverage. For full
Whisper's 2,050 deduplicated candidates, linked-entity counts rose from 3,360 to
6,972, with all 2,050 candidates now linked. Only 30 of 438 utterances have
different relational scores across their candidates, limiting this model's
ability to change rankings. This is coverage evidence, not a measured term recall.

Corrected full-graph best normalized WER: LoRA 12.548% (gamma 0), full Whisper
10.201% (gamma 0), CTC 25.432% (gamma .1). None beats its original decoder's
one-best. Neither corrected relation-removal variant beats original one-best
either. The earlier legacy one-word CTC gain is not reproduced under corrected
semantics and is retained only as historical evidence, with its matching CI.

See [the corrected results and provenance](../results/evaluation/m45_corrected_linker_validation.json).
The acoustic numbers passed the audit; RDL's downstream result remains negative
after correcting its actual inference defect. No further ASR training was needed.

## Completion requirements after this audit

- M4: package the selected LoRA/full checkpoints and scored candidates; finish
  terminology metrics, final repeated timing and final-test comparison with the
  exact unchanged Whisper parent, then the reproducible report/private handoff.
- M5: finalize corrected-linker graph artifacts and fresh fusion/ablation results;
  finish terminology/relation-coverage reporting, joint end-to-end timing and the
  final matched CTC/acoustic/RDL test comparisons and artifact handoff.
- Freeze the reference versions, checkpoints and all final systems before test
  scoring. Test gold is still closed. Owner authority supplies technical choices;
  it does not turn unreviewed minutes into verified verbatim transcripts or invent
  another member's review/acknowledgment.
- Single-seed reporting is allowed under adopted M6 deadline policy and D36;
  seeds 42/2026 are optional follow-up robustness work, not a new completion gate.
