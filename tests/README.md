# Meaningful test policy

Members 4/5 now have focused offline tests using tiny random models and synthetic
fixtures; they produce no benchmark results. Packaging imports validate only the
package layout. Tests are added alongside actual behavior, using hand-checkable fixtures;
no dataset/model download, GPU or live API is allowed in unit tests. Small
synthetic text fixtures in future tests must be clearly labeled and never become
claimed dataset transcripts or experiment predictions.

Highest-value checks when the corresponding implementation is introduced:

- Stable IDs, duplicate/group overlap, manifest hashes and frozen split integrity.
- Strict inference allowlist, absent reference access and gold-invariance checks.
- Turkish dotted/dotless I, normalizer idempotence and manual WER/CER edit counts.
- CTC/Seq2Seq score conventions, candidate identity and deterministic fusion ties.
- Checkpoint/tokenizer/base compatibility and explicit failed/missing prediction handling.
- Held-out graph edges/reverse duplicates and reference-derived feature exclusion.

Model owners write component tests; M6 coordinates shared contract/evaluation and
integration checks. Real-data/model integration runs are separately declared and
opt-in, with lawful hashed artifacts. Do not generate speculative placeholder tests
or coverage targets for empty packages. Record checks actually run in each PR.
