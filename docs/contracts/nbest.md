# N-best and score/rescoring contract — 0.2 (proposed by M6, pending review by M4/M5 and M3)

N-best is a separate UTF-8 JSONL artifact, one object per utterance. It requires
schema/run/system/release/split IDs, utterance/audio/split hashes, decoder and model
hashes, `requested_n`, `actual_n`, `score_convention_id`, and a `candidates` array.
Do not pad a short list with invented alternatives. Preserve original rank order
and candidate IDs after deduplication; record the raw-to-deduplicated mapping.
Rows keep `schema_version: "0.1"`: contract 0.2 makes exact what 0.1 left open
and adds only the optional row key `dedup_map`.

## Deduplication and ranks

The optional row-level object `dedup_map` maps each removed raw candidate ID to
the candidate ID that was kept. After deduplication, the candidate `text` values
in a row are pairwise distinct, compared exactly, code point for code point; a
repeated text is invalid even when `dedup_map` is present. `actual_n` is the
number of kept candidates and is at most `requested_n`. Kept candidates keep
their original `rank` values: integers, strictly increasing, from 1 to
`requested_n`, with gaps allowed (for example 1, 3, 4). Candidates whose raw
texts differ but match after `tr-normalized-v1` are allowed and reported as a
warning.

## Candidates

Each candidate has `candidate_id`, original `rank` (1-based), raw `text`,
`token_ids`, `token_count`, `acoustic_log_score`, `acoustic_score_kind`,
`decoder_sequence_score`, `lm_log_score`, `rdl_log_score`, `combined_score`,
and `score_details`. IDs are unique within utterance/run and stable on rescoring.
Token IDs/count refer to that model's tokenizer, not whitespace words. Score
fields unavailable for a condition are null with an explicit reason in details:
`score_details.null_reasons` maps each null field among `decoder_sequence_score`,
`lm_log_score`, `rdl_log_score` and `combined_score` to a non-empty string.
The originating model score is mandatory for every N-best candidate.

## Score meanings

- All canonical log scores use natural logarithms; higher is better. Finite
  negative log-likelihoods/costs must be negated and their original sign/base
  recorded. Never mix loss, log probability, beam rank and probability.
- Seq2Seq acoustic score is the teacher-forced sum of model log probabilities
  of generated output tokens conditioned on audio/prompt. Record inclusion of
  BOS/EOS, language/task tokens and prompt, temperature, forced tokens, and
  token exclusion policy. An integrated speech-language score must declare that
  it is joint; do not falsely separate an unavailable acoustic-only component.
- CTC sequence probability marginalizes alignments when computed exactly. A beam
  approximation must be labeled `ctc_beam_log_score` with pruning details; a
  Viterbi path score is `ctc_path_log_score`. Never call a frame/path sum an exact
  transcript probability. An LM-fused beam total is distinct from acoustic score.
- Preserve framework `decoder_sequence_score` separately with its library/version,
  length penalty and exact formula. Reconstruct token sums or export sufficient
  per-token evidence; do not assume a library beam score is an unnormalized sum.
- `lm_log_score` is an optional external LM token log-probability sum with its
  vocabulary, EOS and tokenization policy. Set its weight to zero when disabled.
- `rdl_log_score` is the trainable graph compatibility score specified in the
  RDL contract. It is not an acoustic probability or a hand-authored confidence.

`score_details` records original raw score, base/sign conversion, exact versus
approximate status, tokenizer hash, token count policy, score length divisor,
N-best algorithm (beam/lattice), pruning, seed, and duplicate policy. Adapters
without real scored N-best must report unsupported capability. A single decoded
string repeated N times, artificial scores, or fabricated perturbations are invalid.

## Joint selection

For candidate h, use the versioned formula:

`S(h) = alpha * A(h) + beta * L(h) + gamma * R(h)`

`A = acoustic_log_score / max(1, token_count)^p_a`;
`L = lm_log_score / max(1, lm_token_count)^p_l`;
`R = rdl_log_score` (already averaged over extracted relations). Exponents,
weights, candidate generation parameters, and any validation-fitted affine
calibration are frozen in decoding configuration. If beta/gamma is zero the
corresponding value may be null and its term is exactly zero; an enabled term
must be present. Calibration statistics are fit on train/validation only.
No direct score comparison across different models/tokenizers is implied.

Select the maximum combined score; ties resolve by original decoder rank and
then candidate ID. Keep original ranks and component scores in the artifact and
record `selected_candidate_id` in predictions. RDL must not use gold to choose
candidates, tune weights, or construct graph features. Compare acoustic-only,
acoustic+LM, acoustic+RDL, and all-enabled conditions on identical candidate sets.
Do not double-count a fused LM already inside the exported acoustic score.

N-best oracle WER is an evaluator-only upper-bound diagnostic using protected
references; never use oracle choice as a deployable prediction. Record actual N
and the ceiling imposed by candidate coverage, including identical/empty lists.

## Version history

- 0.1 (2026-11-17): initial contract.
- 0.2 (proposed by M6 on 2026-11-17, pending review by M4/M5 and M3;
  [D18](../decisions.md)): names `score_details.null_reasons`, adds the optional
  row-level `dedup_map`, requires distinct candidate texts after deduplication,
  and keeps original rank values with gaps allowed.
