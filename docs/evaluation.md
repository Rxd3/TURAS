# Evaluation and timing policy — 0.2 (proposed by M6, pending review by M1 and model owners M2–M5)

Owner adoption for M4/M5 integration: 2026-11-19; see [D24](owner_decisions.md).
This records project approval, not independent peer-review sign-off.

M6 owns the shared offline evaluator, now integrated on the M4/M5 branch. All baselines,
fine-tuned systems and RDL conditions must be reported on the same frozen test
IDs and reference release. A missing or incompatible prediction bundle blocks
comparison; never silently score its successful subset.

## WER/CER and aggregation

Report both [raw and Turkish-normalized](contracts/normalization.md) WER and CER.
WER uses word-token Levenshtein alignment; CER uses Unicode code-point alignment
with the documented internal-space policy. For each unit type, export substitution
S, deletion D, insertion I and reference count N per utterance and corpus.

`corpus_error_rate = (sum(S) + sum(D) + sum(I)) / sum(N)`

Do not average per-utterance percentages to produce corpus WER/CER. Report
denominators and rates as both ratios and clearly labeled percentages when used.
Rates may exceed 1. Deterministic alignment ties prefer substitution, then
deletion, then insertion so error-type counts remain stable. Alignment direction
is reference to hypothesis; the backtrace runs from the end of both sequences
(exact rule and golden test in
[normalization](contracts/normalization.md#alignment-tie-break)). Empty hypotheses incur deletions; an empty reference
adds insertions and zero denominator. Report zero-reference cases separately;
if total N is zero, the aggregate rate is undefined, not zero.

Failures retain all frozen IDs. Strict export validation catches missing/extra
records and hard errors before scoring. A declared reliability report additionally
maps explicit `error` records to empty hypotheses for an all-ID error-inclusive
view, counts errors separately, and retains their original status. Successful-only
scores, if diagnostic, must state their denominator and cannot replace the primary
matched all-ID comparison. Do not silently impute absent rows.

## Metric names and course mapping

The Hugging Face audio course (chapter 5) asks to justify WER, CER and any
additional metric. Reports name metrics as follows:

| Course name | TURAS metric |
| --- | --- |
| Orthographic WER | WER under `raw-v1` |
| Normalised WER | WER under `tr-normalized-v1` |
| CER | CER under each view, with the space policy stated |
| WAcc | `1 - WER` under the same view |
| RTFx | `1 / RTF` under the same timing condition |

WER is the course's headline ASR metric. CER also gives partial credit inside
long suffixed Turkish words, where one wrong suffix makes the whole word wrong.
The normalized view separates spelling conventions (case, punctuation) from
recognition errors; the raw view keeps them visible. The course's
`BasicTextNormalizer` is not used: it is not Turkish-aware and splits `İZMİR`
into three tokens.

## Domain terminology

M1/M6 freeze a versioned domain lexicon with canonical IDs, approved aliases,
topic/entity/quantity categories and provenance before test evaluation. Train-only
and audited independent external sources may build inference vocabulary; held-out
reference annotations remain evaluator-only. For the M4/M5 final protocol, owner
decision D38 freezes `canonical-occurrence-m1-v1`: apply M1's existing raw-text
`screening_tokens` and lexicon matcher, longest spans left-to-right without
overlap, canonical-ID ties, prefix aliases and exclusions. This amends the earlier
normalized-span proposal because normalization can erase apostrophe boundaries
needed by frozen aliases. WER/CER normalization remains unchanged. The frozen
lexicon hash and ambiguity rules are recorded in the final protocol.

Report exact canonical term precision, recall and F1 from occurrence-count
multisets per utterance (TP = minimum reference/predicted count per canonical
term, then FP/FN); aggregate micro counts, macro over supported canonical terms,
and support by topic/entity/quantifier. Zero predicted/reference supports must be
visible; use zero for undefined per-term P/R/F1 and exclude zero-reference terms
from supported-term macro recall/F1. Report false positives for unsupported terms.
Also report WER on utterances containing domain terms, clearly labeled as a
subset diagnostic. Do not substitute approximate entity/number matching for the
mandatory exact term metric; any relaxed quantity-value view is separate.

## Matched comparisons and uncertainty

Pair B1 with Whisper-small LoRA/full; B2 with any large-v3 adaptation; B4 with
fine-tuning of the identical checkpoint, head, vocabulary and processor. Declare
changes such as vocabulary extension as separate ablations. B3 and the trained
from-scratch B5 are independent baseline tracks, not untrained placeholders.

Hold candidate generation fixed for acoustic/LM/RDL rescoring comparisons. Choose
hyperparameters, checkpoint steps, fusion weights, normalizer, lexicon, graph
snapshot and stopping policy on train/validation before opening final test scores.
Freeze the selected configuration and timestamp. Post-test hypotheses are
exploratory, labeled, and require a fresh holdout for confirmatory claims.

Report paired differences and, when support permits, 95% group-bootstrap
intervals resampling frozen leakage groups (2,000 replicates, evaluation seed
2026 as the initial protocol). Preserve pairing across systems. Report actual
group count and uncertainty limitations; one split/seed is not proof of robustness.
Separate preregistered comparisons from exploratory multiple comparisons.

## Real-time factor and latency

`RTF = total measured wall-clock seconds / total decoded audio seconds`.
Use sum/sum, not a mean of utterance ratios. RTF < 1 means faster than real time
for the measured hardware, scope and batching; it is not a streaming guarantee.

Record CPU model/threads, GPU model/count/VRAM/driver, RAM, OS, framework/CUDA,
precision/quantization, sample rate/resampling, storage/cache condition, batch size,
length bucketing, beam/N-best size, concurrency and synchronization method.
Synchronize accelerators before/after timed sections. Use a monotonic wall clock.

Separate cold model/graph load time from warm inference. Warm up with at least
three batches from train/validation only, record exact count and exclude them
from numerator/denominator. Repeat the frozen timed workload three times where
feasible; report actual repeats, total duration and dispersion. Batch-1 latency
and fixed-batch throughput are separate conditions; do not compare models with
undisclosed batching or different hardware as direct speed wins.

End-to-end timed scope begins before audio reading/decoding/resampling and ends
after text decoding, N-best extraction, optional LM/RDL lookup/scoring and result
serialization. Report core model time separately if useful. State whether input
queue/network/storage and output write/flush are included. For batch runs report
batch latency plus throughput RTF; do not assign the whole batch time to each
utterance. Report p50/p95 request latency only from actual request measurements,
with sample count; separate single-utterance and batched workloads. Report memory
and failures/OOMs. CPU/GPU RDL transfer costs belong in joint end-to-end timing.

Required timing sidecar fields: timing record ID, run ID, utterance IDs in batch,
audio duration, batch size, warm-up flag, cold-load time, per-stage and end-to-end
seconds, device synchronization, hardware/environment hash and measurement scope.
Training wall time/peak memory is separate from inference RTF.

The headline condition is end-to-end RTF at batch size 1, with 3 timed repeats
over a frozen 30-minute timing subset of test IDs, sampled by recording group
with seed 2026 when the release freezes. Each timed boundary synchronizes the GPU
with `torch.cuda.synchronize()` when CUDA is in use and reads `time.perf_counter()`.
A secondary core-model RTF times only the model call, after audio is loaded and
before results are serialized; it is always labeled core-model and never replaces
the end-to-end figure. Batched throughput is an optional, labeled secondary view.

Before and after each timed repeat, record from `nvidia-smi` the GPU name,
driver version, SM clock and maximum SM clock, power draw, temperature and
`clocks_event_reasons.active`, `.hw_slowdown`, `.hw_thermal_slowdown`,
`.sw_thermal_slowdown` and `.sw_power_cap`. A sample in which a hardware or
thermal slowdown reason is `Active` is flagged, and the number of flagged samples
is reported. `sw_power_cap` alone is normal on laptops and is only recorded.
When `nvidia-smi` is unavailable, the record states the error and the run
continues.

## Final report requirements

Include system and parent baseline, split/release/hash, raw/normalized edit counts
and WER/CER, terminology supports/P/R/F1, RTF/latency scope/hardware, actual seed
count, uncertainty, artifact/run hashes, N-best coverage, failure counts, ablations
and limitations. No numeric result exists until a traceable run is completed.

## Declared limitations

The Monday 2026-11-23 deadline forces these cuts. Each is reported as a
limitation, never hidden:

- One training seed per system, labeled single-seed, instead of seeds 17/42/2026.
- Timing uses batch 1 and 3 repeats on a frozen 30-minute timing subset instead
  of a 1-hour subset.
- Terminology P/R/F1 is computed only if Member 1 freezes a lexicon by Thursday
  2026-11-19; otherwise the report states that it was not done.
- The 0.2 contract amendments land as one combined documentation pull request
  with decision entries D17–D21, not as separate contract pull requests.
- Run-manifest and timing checks cover required keys only.
- F2 (Whisper full fine-tuning) and the R1 ablations run only if GPU time remains
  after the headline systems. They stay in scope; if they do not run, the report
  says so, with measured hardware feasibility evidence for F2.

## Version history

- 0.1 (2026-11-17): initial policy.
- 0.2 (proposed by M6 on 2026-11-17, pending review by M1 and model owners
  M2–M5; [D21](decisions.md)): fixes the timing headline (end-to-end, batch 1,
  3 repeats, 30-minute subset), labels core-model RTF as secondary, records
  `clocks_event_reasons`, maps the course metric names and lists the declared
  limitations.
