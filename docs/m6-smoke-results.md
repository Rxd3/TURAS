# Smoke results for Members 2–5 (2026-11-17, re-run 2026-11-18)

Member 6 ran the smoke kit on the M6 RTX 5070 Ti laptop on 2026-11-17. On
2026-11-18 a follow-up check looked at the duplicate candidates, the four beam
systems were re-run with the shared-export dedup (smoke kit at `dfc1bee`), and
a timing slowdown in that re-run, caused by another GPU-heavy application
running at the same time, was diagnosed. After that application was closed, a
second four-model re-run at `dfc1bee` (called the second re-run below)
repeated all four beam systems. The measurements here describe those smoke runs.
Track labels follow the [submission guide](../SUBMISSION.md#scope-and-remaining-work). F1 is Whisper LoRA, F2 is
Whisper full fine-tuning (both Member 4), and F3 is XLS-R CTC fine-tuning
(Member 5).

The [M4/M5 final protocol](../configs/evaluation/m45_final_protocol.json) was
frozen on 2026-11-19 before test references were opened and is now on `main`
through [PR #5](https://github.com/Rxd3/TURAS/pull/5). The
[decoder erratum](m45_decoding_erratum.md) gives the effective widths
(Whisper 5 beams and CTC prefix beam 10). On 2026-11-20, Member 6 accepted
the [frozen timing results](../results/final/M4_M5_FINAL_REPORT.md#repeated-timing)
as the official RTF for those eight conditions.

Every result number carries one of three labels:

- **measured**: read from a smoke run's output.
- **estimate**: arithmetic on measured numbers; nothing was run to check it.
- **UNMEASURED**: not run by the M6 smoke kit; "not run" in a table means the
  same.

Settings (batch size, beam width, learning rate, model revisions, software
versions, model limits) describe what was run and carry no label.

## What ran

- Hardware: the M6 RTX 5070 Ti laptop. Its GPU reports 11.94 GiB of VRAM
  (measured); about 10.77 GiB was free before the training-probe load
  (measured, one reading).
- Software: torch 2.14.1+cu130, CUDA 13.0, transformers 5.23.0, PyTorch
  Geometric 2.8.1. Smoke kit: branch `feat/m6-smoke-tests` at `793ff9c`.
- Inference: 3 warm-up clips, then 20 timed clips with 287.6 s of audio in
  total (measured). Batch 1, one repeat, `max_new_tokens` 440, language `tr`.
  Thermal flags: 0 (measured). The first two timed clips come from one
  recording (`tr_28_3_1`); the run output does not say how many recordings
  the 20 clips span, so treat them as a narrow sample.
- Re-runs on 2026-11-18: the same 20 timed clips (287.58 s of audio,
  measured), the smoke kit at `dfc1bee`, and `793ff9c` for the old-code
  comparison runs. The re-run reports do not restate library versions.
- These runs used no references, so smoke-run accuracy is UNMEASURED.
- These runs used Member 6's smoke wrappers. See the [project scope](../SUBMISSION.md#scope-and-remaining-work)
  for the current status of the owners' adapters and benchmark runs.

Pinned models (Hugging Face commit SHAs):

| Model | Revision | Precision |
| --- | --- | --- |
| `openai/whisper-small` | `973afd24965f72e36ca33b3055d56a652f456b4d` | float16 |
| `openai/whisper-large-v3` | `06f233fe06e710322aca913c1bc4249a0d71fce1` | float16 |
| `Qwen/Qwen3-ASR-1.7B-hf` | `bcd2b5b7f32b480ab5790554cfa8347f246a14f3` | bfloat16 |
| `Qwen/Qwen3-ASR-0.6B-hf` | `7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c` | bfloat16 |
| `mpoyraz/wav2vec2-xls-r-300m-cv7-turkish` | `708639f50559d7970f462e13ec64d3f059ca89f6` | float32 |
| `mpoyraz/wav2vec2-xls-r-300m-cv8-turkish` | `bdd0bb878d8bf59509611631e334eb073ba57773` | float32 |

## Inference

The table below is the 2026-11-17 run with the smoke kit at `793ff9c`.

| System | Informs | Precision | Decoding | Peak VRAM GiB | Peak reserved GiB | End-to-end RTF | Core RTF | N-best as exported | Label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| whisper-small | B1 | float16 | beam 10, 10 returned | 1.37 | 1.94 | 0.049 | 0.049 | invalid, duplicate text only (24 errors); 8/20 lists all distinct | measured |
| whisper-large-v3 | B2 | float16 | beam 10, 10 returned | 5.67 | 6.50 | 0.167 | 0.166 | invalid, duplicate text only (5 errors); 16/20 lists all distinct | measured |
| qwen3-asr-1.7b | B3 | bfloat16 | beam 5, 5 returned | 4.56 | 5.63 | 0.098 | 0.097 | invalid, duplicate text only (16 errors); 8/20 lists all distinct | measured |
| qwen3-asr-0.6b | B3 | bfloat16 | beam 5, 5 returned | 2.22 | 3.30 | 0.090 | 0.090 | invalid, duplicate text only (15 errors); 9/20 lists all distinct | measured |
| xlsr-tr-cv7 | B4 | float32 | greedy, 1 returned | 1.62 | 2.77 | 0.006 | 0.006 | valid (1-best only) | measured |
| xlsr-tr-cv8 | B4 | float32 | greedy, 1 returned | 1.62 | 2.77 | 0.006 | 0.005 | valid (1-best only) | measured |

Every system returned 20 `ok` results, 0 `empty` and 0 `error`, and every
acoustic log score was finite (measured).

After the shared export deduplicated them, all four beam systems' N-best lists
validated with 0 errors (measured, 2026-11-18):

| System | `actual_n` mean | `actual_n` min | Rows with `dedup_map` (of 20) | Validation |
| --- | --- | --- | --- | --- |
| whisper-small (beam 10) | 8.55 | 5 | 12 | valid, 0 errors |
| whisper-large-v3 (beam 10) | 9.75 | 8 | 4 | valid, 0 errors |
| qwen3-asr-1.7b (beam 5) | 3.85 | 2 | 12 | valid, 0 errors |
| qwen3-asr-0.6b (beam 5) | 4.05 | 2 | 11 | valid, 0 errors |

Both 2026-11-18 re-runs gave the same statistics, and peak VRAM matched
2026-11-17 for all four systems (measured).

**Do not rank models on these RTFs.** They mix beam 10 (Whisper), beam 5
(Qwen3-ASR) and greedy decoding (XLS-R), cover 20 clips and come from one
repeat. RTFs also shift between sessions: see
[Timing varies between sessions](#timing-varies-between-sessions). The
official RTF for B1, B4, F1–F3 and the `--rdl-full` joint systems of F1–F3
comes from the [frozen timing results](../results/final/M4_M5_FINAL_REPORT.md#repeated-timing).
The [evaluation timing protocol](evaluation.md#real-time-factor-and-latency)
applies to future official runs for the remaining systems. Batch > 1 inference
and three-repeat timing are UNMEASURED by the smoke kit.

All seq2seq baselines fit at batch 1 in these precisions. The heaviest,
whisper-large-v3 at beam 10, used 6.50 GiB reserved (measured) of about
10.77 GiB free (measured).

Memory budget on this laptop: plan for about 10.77 GiB, not 11.94 GiB.
Exceeding VRAM on this Windows laptop does not reliably give an out-of-memory
error. The whisper-large-v3 full fine-tuning probe reached 15.65 GiB reserved
(measured) with no OOM, through Windows system-memory fallback, but a larger
overflow (the Whisper trap call below) did raise a CUDA OOM that left a sticky
error. After every probe, compare `torch.cuda.max_memory_reserved()` against
about 10.77 GiB. How much the fallback slows a run, and the effect of the
driver's fallback policy, are UNMEASURED. A teammate on Windows can believe a
configuration fits when it does not.

### Timing varies between sessions

The 2026-11-18 re-runs showed that the same code on the same laptop can time
very differently.

- With another GPU-heavy application running at the same time, beam decoding
  was 3–8x slower (estimate, from measured RTFs such as whisper-small 0.049
  against 0.291). In the whisper-small runs, the GPU sat at 100% utilisation
  while drawing only about 33 W (measured): it was busy with the other
  application's work. Closing that application brought whisper-small from
  about 0.29 to 0.0755 (measured), removing about 4x of the slowdown
  (estimate); the bullets below cover the remaining gap to 0.049. Only
  whisper-small was A/B tested.
- The smoke-kit change from `793ff9c` to `dfc1bee` did not change
  whisper-small speed: with no other GPU workload running and no monitor,
  four interleaved runs gave 0.0660 and 0.0665 with the old code and 0.0675
  and 0.0659 with the new code (measured), new over old about 1.007
  (estimate). The other systems were not A/B tested.
- The four interleaved whisper-small runs agreed within 2.5% (estimate, from
  measured RTFs); a fifth whisper-small run, in the second four-model re-run,
  gave 0.0697 (measured). Across sessions, the same decoding work ran 1.2–1.4x
  slower on 2026-11-18 than on 2026-11-17 (estimate, from measured RTFs). For
  whisper-small, the old `793ff9c` code itself gave 0.066 against 0.049, and
  a 2026-11-18 run's output text matched the 2026-11-17 run on all 20 clips
  (measured). The cause is unexplained. The 2026-11-17 machine state was not
  recorded, and background programs were active on 2026-11-18.
- A run with a GPU monitor polling every 0.5 s, made after the other
  application was closed, was about 14% slower than the four interleaved runs
  without the monitor (estimate, one run). A small part of the other
  application was still listed on the GPU in that run and in the four runs
  without the monitor.
- 2026-11-18 RTFs: whisper-small 0.066–0.070 over the five runs with no other
  GPU workload running and no monitor (the four interleaved runs and the
  second re-run's one; measured). From the second four-model re-run:
  qwen3-asr-0.6b 0.109 (measured), with an OS update using CPU just before
  that run.
  whisper-large-v3 0.209 (measured) ran while about 25 python.exe processes on
  the run's own interpreter were listed on the GPU and total CPU load was 100%
  (both measured, in the snapshot before the run). Those processes were very
  likely the run's own process tree; no outside job is known. qwen3-asr-1.7b
  0.118 (measured) ran with the same processes present. Treat both as
  same-session measurements: their ratios to 2026-11-17 (1.25 and 1.20;
  estimate, from measured RTFs) match the other systems.

Compare RTFs only within one timing session. The eight frozen M4/M5 conditions
(B1, B4, F1–F3 and the `--rdl-full` joint systems of F1–F3) were timed serially
under the [frozen protocol](../configs/evaluation/m45_final_protocol.json) and
[decision D40](owner_decisions.md). Report the spread across repeats and use it
when interpreting CTC timing. [M4/M5 completion record](m45_completion.md)

For official comparisons involving B2, B3 or B5, time all compared systems
back to back in the same session and use 3 repeats. Record GPU processes,
power source and plan, CPU load and top CPU processes. Close other GPU
applications, pause cloud-drive sync and wait for active OS updates to finish.
This includes comparisons with a system from the frozen M4/M5 block, such as
B2 against B1.

## Training probes

Three AdamW steps per probe at lr 1e-5, in pure bf16 (weights, gradients and
AdamW states all bf16), with non-reentrant gradient checkpointing. All losses
were finite (measured).

| Probe | Informs | Batch | Trainable parameters | Peak alloc GiB | Peak reserved GiB | Fits | s/step | Label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| whisper-small full | F2 on whisper-small | 8 | 241,734,912 of 241,734,912 | 2.39 | 2.64 | yes | 0.72 | measured |
| whisper-large-v3 LoRA | F1 on large-v3 | 2 | 15,728,640 of 1,559,219,200 | 3.59 | 4.05 | yes | 0.87 | measured |
| whisper-large-v3 full | F2 on large-v3 | 1 | 1,543,490,560 | 14.55 | 15.65 | no | 3.26, not representative (system-memory fallback) | measured |

LoRA settings: rank 32, alpha 64, dropout 0, targets `q_proj` and `v_proj`,
bias none.

**Pure-bf16 caveat.** In pure bf16 at lr 1e-5, an Adam update of about 1e-5 is
lost under round-to-nearest for any weight with |w| above about 2.6e-3 to
5.1e-3 (estimate). What fraction of Whisper's weights that covers is
UNMEASURED, and a 3-step probe cannot show whether the model learns. Prefer
fp32 master weights with bf16 autocast, keep LoRA adapters in fp32, and pass
the dtype explicitly. The memory figures above also understate that recipe.

fp32-master estimates. fp32 master weights with bf16 autocast double static
memory from 8 to 16 bytes per parameter (estimate):

| Recipe | Memory | Label |
| --- | --- | --- |
| whisper-small full fine-tuning, batch 8 | about 4.2–5.2 GiB; fits in about 10.77 GiB | estimate |
| whisper-large-v3 full fine-tuning, batch 1 | about 26–29 GiB, about 2.4–2.7x what is free | estimate |
| whisper-large-v3 LoRA: fp32 master copies of the adapter weights | adds about 0.12 GiB | estimate |
| whisper-large-v3 LoRA: frozen base loaded in fp32 | adds about 2.9 GiB; fits but wastes memory, so keep the base in bf16 or fp16 | estimate |
| any fp32-master recipe on real hardware | not run | UNMEASURED |

The upper bounds assume the probe's residual memory (about 2.1–2.6 bytes per
parameter, estimate) scales with parameter precision. Measure it with a 3-step
fp32-master probe.

whisper-large-v3 full fine-tuning does not fit in the probed recipe (batch 1,
bf16 weights, gradients and AdamW states, gradient checkpointing): 14.55 GiB
allocated and 15.65 GiB reserved on an 11.94 GiB card with about 10.77 GiB
free (measured). Record that as the measured hardware block TEAM_TASKS allows,
naming the recipe. Leaner recipes (Adafactor, 8-bit optimizers, a frozen
encoder, CPU offload) are UNMEASURED.

Epoch time on a 20–30 h train set, GPU steps only: whisper-small full
fine-tuning about 7–11 minutes per epoch at batch 8; whisper-large-v3 LoRA
about 35–52 minutes per epoch at batch 2 (estimate). These exclude audio
decoding, feature extraction, data loading, evaluation and checkpoint saving.
The 3-step mean includes warm-up, fp32-master may run slower, and other batch
sizes were not run.

Not probed (UNMEASURED): whisper-small LoRA, XLS-R (F3) and Qwen3-ASR
training memory, B5, and every fp32-master recipe.

## Whisper N-best rule

- Build Whisper N-best with the core transformers `GenerationMixin.generate`
  beam path, as the smoke run did. On that path, `num_beams` 10 with
  `num_return_sequences` 10 returned real beam candidates.
- Never call `WhisperForConditionalGeneration.generate` with
  `num_return_sequences > 1`. On whisper-small it returned copies of the
  1-best: 180 duplicate texts across 20 of 20 utterances (measured).
- Run each system in its own process, and never make that call in a process
  that later does timed runs. On whisper-large-v3 it hit a CUDA OOM (100
  sequences) that left a sticky CUDA error (measured); it ran after the main
  run, so the main beam-10 run was not affected. The same OOM happened again
  on 2026-11-18, again after the main run, which it did not affect (measured).
- The core path skips Whisper's own wrapper, so the adapter must supply and
  record the forced prompt (SOT, `<|tr|>`, transcribe, notimestamps), strip it
  from `token_ids` and `token_count`, and keep `max_new_tokens` plus the prompt
  at or under 448. Get `acoustic_log_score` from a separate teacher-forced
  pass. Record `decoder_sequence_score` as length-normalized.
- The smoke kit did not compare rank 1 from the core path with the model
  wrapper's one-best. Use the [rank-1 checker](../scripts/smoke_check_whisper_rank1.py)
  to compare them on a few clips with
  `WhisperForConditionalGeneration.generate(input_features, language="tr", task="transcribe", return_timestamps=False, num_beams=5, num_return_sequences=1, length_penalty=1.0, do_sample=False, max_new_tokens=440)`,
  run on the same input features at the core path's precision (bf16 for B1)
  and beam width (the frozen 5 for B1; this page asks B2 to use 5 too).

## Duplicate candidates

On 2026-11-17, each of the four beam systems' N-best artifacts failed
validation, only for duplicate raw text. Lists with at least one duplicate:
whisper-small 12/20, whisper-large-v3 4/20, Qwen3-ASR-1.7B 12/20,
Qwen3-ASR-0.6B 11/20 (measured).
Duplicates are not always adjacent: in whisper-small `tr_28_3_1_seg00204`,
c01 = c04 and c02 = c05.

The 2026-11-18 follow-up found 76 duplicate-text pairs across the four systems
(whisper-small 29, whisper-large-v3 5, Qwen3-ASR-1.7B 23, Qwen3-ASR-0.6B 19),
and 0 of them had identical `token_ids` (measured). Every duplicate is a
different token path that decodes to the same text.

What this changes (N-best contract 0.2, proposed by M6;
[who deduplicates](contracts/nbest.md#who-deduplicates)):

- The [shared export](../src/turkish_asr_rdl/decoding/export.py) deduplicates
  the decoder's raw list. Adapters return that list with duplicates included.
- The export keeps the first (lowest-rank) occurrence of each exact raw text
  with its original rank and `candidate_id`, writes `dedup_map` only when it
  removed something, and sets `actual_n` to the kept count. `requested_n`
  stays the requested N (for the beam systems above it also equals the beam
  width).
- Candidates with the same text and identical `token_ids` are copies of one
  hypothesis (the Whisper trap above), and the export turns that result into an
  error row.

Mean distinct candidates per utterance after deduplication, on these 20 clips
(measured in the 2026-11-18 re-run, the same values as estimated from the
2026-11-17 run): whisper-large-v3 9.75 of 10, whisper-small 8.55 of 10,
Qwen3-ASR-1.7B 3.85 of 5, Qwen3-ASR-0.6B 4.05 of 5. The beam widths differ.
Usefulness for R1 (oracle WER) is UNMEASURED.

Qwen3-ASR: the pinned environment generates finite-scored beam-5 candidate
lists from the `-hf` checkpoints through `AutoModelForMultimodalLM` and
`AutoProcessor.apply_transcription_request` (measured). Label the score as
joint ([N-best contract](contracts/nbest.md#score-meanings)). Never install the
`qwen-asr` pip package into the main environment; it pins transformers 4.57.6.

## CTC, B4 and B5

- B4: the mpoyraz XLS-R cv7 and cv8 checkpoints in fp32 with greedy decoding
  give valid 1-best output (measured), provided the processor is built
  explicitly (`Wav2Vec2CTCTokenizer` + `Wav2Vec2FeatureExtractor`, no LM).
  For the paired comparison with F3, the frozen protocol's B4
  (`B4-xlsr-tr-cv7`) is cv7 at `708639f5`, decoded like F3: bf16 on CUDA,
  prefix beam 10, requested N 5, rank 1 as the one-best. Report cv8 only as a
  labelled secondary system, and fp32 greedy only as a labelled secondary
  setting. Neither checkpoint's vocabulary has digits (cv7's has 38 tokens), so
  numerals in references become systematic errors for B4; report it as a
  limitation. F3 extends the
  head to 52 tokens. The 14 added characters include the digits.
  [Decision D33](owner_decisions.md)
- The frozen M4/M5 benchmark used [CTC N-best](../src/turkish_asr_rdl/decoding/ctc.py).
  It is UNMEASURED by the smoke kit. In that decoder,
  `prefix_beam_search` (beam 10) ranks its prefixes by approximate beam mass.
  `ctc_nbest` takes the first `requested_n` of them in that order (the top 5
  of 10) and keeps that order as the rank. Only those get an exact score from
  `sequence_log_probability`, a forward dynamic program over all CTC
  alignments, written as `acoustic_log_score` with `acoustic_score_kind`
  `ctc_sequence_log_probability`; the rest are discarded unscored, and the
  list is not re-sorted by exact score. Rank 1, the one-best, is the highest
  approximate beam mass, not necessarily the highest exact score. The
  approximate beam mass goes in `decoder_sequence_score`, and `score_details`
  sets `sequence_score_exact` and `beam_score_approximate` to true. A
  candidate whose exact score is not finite is dropped, not given an
  artificial score, and is not replaced. That branch does not fire on finite
  fp32 frames: every alignment counted in a prefix's beam mass is also an
  alignment of that prefix, so the exact score is at least the beam mass,
  which is finite. In practice the exported CTC list falls below N when
  distinct prefixes decode to the same text (for example a leading or trailing
  word delimiter `|`, or a skipped special token), because the shared export
  removes those duplicates. The
  CTC path in `decoding/adapted.py` casts the logits to fp32, applies
  `log_softmax` and uses the pad token as blank. Member 3 owns
  `decoding/ctc*` in TEAM_TASKS: review
  `decoding/ctc.py` and build on it rather than writing a second scorer.
  `decoding/ctc.py`, like `decoding/adapted.py` and `decoding/whisper.py`, is
  hash-pinned in the frozen protocol (`critical_code_sha256`), and the frozen
  run and audit scripts stop if it changes: call it unchanged for B4 and put
  any extension in a new module.
- To cross-check CTC scores with `torch.nn.CTCLoss`, use `reduction='none'`
  and `zero_infinity=False` on the same frames as the frozen path: the bf16
  model's logits cast to fp32 before `log_softmax` (`decoding/adapted.py`),
  with blank = `pad_token_id` (0 for both mpoyraz checkpoints, and for F3,
  whose D33 extension keeps the blank ID). Each exact score should then equal
  minus that loss within float tolerance (for example rel 1e-4), not bit for
  bit: `ctc.py` sums in Python floats (float64), while `CTCLoss` on fp32
  tensors works in fp32. Do not re-run the model in fp32 for this check.
  `model(labels=...).loss` is no
  check: both mpoyraz checkpoints set `ctc_loss_reduction` to `"mean"` in
  `config.json`, so that loss is length-normalised, not a log probability.
- B5 (BiLSTM-CTC from scratch): memory, step time, epoch time and whether it
  runs on CPU are all UNMEASURED. Before committing to a size, run a 3-step
  memory probe and time one epoch on the machine that will train it. Build the
  vocabulary from the train split only.
- F3 (XLS-R CTC fine-tuning): training memory is UNMEASURED by M6. The
  1.62 GiB inference peak (measured) is only a floor. The M4/M5
  [training-memory probes](m45_gpu_profile.md) are a separate set of runs.
  The [F3 training record](m45_execution_2026-11-19.md) documents the cv7
  parent, the head extended to 52 tokens and a train-target length-feasibility
  check. The [training code](../src/turkish_asr_rdl/training/ctc.py) uses a
  frozen feature encoder, non-reentrant gradient checkpointing and
  `ctc_zero_infinity=False`. The probe
  advice applies only to a new F3 recipe: run a 3-step probe on the training
  machine with fp32 master weights before choosing a batch size, and keep
  `ctc_zero_infinity=False` only with a train-target length-feasibility check
  like F3's; otherwise set it to `True` or filter long labels.

## PyTorch Geometric

PyTorch Geometric 2.8.1 works on the pinned torch 2.14.1+cu130: two `HeteroConv`
`SAGEConv` layers, forward and backward on CUDA, passed in fp32 and in bf16
(measured), with a peak of 0.06 GiB on a toy graph (measured). Train
full-batch, add reverse edge types, and do not depend on samplers that need
`pyg-lib`. Memory and time on a real-sized graph are UNMEASURED.

## raw-v1 edge spaces

Whisper hypotheses start with a space; Qwen3-ASR hypotheses do not (measured,
from the smoke run's sample outputs). Under `raw-v1`, which compares code
points as emitted, each such leading space counts as one CER insertion against
a reference without one; that is inferred, not measured. Whether xlsr-cv8
output really ends in a space is unclear, because only an 80-character excerpt
was kept. Edge spaces do not affect `raw-v1` WER or `tr-normalized-v1` WER and
CER.

`raw-v1` keeps model output byte for byte, because the
[normalization contract](contracts/normalization.md#raw-v1) forbids silent
correction. Return the decoder's text unchanged. A per-system count of
hypotheses where `text != text.strip()` would quantify the effect's coverage.
If every hypothesis were affected and clips had
150–200 code points, raw-v1 CER would rise by at most about 0.5 to 0.67 points
(estimate); the affected count is UNMEASURED.

## Actions per member

B1, B4, F1–F3 and the `--rdl-full` joint systems of F1–F3 take their official
RTF from the [frozen M4/M5 timing block](../results/final/M4_M5_FINAL_REPORT.md#repeated-timing).
For a comparison involving any other system, re-time all compared systems
together. [Timing varies between sessions](#timing-varies-between-sessions)

| Member | Tracks | Next actions |
| --- | --- | --- |
| Member 2 | B1, B2, B3 | Build Whisper N-best on the core `GenerationMixin` path and never call `WhisperForConditionalGeneration.generate` with `num_return_sequences > 1`; one process per system. Return the raw candidate list; the shared export deduplicates. Run the rank-1 check on a few clips. Qwen3-ASR: use the `-hf` checkpoints, label the score joint, keep `qwen-asr` out of the main environment. Return text exactly as decoded. Decode B1 like the frozen protocol's `B1-whisper-small` (bf16 on CUDA, `num_beams` = requested N = 5, `num_return_sequences` 5, `max_new_tokens` 440), and B2 the same way; the smoke run's fp16 beam 10 is not the frozen setting. Beam width alone is not enough (`docs/m45_decoding_erratum.md`): also match B1's checkpoint, the pinned `openai/whisper-small` at `973afd24` (weight SHA-256 `1d7734884874…` in the frozen protocol; decision D23), and the frozen `decoding/whisper.py` scoring and decoding policy (and the score convention `whisper-original-logits-generated-plus-eos-v1` that `decoding/adapted.py` assigns): the Turkish forced prompt of the [Whisper N-best rule](#whisper-n-best-rule) with `forced_decoder_ids` cleared, EOS kept in `token_ids` and `token_count`, a teacher-forced score from the original logits (no suppression or temperature in the score; the search itself uses a copy of the checkpoint's `generation_config` with only `forced_decoder_ids` cleared) over the generated tokens plus EOS with the prompt excluded, `length_penalty` 1.0, no sampling. B3's beam is your validation choice. Freeze beam size and requested N in each decoder config before the timed re-run. |
| Member 3 | B4, B5 | B4 uses cv7 at `708639f5` with the frozen B4/F3 decoding settings (bf16 on CUDA, prefix beam 10, requested N 5, rank 1 as one-best). Label cv8 and fp32 greedy as secondary and report the cv7 head's missing digits. Reuse the hash-pinned [CTC decoder](../src/turkish_asr_rdl/decoding/ctc.py) and keep extensions in a separate module. See [CTC, B4 and B5](#ctc-b4-and-b5) for the exact `CTCLoss` cross-check. B5 still needs a 3-step memory probe, a timed epoch on the training machine and train-only vocabulary. |
| Member 4 | F1, F2 | The frozen working benchmark trained both from Whisper-small and compared them with `B1-whisper-small`. See the separate [M4/M5 training-memory probes](m45_gpu_profile.md). M6 did not measure Whisper-small LoRA memory. M6 estimated 4.2–5.2 GiB at batch 8 for full fine-tuning with fp32 master weights. Large-v3 LoRA fit at batch 2 (4.05 GiB reserved, measured), and reporting its quality would require a B2-matched experiment. Record the measured hardware block for the probed large-v3 full fine-tuning recipe and follow the [Whisper N-best rule](#whisper-n-best-rule). |
| Member 5 | F3, R1 | See the [F3 training record](m45_execution_2026-11-19.md) for the 52-token head, train-target length-feasibility check and `ctc_zero_infinity=False` recipe. Run the 3-step memory probe for a new F3 recipe and apply the length safeguards in [CTC, B4 and B5](#ctc-b4-and-b5). The frozen [Whisper](../src/turkish_asr_rdl/decoding/whisper.py) and [CTC](../src/turkish_asr_rdl/decoding/ctc.py) R1 lists passed shared N-best validation in the M4/M5 run. The smoke wrappers' separate lists validated on 20 clips after export deduplication (measured 2026-11-18). The [frozen protocol](../configs/evaluation/m45_final_protocol.json) pins decoder settings, test inputs and validation selection reports. |

## Open questions

1. **Who has a GPU, and how much VRAM?** Members 2 and 3 (B1–B5): please say
   which GPU, how much VRAM and which OS. The M4/M5 runs used an RTX 5070
   and completed 3 epochs for F1–F3. GPU steps only, from 3-step probes and a
   20–30 h train set: about 21–32 minutes for 3 epochs of
   whisper-small full fine-tuning, and about 1.7–2.6 h for 3 epochs of
   whisper-large-v3 LoRA, the latter only if a large-v3 F1 is pursued with its
   B2-matched experiment (estimate). These planning figures now apply only to
   new recipes. whisper-small LoRA, XLS-R F3 and CPU training times are
   UNMEASURED by M6. [M4/M5 GPU profile](m45_gpu_profile.md)
2. **F1 parent (settled by owner decision D23).**
   [TEAM_TASKS](../TEAM_TASKS.md#member-4--whisper-fine-tuning)
   makes whisper-small (B1) the adaptation parent, and the frozen protocol
   follows it: F1 and F2 were trained from whisper-small, with
   `B1-whisper-small` as their control. A whisper-large-v3 F1 needs its own
   B2-matched experiment. [Decision D23](owner_decisions.md)
3. **Beam sizes (settled for the frozen protocol).** The smoke run used beam
   10 for Whisper, beam 5 for Qwen3-ASR and greedy decoding for XLS-R. The
   frozen protocol's decoder block is bf16 on CUDA, requested N 5, beam size
   10 and `max_new_tokens` 440. Whisper (B1, F1, F2) uses `num_beams` 5 with 5
   returned; the beam size 10 applies only to CTC (B4, F3), as prefix beam 10
   with rank 1 as the one-best. B2 and B3 are outside the frozen protocol.
   The smoke follow-up proposes B1-matched decoding for B2 and
   validation-based beam selection for B3. [Decoder erratum](m45_decoding_erratum.md)
   Each owner freezes beam size and requested N in the decoder config before
   timing, so the re-run times the configuration that is reported.
