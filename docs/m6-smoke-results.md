# Smoke results for Members 2–5 (2026-11-17, re-run 2026-11-18)

Member 6 ran the smoke kit on the M6 RTX 5070 Ti laptop on 2026-11-17. On
2026-11-18 a follow-up check looked at the duplicate candidates, the four beam
systems were re-run with the shared-export dedup (smoke kit at `dfc1bee`), and
a timing slowdown in that re-run was diagnosed. This page is evidence from
these runs, not a specification: it says what was measured and what each model
owner should do next. Track labels follow the [README](../README.md#scope): F1
is Whisper LoRA, F2 is Whisper full fine-tuning (both Member 4), F3 is XLS-R
CTC fine-tuning (Member 5).

Every result number carries one of three labels:

- **measured**: read from a smoke run's output.
- **estimate**: arithmetic on measured numbers; nothing was run to check it.
- **UNMEASURED**: not run at all.

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
- Re-runs on 2026-11-18: the same 20 timed clips and the same software, with
  the smoke kit at `dfc1bee`.
- No references were used: accuracy is UNMEASURED.
- These runs used Member 6's smoke wrappers, not the owners' adapters. Each
  owner's B1–B4 adapter needs its own check once it exists.

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
official RTF comes from the re-run protocol in
[evaluation](evaluation.md#real-time-factor-and-latency). Batch > 1 inference
and the 3-repeat timing protocol are UNMEASURED.

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

- Another GPU-heavy application running at the same time made beam decoding
  3–8x slower (measured). whisper-small went from 0.049 to about 0.29, with
  the GPU at 100% utilisation drawing only about 33 W (measured): it was busy
  with the other application's work. Closing that application removed the
  slowdown (measured).
- The smoke-kit change from `793ff9c` to `dfc1bee` did not change speed. On a
  free GPU, four interleaved whisper-small runs gave 0.0660 and 0.0665 with
  the old code and 0.0675 and 0.0659 with the new code (measured).
- Within one session, repeats agreed within 2.5% (measured). Across sessions,
  the same decoding work ran 1.2–1.4x slower on 2026-11-18 than on 2026-11-17
  (measured ratios). For whisper-small, the old `793ff9c` code itself gave
  0.066 against 0.049, and a 2026-11-18 run's output text matched the
  2026-11-17 run on all 20 clips (measured). The cause is unexplained. The
  2026-11-17 machine state was not recorded, and background programs were
  active on 2026-11-18.
- A run with a GPU monitor polling every 0.5 s, and part of the other
  application still on the GPU, was about 14% slower than the four runs
  without them (estimate, one run).
- 2026-11-18 RTFs: whisper-small 0.066–0.070 over five runs on a free GPU
  (measured); qwen3-asr-0.6b 0.109 (measured), with an OS update using CPU
  just before that run. whisper-large-v3 0.209 and qwen3-asr-1.7b 0.118
  (measured) ran while another program's processes held the GPU and total CPU
  load was 100%, so treat those two as possibly contended.

Rule for everyone: compare RTFs only within one timing session. The official
timing runs every system back to back in one session, records the machine
state (GPU processes, power source and plan, CPU load) and uses 3 repeats. No
other GPU application is open, cloud-drive sync is paused and no OS update is
installing.

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
- Not yet run (UNMEASURED): on a few clips, rank 1 from the core path should
  equal the 1-best of
  `WhisperForConditionalGeneration.generate(num_beams=10, num_return_sequences=1)`.

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

- Deduplication is now done by the shared export (on `main` after PR-B).
  Adapters return the decoder's raw list, duplicates included, and never
  pre-deduplicate.
- The export keeps the first (lowest-rank) occurrence of each exact raw text
  with its original rank and `candidate_id`, writes `dedup_map` only when it
  removed something, and sets `actual_n` to the kept count. `requested_n`
  stays the requested beam width.
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
  Their vocabulary has no digits, so numerals in references become systematic
  errors for B4, and for F3 if F3 keeps the head. Report it as a limitation, or
  run vocabulary extension as a separate ablation.
- CTC N-best does not exist yet and is UNMEASURED. Planned design: a small
  numpy prefix beam search proposes the candidates, and each candidate's exact
  score is `-CTCLoss(reduction='none')`, labeled per the
  [N-best contract](contracts/nbest.md#score-meanings). Until it exists, no CTC
  system can feed R1.
- B5 (BiLSTM-CTC from scratch): memory, step time, epoch time and whether it
  runs on CPU are all UNMEASURED. Before committing to a size, run a 3-step
  memory probe and time one epoch on the machine that will train it. Build the
  vocabulary from the train split only.
- F3 (XLS-R CTC fine-tuning): memory is UNMEASURED; the 1.62 GiB inference
  peak (measured) is only a floor. Before choosing a batch size, run a 3-step
  probe on the training machine with fp32 master weights, a frozen feature
  encoder and gradient checkpointing. Set `ctc_zero_infinity=True` or filter
  long-label samples.

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
correction: return the decoder's text unchanged. The evaluation report will
state the CER space policy and give a per-system count of hypotheses where
`text != text.strip()` (on `main` after PR-E). If every hypothesis were affected and clips had
150–200 code points, raw-v1 CER would rise by at most about 0.5 to 0.67 points
(estimate); the affected count is UNMEASURED.

## Actions per member

Everyone times only in the shared timing session (see
[Timing varies between sessions](#timing-varies-between-sessions)) and does
not quote an RTF measured in another session.

| Member | Tracks | Next actions |
| --- | --- | --- |
| Member 2 | B1, B2, B3 | Build Whisper N-best on the core `GenerationMixin` path and never call `WhisperForConditionalGeneration.generate` with `num_return_sequences > 1`; one process per system. Return the raw candidate list; the shared export deduplicates. Run the rank-1 check on a few clips. Qwen3-ASR: use the `-hf` checkpoints, label the score joint, keep `qwen-asr` out of the main environment. Return text exactly as decoded. Freeze beam size and requested N in each decoder config before the timed re-run. |
| Member 3 | B4, B5 | B4: build the XLS-R processor explicitly, fp32, greedy; report the missing digits as a limitation. CTC N-best (prefix beam search, exact `-CTCLoss` score) is UNMEASURED; until it exists B4 gives 1-best only. B5: run a 3-step memory probe and time one epoch on the training machine before choosing a size; train-only vocabulary. |
| Member 4 | F1, F2 | F1: confirm the parent (see open questions); run a 3-step whisper-small LoRA probe before committing (UNMEASURED today). A large-v3 LoRA fits (4.05 GiB reserved at batch 2, measured) but is claimed only with its own B2-matched experiment. Train with fp32 master weights and bf16 autocast, LoRA adapters in fp32, the frozen base in bf16 or fp16, dtype passed explicitly. F2: run a 3-step fp32-master whisper-small probe (about 4.2–5.2 GiB, estimate); record large-v3 full fine-tuning as a measured hardware block naming the probed recipe. Follow the Whisper N-best rule and return raw lists. |
| Member 5 | F3, R1 | F3: run the 3-step probe (fp32 master, frozen feature encoder, gradient checkpointing) before choosing a batch size; set `ctc_zero_infinity=True` or filter long labels; report the missing digits if the head is kept. R1: PyTorch Geometric works (full-batch, reverse edge types, no `pyg-lib` samplers). No validated N-best producer exists yet: the planned producers are Member 4's Whisper N-best (whisper-small parent) and Member 3's CTC export. Pick the producer early and freeze its validation and test candidate sets. |

## Open questions

1. **Who has a GPU, and how much VRAM?** Members 2–5: please say which
   GPU, how much VRAM and which OS. If Member 4 or Member 5 has none, the M6
   RTX 5070 Ti might host a training queue; that reverses the rule that it runs
   only Member 6's jobs, so it is Member 6's call. GPU steps only, from 3-step
   probes and a 20–30 h train set: about 21–32 minutes for 3 epochs of
   whisper-small full fine-tuning, and about 1.7–2.6 h for 3 epochs of
   whisper-large-v3 LoRA, the latter only if a large-v3 F1 is pursued with its
   B2-matched experiment (estimate). whisper-small LoRA, XLS-R F3 and CPU
   training times are UNMEASURED.
2. **F1 parent.** [TEAM_TASKS](../TEAM_TASKS.md#member-4--whisper-fine-tuning)
   makes whisper-small (B1) the adaptation parent. A whisper-large-v3 F1 needs
   its own B2-matched experiment. Members 2 and 4: confirm whisper-small.
3. **Beam sizes.** The smoke run used beam 10 for Whisper, beam 5 for
   Qwen3-ASR and greedy decoding for XLS-R. Each owner freezes beam size and
   requested N in the decoder config before timing, so the re-run times the
   configuration that is reported.
