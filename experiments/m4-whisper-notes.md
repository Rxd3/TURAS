# Member 4: Whisper adaptation implementation

Updated 2026-11-19. Code and offline synthetic tests are being delivered; no
pretrained weight was downloaded and no corpus training/evaluation has run.

## Parent and planned comparison

Proposed B1 parent: `openai/whisper-small`, revision
`973afd24965f72e36ca33b3055d56a652f456b4d`. Its published `model.safetensors`
SHA-256 is `1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b`.
Metadata was inspected through the official Hub API; the weight bytes have not
been downloaded/independently verified. M2/M6 must accept this exact parent and
baseline evidence before training. No WorldSpeech-trained checkpoint is allowed
for this release because M1 reports upstream-training overlap with test audio.

Separate draft profiles:

- `configs/fine-tuning/whisper_small_lora.json`: rank 16, alpha 32, dropout .05,
  attention q/v projections in encoder and decoder; frozen base, no trainable bias.
- `configs/fine-tuning/whisper_small_full.json`: all parameters trainable.

Both propose batch one, accumulation 16, three epochs, bf16 CUDA, seed 17.
LoRA LR is 1e-4; full LR is 1e-5. These are proposed budgets, not selected settings.
Seeds 42 and 2026 remain later repetitions, subject to actual resources. Checkpoints
are saved every epoch; M6 chooses using validation normalized WER, raw WER and then
earlier epoch for ties. No test references enter these commands. A wider search
must be preregistered before test access.

## Current data and review gates

M1 reports release `turas-tr-econ-legis-v0.1` with split manifest SHA-256
`a8fff65b99fcf11d9452685e438e250207032ecfe11152741562147126d095b8` in
[PR 2](https://github.com/Rxd3/TURAS/pull/2). The user confirmed that the inference
and train-reference bundles are not ready locally. They are required inputs;
the repaired test-only archive is never a training source.

Required inputs are a train-only inference JSONL, train-only references JSONL,
approved audio root, split manifest and train ID file, file hashes, and a real
compatible baseline report. The report must identify `status=completed`,
`training_mode=pretrained`, `model_id`, `revision`, `dataset_release` and
`split_manifest_sha256`. This is a minimal execution gate, not proof that every
course baseline has been completed. M2/M3/M6 retain the broader baseline audit.

The release uses sampled listening verification and recording-level grouping;
M1 reports unknown speaker identities and reference noise. Its approval is reported
by M1, not independently verified by this workstream. M6 must review that departure
from the original full-reference-verification specification and professor's speaker
rule before final claims. This code does not silently upgrade verification status.

## Commands implemented

Install the package in the intended environment. The current local `.venv` is a
CPU development environment used for tiny random-model tests; it is not a verified
RTX 5070 training environment. Run from the checkout:

```powershell
python -m turkish_asr_rdl.training.whisper --config configs/fine-tuning/whisper_small_lora.json --preflight
python -m turkish_asr_rdl.training.whisper --config configs/fine-tuning/whisper_small_full.json --preflight
```

Both print explicit blockers while required inputs/reviews are absent. Once M2/M6
review the protocol, create an ignored resolved config with real paths/hashes and
run:

```powershell
python -m turkish_asr_rdl.training.whisper --config artifacts/m4/resolved-lora.json --output artifacts/m4/lora-seed17
```

That resolved path is an instruction for future creation, not an existing run.
The thin `scripts/m4_train_whisper.py` wrapper invokes the same command.

Train rows must match frozen IDs exactly; duplicates, mismatched release/audio
hashes, gold fields in inference, non-train references, path escapes, overlong
audio/targets and non-finite loss fail explicitly. No rows are silently removed.
M1's references may be locally unreviewed under the sampled-verification release;
their exact file hash is recorded, and rejected/in-review rows are refused.
Training uses raw targets and the pinned Turkish transcribe prefix. Padding is
masked with -100; targets and audio are never silently truncated.

## N-best handoff to M5

`decoding/whisper.py` performs genuine beam search through core GenerationMixin
with the explicit Turkish/task/no-timestamp prefix. It then teacher-forces each
candidate against the original model logits. Natural-log scores include generated
tokens and EOS when present, exclude the forced prefix and padding, and preserve
framework beam scores separately. Generation suppression/length processing is not
mistaken for original acoustic log probability.

Candidate IDs retain original ranks; duplicate raw texts map to the retained
candidate; short lists are never padded. Truncated candidates expose termination
status. M5 must use token_count with the same EOS convention. The low-level export
function is implemented; integration into M6's JSONL/timing harness is a subsequent
owner-reviewed handoff, not a delivered CLI or completed baseline evaluation.

## Hardware feasibility still required

Read-only inspection found RTX 5070, 12,227 MiB VRAM, approximately 16 GB RAM.
This is inventory, not an observed training memory/time result. Before a corpus
run, validate a Blackwell-compatible CUDA PyTorch build, bf16 execution, decoder/
PEFT compatibility and three train-only warm-up optimizer steps, followed by a
representative train-only profiling pass. Record peak allocated/reserved VRAM,
system RAM, seconds/step, actual target lengths and estimated full budget.
Reduce microbatch before changing data. Keep the same effective batch if feasible.
Full tuning stays pending until measured; LoRA is not evidence that full tuning fits.

## Required review

Local validation: nine offline tests passed on Python 3.13.14, PyTorch 2.15.0+cpu,
Transformers 5.23.0 and PEFT 0.21.3. They use locally constructed random models and
synthetic fixtures, including genuine beam search, adapter reload, gradients and
partial accumulation. Transformers emits checkpointing API deprecation warnings;
these tests do not establish CUDA memory feasibility or corpus ASR performance.

M2: parent and matching B1 protocol. M6: train-only access, release/verification,
shared `training/acoustic.py`, provenance, budget and evaluator integration. M5:
score fields, deduplication and N-best consumption. No approval is invented.

Sources: [official model card](https://huggingface.co/openai/whisper-small),
[Whisper API](https://huggingface.co/docs/transformers/v5.23.0/model_doc/whisper),
[generation API](https://huggingface.co/docs/transformers/v5.23.0/main_classes/text_generation),
[PEFT LoRA](https://huggingface.co/docs/peft/v0.21.0/en/package_reference/lora).
