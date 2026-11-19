# Member 2: Seq2Seq and Speech-LLM Baselines (B1, B2, B3)

Status: read-only checkpoint audit and proposed baseline protocols. Updated 2026-11-19.
No pretrained weights were downloaded, no audio was processed, no models were executed,
and no benchmark evaluation results are claimed or passed.

## Track Scope and Checkpoint Audit

Member 2 owns unmodified, reference-free pretrained baseline adapters for three tracks:
- **B1:** `openai/whisper-small`
- **B2:** `openai/whisper-large-v3`
- **B3:** `Qwen3-ASR` candidates (retaining both `-hf` candidates without selection)

The table below summarizes upstream releases, licenses, Turkish support, training data disclosures, and WorldSpeech eligibility. Original upstream release dates are distinguished from pinned Hugging Face revisions.

| Baseline Track & Model ID | Original Upstream Release Date | Pinned Hugging Face Revision | License | Turkish Support | Documented Training Data & Direct Sources | WorldSpeech Eligibility (`disco-eth/WorldSpeech` `tr_tr`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B1:** `openai/whisper-small` | September 21, 2022 ([OpenAI Release](https://github.com/openai/whisper)) | [`973afd24...`](https://huggingface.co/openai/whisper-small/commit/973afd24965f72e36ca33b3055d56a652f456b4d) | [Apache 2.0 (pinned Hugging Face model-card metadata)](https://huggingface.co/openai/whisper-small/blob/973afd24965f72e36ca33b3055d56a652f456b4d/README.md) | Supported (`tr` / `<\|tr\|>`) ([Radford et al., 2022](https://arxiv.org/abs/2212.04356)) | 680,000 h weakly supervised web audio (~1,000+ h Turkish in Fig 9; source URLs undisclosed) ([Radford et al., 2022](https://arxiv.org/abs/2212.04356)) | **unverified** (Packaged dataset 2026 postdates release 2022, but raw TBMM web audio predates 2022; web crawl URLs undisclosed; missing disclosure does not prove eligibility) |
| **B2:** `openai/whisper-large-v3` | November 6, 2023 ([OpenAI Discussion #1762](https://github.com/openai/whisper/discussions/1762)) | [`06f233fe...`](https://huggingface.co/openai/whisper-large-v3/commit/06f233fe06e710322aca913c1bc4249a0d71fce1) | [Apache 2.0 (pinned Hugging Face model-card metadata)](https://huggingface.co/openai/whisper-large-v3/blob/06f233fe06e710322aca913c1bc4249a0d71fce1/README.md) | Supported (`tr` / `<\|tr\|>`, 128 Mel bins) ([HF Model Card](https://huggingface.co/openai/whisper-large-v3)) | 5M h audio (1M weakly labeled, 4M pseudo-labeled; source URLs undisclosed) ([HF Model Card](https://huggingface.co/openai/whisper-large-v3)) | **unverified** (Packaged dataset 2026 postdates release 2023, but raw TBMM audio was public; web crawl URLs undisclosed; WorldSpeech `tr_tr` upstream test references filtered with `whisper-large-v3-turbo` agreement [CER < 0.3], introducing circularity/bias) |
| **B3 (Candidate 1, unselected):**<br>`Qwen/Qwen3-ASR-1.7B-hf` | January 28, 2026 ([Qwen3-ASR Repo](https://github.com/QwenLM/Qwen3-ASR)) | [`bcd2b5b7...`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf/commit/bcd2b5b7f32b480ab5790554cfa8347f246a14f3) | [Apache 2.0](https://github.com/QwenLM/Qwen3-ASR/blob/main/LICENSE) | Supported (1 of 52 languages/dialects) ([HF Model Card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)) | ~40M h pseudo-labeled audio for audio-encoder pretraining ([Qwen3-ASR Technical Report, Section 2.2, arXiv:2601.21337](https://arxiv.org/abs/2601.21337)); specific training corpora, hours, and Turkish URLs undisclosed | **unverified** (Contemporaneous release 2026; training audio sources undisclosed; missing disclosure does not prove eligibility) |
| **B3 (Candidate 2, unselected):**<br>`Qwen/Qwen3-ASR-0.6B-hf` | January 28, 2026 ([Qwen3-ASR Repo](https://github.com/QwenLM/Qwen3-ASR)) | [`7f1569a4...`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf/commit/7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c) | [Apache 2.0](https://github.com/QwenLM/Qwen3-ASR/blob/main/LICENSE) | Supported (1 of 52 languages/dialects) ([HF Model Card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf)) | ~40M h pseudo-labeled audio for audio-encoder pretraining ([Qwen3-ASR Technical Report, Section 2.2, arXiv:2601.21337](https://arxiv.org/abs/2601.21337)); specific training corpora, hours, and Turkish URLs undisclosed | **unverified** (Contemporaneous release 2026; training audio sources undisclosed; missing disclosure does not prove eligibility) |

The published `model.safetensors` SHA-256 for `openai/whisper-small` is
`1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b` (inspected via
Hub API metadata; bytes unverified locally). Other model weight hashes remain null until
local download verification.

## Smoke Test Observations vs. Benchmark Results

Accuracy is **UNMEASURED** in the smoke test: no reference transcripts were loaded, and neither
WER nor CER was evaluated. Upstream claims cannot be substituted for measured project evaluations.

Smoke observations (conducted by Member 6 on the M6 RTX 5070 Ti laptop, branch `feat/m6-smoke-tests`;
see `docs/m6-smoke-results.md`) are technical feasibility measurements and sanity checks, not official
project benchmark results. Precision, decoding parameters, and resource measurements describe what ran
in that narrow feasibility test:
- **Whisper-small (B1):** 20/20 `ok` results (batch 1, fp16). Peak VRAM 1.37 GiB alloc / 1.94 GiB
  reserved; RTF 0.049. Standard `WhisperForConditionalGeneration.generate(num_return_sequences > 1)`
  failed (returned 180 duplicate copies of 1-best with identical token IDs across 20 utterances).
  Core `GenerationMixin.generate(num_beams=10, num_return_sequences=10)` succeeded. Beam search
  returned 29 duplicate text pairs across 12/20 clips (all distinct token paths); valid after shared-export
  deduplication (mean kept N: 8.55/10). Hypotheses start with leading space (`raw-v1`).
- **Whisper-large-v3 (B2):** 20/20 `ok` results (batch 1, fp16). Peak VRAM 5.67 GiB alloc / 6.50 GiB
  reserved; RTF 0.167. Standard wrapper call caused a sticky CUDA OOM (100 sequences). Core
  `GenerationMixin` succeeded. Beam search returned 5 duplicate text pairs across 4/20 clips (distinct
  token paths); valid after shared export deduplication (mean kept N: 9.75/10). Hypotheses start with
  leading space (`raw-v1`).
- **Qwen3-ASR 1.7B-hf (B3 Candidate 1):** 20/20 `ok` results (batch 1, bf16). Peak VRAM 4.56 GiB alloc /
  5.63 GiB reserved; RTF 0.098. Beam 5 returned 23 duplicate text pairs across 12/20 clips (distinct
  token paths); valid after shared export deduplication (mean kept N: 3.85/5). Hypotheses do not start
  with leading space. Native `-hf` path verified; `qwen-asr` pip package is prohibited (pins transformers 4.57.6).
- **Qwen3-ASR 0.6B-hf (B3 Candidate 2):** 20/20 `ok` results (batch 1, bf16). Peak VRAM 2.22 GiB alloc /
  3.30 GiB reserved; RTF 0.090. Beam 5 returned 19 duplicate text pairs across 11/20 clips (distinct
  token paths); valid after shared export deduplication (mean kept N: 4.05/5). Hypotheses do not start
  with leading space. Native `-hf` path verified.

Both B3 candidates remain unselected. Selection will be determined by measured validation performance
and resource constraints once the validated dataset is opened for evaluation.

## Proposed Baseline Decoding Protocols

All decoding settings are **proposed** and require freeze on validation before official timing.
Dataset binding remains pending the approved, verified release from M1/M6 (all dataset fields set to null).

- `configs/baselines/whisper_small.json`: Proposed beam 10, requested N 10, max new tokens 440,
  precision float16, forced prefix `["<|startoftranscript|>", "<|tr|>", "<|transcribe|>", "<|notimestamps|>"]`.
  Score convention: `whisper-original-logits-generated-plus-eos-v1`.
- `configs/baselines/whisper_large_v3.json`: Proposed beam 10, requested N 10, max new tokens 440,
  precision float16, same forced prefix and score convention as B1.
- `configs/baselines/qwen3_asr_1_7b.json`: Proposed beam 5, requested N 5, max new tokens 440,
  precision bfloat16, language `tr`. Score convention: `qwen3-asr-joint-sequence-score-v1`.
- `configs/baselines/qwen3_asr_0_6b.json`: Proposed beam 5, requested N 5, max new tokens 440,
  precision bfloat16, language `tr`. Score convention: `qwen3-asr-joint-sequence-score-v1`.

### Implementation Rules
1. **Whisper N-best:** Adapter MUST use core `GenerationMixin.generate` with manual forced prefix
   tokens, strip the prompt from token IDs/count, and compute `acoustic_log_score` via a separate
   teacher-forced pass.
2. **Qwen3-ASR Score Semantics:** Sequence scores MUST be labeled as **joint** scores
   (`joint_speech_language_beam_score`) under N-best contract 0.2, as acoustic-only components cannot be isolated.
3. **Environment:** Native `-hf` checkpoints run natively in transformers 5.23.0. Do not install `qwen-asr`.
4. **Deduplication:** Adapters return raw candidate lists. Shared export owns deduplication.
5. **Leading Whitespace:** Whisper outputs leading spaces; Qwen does not. Impact under `raw-v1` CER
   is flagged for M6 evaluation reporting.

## Required Reviews Flagged for Teammates

### For Member 4 (Whisper Adaptation)
- **Parent Confirmation:** Confirm B1 `openai/whisper-small` (revision `973afd24965f72e36ca33b3055d56a652f456b4d`)
  as the common parent for F1 (LoRA) and F2 (full fine-tuning).
- **N-best Alignment:** M4 proposed `requested_n: 5`, while the smoke test evaluated beam 10 returning 10.
  Agree on frozen beam width and requested N before timing.
- **Acoustic Score Convention:** Align on `whisper-original-logits-generated-plus-eos-v1` teacher-forced
  log-probability sums excluding prompt tokens.

### For Member 6 (Integration, Evaluation & QA)
- **B3 Candidate Selection:** Evaluate 1.7B vs 0.6B once validation accuracy can be measured, trading
  off compute/latency against task accuracy.
- **Decoding Freeze:** Review and freeze beam widths across all baselines before the standardized timing session.
- **Joint Score Convention:** Approve `qwen3-asr-joint-sequence-score-v1` under N-best contract 0.2.
- **CER Policy on Edge Whitespace:** Review evaluation handling for Whisper's emitted leading space under `raw-v1`.
- **Timing Protocol:** Ensure timing executes back-to-back with 3 repeats on the 30-minute frozen subset
  with background GPU processes closed.

## Pending Technical Steps

- [ ] Confirm dataset binding once M1/M6 release is approved and frozen.
- [ ] Obtain M4 confirmation of B1 parent and M6 baseline protocol review.
- [ ] Implement offline unit tests for B1/B2/B3 adapters using synthetic mock inputs (no weights downloaded).
- [ ] Implement reference-free adapters implementing the `AsrAdapter` protocol (`load`, `transcribe`).
- [ ] Verify local weights and safetensors hashes upon authorized download.
- [ ] Export full-ID prediction bundles for validation and test splits with zero test reference access.
