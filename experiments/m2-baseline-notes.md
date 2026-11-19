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
| **B1:** `openai/whisper-small` | September 21, 2022 ([OpenAI Release](https://github.com/openai/whisper)) | [`973afd24...`](https://huggingface.co/openai/whisper-small/commit/973afd24965f72e36ca33b3055d56a652f456b4d) | [Apache 2.0 (pinned Hugging Face model-card metadata)](https://huggingface.co/openai/whisper-small/blob/973afd24965f72e36ca33b3055d56a652f456b4d/README.md) | Supported (`tr` / `<\|tr\|>`) ([Radford et al., 2022](https://arxiv.org/abs/2212.04356)) | 680,000 h weakly supervised web audio (~2,286 h Turkish ASR and ~2,241 h Turkish-to-English translation in Figure 11, Appendix E; source URLs undisclosed) ([Radford et al., 2022](https://arxiv.org/abs/2212.04356)) | **eligible** (`disco-eth/WorldSpeech` was created on the Hub on 2026-04-28 [paper arXiv:2605.09167 submitted 2026-05-09]; upstream release 2022 predates WorldSpeech, and TURAS test sessions [2026-01-06 to 2026-04-22] postdate release, so B1 cannot have seen WorldSpeech or test audio) |
| **B2:** `openai/whisper-large-v3` | November 6, 2023 ([OpenAI Discussion #1762](https://github.com/openai/whisper/discussions/1762)) | [`06f233fe...`](https://huggingface.co/openai/whisper-large-v3/commit/06f233fe06e710322aca913c1bc4249a0d71fce1) | [Apache 2.0 (pinned Hugging Face model-card metadata)](https://huggingface.co/openai/whisper-large-v3/blob/06f233fe06e710322aca913c1bc4249a0d71fce1/README.md) | Supported (`tr` / `<\|tr\|>`, 128 Mel bins) ([HF Model Card](https://huggingface.co/openai/whisper-large-v3)) | 5M h audio (1M weakly labeled, 4M pseudo-labeled; source URLs undisclosed) ([HF Model Card](https://huggingface.co/openai/whisper-large-v3)) | **eligible** (Upstream release November 2023 predates WorldSpeech creation 2026-04-28; test sessions [2026-01-06 to 2026-04-22] postdate release, so B2 cannot have seen WorldSpeech or test audio; note alignment filter family bias limitation) |
| **B3 (Candidate 1, unselected):**<br>`Qwen/Qwen3-ASR-1.7B-hf` | January 28, 2026 ([Qwen3-ASR Repo](https://github.com/QwenLM/Qwen3-ASR)) | [`bcd2b5b7...`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf/commit/bcd2b5b7f32b480ab5790554cfa8347f246a14f3) | [Apache 2.0](https://github.com/QwenLM/Qwen3-ASR/blob/main/LICENSE) | Supported (1 of 52 languages/dialects) ([HF Model Card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)) | ~40M h pseudo-labeled audio for audio-encoder pretraining ([Qwen3-ASR Technical Report, Section 2.2, arXiv:2601.21337](https://arxiv.org/abs/2601.21337)); specific training corpora, hours, and Turkish URLs undisclosed | **unverified / open** (Pinned `-hf` repo created 2026-06-26 postdates WorldSpeech 2026-04-28; card does not state it holds January weights; weights differ in size [4.08 GB in 1 shard vs 4.70 GB in 2 shards]; requires conversion citation or tensor check; 6–27 Jan test audio predates Jan 28 release) |
| **B3 (Candidate 2, unselected):**<br>`Qwen/Qwen3-ASR-0.6B-hf` | January 28, 2026 ([Qwen3-ASR Repo](https://github.com/QwenLM/Qwen3-ASR)) | [`7f1569a4...`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf/commit/7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c) | [Apache 2.0](https://github.com/QwenLM/Qwen3-ASR/blob/main/LICENSE) | Supported (1 of 52 languages/dialects) ([HF Model Card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf)) | ~40M h pseudo-labeled audio for audio-encoder pretraining ([Qwen3-ASR Technical Report, Section 2.2, arXiv:2601.21337](https://arxiv.org/abs/2601.21337)); specific training corpora, hours, and Turkish URLs undisclosed | **unverified / open** (Pinned `-hf` repo created 2026-06-26 postdates WorldSpeech 2026-04-28; card does not state it holds January weights; weights differ in size [1.56 GB vs 1.88 GB]; requires conversion citation or tensor check; 6–27 Jan test audio predates Jan 28 release) |

Published model weight files and SHA-256 LFS OIDs inspected via Hugging Face Hub metadata across all four pinned revisions (labeled "Hub LFS oid; bytes verified at download"):
- **B1 (`openai/whisper-small`):** `model.safetensors` SHA-256 `1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b` (966,995,080 bytes).
- **B2 (`openai/whisper-large-v3`):** `model.safetensors` SHA-256 `a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95` (3,087,130,976 bytes; pinned by filename because repo also contains fp32 shards).
- **B3 Candidate 1 (`Qwen/Qwen3-ASR-1.7B-hf`):** `model.safetensors` SHA-256 `2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1` (4,076,193,080 bytes).
- **B3 Candidate 2 (`Qwen/Qwen3-ASR-0.6B-hf`):** `model.safetensors` SHA-256 `d3f212dd20abecd315d830bc54ae3865e56ebfc3276484e57b771288ba27fd35` (1,564,928,088 bytes).

### B1/B2 and B3 Checkpoint Eligibility (WorldSpeech / TBMM)

#### Project Rules and Disclosure Boundaries
Under the binding project methodology in [docs/dataset.md](../docs/dataset.md#L55), Member 2 must:
- *"Report limitations from unknown model pretraining contamination."*

Additionally, [docs/contracts/artifacts.md](../docs/contracts/artifacts.md#L47) mandates that experiment manifests document `contamination risk, omissions and reviewer evidence` under run limitations, while [TEAM_TASKS.md](../TEAM_TASKS.md#L96-L98) and [docs/decisions.md](../docs/decisions.md) (D05) require verifying constraints before committing to dependencies or downloads.

The project evaluates models on curated Turkish speech derived from `disco-eth/WorldSpeech` (`tr_tr`), sourced from Turkish parliamentary proceedings (TBMM). For baseline comparisons to be sound, pretraining exposure to evaluation audio must be verified against release timelines and dataset provenance.

#### B1 and B2 Eligibility Finding
`disco-eth/WorldSpeech` was created on Hugging Face on 2026-04-28 (paper arXiv:2605.09167 submitted 2026-05-09). The TURAS test sessions span 2026-01-06 to 2026-04-22.
- **B1 (`openai/whisper-small`):** Released September 21, 2022, predating WorldSpeech creation and test sessions by over 3 years. B1 cannot have seen the WorldSpeech benchmark or test audio: **eligible**.
- **B2 (`openai/whisper-large-v3`):** Released November 6, 2023, predating WorldSpeech creation and test sessions by over 2 years. B2 cannot have seen the WorldSpeech benchmark or test audio: **eligible**.

#### ASR-Agreement Filter and Stated Limitation
TURAS keeps `cer <= 0.25` on every row in all three splits (applied by Member 1; 3,886 rows removed), comparing parliamentary minutes with an upstream ASR transcript. Member 1 measured upstream `tr_tr` CER at mean 0.075 and max 0.50. Which specific ASR model produced the alignment transcript for `tr_tr` is not confirmed (the WorldSpeech paper names `whisper-large-v3-turbo` as its most common alignment model, but does not provide a language-specific confirmation for `tr_tr`). Potential bias toward the alignment model's architectural family (possibly favoring B2 and B1) is a recognized and documented limitation; Member 6 reports verified-subset WER alongside full test WER to quantify any discrepancy.

#### Examination of Official Qwen3-ASR Sources (B3)
The official model cards at pinned revisions [`bcd2b5b7...`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf/blob/bcd2b5b7f32b480ab5790554cfa8347f246a14f3/README.md) (`Qwen/Qwen3-ASR-1.7B-hf`) and [`7f1569a4...`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf/blob/7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c/README.md) (`Qwen/Qwen3-ASR-0.6B-hf`), alongside the *Qwen3-ASR Technical Report* ([arXiv:2601.21337](https://arxiv.org/abs/2601.21337), Section 2.2), disclose the following:
- **Training Pipeline Stages (arXiv:2601.21337, Section 2.2):** Three sequential training phases:
  1. *Audio Transformer (AuT) Pretraining:* ~40 million hours of pseudo-labeled ASR data under an AED framework, predominantly Chinese and English.
  2. *Omni Pretraining:* Foundation pretraining of the Qwen3-Omni base model on ~3 trillion multi-modal tokens.
  3. *ASR Post-Training / Alignment:* Supervised fine-tuning (SFT) on a multilingual/dialect corpus supporting 52 languages/dialects (including Turkish `tr`), followed by reinforcement learning via Group Sequence Policy Optimization (GSPO) on ~50k utterances.
- **Specific Mentions of WorldSpeech or TBMM:** Neither the model cards nor the technical report mentions `disco-eth/WorldSpeech`, Turkish parliamentary debates, or TBMM audio in any section, table, or benchmark list.
- **Repository Creation & Weight Discrepancies:** The pinned `-hf` repositories were created on 2026-06-26, after WorldSpeech (2026-04-28). The model cards do not explicitly state that `-hf` checkpoints retain the exact January 28, 2026 weights (`Qwen/Qwen3-ASR-1.7B` was created on 2026-01-28). Weight files differ in structure and size: 1.7B is 4.08 GB in a single `model.safetensors` file vs 4.70 GB across two shards in the original repo; 0.6B is 1.56 GB vs 1.88 GB.

#### B3 Eligibility Status and Resolution Boundary
Because the pinned `-hf` repos postdate WorldSpeech and weight formats differ, **B3 eligibility remains unverified / open**.
- **Verification Paths:**
  1. Upstream documentation or citation from Qwen confirming `-hf` is strictly a format conversion of the January 28, 2026 release.
  2. Post-download comparison of shared tensor values between original and `-hf` checkpoints (M6 can execute this on the RTX 5070 Ti).
- **If Verified Identical:** The January 28, 2026 release date applies, with the stated limitation that test sessions from 2026-01-06 to 2026-01-27 predate it.
- **Candidate Status:** Neither candidate is selected, and both configuration files (`configs/baselines/qwen3_asr_1_7b.json` and `configs/baselines/qwen3_asr_0_6b.json`) remain unselected (`candidate_status: "unselected-candidate"`).

## Smoke Test Observations vs. Benchmark Results

Accuracy is **UNMEASURED** in the smoke test: no reference transcripts were loaded, and neither
WER nor CER was evaluated. Upstream claims cannot be substituted for measured project evaluations.
Smoke RTF is **not** a selection criterion (measured over 20 clips with one repeat and differing beam widths). B3 candidate selection will be decided on validation WER/CER plus standardized RTF measured in the unified M6 timing session.

Smoke observations (conducted by Member 6 on the M6 RTX 5070 Ti laptop, branch `feat/m6-docs` at commit `9f09a5f`;
see `docs/m6-smoke-results.md`) are technical feasibility measurements and sanity checks, not official
project benchmark results:
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

- `configs/baselines/whisper_small.json`: Proposed beam 10, requested N 10 (provisional 5 or 10, matched with F1/F2 and frozen on validation), max new tokens 440, length penalty 1.0, early stopping false, repetition penalty 1.0, no repeat ngram size 0, precision float16, do_sample false, temperature null, forced prefix `["<|startoftranscript|>", "<|tr|>", "<|transcribe|>", "<|notimestamps|>"]`. Score convention: `whisper-original-logits-generated-plus-eos-v1`.
- `configs/baselines/whisper_large_v3.json`: Proposed beam 10, requested N 10 (provisional 5 or 10, frozen on validation), max new tokens 440, length penalty 1.0, early stopping false, repetition penalty 1.0, no repeat ngram size 0, precision float16, do_sample false, temperature null, same forced prefix and score convention as B1. Pinned to `model.safetensors` by name.
- `configs/baselines/qwen3_asr_1_7b.json`: Proposed beam 5, requested N 5 (provisional, validation choice), max new tokens 440, length penalty 1.0, early stopping false, repetition penalty 1.0, no repeat ngram size 0, precision bfloat16, do_sample false, temperature null, language `tr`, prompt via `AutoProcessor.apply_transcription_request(language='tr')`. Score convention: `smoke-qwen3-asr-joint-token-logprob-sum-v1`.
- `configs/baselines/qwen3_asr_0_6b.json`: Proposed beam 5, requested N 5 (provisional, validation choice), max new tokens 440, length penalty 1.0, early stopping false, repetition penalty 1.0, no repeat ngram size 0, precision bfloat16, do_sample false, temperature null, language `tr`, prompt via `AutoProcessor.apply_transcription_request(language='tr')`. Score convention: `smoke-qwen3-asr-joint-token-logprob-sum-v1`.

### Audio Constraints
TURAS audio clips are 10–20 s (from Member 1 source audit), comfortably within Whisper's 30 s window and Qwen3-ASR's input frame limits. No sliding-window chunking or audio segmentation is required for baseline inference.

### Implementation Rules
1. **Whisper N-best:** Adapter MUST use core `GenerationMixin.generate` with manual forced prefix
   tokens, strip the prompt from token IDs/count, and compute `acoustic_log_score` via a separate
   teacher-forced pass.
2. **Qwen3-ASR Score Semantics:** Sequence scores MUST be labeled as **joint** scores
   (`speech_llm_joint_token_log_prob_sum`) under convention `smoke-qwen3-asr-joint-token-logprob-sum-v1`, computed via `transformers.GenerationMixin.compute_transition_scores` over generation output (`out.sequences`, `out.scores`, `out.beam_indices`), summed over generated tokens through end-of-turn (EOS), excluding prompt and padding tokens. This score originates from generation transition scores rather than a separate teacher-forced forward pass on the original model logits; these two methods differ in execution and cannot be assumed interchangeable without concrete empirical evidence. Formal confirmation of this convention for the official B3 baseline run remains pending team sign-off. The library beam score (`sequences_scores` with length penalty) is preserved separately in `decoder_sequence_score` (annotated with null reasons if missing).
3. **Environment:** Native `-hf` checkpoints run natively in transformers 5.23.0. Do not install `qwen-asr`.
4. **Deduplication:** Adapters return unpadded raw candidate lists. Shared export owns deduplication under the contract duplicate policy: `"exact raw-text match; first (lowest-rank) occurrence kept; removed IDs listed in dedup_map"`.
5. **Leading Whitespace:** Whisper outputs leading spaces; Qwen does not. Under `raw-v1`, adapters preserve emitted leading whitespace without stripping. M6 reports per-system edge whitespace counts alongside raw CER.
6. **Matched Decoding:** B1 is the common parent of F1/F2; B1 and F1/F2 must decode under a single frozen beam/N setting chosen on validation (provisional 5 vs 10). B2 and B3 beam widths are likewise frozen on validation.

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
  off compute/latency against task accuracy. Both candidates remain strictly unselected (`candidate_status: "unselected-candidate"`).
- **B3 Checkpoint Eligibility:** Resolve pretraining contamination / timeline audit independently: confirm whether `-hf` weights (June 2026) are bitwise/tensor identical to the original January 28, 2026 release, or obtain an upstream conversion citation before official use.
- **B3 Scoring Convention Approval:** Formally confirm and approve `smoke-qwen3-asr-joint-token-logprob-sum-v1` under N-best contract 0.2 (using generation transition scores through EOS rather than a separate teacher-forced forward pass) for the official B3 run (currently pending team sign-off).
- **Decoding Freeze:** Review and freeze beam widths across all baselines on the validation set before the standardized timing session (B1 and F1/F2 aligned on provisional 5 vs 10; N=10 is not final).
- **CER Policy on Edge Whitespace:** Review evaluation handling for Whisper's emitted leading space under `raw-v1`.
- **Timing Protocol:** Ensure timing executes back-to-back with 3 repeats on the 30-minute frozen subset
  with background GPU processes closed.

## Baseline Run and Handoff Checklist (B1, B2, B3)

This checklist organizes the remaining execution steps for Member 2 across four sequential phases, strictly respecting shared contracts, model boundaries, and handoff protocols.

### Phase 1: Offline Adapter, Smoke Runner, and Batch Exporter Implementation
- [x] **Contract Compliance in Adapter:**
  - Verified that adapter input handling strictly follows [docs/contracts/predictions.md](../docs/contracts/predictions.md) (allowing only `utterance_id`, `audio_uri`, `audio_sha256`, `duration_s`, `sample_rate_hz`, `channels`, and `language`; zero access to reference text).
  - Validated adapter wiring against pinned configurations in `configs/baselines/` without downloading weights.
- [x] **Whisper Decoding & Scoring Tests (B1 & B2):**
  - Verified that the adapter uses core `GenerationMixin.generate` with forced prefix tokens `["<|startoftranscript|>", "<|tr|>", "<|transcribe|>", "<|notimestamps|>"]`.
  - Checked that prompt tokens are excluded from returned token IDs and token counts.
  - Checked that hypotheses preserve decoded text exactly as emitted, including leading whitespace under `raw-v1`.
  - Verified that `acoustic_log_score` is computed via teacher-forced log-probabilities on generated tokens and kept separate from framework `decoder_sequence_score`.
  - Ensured that if `decoder_sequence_score` is unavailable, it is preserved honestly as `null` with explanation in `score_details`.
  - Verified raw candidate output format and integration with shared export deduplication (`export_nbest_records`).
- [x] **Qwen Decoding & Scoring Tests (B3):**
  - Verified that candidate sequence scores are labeled strictly as `speech_llm_joint_token_log_prob_sum` under convention `smoke-qwen3-asr-joint-token-logprob-sum-v1` computed via generation transition scores through EOS.
  - Kept checkpoint selection between Candidate 1 (1.7B) and Candidate 2 (0.6B) pending.
- [x] **Whisper Rank-1 Smoke Runner (`scripts/smoke_check_whisper_rank1.py`):**
  - Maintained `--synthetic` mode for deterministic zero-dependency tests.
  - Added real-audio mode supporting individual clips (`--audio`) and reference-free manifests (`--manifest`).
  - Enforced local-only file paths, verified checkpoint byte SHA-256 before model loading, and suppressed decoded transcripts by default (`--show-transcripts` required).
- [x] **Baseline Batch Inference & Exporter (`src/turkish_asr_rdl/baselines/runner.py` & `scripts/run_baseline_batch.py`):**
  - Accepts baseline JSON config, local checkpoint directory/file, reference-free input manifest, and output directory outside the repository.
  - Verifies checkpoint inventory and SHA-256 before loading (rejects remote URLs, missing files, corrupt hashes, and unindexed shards).
  - Parses reference-free manifest strictly enforcing allowlist (rejects any manifest containing `reference_text` or `gold`).
  - Preserves utterance order, utterance IDs, and per-clip error captures without dropping items.
  - Exports schema-validated `predictions.jsonl`, `nbest.jsonl`, and `manifest.json`.

### Phase 2: Authorized Smoke Runs (Real Weights on Local Pinned Runtime)
- [ ] **Environment Verification:** Execute all runs inside the pinned Windows environment (`$env:LOCALAPPDATA\TURAS\venv-313\Scripts\python.exe`, lock `requirements/windows-cu130.txt` at `bf4e0bf1319aa573f90f73e3cf8ec0a29a197608`).
- [ ] **Whisper Rank-1 Equivalence Check (B1 & B2):**
  - On authorized local audio inputs, run Whisper beam search at beam width $K$ returning $K$ candidates (`num_beams=K`, `num_return_sequences=K`).
  - Run standard single-result generation at the same beam width $K$ (`num_beams=K`, `num_return_sequences=1`).
  - Verify that candidate rank 1 from the N-best generation produces text and token IDs identical to the single-result output.
- [ ] **Resource and Feasibility Verification:**
  - Confirm fp16 execution on CUDA, peak VRAM within hardware budget (12 GiB RTX 3060), and measure RTF sanity.
  - For Qwen candidates, verify native `-hf` execution in transformers 5.23.0 (`qwen-asr` package prohibited).

### Phase 3: Validation-Based Settings Freeze
- [ ] **Dataset Release Binding:**
  - Update `dataset_release` and `split_manifest_sha256` in baseline JSON configs only after M1/M6 officially release and freeze the dataset.
  - Ensure all input paths and split IDs point to immutable release manifests.
- [ ] **Decoding Parameter Freeze:**
  - Freeze validation beam width, requested N, and max new tokens on the validation split prior to test split execution.
  - Update `status` from `proposed-baseline-protocol` to `frozen-baseline-protocol` upon review.
- [ ] **B3 Candidate Resolution:**
  - Review measured validation accuracy, peak VRAM, and RTF trade-offs with M6.
  - Formally select either Candidate 1 (1.7B) or Candidate 2 (0.6B) for the B3 track, update `candidate_status` to `selected-baseline`, and record the selection decision in `docs/decisions.md`.

### Phase 4: Final Prediction Export and Handoff to M6
- [ ] **Prediction JSONL Export ([docs/contracts/predictions.md](../docs/contracts/predictions.md)):**
  - Generate full-ID predictions for validation and test splits using frozen settings.
  - Ensure every record includes `schema_version: "0.1"`, `run_id`, `system_id`, `dataset_release`, `split`, `split_manifest_sha256`, `utterance_id`, `audio_sha256`, `status`, `hypothesis_text`, `selected_candidate_id`, `nbest_artifact_sha256`, `model_artifact_sha256`, `decoder_config_sha256`, `normalization_version: "raw-v1"`, `timing_record_id`, and `error`.
  - Explicitly record failures: emit status `error` with sanitized error details, or `empty` for zero-token decodes; never drop an utterance or emit a fabricated transcript.
- [ ] **N-Best JSONL Export ([docs/contracts/nbest.md](../docs/contracts/nbest.md)):**
  - Generate matching N-best JSONL artifact containing ranked, unpadded candidate lists with distinct acoustic and sequence scores, score details, and shared deduplication mappings.
- [ ] **Manifest Generation ([docs/contracts/artifacts.md](../docs/contracts/artifacts.md)):**
  - Model Manifest: Document system ID, pinned Hugging Face revision, weight file SHA-256, runtime lock hash, and processor settings.
  - Experiment Run Manifest: Document run ID, code git commit, dirty diff hash (if any), resolved config hash, exact CLI command, hardware environment, timing scope, and explicit contamination risk limitations.
- [ ] **Integrity Checks Prior to Handoff:**
  - Verify exact 1-to-1 match between input split manifest IDs and output prediction IDs (no missing, extra, or duplicate records).
  - Verify audio byte SHA-256 matches input records.
  - Check that all numerical scores are finite and JSONL files parse strictly.
- [ ] **Handoff Boundary to M6:**
  - Deliver prediction bundles, N-best artifacts, and run manifests to M6.
  - Leave official evaluation (raw and normalized WER/CER, Turkish I / apostrophe normalization under `tr-normalized-v1`, domain terminology metrics, and standardized timing evaluation) to Member 6.
  - Record handoff acknowledgment and artifact hashes in [contributions/member_2.md](member_2.md).

## Workflow Execution Guide & Handoff Sequence

All commands run under PowerShell with the pinned Python 3.13 virtual environment:
`$PYTHON = "$env:LOCALAPPDATA\TURAS\venv-313\Scripts\python.exe"`

### Execution Order & Handoff Protocol

Execution of pretrained baseline workflows must proceed in three strict sequential stages:
1. **Stage 1 (First Real Run): B1/B2 Smoke Check on Approved Local Clips & Verified Weights:**
   Before any batch inference on dataset splits, execute `scripts/smoke_check_whisper_rank1.py` on authorized local audio clips once model weights are downloaded to local storage and their byte SHA-256 digests are verified. This confirms exact rank-1 parity between the shared N-best decoder and Whisper's standard `generate(num_return_sequences=1)` path at identical beam widths.
2. **Stage 2: Validation Evaluation & Parameter Freeze:**
   Execute batch decoding on the official reference-free validation split (`val_reference_free.jsonl`) to evaluate candidate decoding parameters. Coordinate with Member 4 to evaluate provisional beam/N settings (5 vs 10) so the common parent B1 and fine-tuned models F1/F2 share identical decoding settings. **Neither N=10 nor N=5 is final** until frozen on the validation set (`status: "frozen-baseline-protocol"`). Both Qwen candidates remain strictly unselected.
3. **Stage 3: Official Test Export Using Frozen Settings:**
   Execute `scripts/run_baseline_batch.py` on the verified reference-free test split manifest using the frozen decoding settings and verified weights. Delivers validated `predictions.jsonl`, `nbest.jsonl`, and `manifest.json` outside the repository for Member 6's benchmark evaluation.

### 1. Whisper Rank-1 Smoke Check (`scripts/smoke_check_whisper_rank1.py`)

#### A. Synthetic Offline Mode
Verifies parity on synthetic Whisper weights without downloading pretrained checkpoints:
```powershell
& "$env:LOCALAPPDATA\TURAS\venv-313\Scripts\python.exe" scripts/smoke_check_whisper_rank1.py `
  --synthetic `
  --shared-decoder-path "path/to/turas_shared_export" `
  --beam-width 2 `
  --max-new-tokens 10
```

#### B. Real-Audio Mode (B1 Whisper-small or B2 Whisper-large-v3)
First real execution step on local audio clips with verified local checkpoint weights:
```powershell
# For B1 Whisper-small on explicit local clip paths:
& "$env:LOCALAPPDATA\TURAS\venv-313\Scripts\python.exe" scripts/smoke_check_whisper_rank1.py `
  --config configs/baselines/whisper_small.json `
  --checkpoint "C:\turas_weights\whisper_small" `
  --audio "C:\turas_data\clip1.flac" "C:\turas_data\clip2.flac" `
  --device "cpu"

# For B2 Whisper-large-v3 using an input manifest:
& "$env:LOCALAPPDATA\TURAS\venv-313\Scripts\python.exe" scripts/smoke_check_whisper_rank1.py `
  --config configs/baselines/whisper_large_v3.json `
  --checkpoint "C:\turas_weights\whisper_large_v3" `
  --manifest "C:\turas_data\smoke_manifest.jsonl" `
  --audio-root "C:\turas_data\audio" `
  --device "cuda"
```

Expected output format:
```text
=== Whisper Rank-1 Real-Audio Smoke Check ===
Clip clip-01: MATCH | Text Match: True | Tokens Match: True
Clip clip-02: MATCH | Text Match: True | Tokens Match: True

=== Smoke Check Summary ===
Total clips: 2
Matches:     2
Mismatches:  0
Errors:      0
Verdict:     PASSED
```
*Note:* Decoded transcripts are hidden by default to prevent accidental transcript leakage; pass `--show-transcripts` to inspect verbatim hypotheses.

### 2. Batch Inference & Shared Exporter (`scripts/run_baseline_batch.py`)

Executes batch decoding across reference-free manifests and generates schema-valid artifacts outside the repository.
*Important:* The `--requested-n` values in the commands below (e.g. 10 or 5) are provisional defaults for testing; the final values used for official test export must be frozen during validation in coordination with Member 4 and Member 6. Both Qwen candidates remain strictly unselected (`candidate_status: "unselected-candidate"`).

#### A. B1 Whisper-small Batch Run (Validation / Test with Frozen N)
```powershell
# Example batch run with provisional requested-n (10 or 5, to be frozen on validation):
& "$env:LOCALAPPDATA\TURAS\venv-313\Scripts\python.exe" scripts/run_baseline_batch.py `
  --config configs/baselines/whisper_small.json `
  --checkpoint "C:\turas_weights\whisper_small" `
  --manifest "C:\turas_data\val_reference_free.jsonl" `
  --output-dir "C:\turas_runs\b1_whisper_small_val01" `
  --audio-root "C:\turas_data\audio" `
  --device "cuda" `
  --requested-n 10 `
  --split "validation"
```

#### B. B2 Whisper-large-v3 Batch Run (Validation / Test with Frozen N)
```powershell
& "$env:LOCALAPPDATA\TURAS\venv-313\Scripts\python.exe" scripts/run_baseline_batch.py `
  --config configs/baselines/whisper_large_v3.json `
  --checkpoint "C:\turas_weights\whisper_large_v3" `
  --manifest "C:\turas_data\val_reference_free.jsonl" `
  --output-dir "C:\turas_runs\b2_whisper_large_v3_val01" `
  --audio-root "C:\turas_data\audio" `
  --device "cuda" `
  --requested-n 10 `
  --split "validation"
```

#### C. B3 Qwen3-ASR Batch Run (Unselected Candidate 1: 1.7B-hf)
*Note:* Candidate 1 and Candidate 2 both remain unselected. Running B3 officially requires both candidate selection and resolution of checkpoint eligibility and scoring approval.
```powershell
& "$env:LOCALAPPDATA\TURAS\venv-313\Scripts\python.exe" scripts/run_baseline_batch.py `
  --config configs/baselines/qwen3_asr_1_7b.json `
  --checkpoint "C:\turas_weights\qwen3_asr_1_7b" `
  --manifest "C:\turas_data\val_reference_free.jsonl" `
  --output-dir "C:\turas_runs\b3_qwen_1_7b_val01" `
  --audio-root "C:\turas_data\audio" `
  --device "cuda" `
  --requested-n 5 `
  --split "validation"
```

Expected output files in `--output-dir`:
1. `predictions.jsonl`: Top-1 predictions adhering to `schemas/predictions.py`.
2. `nbest.jsonl`: Full beam candidate lists adhering to `schemas/nbest.py`.
3. `manifest.json`: Full experiment run manifest adhering to `schemas/report.py`.

Summary printed upon completion:
```text
=== TURAS Baseline Batch Export Summary ===
Run ID:             baseline-whisper-small-test-20261119-152000
System ID:          baseline-whisper-small
Total utterances:   100
  OK:               100
  Empty:            0
  Error:            0
Predictions:        C:\turas_runs\b1_whisper_small_run01\predictions.jsonl
  SHA-256:          <64-hex>
N-best:             C:\turas_runs\b1_whisper_small_run01\nbest.jsonl
  SHA-256:          <64-hex>
Run Manifest:       C:\turas_runs\b1_whisper_small_run01\manifest.json
  SHA-256:          <64-hex>
Validation OK:      True
```

### 3. Verified Test Evidence (63 Tests)

All 63 Member 2 tests pass under `$env:LOCALAPPDATA\TURAS\venv-313\Scripts\python.exe`:
- **Default Offline Suite (no shared decoder checkout):**
  `Ran 63 tests — OK (41 passed, 22 skipped)`
- **Opt-in Integration Suite (with provisional shared decoder at revision `b6afa13063f7fa90bb1bc190c67b4529e40e91a5`):**
  `Ran 63 tests in 9.043s — OK (63 passed, 0 failures, 0 errors, 0 skipped)`

Breakdown by test module:
1. `tests/test_m2_baseline_configs.py` (6 tests): JSON parsing, revision match, unselected status for both Qwen candidates, unset dataset bindings, model/processor IDs, and decoding parameter internal consistency.
2. `tests/test_m2_whisper_adapter.py` (22 tests): B1/B2 config binding, 16 kHz check, sampling/decoding override rejections, SHA-256 validation, prompt token exclusion, empty/error handling, and shared decoder integration (provisional `b6afa13`).
3. `tests/test_m2_qwen_adapter.py` (20 tests): Candidate 1 & 2 config binding, 16 kHz check, transition scoring prompt/padding exclusion, EOS inclusion, and shared decoder integration (provisional `b6afa13`).
4. `tests/test_m2_whisper_rank1.py` (7 tests): Deterministic exact match, mismatch, whitespace preservation, missing candidate, prompt token stripping, decode failure, and real-framework synthetic rank-1 integration.
5. `tests/test_m2_batch_runner.py` (8 tests): Checkpoint verification (success, mismatch, missing, remote, unresolved shards), reference-bearing manifest rejection in reader and batch runner, end-to-end Whisper export, end-to-end Qwen export, per-clip error preservation, and real-audio smoke runner.
