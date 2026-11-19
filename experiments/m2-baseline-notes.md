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

### B3 Checkpoint Eligibility (WorldSpeech / TBMM)

#### Project Rules and Disclosure Boundaries
Under the binding project methodology in [docs/dataset.md](../docs/dataset.md#L55), Member 2 must:
- *"Report limitations from unknown model pretraining contamination."*

Additionally, [docs/contracts/artifacts.md](../docs/contracts/artifacts.md#L47) mandates that experiment manifests document `contamination risk, omissions and reviewer evidence` under run limitations, while [TEAM_TASKS.md](../TEAM_TASKS.md#L96-L98) and [docs/decisions.md](../docs/decisions.md) (D05) require verifying constraints before committing to dependencies or downloads.

The project evaluates models on curated Turkish speech derived from `disco-eth/WorldSpeech` (`tr_tr`), which is sourced from Turkish parliamentary proceedings (TBMM). For baseline comparisons to be sound, pretraining exposure to evaluation audio must be understood. Because training summaries are often high-level, absence of a corpus name from a published description is **not proof of exclusion** ("missing disclosure does not prove eligibility"). Proving that a model was not exposed to evaluation speech would require positive upstream evidence (such as an auditable training manifest or itemized URL list).

#### Examination of Official Qwen3-ASR Sources
The official model cards at pinned revisions [`bcd2b5b7...`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf/blob/bcd2b5b7f32b480ab5790554cfa8347f246a14f3/README.md) (`Qwen/Qwen3-ASR-1.7B-hf`) and [`7f1569a4...`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf/blob/7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c/README.md) (`Qwen/Qwen3-ASR-0.6B-hf`), alongside the *Qwen3-ASR Technical Report* ([arXiv:2601.21337](https://arxiv.org/abs/2601.21337), Section 2.2), disclose the following:
- **Training Pipeline Stages (arXiv:2601.21337, Section 2.2):** The report describes three sequential training phases:
  1. *Audio Transformer (AuT) Pretraining:* ~40 million hours of pseudo-labeled ASR data under an AED framework, predominantly Chinese and English.
  2. *Omni Pretraining:* Foundation pretraining of the Qwen3-Omni base model on ~3 trillion multi-modal tokens.
  3. *ASR Post-Training / Alignment:* Supervised fine-tuning (SFT) on a multilingual/dialect corpus supporting 52 languages/dialects (including Turkish `tr`), followed by reinforcement learning via Group Sequence Policy Optimization (GSPO) on ~50k utterances.
- **Specific Mentions of WorldSpeech or TBMM:** Neither the model cards nor the technical report mentions `disco-eth/WorldSpeech`, Turkish parliamentary debates, or TBMM audio in any section, table, or benchmark list.
- **Scope of Disclosure:** The published disclosure provides high-level volume aggregates across these stages, but does not provide an itemized dataset inventory, web crawl URLs, or per-language breakdowns for the Turkish speech used during AuT pretraining or multilingual SFT.

#### Candidate Analysis
- **Candidate 1 (`Qwen/Qwen3-ASR-1.7B-hf`, revision `bcd2b5b7f32b480ab5790554cfa8347f246a14f3`):**
  - *Confirmed by sources:* Apache 2.0 license; native support for Turkish (`tr`); 1.7B parameters; multi-stage AuT pretraining, Omni base, and multilingual SFT/GSPO.
  - *Remains unknown:* Specific audio corpora, collection dates, and web crawl sources comprising the Turkish training data; whether public TBMM recordings were ingested.
  - *Release timing:* Released January 28, 2026, postdating public TBMM recordings and the WorldSpeech release, so release timing does not preclude exposure.
- **Candidate 2 (`Qwen/Qwen3-ASR-0.6B-hf`, revision `7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c`):**
  - *Confirmed by sources:* Apache 2.0 license; native support for Turkish (`tr`); 0.6B parameters; multi-stage training pipeline described in arXiv:2601.21337.
  - *Remains unknown:* Specific Turkish audio sources and web crawl provenance across pretraining and SFT; whether TBMM audio was included.
  - *Release timing:* Released January 28, 2026.

#### Suggested Verification and Eligibility Finding
- **Suggested verification paths:**
  - Upstream publication by the model authors of an itemized manifest of Turkish training sources or URLs showing whether TBMM audio was included.
  - Empirical transcript collision or n-gram memorization checks against evaluation references, which can flag possible verbatim overlap (though cannot prove complete exclusion across 40M hours).
  - Evaluating on speech recorded strictly after the model release cutoff; this would constitute a separate, post-cutoff evaluation set rather than retroactively clearing the model on the current WorldSpeech benchmark.
- **Eligibility Finding:** Because public disclosures do not specify the Turkish training data sources, **neither Candidate 1 nor Candidate 2 is cleared by public disclosures**. This does not mean either checkpoint is known to be contaminated, but rather that contamination status remains unknown. In accordance with [docs/dataset.md](../docs/dataset.md#L55), this limitation must be reported. Neither candidate is recommended or selected on model size or latency alone, and both configuration files (`configs/baselines/qwen3_asr_1_7b.json` and `configs/baselines/qwen3_asr_0_6b.json`) remain unselected (`candidate_status: "unselected-candidate"`).

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

## Baseline Run and Handoff Checklist (B1, B2, B3)

This checklist organizes the remaining execution steps for Member 2 across four sequential phases, strictly respecting shared contracts, model boundaries, and handoff protocols.

### Phase 1: Offline Adapter and Exporter Tests (No Weights / No Data Downloads)
- [ ] **Contract Compliance in Adapter:**
  - Verify that adapter input handling strictly follows [docs/contracts/predictions.md](../docs/contracts/predictions.md) (allowing only `utterance_id`, `audio_uri`, `audio_sha256`, `duration_s`, `sample_rate_hz`, `channels`, and `language`; zero access to reference text).
  - Validate adapter wiring against pinned configurations in `configs/baselines/` without downloading weights.
- [ ] **Whisper Decoding & Scoring Tests (B1 & B2):**
  - Verify that the adapter uses core `GenerationMixin.generate` with forced prefix tokens `["<|startoftranscript|>", "<|tr|>", "<|transcribe|>", "<|notimestamps|>"]`.
  - Check that prompt tokens are excluded from returned token IDs and token counts.
  - Check that hypotheses preserve decoded text exactly as emitted, including leading whitespace under `raw-v1`.
  - Verify that `acoustic_log_score` is computed via teacher-forced log-probabilities on generated tokens and kept separate from framework `decoder_sequence_score`.
  - Ensure that if `decoder_sequence_score` is unavailable, it is preserved honestly as `null` with explanation in `score_details`.
  - Verify raw candidate output format and integration with shared export deduplication (`export_nbest_records`).
- [ ] **Qwen Decoding & Scoring Tests (B3):**
  - Verify that candidate sequence scores are labeled strictly as `joint_speech_language_beam_score` under convention `qwen3-asr-joint-sequence-score-v1` (no synthetic separation of acoustic-only scores).
  - Keep checkpoint selection between Candidate 1 (1.7B) and Candidate 2 (0.6B) pending.

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
