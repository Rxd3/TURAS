# Project-owner decisions for Members 4 and 5

Effective 2026-11-19. Authority: the project owner (the user in this session), who
explicitly delegated technical choices and asked that decisions be recorded for
the team. No personal name/GitHub identity or other member's approval is inferred.

The owner instructed: "let me be the source of truth and do it as you see fit
best" and requested a reusable record. This record supplies project approval for
the experimental protocols below. It does **not** assert independent code review,
successful training, baseline completion, hardware feasibility or validated gains.
Existing draft PRs and main-branch protection remain separate from these choices.

## Approved choices

| ID | Decision | Implementation / reuse |
| --- | --- | --- |
| D22 | Project owner delegates M4/M5 technical decisions; covered choices do not need repeated inter-member agreement | This document and each config's `approval` object |
| D23 | Primary Whisper parent is multilingual `openai/whisper-small` at `973afd24965f72e36ca33b3055d56a652f456b4d`; primary CTC parent is `mpoyraz/wav2vec2-xls-r-300m-cv7-turkish` at `708639f50559d7970f462e13ec64d3f059ca89f6` | Pinned file hashes in the fine-tuning profiles; corresponding B1/B4 must use exactly these parents |
| D24 | Adopt M6 contract 0.2 text/data/prediction/N-best/evaluation semantics for this integration | Use shared normalizer/scorer, shared export deduplication and `run_timed`; no competing implementations |
| D25 | CTC labels use the shared `tr-normalized-v1` spelling view, unchanged pretrained head/vocabulary, and strict OOV auditing | Raw references stay immutable. Do not invent spoken numeral readings, delete unknown targets, drop clips or silently resize the head. Audit actual train data before downloading weights; any later target/vocabulary experiment is a new documented ablation |
| D26 | Train the typed heterogeneous mean-message-passing model with learned edge scoring; use only train/external provenance | M1 static hashes and train-table hashes are pinned. Reconstruct aggregate co-mentions before training-group holdout; exclude repeated known facts from novel-edge validation. Weak-label and closed-world assumptions remain explicit |
| D27 | LM-disabled acoustic/RDL fusion is the primary comparison; use identical candidate artifacts | Candidate validation weights gamma = 0, 0.1, 0.3, 1.0 with alpha=1, beta=0 and acoustic exponent=1; missing probability=.1. These are a search space, not selected weights. Freeze final values only after actual validation scores |
| D28 | Start acoustic experiments at seed 17, batch one, accumulation 16, three epochs; LoRA LR=1e-4 and full/CTC LR=1e-5 | LoRA rank16/alpha32/dropout.05; q/v targets. Full Whisper tuning remains mandatory where measured hardware permits. Seeds 42/2026 remain later repetitions, not deleted from scope |
| D29 | Baseline evidence is a scientific prerequisite, not an approval vote | Local code can run/check the pinned unmodified parents once audio arrives. Never set a completed baseline flag without real outputs. Initial baseline comparisons use validation; final test scoring waits for the common freeze |
| D30 | Integrate M1/M6 branches locally while preserving original commits; test the combined pipeline before main changes | Integration branch is separate; no teammate's independent review/ACK is invented and no draft PR is automatically merged |
| D31 | Use the verified local Windows CUDA13.0/PyTorch2.14.1 build for M4/M5 execution; retain exact installed package versions | [Runtime lock](../requirements/owner-m45-windows-cu130.txt) and [measured synthetic probes](m45_gpu_profile.md); these do not replace real-data validation |
| D32 | Preserve genuine unmodified-parent predictions for all 438 validation clips before acoustic optimization; permit the fixed three-epoch budget while validation references are unavailable | This clarifies D29: `status=completed, scope=validation_inference_only` means inference completed, with `metrics_status=pending_validation_references`. It does not assert a completed accuracy evaluation. Use the identical decoder for later adapted comparisons; archive every epoch and postpone checkpoint/fusion selection and all test inference |
| D33 | Add a separate `ctc_vocab_extension` contribution because the real train-only audit found OOVs in 3,232/4,376 references | Append the 14 audited characters `+0123456789âîû`, preserving every existing token ID, blank ID, output weight/bias row and the pretrained encoder. Initialize only new rows from the parent's initializer distribution. Keep digits and written references, retain all clips, derive the vocabulary from train only, and save the extended processor with each checkpoint. The original strict-head profile remains blocked; compare this clearly labeled extension against the unmodified parent, with normalization unchanged |
| D35 | Owner-authorized recovery of a separate preliminary validation-minutes reference version from pinned WorldSpeech human text | Use exactly the 438 frozen validation IDs and existing audio hashes, with immutable upstream revision `b76b85395039f50332fbc421f7c098ee2f2653d7`. Keep the recovery private and separately hashed as `owner-recovered-validation-minutes-v1`; never overwrite or represent it as M1's exact corrected reference bundle. Mark every recovered row `unreviewed`, retain the actual `human_transcript` unchanged, and never use `asr_transcript` or predictions as gold. M1's three private validation corrections and listening records cannot be reconstructed from owner approval. Preliminary scoring may use this explicitly identified reference version consistently for all systems; final benchmark claims must state its source/quality and resolve the separate release review. No test inference or training on recovered validation text is authorized by this recovery |
| D36 | The completion study may be reported as single-seed, consistent with the adopted M6 deadline policy | Seed 17 has actual evidence; seeds 42/2026 remain optional follow-up robustness runs, not an invented requirement that blocks the current report. No multi-seed average or robustness claim is permitted |
| D37 | Correct the RDL hypothesis linker to honor M1's frozen alias syntax | The audit found that the legacy linker discarded `*`, losing suffix matches and exclusion semantics. New versioned graph bundles reuse the same learned weights/snapshots but embed the hash-verified frozen lexicon and use M1's existing raw-text matcher. Preserve old artifacts/legacy behavior and rerun all affected validation grids. Acoustic checkpoints and their WER/CER are unaffected. Never treat legacy-linker RDL outcomes as corrected-linker evidence |

The graph prototype uses hidden size64, two layers, dropout.1, LR1e-3 and20
epochs, with an internal 20% **training-group** relation holdout. These are an
approved starting budget; their effectiveness is unknown. Removing a relation
family requires retraining a separately hashed graph. Hypothesis quantity pooling
is explicitly an approximation and must be evaluated, not assumed correct.

## Data and validation authority

Use `turas-tr-econ-legis-v0.1` as a working experimental candidate, conditional on
receipt and verification of private bundles matching split manifest SHA-256
`a8fff65b99fcf11d9452685e438e250207032ecfe11152741562147126d095b8`.
Public split lists/static metadata are now integrated and their hashes can be
verified here. They do not replace private training audio/references.

M1 reports sampled listening verification and recording-based grouping because
speaker identities are unknown. Preserve those limitations; do not relabel the
release fully human-verified or speaker-disjoint. Project approval cannot change
the professor's requirements or establish source/redistribution rights.

The owner may run the shared evaluator locally on a **separately provided
validation-only reference file**. That does not authorize training code to read
it. No final-test reference is needed now. Until validation references or genuine
M6 validation reports arrive, checkpoint/fusion selection remains pending.

M1's supplied Drive links on 2026-11-19 include inference/audio and train gold,
but no validation-only gold. D32 permits fixed-budget optimization only after
actual matching parent validation inference is archived. Training still reads
train gold only; a pending-metrics report never substitutes for checkpoint
selection or the final baseline-versus-contribution comparison. D33 changes
output-head size explicitly, so results must not be labeled unchanged-head CTC.
The tokenizer and CTC-head contract is documented in the official
[Transformers tokenizer documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer)
and [Wav2Vec2 documentation](https://huggingface.co/docs/transformers/model_doc/wav2vec2).

## Things approval cannot replace

- Private inference/audio and train-reference bundles with matching hashes.
- Actual baseline outputs, actual optimization runs, actual validation reports.
- Real-data throughput and memory validation. CUDA execution and five synthetic
  resource probes now pass locally; no corpus training duration has been measured.
- Disclosure of reference noise, unknown speakers, upstream/model contamination.
- Required course baseline-family coverage: the original five-track roster alone
  is insufficient. Keep GMM-HMM and HuBERT/transducer support audits in the plan;
  unsupported checkpoints need official-card exclusions, not wrong-language runs.
- Honest contribution attribution and independent PR-review history.

No completed experiment or test metric is created by this decision. Record a
dated amendment and evidence for changed parents, labels, data, graph access,
normalization or score semantics; do not change frozen artifacts in place.

## What teammates should use

Use the integration branch's five profiles under `configs/fine-tuning/`,
`configs/rdl/heterogeneous_relations.json` and
`configs/decoding/rdl_rescoring.json`. `protocol_approved=true` records owner
approval, while `baseline_evidence_required=true` still requires a genuine report.
Missing file paths/hashes and `fusion_frozen=false` are factual pending work.

Use `AdaptedAsrAdapter` and `RelationalAsrAdapter` through the shared M6
`run_timed` harness for real inference. Shared export preserves component scores
and selected candidate IDs and owns deduplication. Offline rescoring produces an
untimed `.run.json` provenance sidecar, not an independently timed ASR benchmark.

The earlier M4/M5 notes retain their original pre-delegation review requests as
history. For decisions and interface status, this record and the integration
branch supersede those requests. This does not claim any reviewer has signed off.
