# Remaining baseline experiments — 2026-11-22 follow-up

The project owner explicitly requested B2/B3 completion, an eligible Turkish
transducer, a genuine mixture-Gaussian HMM, fresh-environment reproduction and
PR-ready submission artifacts. These are separate follow-up experiments after
earlier working-test exposure. They do not rewrite the frozen M4/M5 study or
claim blind preregistration, independent review or adjudicated references.

The fixed plan is [completion_followup_2026-11-22.json](../configs/evaluation/completion_followup_2026-11-22.json).
Each inference run additionally freezes its model/decoder hashes and environment
before reading audio. Inference receives no references. Shared validators and
the shared evaluator govern every full-ID export and metric.

## Candidate audit

| Track | Pinned candidate and evidence | Experiment |
| --- | --- | --- |
| B2 | `openai/whisper-large-v3` at `06f233fe06e710322aca913c1bc4249a0d71fce1`; Turkish listed, Apache-2.0. [Publisher card](https://huggingface.co/openai/whisper-large-v3/blob/06f233fe06e710322aca913c1bc4249a0d71fce1/README.md) | Re-run locally because the reported M2 ZIP is absent. bf16, Turkish prompt, beam/N=5 and max 440 generated tokens, using the existing core GenerationMixin implementation. |
| B3 | `Qwen/Qwen3-ASR-1.7B-hf` at `bcd2b5b7f32b480ab5790554cfa8347f246a14f3`, and `Qwen/Qwen3-ASR-0.6B-hf` at `7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c`; Turkish and native Transformers support listed. Apache-2.0. [Publisher card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf/blob/bcd2b5b7f32b480ab5790554cfa8347f246a14f3/README.md) | Evaluate both candidates on all 438 validation IDs. Select minimum normalized CER, then WER, then stable key. Run only the selected candidate on test. No hotwords or external context. Joint speech-language scores remain labeled joint. |
| Transducer | `atasoglu/seda-v0.1` at `5cbbda1b34bde5e3004e79b137dc3871587ea282`. Turkish Zipformer2 RNN-T, int8 encoder/joiner and fp32 decoder; publisher names Common Voice 27 and ISSAI acoustic data, not WorldSpeech. Acoustic files list Apache-2.0. [Publisher card](https://huggingface.co/atasoglu/seda-v0.1/blob/5cbbda1b34bde5e3004e79b137dc3871587ea282/README.md) | sherpa-onnx 2.0.2 CPU, one thread, greedy decoding. Optional Wikipedia-trained LM files are not downloaded or used. Raw spoken-form output; no inverse text normalization. |

Publisher disclosures do not supply a complete clip-level training manifest or
prove recording/speaker disjointness. Qwen's large-scale training corpus/cutoff
is insufficiently disclosed to certify zero TURAS exposure. The transducer's
ISSAI sources include some YouTube speech. Report these uncertainties rather
than treating model availability or a publication date as a contamination audit.
The explicitly WorldSpeech-trained transducer in the earlier support audit
remains excluded. No new HuBERT transcription head is implied by this follow-up.

## Completed validation selection

Both Qwen candidates completed all 438 frozen validation IDs with zero inference errors and checked separate predictions/N-best exports.

| Candidate | Validation norm. WER | Validation norm. CER | Selected for follow-up test |
| --- | ---: | ---: | --- |
| Qwen3-ASR-1.7B | 15.343% | 5.423% | yes |
| Qwen3-ASR-0.6B | 22.345% | 7.067% | no |

The minimum-CER rule selected **1.7B** at `2026-11-22T15:38:54.554467+00:00`, before its follow-up test inference. The private `b3-selection.json` freezes both validation-report hashes; it is included in the B3 package. The earlier working-test exposure remains disclosed, and no independent peer approval is inferred.

## Genuine GMM-HMM

The new model has four diagonal Gaussian components per each of three grapheme
states. It uses conventional 13-dimensional MFCCs plus delta and acceleration,
25 ms windows, 10 ms hop and per-utterance CMVN. Initialization uses train-only
uniform alignments and diagonal Gaussian mixtures. Five fixed Viterbi-EM passes
update state alignments and soft component responsibilities. State transitions
have normalized stay/advance probabilities, and character priors use train only.

This is a genuine **mixture-Gaussian grapheme HMM**, with no external LM or
pronunciation lexicon. It is distinct from a conventional phoneme/lexicon speech
recognizer and from the earlier supplied single-Gaussian/DCT baseline. Preserve
that earlier negative result. The component count and five-pass budget are fixed
before validation scoring; no choice uses test scores.

## Execution and reproduction

`scripts/download_completion_models.py` downloads pinned snapshots and verifies
weight hashes. `scripts/run_completion_baseline.py` runs every model through
`run_timed`, reading 16 kHz mono audio inside the measured region. Prediction
and N-best JSONL stay separate. Models lacking supported scored N-best declare
`requested_n=0`.

`scripts/score_completion_study.py` joins references offline, computes shared
raw/normalized edit counts, terminology and recording-bootstrap intervals, and
cross-checks WER/CER with JiWER. Test scoring requires completed accuracy and
repeated-timing exports. The dispatcher runs headline timings serially after
download/training processes finish.

New isolated Python environments install the pinned CUDA lock and the separate
[CPU transducer requirements](../requirements/transducer-windows.txt). A restored
B5 checkpoint has already executed in the newly installed CUDA environment from
archived source and restored audio, matching three archived validation outputs
exactly. This is a fresh Python-environment check on the same machine, not a
second person's review or a new operating-system installation.

The new environment also restored the original M4/M5 archive, verified all 226
indexed members and executed three acoustic models, three online joint graph
runs and 15 cached controls on two validation clips. All 22 child checks passed,
including import origin and joint/cached transcript parity. No references or
test audio were read. The [restoration evidence](../results/final/fresh_environment_restoration.json)
records the archive/source/environment identity and the scope of this subset check.

Private progress, models, logs and reproduction evidence remain under
`artifacts/completion-2026-11-22/`. Public reports and the two-minute presentation
will be refreshed after the actual runs finish. The previous upload delivery
remains a preserved snapshot until a new version is prepared.
