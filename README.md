# TURAS — Project 3

**Turkish Unified Recognition Adaptation System** is a six-member university
project on domain-specific Turkish automatic speech recognition (ASR), acoustic
fine-tuning, and Relational Deep Learning (RDL) for legislative and economic speech.

**[Start with the submission guide](SUBMISSION.md)** for the presentation, final
report, verification commands, and remaining course requirements.

## Results at a glance

On the frozen 636-clip working test set, Whisper-small normalized word error
rate (WER) fell from **24.512%** to **12.368%** with LoRA and **10.371%** with full
fine-tuning. XLS-R CTC went from **37.152%** to **25.150%** after fine-tuning
with a train-derived vocabulary extension. Learned RDL rescoring yielded no
improvement over original one-best on any of the three adapted models.
[Final report](results/final/M4_M5_FINAL_REPORT.md)

The frozen M4/M5 working study and aggregate results are on `main`. B2 and B3
have validation and test scores outside the frozen protocol (below). The
2026-11-22 closeout adds B5 BiLSTM-CTC (47.776% normalized test WER) and the
supplied single-Gaussian character HMM (99.995%); their exports and metrics are
validated. Seda Turkish RNN-T and a four-mixture grapheme GMM-HMM have full
validation/test exports; their repeated timing and final reports are in
progress. Reference adjudication and independent sign-off remain pending.
[Follow-up protocol](docs/baseline_completion_followup.md) ·
[Submission status](SUBMISSION.md#scope-and-remaining-work) ·
[HuBERT and transducer support audit](docs/asr_language_support_audit.md)

## Additional baselines (M6 scorer, outside the frozen protocol)

| System | Test normalized WER % | Raw WER % | Normalized CER % | Term F1 (micro, 0–1) | End-to-end RTF vs B1, same session |
| --- | --- | --- | --- | --- | --- |
| B2 Whisper-large-v3 (official) | 12.3192 | 24.5443 | 5.4473 | 0.9776 | 3.05x |
| B2, Member 2 reproduction | 12.3922 | 24.6039 | 5.4391 | 0.9773 | not in this session |
| B3 Qwen3-ASR-1.7B | 16.0943 | 27.5937 | 5.6005 | 0.9123 | 2.40x |

B2 is zero-shot `openai/whisper-large-v3`, decoded like B1; B3 was selected over
Qwen3-ASR-0.6B on validation. Normalized WER differences: reproduction - B2 =
+0.0731 pp (paired recording-group 95% interval [-0.0532, 0.2277]); B3 - B2 =
+3.7751 pp [2.6238, 4.9312]. The test references were kept by CER <= 0.25
agreement with an upstream ASR transcript, whose `tr_tr` model is not confirmed
(`whisper-large-v3-turbo` is the WorldSpeech paper's most common alignment
model); this filter can favour Whisper-family systems such as B2 over B3
(`data/metadata/turas-tr-econ-legis-v0.1.json` line 83,
`experiments/m2-baseline-notes.md` line 45). The RTFs come from one RTX 5070 Ti laptop session
and are not comparable to the frozen-block RTFs in `results/final`; compare them
with frozen systems only through the B1 ratio shown here.

On test, zero-shot B2 (12.32% normalized WER) is level with F1 LoRA (12.37%);
F2 full fine-tuning (10.37%) is lower by about 2 pp. F1 is better on raw WER
(21.50% vs 24.54%) and runs at about B1's RTF, about 3x faster than B2. These
comparisons are unpaired and cross scorers: B2 is scored by the M6 scorer, whose
B1 parity is pending, and F1/F2 by the frozen scorer. There is no paired
interval against F1 or F2 yet: their per-utterance outputs are in the private
M4/M5 handoff bundle, which M6 has not received, and the B1 parity check needs
the same bundle. 611 of the 636 test references are unreviewed edited TBMM
minutes, the style F1 and F2 are fine-tuned on. [B2 and B3 results](results/m6/README.md)

## Repository map

| Path | Contents |
| --- | --- |
| [`src/turkish_asr_rdl/`](src/turkish_asr_rdl/) | Data curation, baseline adapters, acoustic training, decoding, RDL and evaluation |
| [`configs/`](configs/) | Baseline, adaptation, decoding and evaluation settings |
| [`data/`](data/) | Working benchmark metadata, frozen split IDs, static relational tables and verification audits |
| [`scripts/`](scripts/) | Curation, training, evaluation and verification entry points |
| [`experiments/`](experiments/) | Experiment protocols, workstream notes and the smoke kit |
| [`results/`](results/) | Frozen test results, B2/B3 scores, validation and ablation summaries, and audits |
| [`tests/`](tests/) | Offline unit and integration tests |
| [`docs/`](docs/) | Evidence, reproduction, architecture and contracts |
| [`presentation/`](presentation/) | Project presentation and source outline |

See [team tasks](TEAM_TASKS.md), [contribution records](contributions/README.md),
[contributing](CONTRIBUTING.md), and [agent guidance](AGENTS.md) for ownership and
repository workflow.

## License and data rights

The source code uses the MIT License. The working dataset is derived from
`disco-eth/WorldSpeech` (`tr_tr`, CC BY-NC 4.0). The non-commercial attribution
terms and each pretrained model's license apply. Audio and private reference
transcripts are held outside Git. [License](LICENSE) · [Third-party notice](NOTICE)

Architecture references and adaptation boundaries are documented for
NALTRA and RECAP. [Reference review](docs/reference_review.md)

<!-- TURAS_ARTIFACT_LINKS_START -->
## Reproduction downloads

Drive upload is pending. Use the [organized package map](docs/drive_packages.md),
[reproduction guide](docs/final_reproduction.md), and
[final review checklist](docs/final_review_2026-11-22.md). After uploading the
public submission folder, run `python scripts/update_artifact_links.py --public-drive-url "ACTUAL_FOLDER_URL"`.
<!-- TURAS_ARTIFACT_LINKS_END -->
