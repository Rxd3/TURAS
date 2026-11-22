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

The M4/M5 working study is on `main`. The 2026-11-22 local closeout adds B5
BiLSTM-CTC (47.776% normalized test WER) and the supplied single-Gaussian
character HMM (99.995%). Their exports and metrics are validated; local changes
still need repository review/publication. A separate follow-up has completed
local Whisper-large validation/test exports and validation of both Qwen candidates; Qwen
1.7B was selected by the fixed minimum-CER rule. Full validation/test exports
are also available for Seda RNN-T and a genuine four-mixture grapheme HMM.
The selected-Qwen test run, repeated timing and final follow-up reports are
in progress. Reference adjudication and independent sign-off remain pending.
[Follow-up protocol and validation selection](docs/baseline_completion_followup.md) ·
[Submission status](SUBMISSION.md#scope-and-remaining-work) ·
[HuBERT and transducer support audit](docs/asr_language_support_audit.md)

## Repository map

| Path | Contents |
| --- | --- |
| [`src/turkish_asr_rdl/`](src/turkish_asr_rdl/) | Data curation, baseline adapters, acoustic training, decoding, RDL and evaluation |
| [`configs/`](configs/) | Baseline, adaptation, decoding and evaluation settings |
| [`data/`](data/) | Working benchmark metadata, frozen split IDs, static relational tables and verification audits |
| [`scripts/`](scripts/) | Curation, training, evaluation and verification entry points |
| [`experiments/`](experiments/) | Experiment protocols, workstream notes and the smoke kit |
| [`results/`](results/) | Frozen test results, validation and ablation summaries, and audits |
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
