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

The frozen working study and aggregate results are on `main`. Additional
completed runs cover B2 Whisper-large-v3 (**12.387%** normalized test WER),
B3 Qwen3-ASR-1.7B (**16.172%**), B5 BiLSTM-CTC (**47.776%**), Seda Turkish
RNN-T (**29.958%**) and a four-mixture grapheme HMM (**99.971%**). These archived
runs have full frozen evaluation and three-repeat timing; their scores were
rechecked with the current shared evaluator. They are separate follow-up
experiments, with their original runtime source preserved in artifact packages.
Reference review and independent sign-off remain pending.
[Completed baseline results](results/final/BASELINE_COMPLETION_REPORT.md) ·
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
