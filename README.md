# TURAS — Project 3

**Turkish Unified Recognition Adaptation System** is a six-member university
project on domain-specific Turkish ASR, acoustic fine-tuning, and Relational Deep
Learning (RDL).

**Status on this feature branch: owner-directed M1/M4/M5/M6 integration.** Guarded
Whisper LoRA/full and CTC training, scored N-best extraction, heterogeneous graph
learning and relational rescoring are implemented with offline tests. Seed-17
LoRA, full Whisper and vocabulary-extended CTC training have completed, together
with matched parent validation, checkpoint selection and corrected-linker RDL
ablations. The frozen final working benchmark and private model/run handoff are
complete. On 636 test clips, normalized WER is 24.512% for the Whisper parent,
12.368% for LoRA, 10.371% for full tuning, 37.152% for the CTC parent and 25.150%
for vocabulary-extended CTC. Primary learned RDL does not beat original one-best
on any adapted model. See the [final report](results/final/M4_M5_FINAL_REPORT.md)
and [completion/handoff record](docs/m45_completion.md). Independent review is
still pending; this does not complete other members' baseline tracks.

The [technical closeout](results/final/m45_technical_closeout.json) records
conflict-free PR heads, passing CI, restored-checkpoint acceptance and the final
source supplement. The [decoder erratum](docs/m45_decoding_erratum.md) confirms
actual Whisper beam 5 and CTC prefix width 10; frozen outputs/metrics are unchanged.
Technical choices are approved in [the owner decision record](docs/owner_decisions.md).
See the actual [working validation](docs/m45_validation_complete.md),
[independent numerical/linker audit](docs/m45_result_audit.md) and
[frozen final protocol](configs/evaluation/m45_final_protocol.json).
See [Member 4 notes](experiments/m4-whisper-notes.md) and
[Member 5 notes](experiments/m5-ctc-rdl-notes.md).

Private audio/train inputs have been hash verified. Validation uses the explicitly
versioned recovered upstream minutes, which omit M1's private corrections. The
working benchmark is not a fully listening-verified verbatim or proven
speaker-disjoint release, and independent M6 approval is not asserted. The
[readiness record](docs/m45_readiness.md) preserves earlier handoff history;
[GPU probes](docs/m45_gpu_profile.md) are synthetic, while the dated run reports
contain real corpus evidence.

The five baseline tracks below are the original project roster. The professor's
later requirements also call for a GMM-HMM track and explicit HuBERT/transducer
language-support audits. That expanded baseline coverage remains pending with
Members 2/3/6; adaptation code does not establish baseline compliance.

## Scope

The project preserves all five baseline tracks and all adaptation tracks:

| ID | Track | Owner | Current state |
| --- | --- | --- | --- |
| B1 | Pretrained Whisper-small, Seq2Seq | Member 2 | Owner-run pinned parent validation/test and repeated timing complete; peer review pending |
| B2 | Pretrained Whisper-large-v3, Seq2Seq | Member 2 | Planned |
| B3 | Pretrained Qwen3-ASR, speech-language-model family | Member 2 | Planned; checkpoint/capability audit required |
| B4 | Pretrained Turkish-capable Wav2Vec2/XLSR CTC | Member 3 | Owner-run pinned Turkish CV7 parent validation/test and repeated timing complete; peer review pending |
| B5 | Alternative neural BiLSTM-CTC, trained from scratch | Member 3 | Working architecture choice; not implemented |
| F1 | Whisper LoRA fine-tuning | Member 4 | Selected epoch 3; final working test, terminology, repeated timing and private handoff complete |
| F2 | Whisper full fine-tuning where hardware permits | Member 4 | Selected epoch 2; final working test, terminology, repeated timing and private handoff complete |
| F3 | Wav2Vec2/XLSR CTC fine-tuning | Member 5 | Vocabulary extension declared; selected epoch 3; final working evaluation and private handoff complete |
| R1 | Trainable heterogeneous graph relation scorer and N-best rescoring | Member 5 | Learned graph, retrained ablations, random control and final working test/timing complete; primary learned RDL has no gain over original one-best |

The dataset contribution is a **curated Turkish legislative/financial speech
benchmark**. [WorldSpeech](https://huggingface.co/datasets/disco-eth/WorldSpeech),
configuration `tr_tr`, is the initial primary candidate. Selection, source terms,
domain coverage, local human verification, and release approval remain open.
Upstream transcripts or splits do not automatically constitute a TURAS release.

The intended work includes provenance-aware curation, verified references,
leakage-resistant frozen splits, relational CSV tables, learned heterogeneous
graph scoring, model-scored N-best extraction, acoustic/optional LM/relational
rescoring, matched baseline/fine-tuned/RDL comparisons, raw and normalized WER
and CER, domain terminology metrics, RTF, latency, and controlled ablations.

## Start here

1. Read [TEAM_TASKS.md](TEAM_TASKS.md) for ownership, exact handoffs, and next tasks.
2. Read [CONTRIBUTING.md](CONTRIBUTING.md) for branch and PR rules, and
   [AGENTS.md](AGENTS.md) for the six rules every coding agent follows.
3. Review [architecture](docs/architecture.md), [contracts](docs/contracts/README.md),
   and [working decisions](docs/decisions.md) before implementation.
4. Start the assigned `feat/m<N>-...` branch after identity/ownership assignment.

Python 3.11+ is the packaging target. Actual Windows execution used Python 3.13.14
and the [pinned CUDA runtime](requirements/owner-m45-windows-cu130.txt).
An optional packaging-only setup is:

```text
python -m venv .venv
python -m pip install --no-deps -e .
```

Use the environment's Python after activating it (`.\.venv\Scripts\Activate.ps1`
in PowerShell or `source .venv/bin/activate` in bash). Build isolation may install
the lightweight setuptools backend. No runtime dependencies or command-line
entry points are required for a packaging-only check. Training and inference
require the separate pinned runtime and private inputs. Importing the package
does not automatically load models, download data or recognize speech.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/turkish_asr_rdl/` | Package boundaries and Members 4/5 training/decoding/RDL implementations |
| `configs/` | Separate dataset, baseline, fine-tuning, RDL, decoding, evaluation settings areas |
| `docs/` | Architecture, working decisions, data/evaluation policies, contracts, reproducibility |
| `data/` | Future metadata, manifests, split definitions, verification and relational schema documentation |
| `scripts/` | Lightweight entry points, training/rescoring and M6 environment checks |
| `experiments/` | Preregistered comparison and ablation protocol; future notes |
| `results/` | Actual training, graph, working validation and audit summaries; final reports when completed |
| `tests/` | Offline synthetic-fixture and random-model tests; no benchmark data |
| `contributions/` | Individual contribution records and owner-assisted integration evidence |
| `presentation/` | Evidence requirements for the future final presentation |
| `.github/` | PR template and CI (packaging checks and unit tests) |

Repository: [Rxd3/TURAS](https://github.com/Rxd3/TURAS), public, renamed from
`Rxd3/project3-turkish-asr-rdl` (the old URL redirects). `main` is protected;
changes reach it through reviewed pull requests.
Architecture references are [NALTRA](https://github.com/Rxd3/NALTRA) and
[RECAP](https://github.com/Rxd3/RECAP); see the inspected revisions and adaptation
boundaries in [reference review](docs/reference_review.md). No unrelated
implementations or existing members' identities have been copied.
