# TURAS — Project 3

**Turkish Unified Recognition Adaptation System** is a six-member university
project on domain-specific Turkish ASR, acoustic fine-tuning, and Relational Deep
Learning (RDL).

**This is the Member 4 component branch of the completed M4/M5 integration
stack.** It contains guarded Whisper LoRA/full training, scored beam extraction
and offline tests. Actual training, selected checkpoints, frozen test results and
the verified private handoff are documented in
[integration PR #5](https://github.com/Rxd3/TURAS/pull/5). The table below describes
this component branch's contents; the integration branch carries final project
status and measured evidence. Independent review remains pending.

The five baseline tracks below are the original project roster. The professor's
later requirements also call for a GMM-HMM track and explicit HuBERT/transducer
language-support audits. That expanded baseline coverage remains pending with
Members 2/3/6; adaptation code does not establish baseline compliance.

## Scope

The project preserves all five baseline tracks and all adaptation tracks:

| ID | Track | Owner | Current state |
| --- | --- | --- | --- |
| B1 | Pretrained Whisper-small, Seq2Seq | Member 2 | Planned |
| B2 | Pretrained Whisper-large-v3, Seq2Seq | Member 2 | Planned |
| B3 | Pretrained Qwen3-ASR, speech-language-model family | Member 2 | Planned; checkpoint/capability audit required |
| B4 | Pretrained Turkish-capable Wav2Vec2/XLSR CTC | Member 3 | Planned; ASR checkpoint/head selection required |
| B5 | Alternative neural BiLSTM-CTC, trained from scratch | Member 3 | Working architecture choice; not implemented |
| F1 | Whisper LoRA fine-tuning | Member 4 | Code and offline tests here; completed experiments in integration PR #5 |
| F2 | Whisper full fine-tuning where hardware permits | Member 4 | Code and offline tests here; completed experiments in integration PR #5 |
| F3 | Wav2Vec2/XLSR CTC fine-tuning | Member 5 | Planned |
| R1 | Trainable heterogeneous graph relation scorer and N-best rescoring | Member 5 | Planned |

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
2. Read [CONTRIBUTING.md](CONTRIBUTING.md) for branch and PR rules.
3. Review [architecture](docs/architecture.md), [contracts](docs/contracts/README.md),
   and [working decisions](docs/decisions.md) before implementation.
4. Start the assigned `feat/m<N>-...` branch after identity/ownership assignment.

Python 3.11+ is the packaging target; ML compatibility is not yet verified.
An optional packaging-only setup is:

```text
python -m venv .venv
python -m pip install --no-deps -e .
```

Use the environment's Python after activating it (`.\.venv\Scripts\Activate.ps1`
in PowerShell or `source .venv/bin/activate` in bash). Build isolation may install
the lightweight setuptools backend. No runtime dependencies or command-line
entry points are declared. Importing `turkish_asr_rdl` only checks the empty
package layout; it does not recognize speech. Do not download data or models as
part of repository initialization.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/turkish_asr_rdl/` | Package boundaries and Member 4 training/decoding implementation |
| `configs/` | Separate dataset, baseline, fine-tuning, RDL, decoding, evaluation settings areas |
| `docs/` | Architecture, working decisions, data/evaluation policies, contracts, reproducibility |
| `data/` | Future metadata, manifests, split definitions, verification and relational schema documentation |
| `scripts/` | Lightweight entry points, Member 4 training and M6 environment checks |
| `experiments/` | Preregistered comparison and ablation protocol; future notes |
| `results/` | Future benchmark outputs and reports; no results |
| `tests/` | Offline synthetic-fixture, random-model and environment checks; no benchmark data |
| `contributions/` | Six individual, initially empty contribution records |
| `presentation/` | Evidence requirements for the future final presentation |
| `.github/` | PR template and CI (packaging checks and unit tests) |

Repository: [Rxd3/TURAS](https://github.com/Rxd3/TURAS).
Architecture references are [NALTRA](https://github.com/Rxd3/NALTRA) and
[RECAP](https://github.com/Rxd3/RECAP); see the inspected revisions and adaptation
boundaries in [reference review](docs/reference_review.md). No unrelated
implementations or existing members' identities have been copied.
