# TURAS — Project 3

**Turkish Unified Recognition Adaptation System** is a six-member university
project on domain-specific Turkish ASR, acoustic fine-tuning, and Relational Deep
Learning (RDL).

**Status on `main`: M1–M6 merged; M4/M5 frozen test study complete.** Guarded
Whisper LoRA/full and CTC training, scored N-best extraction, heterogeneous graph
learning, relational rescoring, M2 baseline adapters (B1/B2/B3), M3 baseline
configs (B4/B5), and the M6 evaluation/harness/smoke kit are merged on `main`
with offline tests passing in CI. On the 636-clip `turas-tr-econ-legis-v0.1`
test set, normalized WER is **24.512%** for the Whisper-small parent (`B1`),
**12.368%** for LoRA (`F1`), **10.371%** for full fine-tuning (`F2`),
**37.152%** for the CTC parent (`B4`), and **25.150%** for vocabulary-extended
CTC (`F3`). Primary learned RDL does **not** beat original one-best on any
adapted model (+0.005 / +0.058 / +0.039 pp normalized WER on F1/F2/F3, all 95%
CIs include 0), and on F2 the untrained random-graph control beats the learned
graph (`-0.024 pp`, 95% CI `[-0.049, -0.005]`). See the
[final report](results/final/M4_M5_FINAL_REPORT.md) and
[completion/handoff record](docs/m45_completion.md).

## For graders — Quickstart & Verification

```bash
# 1. Create a virtual environment (Python 3.11+ for packaging/lean tests; 3.13 used for CUDA runtime)
python -m venv .venv
# PowerShell: .\.venv\Scripts\Activate.ps1   |   bash: source .venv/bin/activate

# 2. Install package + lean test dependencies (no GPU/torch required)
python -m pip install --no-deps -e .
python -m pip install -r requirements/dev.txt

# 3. Run the offline test suite (1,100+ tests; torch/GPU tests skip cleanly without torch)
pytest
```

Verify that all 56 pinned Python code hashes in `configs/evaluation/m45_final_protocol.json`
match the repository and that all 20 systems in `results/final/m45_metrics.csv`
recompute exactly from the stored S/D/I counts in `results/final/m45_test_summary.json`:

```bash
python -c "import csv, hashlib, json, pathlib; p = json.loads(pathlib.Path('configs/evaluation/m45_final_protocol.json').read_text('utf-8')); bad = [k for k, v in p['critical_code_sha256'].items() if hashlib.sha256(pathlib.Path(k).read_bytes()).hexdigest() != v]; assert not bad, bad; s = json.loads(pathlib.Path('results/final/m45_test_summary.json').read_text('utf-8'))['systems']; rows = {r['system']: r for r in csv.DictReader(open('results/final/m45_metrics.csv', encoding='utf-8'))}; assert len(s) == len(rows) == 20; print('OK: 56 pinned code hashes and 20 system metric rows verified.')"
```

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
| B1 | Pretrained Whisper-small, Seq2Seq | Member 2 | Pinned parent validation/test and repeated timing complete (`m45_metrics.csv`); M2 adapter merged |
| B2 | Pretrained Whisper-large-v3, Seq2Seq | Member 2 | Adapter/config merged and smoke-checked on RTX 5070 Ti; full test-set run **not executed** |
| B3 | Pretrained Qwen3-ASR, speech-language-model family | Member 2 | 0.6B/1.7B adapters/configs merged and smoke-checked on RTX 5070 Ti; full test-set run **not executed** |
| B4 | Pretrained Turkish-capable Wav2Vec2/XLSR CTC | Member 3 | Pinned Turkish CV7 parent validation/test and repeated timing complete (`m45_metrics.csv`); M3 config merged |
| B5 | Alternative neural BiLSTM-CTC, trained from scratch | Member 3 | Protocol config merged (`configs/baselines/bilstm_ctc.json`); training **not executed** |
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
| `src/turkish_asr_rdl/` | Data curation, M2 baseline adapters, M4/M5 training/decoding/RDL, and M6 evaluation/schemas/harness |
| `configs/` | Dataset, baseline, fine-tuning, RDL, decoding, and frozen evaluation protocol configs |
| `docs/` | Architecture, working decisions, data/evaluation policies, contracts, smoke results, and audit records |
| `data/` | `turas-tr-econ-legis-v0.1` metadata, split ID lists, relational CSV tables, and verification audits |
| `scripts/` | Dataset curation (`m1_*`), baseline runners, M4/M5 training/evaluation pipelines, and `check_env.py` |
| `experiments/` | Preregistered protocol, member experiment notes, and M6 smoke-test kit |
| `results/` | Frozen final test report/metrics (`results/final/`), validation/ablation summaries, and audits |
| `tests/` | Offline synthetic-fixture, schema, evaluator, harness, and adapter unit tests |
| `contributions/` | Individual member contribution records |
| `presentation/` | Evidence requirements for the final presentation |
| `.github/` | PR template and CI (packaging, unit tests, and offline adaptation checks) |

## License and data rights

Source code is licensed under the **MIT License** — see [LICENSE](LICENSE).
The derived dataset `turas-tr-econ-legis-v0.1` originates from
`disco-eth/WorldSpeech` (`tr_tr`, **CC BY-NC 4.0**, non-commercial attribution
required); raw audio and private references are not redistributed in Git.

Repository: [Rxd3/TURAS](https://github.com/Rxd3/TURAS), public, renamed from
`Rxd3/project3-turkish-asr-rdl` (the old URL redirects). `main` is protected;
changes reach it through reviewed pull requests.
Architecture references are [NALTRA](https://github.com/Rxd3/NALTRA) and
[RECAP](https://github.com/Rxd3/RECAP); see the inspected revisions and adaptation
boundaries in [reference review](docs/reference_review.md). No unrelated
implementations or existing members' identities have been copied.
