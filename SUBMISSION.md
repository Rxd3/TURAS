# TURAS Project 3 submission

TURAS studies acoustic adaptation and relational rescoring for Turkish
legislative and economic speech. The completed M4/M5 study evaluates 20 systems
and rescoring conditions on the same frozen 636 clips against working references.

## Materials for grading

| Material | What it covers |
| --- | --- |
| [Two-minute presentation](presentation/TURAS-2-minute.pptx) and [full deck](presentation/TURAS.pptx) | Four timed slides for delivery and 12 slides for questions |
| [Final report](results/final/M4_M5_FINAL_REPORT.md) | Raw and normalized WER/CER, terminology, timing, controls and paired uncertainty |
| [Metrics CSV](results/final/m45_metrics.csv) and [evaluation summary](results/final/m45_test_summary.json) | Per-system aggregates, edit counts and evaluation provenance |
| [Error analysis](docs/error_analysis.md) | Error profiles, terminology and quantity diagnostics, and the RDL null result |
| [Frozen protocol](configs/evaluation/m45_final_protocol.json) and [decoder erratum](docs/m45_decoding_erratum.md) | Pre-test protocol, effective decoder settings, and validation-only selection |
| [Dataset handoff](docs/m1_deadline_handoff.md) and [contribution records](contributions/README.md) | Working reference status, artifact boundaries and team ownership |

| Native comparison | Parent normalized WER | Adapted normalized WER |
| --- | ---: | ---: |
| Whisper-small → LoRA | 24.512% | 12.368% |
| Whisper-small → full tuning | 24.512% | 10.371% |
| XLS-R Turkish CV7 → fine-tuning with vocabulary extension | 37.152% | 25.150% |

Each paired recording-group confidence interval for these three gains lies below
zero. Learned RDL changed normalized WER by +0.005, +0.058 and +0.039 percentage
points for LoRA, full Whisper and adapted CTC, respectively. All three 95%
confidence intervals include zero. The study uses a single training seed.
[Final report](results/final/M4_M5_FINAL_REPORT.md)

## Verify from a fresh clone

The first command uses only the Python standard library. The full public offline
suite uses Python 3.11+ and the lean development dependencies. Run from the
repository root.

```text
python scripts/verify_submission.py
python -m venv .venv
```

Activate with `.\.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` in bash, then run:

```text
python -m pip install --no-deps -e .
python -m pip install -r requirements/dev.txt
python -m pytest -q
```

The verifier checks the frozen protocol, split files, and all 56 pinned code
hashes. It recomputes published WER/CER from stored edit counts for all 20
conditions and checks split-ID uniqueness and disjointness across 5,450 clips.
The lean suite skips tests whose optional dependencies are unavailable. [Verifier](scripts/verify_submission.py)

## Reproduce with authorized artifacts

Full benchmark reproduction uses the pinned Windows CUDA runtime, the private
model and run archive, and separate audio and evaluator-only references. Ask the
project team for the appropriate bundles and access. The documented restore
exercised three acoustic systems, three joint systems and 15 cached controls on
validation audio in a fresh checkout using the existing runtime. Independent
clean-environment reproduction, non-author review and consumer acknowledgment
remain pending.
[Reproduction guide](docs/m45_final_reproduction.md) ·
[Delivery acceptance](docs/m45_delivery_acceptance.md)

## Scope and remaining work

| Track | Evidence and status |
| --- | --- |
| B1 (Whisper-small) and B4 (XLS-R Turkish CV7) | Full matched test evaluation and repeated timing |
| B2 (Whisper-large-v3) and B3 (Qwen3-ASR) | Adapters and smoke evidence. Full frozen test runs and official repeated timing are pending |
| B5 (BiLSTM-CTC) | Model, adapter and configuration with a synthetic three-step CPU probe and checkpoint reload check. Corpus training and frozen evaluation are pending |
| F1–F3 (Whisper LoRA, full Whisper and adapted CTC) and R1 (learned relational rescoring) | Complete working test results, adapted checkpoints, graph controls and timing |
| Additional course requirements | Turkish support audited from published HuBERT and transducer sources. A matched transducer run and the GMM-HMM baseline are pending. [Support audit](docs/asr_language_support_audit.md) |

M6's dated GPU smoke report covers a reference-free sample of 20 timed clips.
The four beam systems passed N-best validation after shared-export deduplication.
The matched results above come from the frozen test protocol, with separate
three-repeat timing for eight conditions. [Smoke report](docs/m6-smoke-results.md) ·
[Final report](results/final/M4_M5_FINAL_REPORT.md)

The test references are edited parliamentary minutes. The working set contains
611 unreviewed test clips and 25 clips with historical single-reviewer tags.
Recording groups are disjoint across splits, and speaker disjointness remains
unproven. The recovered validation minutes omit three private reference
corrections. All 249 primary decisions for the separate review plan have been
recorded. Independent adjudication of 55 items and release approval are pending. [Dataset handoff](docs/m1_deadline_handoff.md) ·
[Completion record](docs/m45_completion.md)
