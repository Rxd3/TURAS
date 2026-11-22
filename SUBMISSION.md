# TURAS Project 3 submission

TURAS studies acoustic adaptation and relational rescoring for Turkish
legislative and economic speech. The completed M4/M5 study evaluates 20 systems
and rescoring conditions on the same frozen 636 clips against working references.

## Materials for grading

| Material | What it covers |
| --- | --- |
| [Final two-minute presentation](presentation/TURAS-final-2-minute.pptx) and [full deck](presentation/TURAS.pptx) | Four timed slides for delivery and 12 slides for questions |
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
reproduction by a non-author and consumer acknowledgment remain pending.
The follow-up installed a fresh isolated Python environment and matched three
archived B5 validation transcripts after restoring source, weights and audio.
This owner-assisted same-machine subset check does not establish independent
review or full-test parity for every system.
[Reproduction guide](docs/m45_final_reproduction.md) ·
[Delivery acceptance](docs/m45_delivery_acceptance.md)

## Scope and remaining work

| Track | Evidence and status |
| --- | --- |
| B1 (Whisper-small) and B4 (XLS-R Turkish CV7) | Full matched test evaluation and repeated timing |
| B2 (Whisper-large-v3) | Local GPU rerun: all 438 validation and 636 test predictions/N-best outputs checked, with zero inference error rows. Repeated timing and official test scoring remain pending. M2's reported ZIP remains unreceived and is not used as measured evidence here. |
| B3 (Qwen3-ASR) | Both candidates completed full validation; 1.7B selected by minimum normalized CER (5.423% versus 7.067%). Its test inference is running; timing and test scoring remain pending. |
| B5 (BiLSTM-CTC) | Epoch 22 selected after GPU continuation; full test WER 47.776%, CER 13.503%; checked predictions/N-best and 3-repeat timing |
| Supplied single-Gaussian character HMM | Three EM passes; full test WER 99.995%, CER 88.760%; a weak classical baseline, not a conventional mixture-Gaussian phoneme HMM |
| F1–F3 (Whisper LoRA, full Whisper and adapted CTC) and R1 (learned relational rescoring) | Complete working test results, adapted checkpoints, graph controls and timing |
| Additional baseline coverage | Seda Turkish RNN-T and a new genuine four-mixture grapheme HMM have completed full validation/test exports. Repeated timing and final offline reports are in progress. The earlier supplied single-Gaussian HMM remains preserved separately. [Follow-up audit](docs/baseline_completion_followup.md) |

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

## 2026-11-22 closeout

See the [final review](docs/final_review_2026-11-22.md), [M3 results](results/final/m3_test_summary.json),
[Drive packages](docs/drive_packages.md), and [final reproduction commands](docs/final_reproduction.md).
The new local deliverables still require PR publication and genuine non-author review.
