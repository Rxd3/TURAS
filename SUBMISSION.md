# TURAS Project 3 submission

TURAS studies acoustic adaptation and relational rescoring for Turkish
legislative and economic speech. The completed M4/M5 study evaluates 20 systems
and rescoring conditions on the same frozen 636 clips against working references.

## Materials for grading

| Material | What it covers |
| --- | --- |
| [Final two-minute presentation](presentation/TURAS-final-2-minute.pptx) and [full deck](presentation/TURAS.pptx) | Title slide and four timed slides (about 1:15) for delivery; 12-slide full deck for questions |
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

**Clone size:** the repository holds the dataset (about 2.1 GB, mostly 5,450 FLAC
clips, without Git LFS), so `git clone` downloads about 2.1 GB and CI checkout
takes 2–3 minutes. `git clone --depth 1` still includes the data.

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
conditions, checks split-ID uniqueness and disjointness across 5,450 clips, and
verifies the SHA-256 of each public result file listed in the
[handoff manifest](results/final/m45_handoff_manifest.json).
The lean suite skips tests whose optional dependencies are unavailable. [Verifier](scripts/verify_submission.py)

## Reproduce with authorized artifacts

Full benchmark reproduction uses the pinned Windows CUDA runtime and the private
model and run archive. Audio and references are committed under `data/cleaned/`
([D44](docs/owner_decisions.md)); test references are used only for final scoring.
Ask the project team for the model and run archive. The documented restore
exercised three acoustic systems, three joint systems and 15 cached controls on
validation audio in a fresh checkout using the existing runtime. After the freeze,
M6 (a non-author) reproduced the F2 seed 17 training, the B1 decode, the B5 scores and
the B8 decode ([M6 reproductions](results/m6/repro-2026-11-23/README.md)). Consumer
acknowledgment, reproduction from the private M4/M5 bundle, and F1/F3 reproduction remain pending.
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
| B2 (Whisper-large-v3) and B3 (Qwen3-ASR-1.7B) | Official validation and test scores from the M6 scorer, outside the frozen protocol, and same-session timing on an RTX 5070 Ti laptop. B3 was selected over 0.6B on validation. Member 2's B2 handoff is scored as an independent reproduction. M6's B1 re-decode matches the frozen B1 normalized WER within 0.04 pp; scorer parity on the frozen B1 predictions and paired intervals against F1/F2 are pending the private M4/M5 bundle. The owner's separate local B2/B3 follow-up is a reproduction, not the reported result. [B2 and B3 results](results/m6/README.md) |
| B5 (BiLSTM-CTC) | Epoch 22 selected after GPU continuation; full test WER 47.776%, CER 13.503%; checked predictions/N-best and 3-repeat timing |
| Supplied single-Gaussian character HMM | Three EM passes; full test WER 99.995%, CER 88.760%; a weak classical baseline, not a conventional mixture-Gaussian phoneme HMM |
| F1–F3 (Whisper LoRA, full Whisper and adapted CTC) and R1 (learned relational rescoring) | Complete working test results, adapted checkpoints, graph controls and timing |
| B7 four-mixture MFCC GMM-HMM and B8 Seda Turkish RNN-T | Aggregate validation/test results, edit counts, export hashes and three-repeat timing are committed (B8 29.96% WER / 12.81% CER, B7 99.97% / 64.55%); the per-utterance exports and the B7 model file are external and not verified in this repository. The earlier single-Gaussian HMM is kept separately. [Completion report](results/final/BASELINE_COMPLETION_REPORT.md) |
| Post-freeze M6 reproductions (2026-11-23) | Run after test access and reported apart from the main results: F2 re-trained with seed 17 (10.468% normalized test WER against the frozen 10.371%; seed 18 pending), B1 re-decoded (24.551% against 24.512%), B5 re-scored (identical) and B8 re-decoded (byte-identical exports). A number-invariant WER view cuts the B1 → F2 gain of M6's runs from 14.1 to 6.8 points. F1 and F3 were not reproduced. [M6 reproductions](results/m6/repro-2026-11-23/README.md) |

M6's dated GPU smoke report covers a reference-free sample of 20 timed clips.
The four beam systems passed N-best validation after shared-export deduplication.
The B1, B4, F1–F3 and R1 results above come from the frozen test protocol, with separate
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
and [final reproduction commands](docs/final_reproduction.md).
On 2026-11-23, PR #26 is open; M6 (non-author) reviewed it at `a70b8e3` and `dd7654b` and requested changes.
