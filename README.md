# TURAS — Project 3

**Turkish Unified Recognition Adaptation System** is a six-member university
project on domain-specific Turkish automatic speech recognition (ASR): speech
from Turkish Grand National Assembly (TBMM) debates on the economy and
legislation. We build the dataset, run the course baselines, fine-tune Whisper
and wav2vec 2.0 models, and test relational (graph) rescoring.

[Submission guide](SUBMISSION.md) · [Final report](results/final/M4_M5_FINAL_REPORT.md) ·
[Presentation](presentation/TURAS-final-2-minute.pptx)

## Dataset

**Name:** `turas-tr-econ-legis-v0.1`, a custom domain-specific subset of
WorldSpeech Turkish ([`disco-eth/WorldSpeech`](https://huggingface.co/datasets/disco-eth/WorldSpeech),
config `tr_tr`, revision `b76b8539`, CC BY-NC 4.0). WorldSpeech `tr_tr` holds
239,654 clips (about 1,008 h) of TBMM sessions from 2017 to 2026, with the
official parliamentary minutes as transcripts. [Source audit](data/metadata/worldspeech-tr_tr-source-audit.md)

**Cleaning.** [`scripts/m1_build_release.py`](scripts/m1_build_release.py) applies
these filters in order and logs the first reason for every removed clip:

| Step | Removed | Clips left |
| --- | ---: | ---: |
| All `tr_tr` clips | | 239,654 |
| Not parliament (Ankara city council) | 2,451 | 237,203 |
| Non-spoken or truncated text (speaker headers, stage directions, brackets) | 16,768 | 220,435 |
| Duplicate text anywhere in the corpus | 3,175 | 217,260 |
| Text does not match audio (upstream ASR agreement CER > 0.25) | 3,886 | 213,374 |
| Not in domain (see below) | 184,567 | 28,807 |
| Recording also in the upstream test split (quarantined) | 1,069 | 27,738 |
| Eligible but over the per-recording cap | 22,288 | **5,450** |

**Domain rule.** A clip is kept only if its transcript contains at least two
distinct economic terms, or one economic term plus a legislative term or an
amount/percentage, from our 90-term
[economic and legal lexicon](configs/datasets/turas-domain-lexicon-v0.1.json)
(budget, tax, inflation/interest/currency, banking, wages/pensions/social
security, trade/investment, laws). Within each recording, clips are ranked by
domain-term density.

**Split rule.** The dataset has no speaker IDs, so we group by the closest unit
it has: the recording. One recording is one parliamentary session, and all clips
of a recording go to the same split. The test split is every 2026 session
(temporal hold-out); validation is 15 sessions from Sep–Dec 2025 drawn with
seed 2026; train is every other session. The split is made by
[`scripts/m1_build_release.py`](scripts/m1_build_release.py) with
[`turkish_asr_rdl.data.selection`](src/turkish_asr_rdl/data/selection.py), and the
IDs are committed in [`data/splits/`](data/splits/turas-tr-econ-legis-v0.1/).
We did not use the upstream WorldSpeech split: all 188 of its test recordings
also appear in its train split.

| Split | Speakers | Utterances | Hours | Recordings (sessions) | Session dates |
| --- | --- | ---: | ---: | ---: | --- |
| Train | unknown | 4,376 | 20.0 | 129 | 2017-01-10 – 2025-12-23 |
| Validation | unknown | 438 | 2.0 | 15 | 2025-10-01 – 2025-12-24 |
| Test | unknown | 636 | 3.0 | 28 | 2026-01-06 – 2026-04-22 |

- **Test sentences that appear word for word in train: 0 of 636** (normalized
  text; [split manifest](data/splits/turas-tr-econ-legis-v0.1/split_manifest.json),
  `leakage_checks`). No recording or session date appears in two splits.
- **Speakers are not separated.** Members of parliament speak in many sessions,
  so the same speaker can appear in train and test. Recording grouping and the
  temporal test split are the closest available control.
- Clips are 10–20 s (about 16.5 s on average), converted to 16 kHz mono FLAC.
- **Listening check.** The team reviewed 249 clips by listening (224 of the 636
  test clips, plus 15 validation and 10 train clips): 245 accepted, 4 corrected.
  57 have per-clip records and 192 are recorded as one batch confirmation.
  Independent second-listener adjudication of 55 items is pending. Reported
  scores use the original references, which are the official edited minutes.
  [Review record](docs/m1_deadline_handoff.md)

**Data in this repository:**

- **Raw:** all 239,654 upstream rows (transcripts and metadata, no audio) in
  [`data/raw/worldspeech-tr_tr-b76b8539/`](data/raw/worldspeech-tr_tr-b76b8539/README.md).
- **Cleaned:** 5,450 FLAC clips, train, validation and test references, inference
  inputs, relational tables and checksums in
  [`data/cleaned/turas-tr-econ-legis-v0.1/`](data/cleaned/turas-tr-econ-legis-v0.1/README.md).
- **Removal log:** the reason every upstream row was kept or removed, in
  [`exclusion_ledger.jsonl.gz`](data/cleaned/turas-tr-econ-legis-v0.1/exclusion_ledger.jsonl.gz).
  [`scripts/m1_export_raw.py`](scripts/m1_export_raw.py) rebuilds it from the raw
  data and checks that it matches the frozen release byte for byte.
- **Fixed by listening:** 13 references (3 train, 10 test) were corrected after
  listening; 35 rows are marked `verified`.

Full details: [dataset card](docs/dataset.md).

## Metrics and why we use them

Following the [Hugging Face audio course, chapter 5 "Evaluation"](https://huggingface.co/learn/audio-course/en/chapter5/evaluation):

- **WER** (word error rate) is the standard ASR metric: substitutions, deletions
  and insertions divided by the number of reference words. It matches how a
  reader judges a transcript of a debate.
- **CER** (character error rate) is reported because Turkish is agglutinative: one
  wrong suffix makes a whole long word wrong under WER, while CER gives partial
  credit.
- **Normalized and raw views.** The headline numbers use one Turkish-aware
  normalizer (`tr-normalized-v1`: Turkish casing for `I/ı` and `İ/i`, punctuation
  removed) for every system, so spelling conventions are separated from
  recognition errors. The raw (orthographic) view is also reported. The course's
  `BasicTextNormalizer` is not used because it is not Turkish-aware.
- **Domain-term F1** measures whether economic and legal terms are recognized,
  and **RTF** (processing time / audio time) measures decoding cost.

All systems are scored on the same 636 test clips, with the same references and
the same scoring code ([`turkish_asr_rdl.evaluation`](src/turkish_asr_rdl/evaluation/)).
[Evaluation details](docs/evaluation.md)

## Baselines (no training by our group, or classical/CTC models we trained from scratch)

Normalized test WER/CER (%) on the 636 test clips.

| ID | Family | Exact model and revision | WER | CER |
| --- | --- | --- | ---: | ---: |
| B1 | Attention encoder-decoder (Whisper) | `openai/whisper-small` @ `973afd24` | 24.51 | 10.08 |
| B2 | Attention encoder-decoder (Whisper) | `openai/whisper-large-v3` @ `06f233fe` (M6 scorer, B1 parity pending) | 12.32 | 5.45 |
| B3 | Speech LLM | `Qwen/Qwen3-ASR-1.7B-hf` @ `bcd2b5b7` (0.6B rejected on validation; M6 scorer, B1 parity pending) | 16.09 | 5.60 |
| B4 | wav2vec 2.0 + CTC head, fine-tuned for Turkish by its publisher | `mpoyraz/wav2vec2-xls-r-300m-cv7-turkish` @ `708639f5` | 37.15 | 13.17 |
| B5 | CTC encoder (BiLSTM-CTC), trained by us from scratch | [`configs/baselines/bilstm_ctc.json`](configs/baselines/bilstm_ctc.json) | 47.78 | 13.50 |
| B6 | Classical HMM, single diagonal Gaussian per character state, trained by us | [`scripts/m3_run_gaussian_hmm.py`](scripts/m3_run_gaussian_hmm.py) | 100.00 | 88.76 |
| B7 | Classical MFCC-39 + GMM-HMM (4 diagonal Gaussians, 3-state graphemes), trained by us | [`scripts/train_mixture_hmm.py`](scripts/train_mixture_hmm.py) | 99.97 | 64.55 |
| B8 | Transducer (Zipformer2 RNN-T), CPU, greedy, no external LM | `atasoglu/seda-v0.1` @ `5cbbda1b` | 29.96 | 12.81 |

B1 and B4 come from the frozen protocol ([metrics](results/final/m45_metrics.csv)).
B2 and B3 are scored with the same library calls by
[`scripts/m6_score_baselines.py`](scripts/m6_score_baselines.py), outside the frozen
protocol ([B2/B3 results](results/m6/README.md)). B5 and B6:
[M3 summary](results/final/m3_test_summary.json), scored by the `m3_*` scripts outside the frozen protocol with the shared evaluation library, with checkpoints, predictions,
N-best lists, timing and training history in [`results/m3/`](results/m3/README.md).
B7 and B8: [baseline completion report](results/final/BASELINE_COMPLETION_REPORT.md)
([CSV](results/final/baseline_completion_metrics.csv), [summary](results/final/baseline_completion_summary.json)).
That report also has a separate local rerun of B2 (12.39 / 5.46) and B3 (16.17 / 5.62),
scored with the frozen evaluator; the table keeps M6's runs, chosen before test access.
B7 is a real GMM-HMM but has no pronunciation lexicon or language model, so it
recognizes almost no whole words; its CER is reported for that reason. The test references were kept by
agreement with an upstream ASR transcript (CER ≤ 0.25), which can favour
Whisper-family systems.

**Course baselines not run, because the model does not support Turkish:**

| Model | Reason |
| --- | --- |
| `nvidia/canary-1b-v2` @ `d4557063` | The card lists 25 European languages; Turkish is not one of them. [Model card](https://huggingface.co/nvidia/canary-1b-v2/blob/d455706339a6b32e1aa40f82c713a482a0c938e2/README.md) |
| `nvidia/parakeet-tdt-0.6b-v3` @ `541d1f99` | The card lists 25 European languages; Turkish is not one of them. [Model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/blob/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/README.md) |
| HuBERT + CTC (`facebook/hubert-large-ls960-ft`, `facebook/hubert-base-ls960`) | No Turkish ASR head: the CTC model is English-only and lacks Turkish letters; the base model is an encoder without a head. |

[Language support audit](docs/asr_language_support_audit.md)

## Our contributions (trained by our group)

Same test clips, references and scoring as the baselines. Training seed 17;
model and checkpoint selection on validation only; test used once after the
[protocol was frozen](configs/evaluation/m45_final_protocol.json).

| ID | What we trained | Parent | WER | CER | Change vs parent (95% CI, points) |
| --- | --- | --- | ---: | ---: | --- |
| F1 | Whisper LoRA fine-tuning | B1 | 12.37 | 4.15 | −12.14 [−13.46, −10.77] |
| F2 | Whisper full fine-tuning | B1 | **10.37** | **3.69** | −14.14 [−15.42, −12.82] |
| F3 | XLS-R CTC fine-tuning with vocabulary extension (38 → 52 tokens) | B4 | 25.15 | 7.58 | −12.00 [−13.13, −10.90] |
| R1 | Learned relational (GNN) rescoring of 5-best lists, on F1 / F2 / F3 | F1–F3 | 12.37 / 10.43 / 25.19 | 4.13 / 3.72 / 7.57 | +0.005 / +0.058 / +0.039, all intervals include 0 |

Intervals are paired recording-group bootstrap (28 groups, 2,000 replicates,
seed 2026). [Final report](results/final/M4_M5_FINAL_REPORT.md) ·
[Error analysis](docs/error_analysis.md)

## Reproduce

1. Check the published numbers (standard library only): `python scripts/verify_submission.py`
2. Install: `python -m venv .venv`, activate it, then
   `python -m pip install --no-deps -e .` and `python -m pip install -r requirements/dev.txt`
3. Tests: `python -m pytest -q` (tests needing absent optional packages are skipped)
4. GPU environment (Python 3.13, CUDA 13.0): `pip install -r requirements.txt`, then
   `python scripts/check_env.py`
5. Full rerun: [reproduction guide](docs/m45_final_reproduction.md) and
   [follow-up baselines](docs/final_reproduction.md)

Main entry points in [`scripts/`](scripts/README.md): `m1_build_release.py` (dataset),
`m1_export_raw.py` (raw data and removal log),
`run_baseline_batch.py` (baselines), `m4_train_whisper.py`, `m5_train_ctc.py`,
`m5_train_rdl.py` (training), `m45_run_final.py` / `m45_final_evaluate.py` (frozen
benchmark), `m6_score_baselines.py` (B2/B3), `m3_*.py` and `train_mixture_hmm.py`
(B5–B7).

## Repository map

| Path | Contents |
| --- | --- |
| [`src/turkish_asr_rdl/`](src/turkish_asr_rdl/) | Python package: data curation, baseline adapters, training, decoding, RDL and evaluation |
| [`scripts/`](scripts/) | Command-line entry points for each step |
| [`configs/`](configs/) | Dataset, baseline, fine-tuning, decoding and evaluation settings |
| [`data/`](data/) | Cleaned dataset (audio and references), metadata, frozen split IDs, relational tables and verification records |
| [`results/`](results/) | Test results, B2/B3 scores, B5/HMM checkpoints and outputs, validation and ablation summaries |
| [`experiments/`](experiments/) | Experiment protocol and per-workstream notes |
| [`tests/`](tests/) | Offline unit and integration tests for data, evaluation, models and scoring |
| [`docs/`](docs/) | Dataset card, evaluation, architecture, contracts and run records |
| [`presentation/`](presentation/) | Final two-minute deck, its generator and the speaker script |
| [`contributions/`](contributions/) | One contribution record per team member |
| [`requirements/`](requirements/) | Pinned environments: `dev.txt` (tests), `windows-cu130.txt` (GPU), others for specific runs |
| [`licenses/`](licenses/) | License texts for third-party model files |
| [`.github/`](.github/) | CI workflows (tests and packaging) and the pull-request template |
| `requirements.txt` | Pinned GPU environment used for the reported runs (copy of `requirements/windows-cu130.txt`) |
| `pyproject.toml` | Package metadata for `turkish_asr_rdl` |
| `LICENSE`, `NOTICE` | MIT license for our code; attribution for the dataset and models |
| `SUBMISSION.md` | Course submission guide |
| `TEAM_TASKS.md`, `CONTRIBUTING.md` | Team ownership and review workflow |
| `.gitignore`, `.gitattributes` | Ignored files; line endings and byte-exact hashed files |

## License and data rights

Our code uses the [MIT License](LICENSE). The dataset is derived from
`disco-eth/WorldSpeech` (`tr_tr`, CC BY-NC 4.0, TBMM recordings); its
non-commercial attribution terms and each pretrained model's license apply.
[Third-party notice](NOTICE)
