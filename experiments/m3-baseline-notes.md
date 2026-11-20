# Member 3 — CTC & Alternative Neural Baselines Audit and Protocol (B4 & B5)

Status: **Protocol and checkpoint audit only** (no model weights downloaded locally, no splits bound, no audio evaluated yet; awaiting M1 re-freeze hash and M5/M6 review).

## 1. Track B4: Pretrained Turkish CTC Baseline (`Wav2Vec2` / `XLS-R`)
* **Primary Checkpoint (matched to M5 F3 `B4-xlsr-tr-cv7`):**
  * `mpoyraz/wav2vec2-xls-r-300m-cv7-turkish` @ `708639f50559d7970f462e13ec64d3f059ca89f6` (committed `2022-03-23`, license `cc-by-4.0`, attribution required; `pytorch_model.bin` `1,262,079,473` bytes, Hub LFS oid `008d69c82bf45c82ec5485e550abc86ddabefe75080f22e1ebfdc1699889d82e`, no `safetensors`).
  * Labeled secondary comparison: `mpoyraz/wav2vec2-xls-r-300m-cv8-turkish` @ `bdd0bb878d8bf59509611631e334eb073ba57773` (committed `2022-03-23`, license `apache-2.0`; `pytorch_model.bin` `1,262,079,473` bytes, Hub LFS oid `dbd7a9f776a232c0dd4cbbee44025c8056246d83ad4de930b049e649e2fe402c`, no `safetensors`).
* **Held-Out Eligibility (per checkpoint):**
  * `cv7` was trained on Common Voice 7.0 Turkish + MediaSpeech, and `cv8` on Common Voice 8.0 Turkish, both atop `facebook/wav2vec2-xls-r-300m`.
  * Because both checkpoints were committed on **2022-03-23**, neither can have seen TURAS validation sessions (starting **2025-10-01**) or test sessions (**2026-01-06 to 2026-04-22**). Pre-2022 exposure of train audio via MediaSpeech or self-supervised pretraining cannot be excluded (affects `train` only).
* **Processor, Vocabulary & KenLM Exclusion:**
  * Both repos ship a `language_model/` (KenLM) directory and set `Wav2Vec2ProcessorWithLM` in `preprocessor_config.json`. B4 constructs `Wav2Vec2CTCTokenizer` + `Wav2Vec2FeatureExtractor` explicitly without LM so KenLM is never loaded.
  * Token IDs: `vocab_size = 38`, `<pad>` / CTC blank = `0`, `<unk>` = `3`, word delimiter `|` = `4`.
  * **Known Limitation:** The 38-token head cannot emit digits (`0–9`), `+`, circumflex vowels (`â`, `î`, `û`), uppercase characters, or punctuation other than `'`. No numeral or diacritic post-processing is applied to B4 output. Because M5's F3 extends the head to 52 tokens (`D33`), the B4 $\rightarrow$ F3 delta measures acoustic fine-tuning plus vocabulary extension together.
* **Matched Decoding & N-Best Contract (`decoding/ctc.py`):**
  * To match M5's F3 parent control (`configs/evaluation/m45_final_protocol.json`), B4 decodes in `bfloat16` using CTC prefix beam search (`beam_size = 10`, `requested_n = 5`) and takes rank 1 as the one-best (`float32` greedy is retained only as a labeled secondary diagnostic).
  * M3 will review and adopt M5's forward-algorithm prefix beam scorer in `src/turkish_asr_rdl/decoding/ctc.py` (`b900c2e`) rather than building a separate NumPy scorer, keeping exact sequence log-probability separate from approximate beam mass. Any `CTCLoss` cross-check uses `reduction="none"`, `zero_infinity=False`, and `float32` `log_softmax` (never `model(labels=...).loss`, which uses `ctc_loss_reduction="mean"`).

## 2. Track B5: Train-From-Scratch Neural Baseline (`BiLSTM-CTC`)
* **Architecture:** 80-bin log-Mel features (16 kHz, 25 ms window, 10 ms hop, per-utterance CMVN) $\rightarrow$ two stride-2 Conv2D subsampling layers (total subsampling factor `4`) $\rightarrow$ provisional 4-layer bidirectional LSTM (`hidden_size = 320`, `dropout = 0.2`) $\rightarrow$ linear CTC head (`float32`).
* **Target & Feasibility Rules:** Targets use `tr-normalized-v1` with character vocabulary (`<blank>=0`, `<unk>=1`, `<space>=2`, plus `train` characters). Training sets `zero_infinity=True` and enforces a minimum subsampled frame length check (`T_out >= L_target`), logging any dropped clips.
* **Probe-First & Compute Ceiling:** Before locking model dimensions, a 3-step memory probe and 1-epoch timing run will be executed on the allocated training GPU to fix numeric VRAM and wall-clock ceilings. Validation selection monitors `validation_norm_cer` against `tr-normalized-v1`.
* **Held-Out Exposure:** `none` (trained from random initialization strictly on the WorldSpeech-derived TURAS `train` split with zero validation/test recording overlap).
* **Official Timing:** Local/development RTF numbers are not reported; M6 benchmarks B4 and B5 back-to-back in the unified RTX 5070 Ti timing session.