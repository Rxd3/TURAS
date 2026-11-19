# Member 3 — CTC & Alternative Neural Baselines Audit and Protocol (B4 & B5)

Status: **Protocol and checkpoint audit only** (no model weights downloaded locally, no splits bound, no audio evaluated yet; awaiting M1 re-freeze hash).

## 1. Track B4: Pretrained Turkish CTC Baseline (`Wav2Vec2` / `XLS-R`)
* **Pinned Checkpoints (matching M6 smoke-test revisions):**
  * Primary: `mpoyraz/wav2vec2-xls-r-300m-cv8-turkish` @ `bdd0bb878d8bf59509611631e334eb073ba57773` (`float32`, greedy CTC).
  * Secondary comparison: `mpoyraz/wav2vec2-xls-r-300m-cv7-turkish` @ `708639f50559d7970f462e13ec64d3f059ca89f6` (`float32`, greedy CTC).
* **WorldSpeech Eligibility:** `disco-eth/WorldSpeech` was created on **2026-04-28** and TURAS test sessions span **2026-01-06 to 2026-04-22**. Both Common Voice XLS-R checkpoints predate WorldSpeech and have zero overlap with the test set.
* **Processor & Decoding Rules:**
  * Build processor explicitly via `Wav2Vec2CTCTokenizer` + `Wav2Vec2FeatureExtractor` (no LM).
  * Run in `float32` on 10–20 s mono 16 kHz clips with no chunking; preserve emitted raw text byte-for-byte (no `.strip()`).
  * **Known Limitation (shared with M5 F3):** Neither checkpoint's vocabulary includes digit characters (`0–9`), causing systematic errors on numerals.
  * **N-Best Status:** Currently 1-best greedy only. Planned N-best extension will use a NumPy prefix beam search scored by exact `-CTCLoss(reduction='none')` per the N-best contract.

## 2. Track B5: Train-From-Scratch Neural Baseline (`BiLSTM-CTC`)
* **Architecture:** 2x strided Conv2D subsampling over 80-bin log-Mel features (16 kHz, 25 ms window, 10 ms hop) + provisional 4-layer bidirectional LSTM (`hidden_size=320`, `dropout=0.2`) + linear CTC head (`float32`).
* **Probe-First Commitment:** Memory, step time, and epoch time on allocated training hardware are currently unmeasured. Before locking final layer dimensions, we will run a **3-step memory probe** and **1-epoch timing run** on the frozen train split.
* **Zero-Leakage Guarantee:** Character vocabulary (`<blank>`, `<unk>`, `<space>`, plus train characters) and checkpoint selection rely strictly on `train` and `validation` splits respectively.
* **Official Timing:** Local/development RTF numbers will not be reported; M6 will benchmark B4 and B5 back-to-back on the RTX 5070 Ti.