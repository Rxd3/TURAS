# TURAS Final Presentation Outline (12 Slides)

Every metric below comes directly from committed, hash-verified artifacts
(`results/final/M4_M5_FINAL_REPORT.md`, `results/final/m45_test_summary.json`,
and `docs/error_analysis.md`). Unrun baselines (`B2`, `B3`, `B5`) are labeled
explicitly as not executed on the frozen test set.

1. **Title & Motivation**
   - **TURAS:** Turkish Unified Recognition Adaptation System — domain-specific Turkish legislative/economic ASR, acoustic fine-tuning, and Relational Deep Learning (RDL) N-best rescoring.
   - Why domain adaptation matters: pretrained Turkish ASR misses institutional acronyms (`SGK` 35.8% F1, `Bağ-Kur` 46.4% F1, `TÜİK` 57.1% F1) and spoken parliamentary quantities (`32.1%` F1).
2. **Dataset Curation (`turas-tr-econ-legis-v0.1`)**
   - Derived from `disco-eth/WorldSpeech` (`tr_tr`, CC BY-NC 4.0): **5,450 clips (~25 h)** across 7 economic/legislative topics.
   - Temporal, recording-group-disjoint split: `4,376` train / `438` validation / `636` test (28 test recording groups; 0 split overlap).
   - Honest data limits: `611/636` test clips use unreviewed edited TBMM minutes (`25` single-reviewer verified clips); speaker disjointness unproven across sessions.
3. **Relational Schema & Domain Lexicon**
   - 90 canonical entities, 154 aliases, 90 static relationships, 7 topics (`data/relational/turas-tr-econ-legis-v0.1/`).
   - Deterministic `tr-normalized-v1` Turkish text normalizer (dotted/dotless `İ/i/I/ı`, apostrophes, NFC) + canonical terminology & quantity diagnostics.
4. **Experimental Protocol & Leakage Discipline**
   - Strict pre-test freeze (`configs/evaluation/m45_final_protocol.json`, 56 pinned code hashes); validation-only checkpoint & weight selection; single training seed (`17`).
   - Paired recording-group percentile bootstrap (`2,000` replicates, 28 groups) + 3-repeat RTX 5070 Ti timing on 107 frozen clips (`1,813.8 s`).
5. **System Matrix (Baselines & Adaptation)**
   - **Run on frozen test (636 clips):** `B1` Whisper-small, `B4` Wav2Vec2-XLS-R-300M-cv7, `F1` Whisper LoRA (epoch 3), `F2` Whisper Full (epoch 2), `F3` CTC + vocab extension (`38 -> 52` tokens, epoch 3), `R1` heterogeneous RDL + 4 controls per model (20 total conditions).
   - **Implemented/Configured, not run on test:** `B2` Whisper-large-v3 & `B3` Qwen3-ASR (0.6B/1.7B) adapters merged + smoke-checked on RTX 5070 Ti; `B5` BiLSTM-CTC config merged.
6. **Main Acoustic Results (636 Test Clips)**
   - `B1` → `F1` (LoRA): `24.512% → 12.368%` norm WER (**-12.144 pp**, 95% CI `[-13.46, -10.77]`).
   - `B1` → `F2` (Full): `24.512% → 10.371%` norm WER (**-14.141 pp**, 95% CI `[-15.42, -12.82]`).
   - `B4` → `F3` (CTC + vocab): `37.152% → 25.150%` norm WER (**-12.003 pp**, 95% CI `[-13.13, -10.90]`; combines fine-tuning + vocab extension).
   - Verified-25 subset tracks full 636-clip rankings (`B1: 22.10% → F2: 10.75%`, `B4: 35.96% → F3: 25.09%`).
7. **Quantitative Error Analysis (S / D / I & Quantities)**
   - Whisper `B1 → F2` slashes insertions **-76%** (`1,264 → 303`) and substitutions **-54%** (`3,433 → 1,563`).
   - CTC `B4 → F3` cuts substitutions `-39%` and insertions `-74%`, but **deletions more than double** (`542 → 1,349`, classic CTC blank-overprediction).
   - Quantity F1: `B1` `32.13%` → `F2` **`94.68%`**, while `F3` stays **`0.00%`** (`0/618`, `FP=12`) — exposing Seq2Seq ITN vs monotonic char-CTC on spoken Turkish numbers.
8. **Domain Terminology Breakthrough**
   - `organization` F1 jumps **+31.1 pp** (`64.2% → 95.3%` on `F2`).
   - Largest gains: `LAB_SGK` (`35.8% → 95.3%`), `LAB_BAGKUR` (`46.4% → 98.8%`), `MON_TUIK` (`57.1% → 96.0%`), `MON_AVRO` (`71.4% → 97.1%`).
9. **RDL Rescoring: Honest Null Result & Mechanistic Diagnosis**
   - Learned RDL degrades 1-best slightly on all 3 models: `+0.005` (+1 word), `+0.058` (+12 words), `+0.039 pp` (+8 words) across `20,529` words; all 95% CIs cross 0.
   - Untrained random graph beats learned RDL on F2 (`-0.024 pp`, `[-0.049, -0.005]`); removing co-mentions beats full RDL on F1 (`-0.049 pp`, `[-0.098, -0.010]`).
   - Why: 90-entity/90-edge KG saturates `99.9%` of N-best hypotheses (`2,994/2,994` on F2) with low candidate contrast, and `37.5%` of quantifier queries (`1,074/2,867`) are contextless.
10. **Efficiency & Repeated RTF Timing**
    - Whisper family (`B1/F1/F2`): **RTF ~0.053–0.054** (~18.5–18.9× real-time), peak VRAM `1.16–1.23 GiB`.
    - CTC family (`B4/F3`): **RTF ~0.124–0.128** (~7.8–8.1× real-time), peak VRAM `1.15 GiB`.
    - Joint RDL overhead is negligible (`+0.0008` RTF on F2).
11. **Reproducibility, Verification & Team Contributions**
    - Single-command verification (`python scripts/verify_submission.py`): 56/56 pinned code hashes, 20/20 systems recomputed from S/D/I/N, 5,450 disjoint split clips, `1,133` passing offline tests, MIT `LICENSE` + CC BY-NC 4.0 `NOTICE`.
    - Clear per-member ownership (`contributions/member_{1..6}.md`).
12. **Limitations & Future Work**
    - Run `B2` (`whisper-large-v3`), `B3` (`Qwen3-ASR`), and `B5` (`BiLSTM-CTC`) on test; compute N-best oracle WER (`scripts/compute_nbest_oracle.py`) and add a Turkish neural/n-gram LM rescoring baseline; expand KG scale and human verbatim verification; multi-seed runs.
