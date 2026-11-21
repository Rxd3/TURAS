# TURAS Quantitative Error Analysis and RDL Failure Diagnosis

All numbers below are computed directly from the frozen test benchmark artifact
[`results/final/m45_test_summary.json`](../results/final/m45_test_summary.json)
(`turas-tr-econ-legis-v0.1`, 636 test clips across 28 recording groups,
`20,529` normalized reference words, `146,220` normalized reference characters,
single training seed `17`).

---

## 1. Substitution / Deletion / Insertion (S / D / I) Error Profile

Word-level error breakdown under `tr-normalized-v1` (`N = 20,529` reference words):

| System | Norm WER | Substitutions | Deletions | Insertions | Total Errors | Verified-25 WER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B1-whisper-small` | 24.512% | 3,433 (16.72%) | 335 (1.63%) | 1,264 (6.16%) | 5,032 | 22.103% |
| `F1-whisper-lora` | 12.368% | 1,871 (9.11%) | 311 (1.51%) | 357 (1.74%) | 2,539 | 12.067% |
| `F2-whisper-full` | 10.371% | 1,563 (7.61%) | 263 (1.28%) | 303 (1.48%) | 2,129 | 10.753% |
| `B4-xlsr-tr-cv7` | 37.152% | 5,669 (27.61%) | 542 (2.64%) | 1,416 (6.90%) | 7,627 | 35.962% |
| `F3-ctc-vocab` | 25.150% | 3,449 (16.80%) | 1,349 (6.57%) | 365 (1.78%) | 5,163 | 25.090% |
| `F1-whisper-lora--rdl-full` | 12.373% | 1,881 (9.16%) | 302 (1.47%) | 357 (1.74%) | 2,540 | 11.947% |
| `F2-whisper-full--rdl-full` | 10.429% | 1,571 (7.65%) | 265 (1.29%) | 305 (1.49%) | 2,141 | 10.992% |
| `F3-ctc-vocab--rdl-full` | 25.189% | 3,467 (16.89%) | 1,330 (6.48%) | 374 (1.82%) | 5,171 | 25.209% |

### Key findings from S / D / I and verified-subset tracking

1. **Whisper fine-tuning cuts insertions by 76% and substitutions by 54% (`B1 -> F2`):**
   Insertions fall from `1,264` (`6.16%`) to `303` (`1.48%`), and total word errors drop by **57.7% relative** (`5,032 -> 2,129`). A large driver of the insertion drop is that pretrained `B1` spells out spoken Turkish numbers as multi-word phrases and emits conversational fillers, whereas TBMM parliamentary minutes use compact digit/edited forms.
2. **CTC vocabulary extension (`B4 -> F3`) trades substitutions/insertions for deletions:**
   Substitutions drop `-39.2%` (`5,669 -> 3,449`) and insertions drop `-74.2%` (`1,416 -> 365`), but **deletions more than double** (`542 -> 1,349`, `2.64% -> 6.57%`). This is a classic CTC blank-overprediction signature when extending a pretrained CTC head (`38 -> 52` tokens) over limited fine-tuning epochs.
3. **Adaptation gains hold on the human-verified subset (`n = 25`):**
   Although 611/636 test clips rely on unreviewed edited TBMM minutes, the 25 single-reviewer verified clips track full-set WER closely (`B1: 22.10% -> F2: 10.75%`, `B4: 35.96% -> F3: 25.09%`), confirming that the ~12–14 pp adaptation gains are not an artifact of unreviewed minutes.

---

## 2. Domain Terminology, Entity-Type, and Quantity Analysis

### Performance by entity type (`F1` / `Recall`)

| System | `concept` (n=2,507) | `currency` (n=502) | `legal_text` (n=161) | `organization` (n=215) |
| --- | ---: | ---: | ---: | ---: |
| `B1-whisper-small` | 93.3% (R=89.0%) | 96.2% (R=93.6%) | 92.1% (R=87.0%) | **64.2% (R=48.8%)** |
| `F1-whisper-lora` | 97.4% (R=95.9%) | 98.2% (R=98.0%) | 92.2% (R=87.6%) | 90.4% (R=85.1%) |
| `F2-whisper-full` | 98.5% (R=97.8%) | 98.3% (R=98.2%) | 95.2% (R=93.2%) | **95.3% (R=93.5%)** |
| `B4-xlsr-tr-cv7` | 86.5% (R=77.8%) | 71.0% (R=55.2%) | 83.0% (R=72.7%) | 70.9% (R=57.2%) |
| `F3-ctc-vocab` | 95.5% (R=92.6%) | 93.0% (R=87.8%) | 90.8% (R=85.7%) | 84.2% (R=75.8%) |

### Largest domain-term gains (`B1 -> F2`, support >= 15)

Pretrained Whisper struggles most on Turkish institutional acronyms and domain-specific compounds; full fine-tuning (`F2`) closes almost the entire gap:

| Canonical Term | Support | `B1` F1 | `F2` F1 | Gain (`F2 - B1`) | `F3` F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `LAB_SGK` | 55 | 35.8% | 95.3% | **+59.5 pp** | 91.3% |
| `LAB_BAGKUR` | 43 | 46.4% | 98.8% | **+52.4 pp** | 80.6% |
| `MON_TUIK` | 25 | 57.1% | 96.0% | **+38.9 pp** | 64.9% |
| `MON_AVRO` | 18 | 71.4% | 97.1% | **+25.7 pp** | 61.5% |
| `LAB_PRIM` | 67 | 81.7% | 98.5% | **+16.8 pp** | 92.1% |
| `LAB_UCRET` | 58 | 81.8% | 98.2% | **+16.4 pp** | 94.1% |
| `TRADE_MILLI_GELIR` | 19 | 81.3% | 97.3% | **+16.0 pp** | 81.3% |
| `TAX_OTV` | 19 | 73.3% | 88.9% | **+15.6 pp** | 59.3% |

### Quantity diagnostics (`618` reference quantities: `481` amounts + `137` percentages)

| System | Micro F1 | TP / FP / FN | Amount F1 | Percentage F1 |
| --- | ---: | ---: | ---: | ---: |
| `B1-whisper-small` | 32.13% | 120 / 9 / 498 | 34.52% | 23.08% |
| `F1-whisper-lora` | 93.90% | 570 / 26 / 48 | 94.84% | 90.57% |
| `F2-whisper-full` | 94.68% | 578 / 25 / 40 | 95.39% | 92.13% |
| `B4-xlsr-tr-cv7` | 0.00% | 0 / 0 / 618 | 0.00% | 0.00% |
| `F3-ctc-vocab` | 0.00% | 0 / 12 / 618 | 0.00% | 0.00% |

**Why `F3-ctc-vocab` remains at 0.00% quantity F1 despite vocabulary extension:**
`B4`'s original 38-token CTC head cannot emit digits `0–9` at all (`FP=0`). Extending the head to 52 tokens (`F3`) enables digit emission in principle, yet `F3` emitted only `12` digit-pattern quantities across all 636 clips and matched `0/618` exact `(type, unit, value)` tuples. Character-level CTC maps acoustics monotonically to characters and cannot easily perform the non-monotonic inverse text normalization (ITN) required to convert spoken Turkish number phrases into parliamentary digit formatting, whereas Whisper's Seq2Seq decoder learns this ITN mapping rapidly (`32.13% -> 94.68%` F1).

---

## 3. Why Relational Deep Learning (RDL) Rescoring Did Not Improve WER

Across all three adapted models, learned RDL rescoring slightly degraded normalized WER relative to original one-best:

* `F1--rdl-full`: `+0.005 pp` (**+1 word** out of `20,529`: `2,540` vs `2,539` errors)
* `F2--rdl-full`: `+0.058 pp` (**+12 words** out of `20,529`: `2,141` vs `2,129` errors)
* `F3--rdl-full`: `+0.039 pp` (**+8 words** out of `20,529`: `5,171` vs `5,163` errors)

Four concrete factors explain this negative result:

1. **Coarse KG saturation across N-best candidates:**
   The static relational schema (`data/relational/turas-tr-econ-legis-v0.1/`) contains **90 entities** and **90 static edges**. On `F2`, `2,994 / 2,994` N-best hypotheses (`100%`) linked to entities (`12,402` entity links, `36,996` relation queries). Because almost every candidate in a 5-best beam triggers the same high-frequency parliamentary entities, the graph score has very low discriminative variance across candidates.
2. **Noisy co-mention and ungrounded quantifier signals:**
   On `F2`, **37.5%** of quantifier queries (`1,074 / 2,867`) were `unknown_or_contextless_quantifiers`, injecting ungrounded scoring noise. Consistent with this, removing co-mentions on `F1` improved WER over full RDL (`-0.049 pp`, ~10 words, 95% CI `[-0.098, -0.010]`), and replacing the learned graph on `F2` with an untrained random graph also beat full RDL (`-0.024 pp`, ~5 words, 95% CI `[-0.049, -0.005]`).
3. **Statistical multiplicity across 19 paired bootstrap intervals:**
   The 19 paired comparisons in `M4_M5_FINAL_REPORT.md` report uncorrected 95% percentile bootstrap CIs. Under strict family-wise correction (e.g., Bonferroni `alpha = 0.05 / 19 ≈ 0.0026`), all four acoustic adaptation gains (`-12.14`, `-14.14`, `-12.00`, `-2.00 pp`) remain overwhelmingly significant, whereas the small RDL ablation deltas (`< 0.06 pp`, 1–12 words across 636 clips) represent negligible practical effects.
4. **Unmeasured diagnostics left as explicit future work:**
   Two diagnostics were not executed before freeze and would be the immediate next steps for follow-up research: (a) computing **N-best oracle WER** on the `N=5` beam lists to bound available rescoring headroom, and (b) adding a **classic Turkish n-gram / neural LM rescoring baseline** alongside RDL to separate general language-model priors from structured relational priors.
