# TURAS Quantitative Error Analysis and RDL Results

The reported test metrics come from the frozen benchmark artifact
[`results/final/m45_test_summary.json`](../results/final/m45_test_summary.json)
(`turas-tr-econ-legis-v0.1`, 636 test clips across 28 recording groups,
`20,529` normalized reference words, `146,220` normalized reference characters,
single training seed `17`).

---

## 1. Substitution / Deletion / Insertion (S / D / I) Error Profile

Word-level error breakdown under `tr-normalized-v1` (`N = 20,529` reference words):

| System | Norm WER | Substitutions | Deletions | Insertions | Total Errors | 25-clip pilot WER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B1-whisper-small` | 24.512% | 3,433 (16.72%) | 335 (1.63%) | 1,264 (6.16%) | 5,032 | 22.103% |
| `F1-whisper-lora` | 12.368% | 1,871 (9.11%) | 311 (1.51%) | 357 (1.74%) | 2,539 | 12.067% |
| `F2-whisper-full` | 10.371% | 1,563 (7.61%) | 263 (1.28%) | 303 (1.48%) | 2,129 | 10.753% |
| `B4-xlsr-tr-cv7` | 37.152% | 5,669 (27.61%) | 542 (2.64%) | 1,416 (6.90%) | 7,627 | 35.962% |
| `F3-ctc-vocab` | 25.150% | 3,449 (16.80%) | 1,349 (6.57%) | 365 (1.78%) | 5,163 | 25.090% |
| `F1-whisper-lora--rdl-full` | 12.373% | 1,881 (9.16%) | 302 (1.47%) | 357 (1.74%) | 2,540 | 11.947% |
| `F2-whisper-full--rdl-full` | 10.429% | 1,571 (7.65%) | 265 (1.29%) | 305 (1.49%) | 2,141 | 10.992% |
| `F3-ctc-vocab--rdl-full` | 25.189% | 3,467 (16.89%) | 1,330 (6.48%) | 374 (1.82%) | 5,171 | 25.209% |

### Key findings from S / D / I and the listening pilot

1. **Whisper fine-tuning cuts insertions by 76% and substitutions by 54% (`B1 -> F2`):**
   Insertions fall from `1,264` (`6.16%`) to `303` (`1.48%`), and total word errors drop by **57.7% relative** (`5,032 -> 2,129`). One plausible contributor is the transcript style. Pretrained `B1` spells out some numbers and emits fillers, while edited TBMM parliamentary minutes use compact digit forms and can omit fillers. The aggregate counts do not quantify this contribution.
2. **CTC fine-tuning with vocabulary extension (`B4 -> F3`) reduces substitutions and insertions and increases deletions.**
   Substitutions fall by `39.2%` (`5,669 -> 3,449`) and insertions fall by `74.2%` (`1,416 -> 365`). **Deletions more than double** (`542 -> 1,349`, `2.64% -> 6.57%`). Blank overprediction is one possible explanation. This comparison changes both the CTC head (`38 -> 52` tokens) and the weights, and the edit counts alone cannot isolate a cause.
3. **The 25-clip listening pilot shows gains in the same direction.**
   On the 25 test clips historically marked as verified by one reviewer, WER fell from 22.10% to 10.75% for `B1 -> F2` and from 35.96% to 25.09% for `B4 -> F3`. The other 611 test references use unreviewed edited TBMM minutes. The small pilot and unavailable review records limit what it can establish about how much reference editing affects the full-set gains. [M1 deadline handoff](m1_deadline_handoff.md#where-the-actual-clips-are)

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

**`F3-ctc-vocab` remains at 0.00% on exact digit-based quantity F1.**
`B4`'s original 38-token CTC head cannot emit digits `0–9` (`FP=0`). Extending the head to 52 tokens lets `F3` emit digits. It produced `12` digit-pattern quantities across 636 clips and matched `0/618` exact `(type, unit, value)` tuples. One plausible explanation is that a character-level CTC alignment makes it harder to convert spoken Turkish numbers into the minutes' digit format. Whisper's result (`32.13% -> 94.68%` F1) shows much closer agreement with that format after fine-tuning.

---

## 3. Relational Deep Learning (RDL) Rescoring Results

Learned RDL rescoring produced slightly higher normalized WER point estimates than original one-best across all three adapted models.

* `F1--rdl-full`: `+0.005 pp` (**+1 word** out of `20,529`: `2,540` vs `2,539` errors)
* `F2--rdl-full`: `+0.058 pp` (**+12 words** out of `20,529`: `2,141` vs `2,129` errors)
* `F3--rdl-full`: `+0.039 pp` (**+8 words** out of `20,529`: `5,171` vs `5,163` errors)

The candidate coverage, controls and uncertainty help interpret these results.

1. **Candidate coverage and score variation.**
   The [static relational schema](../data/relational/turas-tr-econ-legis-v0.1/) contains **90 entities** and **90 static edges**. On the `F2` test set, `2,994 / 2,994` N-best hypotheses (`100%`) linked to entities (`12,402` entity links, `36,996` relation queries). In the corrected-linker validation run, only 30 of 438 `F2` utterances had different relational scores across candidates. The scorer had limited opportunities to change rankings in that run. [Linker audit](m45_result_audit.md#corrected-linker-rerun-completed)
2. **Quantifier coverage and graph controls.**
   On `F2`, **37.5%** of quantifier queries (`1,074 / 2,867`) were `unknown_or_contextless_quantifiers`. Removing co-mentions on `F1` lowered WER relative to the full graph (`-0.049 pp`, ~10 words, unadjusted 95% CI `[-0.098, -0.010]`). The untrained random graph also scored lower WER than the learned `F2` graph (`-0.024 pp`, ~5 words, unadjusted 95% CI `[-0.049, -0.005]`). These exploratory controls do not establish what caused the null result.
3. **Uncertainty across 19 paired comparisons.**
   All four reported acoustic comparisons (`-12.14`, `-14.14`, `-12.00`, and `-2.00 pp`) have unadjusted 95% recording-group bootstrap intervals excluding zero. The largest learned-RDL change against original one-best is 12 words across 636 clips, and all three of those intervals include zero. The intervals are per comparison. Family-wise significance was not evaluated. [Paired uncertainty](../results/final/M4_M5_FINAL_REPORT.md#paired-uncertainty)
4. **Unmeasured diagnostics.**
   N-best oracle WER would bound how much the existing five-best candidate lists could improve the result. A Turkish n-gram or neural language-model rescoring baseline would help distinguish general language-model effects from relational effects. Neither diagnostic was included in the frozen study. [Oracle evaluation script](../scripts/compute_nbest_oracle.py)
