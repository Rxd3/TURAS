# TURAS speaker outline

The [12-slide deck](TURAS.pptx) presents the frozen M4/M5 working benchmark.
Use the speaker notes for evidence links and additional metrics.

1. **TURAS.** Explain the task of recognizing Turkish parliamentary speech,
   including institutional terms and quantities.
2. **Working benchmark.** Describe the WorldSpeech provenance and recording-group
   splits (4,376 train, 438 validation, 636 test). The references are edited
   TBMM minutes. Of 636 test clips, 611 remain unreviewed and 25 carry historical
   single-reviewer tags. Speaker identities are unknown.
3. **Relational schema.** Introduce the 90 entities, 154 aliases and 90 static
   relationships. Explain the train-only graph and Turkish normalization.
4. **Protocol.** Discuss validation-only selection, scored N-best decoding,
   the pre-test freeze, 2,000 paired bootstrap draws over 28 recording groups
   and a single training seed.
5. **System coverage.** Five native systems and 15 rescoring conditions ran
   on test. B2 and B3 have smoke evidence, and B5 has synthetic CPU evidence.
   GMM-HMM and matched transducer evaluation are pending. Link the
   [published-source audit](../docs/asr_language_support_audit.md) for Turkish
   HuBERT and transducer support.
6. **Acoustic results.** Normalized WER changed from 24.512% to 12.368% for
   Whisper LoRA and to 10.371% for full Whisper. XLS-R moved from 37.152% to
   25.150% with fine-tuning and a train-derived vocabulary extension.
7. **Word errors.** Full Whisper cut insertions by 76%. Adapted XLS-R reduced
   substitutions and insertions while deletions rose from 542 to 1,349.
8. **Terminology and quantities.** Full Whisper increased organization F1 from
   64.2% to 95.3% and digit-form quantity F1 from 32.13% to 94.68%. The amount
   diagnostic uses exact digit patterns.
9. **RDL controls.** All three learned-versus-one-best intervals include zero.
   Graph linking covered all 2,994 F2 hypotheses, and many candidates shared
   the same cues. Report the random-graph contrast as exploratory.
10. **Repeated timing.** The official runs used an RTX 5070, 107 clips and three
    repeats per condition. Full Whisper had an RTF of 0.0530, rising to 0.0538
    with the joint learned graph.
11. **Reproduction.** Run `python scripts/verify_submission.py` for the public
    artifacts. The full rerun uses authorized audio, checkpoint and graph
    bundles and evaluator-only references. Team ownership is documented under
    [contributions](../contributions/README.md).
12. **Conclusion.** Acoustic adaptation reduced WER on the working benchmark.
    Learned rescoring showed no reliable gain. Complete baseline coverage,
    expand human listening and arrange independent clean-environment reproduction.

[Final report](../results/final/M4_M5_FINAL_REPORT.md) ·
[Error analysis](../docs/error_analysis.md) ·
[Dataset handoff](../docs/m1_deadline_handoff.md) ·
[Reproduction guide](../docs/m45_final_reproduction.md)
