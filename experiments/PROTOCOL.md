# Initial experiment protocol

This is a protocol, not a run log. No experiment has started. Actual model
IDs/revisions, releases, budgets, resolved configurations and run IDs must be
reviewed before execution. Initial selection uses train/validation only.

## Required comparisons

| Group | Conditions | Controlled comparison |
| --- | --- | --- |
| Baselines | B1 Whisper-small; B2 Whisper-large-v3; B3 Qwen3-ASR; B4 Turkish-capable pretrained Wav2Vec2/XLSR CTC; B5 BiLSTM-CTC alternative neural | Same frozen IDs and shared evaluator; report differing compute/context |
| Whisper | B1 vs small LoRA vs small full; any large-v3 tuning vs B2 | Same parent checkpoint/tokenizer/processor, release and declared decoder |
| CTC | B4 vs its fine-tuned counterpart | Same head/vocabulary unless vocabulary change is a separate ablation |
| RDL | Acoustic-only vs acoustic+RDL on each supported N-best producer | Identical candidate artifact, graph trained without held-out references |
| Optional LM | Acoustic+LM vs acoustic+LM+RDL, alongside LM-disabled pair | Same candidate set, LM/model revision and score conventions |

All baselines remain required. B5 is trained from scratch on approved train data;
the others require suitable pretrained ASR checkpoints. Unsupported N-best in a
particular baseline is documented; it does not waive its one-best evaluation.
The primary rescoring producers are Whisper and CTC adaptation tracks.

## Required ablations

- Unmodified acoustic model versus fine-tuned acoustic model, without rescoring.
- LoRA versus full fine-tuning where memory/time permit; record measured resource
  blocks and keep the full experiment explicitly pending if infeasible.
- Relational weight zero versus learned RDL; nonlearned lookup and shuffled-edge
  controls are additional ablations, not replacements for learned RDL.
- Entity/topic/quantifier relation families removed one at a time, retraining the
  graph when its input graph changes; preserve split/knowledge provenance.
- N-best size and score length normalization on validation; freeze selected
  settings and candidate sets before matched test rescoring.
- LM disabled/enabled when a lawful compatible external LM is selected.
- Raw versus Turkish-normalized evaluation for every system, with terminology
  supports and N-best oracle coverage as evaluator-only diagnostics.

Report raw/normalized WER/CER, terminology metrics, RTF/latency/hardware, memory,
paired uncertainty and actual repeated seeds. Do not invent an absent result or
hide a failed run. All configurations and graphs must identify hashes; conclusions
are limited by candidate coverage, domain support, verification quality, pretraining
contamination uncertainty, resource constraints and number of independent runs.

## Execution gates

1. Approve data, contract versions, exact baseline identities and resource budgets.
2. Validate the evaluator with hand-checkable offline fixtures once implemented.
3. Freeze all five baseline protocols, tuning search spaces and run manifests.
4. Execute train/validation work and select configurations without test access.
5. Freeze a pre-test commit/configuration/knowledge snapshot and review readiness.
6. Evaluate final test once per preregistered condition; retain failures and reports.
7. Audit independent reproduction and claims before slides or dataset release.

Initialization stops before gate 1 is satisfied; no execution commands are invented.
