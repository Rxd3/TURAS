# Configuration areas

Reviewed JSON configurations are committed; the frozen protocol is
`evaluation/m45_final_protocol.json`. Owners per directory are below.

| Directory | Contents | Owner |
| --- | --- | --- |
| `datasets/` | Exact candidate revision, rights, curation/verification policy, grouping, seed, frozen release/hash | M1 |
| `baselines/` | Five distinct track identities, processor/tokenizer, generation, B5 training protocol | M2 Whisper/Qwen; M3 CTC/BiLSTM |
| `fine-tuning/` | Whisper LoRA/full and CTC parent, trainable parameters, selection, resources | M4 Whisper; M5 CTC |
| `rdl/` | Typed tables/graph, access filters, train/edge splits, encoder/decoder, negative sampling | M5 with M1 schema review |
| `decoding/` | Beam/N-best, token/score convention, LM/RDL weights, length and missing-relation policy | M6 shared; M5 RDL; model-owner review |
| `evaluation/` | Normalizer, WER/CER/term aggregation, failure policy, timing/hardware, grouped intervals | M6 |

Use portable paths and explicit immutable revisions; do not add placeholder
checkpoint paths or runnable configs that silently select unapproved data.
Resolved configurations and hashes accompany every real experiment. Configurations
are JSON.
