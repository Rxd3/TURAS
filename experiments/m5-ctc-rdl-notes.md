# Member 5: CTC adaptation, heterogeneous RDL and rescoring

Current integration status: owner decisions are approved and M1/M6 interfaces
are integrated on `feat/m4-m5-integration`. CUDA and synthetic resource checks
pass. See [owner decisions](../docs/owner_decisions.md) and
[readiness](../docs/m45_readiness.md); the earlier handoff below is retained as history.

Updated 2026-11-19. This is an implementation/review handoff, not an experiment
report. Training bundles are not ready locally, no pretrained weights were
downloaded, and no corpus model, prediction bundle or benchmark metric exists.

## Delivered components

| Area | Implementation | Actual validation |
| --- | --- | --- |
| CTC adaptation | `training/ctc.py`, `scripts/m5_train_ctc.py` | Tiny random CTC model backward pass; frozen frontend; real tokenizer OOV rejection |
| CTC N-best | `decoding/ctc.py` | Exact sequence probability and prefix search checked against every alignment of a tiny fixture, including repeated labels |
| Checkpoint inference | `decoding/adapted.py` | Real tiny saved Whisper reload/generation and hash-tamper rejection; no Hub weight download |
| Graph construction | `rdl/graph.py` | Typed keys/provenance, group separation, aggregate reconstruction and held-out adjacency checks |
| Trainable heterogeneous graph | `rdl/model.py`, `rdl/train.py` | Learned weights change, synthetic loss improves, typed negatives and safe save/load pass |
| Hypothesis linking/scoring | `rdl/scoring.py` | Reference-free synthetic queries, coverage and no-relation floor |
| Joint fusion | `decoding/rescoring.py`, `rdl/rescore.py` | Hand-computed component/tie cases and offline synthetic artifact roundtrip |

The code shares Member 4's train-only loader/optimization loop. This PR is stacked
on `feat/m4-whisper-finetuning` to retain ownership and avoid copying that code.
M1's release/extractor and M6's harness/evaluator branches remain separate
dependencies; this work does not merge or approve their PRs.

## Candidate CTC parent and spelling policy

Proposed B4: `mpoyraz/wav2vec2-xls-r-300m-cv7-turkish`, revision
`708639f50559d7970f462e13ec64d3f059ca89f6`, published `pytorch_model.bin` SHA-256
`008d69c82bf45c82ec5485e550abc86ddabefe75080f22e1ebfdc1699889d82e`.
The official Hub metadata declares `Wav2Vec2ForCTC`, 38 output labels and blank
ID 0. These are inspected metadata, not independently verified downloaded weights.
M3/M6 must accept the exact checkpoint, training-data/contamination audit,
baseline processor/vocabulary and decoder before an adaptation run.

No head resize or fresh head is allowed. The convolutional feature extractor is
frozen explicitly; transformer/head parameters train, with nonreentrant activation
checkpointing. The candidate profile proposes batch one, accumulation 16, three
epochs, LR 1e-5, bf16 CUDA and seed 17. Other seeds and budgets await review.

Training labels use NFC, Turkish lowercase, punctuation-to-space except ASCII
apostrophe, and whitespace collapse. This is a proposed CTC label policy, not a
change to shared evaluation. Digits and unsupported characters remain visible:
if the exact tokenizer produces UNK/blank/empty targets the run stops before
optimization. M3/M6 need to approve a spoken-number/vocabulary policy for actual
references; unsupported amounts are never silently deleted or examples dropped.

## Graph training boundary

The draft graph config pins actual static/train CSV hashes from M1 commit
`2ebfa78` and release split hash
`a8fff65b99fcf11d9452685e438e250207032ecfe11152741562147126d095b8`.
Every source hash is checked. Static provenance must be independent external
knowledge; dynamic provenance/access must be train-only. Train utterance IDs
must match the frozen train inventory. No validation/test CSV directory is a
permitted training source.

M1's labels are automatically extracted from upstream references. They are weak
supervision, not human-verified relational facts. Quantifier IDs can be shared
across split tables; only nodes actually referenced by the selected training
groups enter this snapshot. Unseen inference values receive a declared floor.

Internal graph selection holds out whole groups from the corpus **training**
annotations. It does not access corpus-validation/test references. Co-mentions
are reconstructed from per-utterance mentions because M1's aggregate records do
not retain all contributing groups. A held-out co-mention already evidenced in a
training group is excluded from the internal evaluation set, with its overlap
count recorded; hiding only the direct edge would leave equivalent mention-path
evidence in training. Novel held-out relations and reverse equivalents never
enter adjacency/labels. Report group/support counts and unavailable evaluation
explicitly if no novel relations remain.

The model has node-type embeddings/projections, separate per-relation/direction
mean message transforms, self transforms, dropout and learned typed bilinear
decoders. Undirected scores are symmetrized. It is genuinely parameter-trained,
not a lookup table. BCE uses type-compatible corruption negatives excluding all
known train/internal-dev positives and their undirected equivalents. Saturated
positives with no possible corruption are counted; unknown edges may still be
true, so this is a closed-world assumption.

Internal average precision (step precision-recall area), filtered tail MRR and
Hits@10 have declared candidates and pessimistic ranking ties. These are relation
diagnostics on a training-group holdout, not ASR performance or independent
corpus-validation results. After configuration selection, refitting with
`validation_fraction=0` is an explicit separate hashed run.

`exclude_relations` can remove relation families before graph learning and
scoring. For example, excluding `rel_expresses` is the quantifier-family
ablation. Retrain for every changed graph; reusing old weights is not an ablation.
Shuffled-graph/nonlearned controls remain protocol tasks, not measured results.

## Reference-free inference and N-best

`AdaptedAsrAdapter` verifies the externally pinned checkpoint-manifest hash and
all inventoried files. Full models load saved safetensors. LoRA verifies the exact
base weight hash and merges the compatible adapter for decoding. The processor
is loaded from the hashed checkpoint inventory. `load()` and `nbest(samples)`
work standalone; `transcribe(batch)` imports M6's dataclasses when their reviewed
`decoding.interface` is integrated. No M6 code is copied. The bridge still needs
an actual shared-harness integration test after that dependency merges.

CTC candidate discovery uses a pruned prefix beam, while the exported acoustic
score is an **exact sum over all CTC alignments** for each discovered transcript.
The approximate discovery score remains separate. The top requested raw ranks
are deduplicated without padding or replacing ranks; decoding uses
`group_tokens=False` because the prefix is already collapsed. RDL comparisons
must share the identical candidate artifact and distinguish native decoder
one-best from acoustic-only exact-score reranking.

The linker uses only frozen external aliases and candidate text. It scores
entity/topic and entity/co-mention queries with learned graph embeddings.
Quantity parsing uses M1's approved `data.relational.extract_quantities`; that
dependency is required for the full scorer and is not independently reimplemented.
Known quantities use the `expresses` decoder with a virtual utterance context
equal to mean linked entity embeddings. **That pooling is a modeling
approximation**, not demonstrated predictive validity, and must be tested against
quantity-disabled controls. Unknown/contextless quantities get the frozen missing
probability. Alias coverage does not measure all linguistic entity mentions.

The relational score is mean log probability, with explicit counts and no-relation
floor. Joint scoring preserves acoustic/LM/RDL components and chooses maximum
score, then original rank, then candidate ID. LM is optional and disabled in the
draft profile; no LM was trained or added. Enabled missing/nonfinite scores fail.

## Commands and current blockers

Run these implemented checks from an environment with the editable package:

```powershell
python -m turkish_asr_rdl.training.ctc --config configs/fine-tuning/ctc_xlsr_turkish.json --preflight
python -m turkish_asr_rdl.rdl.train --config configs/rdl/heterogeneous_relations.json --preflight
python -m turkish_asr_rdl.rdl.rescore --config configs/decoding/rdl_rescoring.json --preflight
```

They currently report missing inputs/reviews. Once real bundles and owner review
are available, create ignored resolved configurations with exact paths/hashes:

```powershell
python -m turkish_asr_rdl.training.ctc --config artifacts/m5/ctc-resolved.json --output artifacts/m5/ctc-seed17
python -m turkish_asr_rdl.rdl.train --config artifacts/m5/rdl-resolved.json --output artifacts/m5/rdl-seed17
python -m turkish_asr_rdl.rdl.rescore --config artifacts/m5/fusion-resolved.json --output results/nbest/rdl-validation.jsonl
```

These resolved paths describe future files, not existing checkpoints/runs.
Thin wrappers exist in `scripts/m5_train_ctc.py`, `m5_train_rdl.py`, `m5_rescore.py`.
Outputs never overwrite an existing run. Graph weights use safetensors and
checksum inventories; code/config/environment/seed evidence accompanies them.
The local `.venv` is CPU-only for development. CUDA setup and real memory/time
profiling await the data; no GPU feasibility or training-duration claim is made.

Rescoring requires exact ordered IDs and artifact hashes; recursive gold fields
are rejected. Validation is the default selection split. Test rescoring requires
frozen fusion and hashed M6 validation-selection evidence. Offline rescoring
timing covers N-best read, graph/fusion and JSONL write; **it is not end-to-end ASR
RTF**. M6 must combine this with actual acoustic decoding and standard timing.

## Review and remaining experimental work

Local validation: 23 offline tests passed on Python 3.13.14, PyTorch 2.15.0+cpu,
Transformers 5.23.0 and PEFT 0.21.3. The fixtures/random models are explicitly
synthetic and their intermediate losses/scores are not benchmark results.

M3/M6: exact CTC baseline, tokenizer/number policy and matching conditions.
M1/M6: CSV provenance, weak labels, aggregate reconstruction and group holdout.
M4/M6: checkpoint bridge, candidate conventions, fusion and timing integration.
M6 additionally reviews the optional test environment/CI. No technical approval
or consumer acknowledgment is invented.

Pending: approved data handoff, corresponding baseline runs, CUDA profiling,
actual CTC/graph training, validation model/fusion selection, real N-best export,
baseline/fine-tuned/RDL and ablation comparisons, end-to-end timing, and independent
reproduction. The repaired evaluator-only test archive remains unextracted and
outside all training inputs.
