# Member 6 checker kit: design

Date: 2026-11-17. Author: Member 6's coding assistant, from conversations with Member 6.
Status: draft for Member 6's approval. Nothing here is implemented.

## 1. Intent and why

Member 6 (M6) is the last stage of TURAS. Members 1–5 build and train their own
tracks on their own machines. M6 then:

1. checks every handoff against the repo contracts,
2. fixes what is broken (through pull requests on the owner's branch),
3. re-runs every system on one machine (the RTX 5070 Ti laptop) with one timing
   protocol and one evaluator, and
4. tunes the rescoring weights on validation and writes the final tables.

Waiting for teammates before writing any of this would put all the checking and
fixing in the last days. Most of M6's tooling depends only on the written
contracts, which already exist. Building it now gives teammates (and their AI
agents) the rules and a one-command checker before they write code, so fewer
fixes are needed at the end.

Binding decisions from Member 6:
- Language is Turkish (WorldSpeech `tr_tr`). The pasted Arabic master plan is a
  template only.
- The repo contracts win wherever they differ from the master plan
  (`docs/contracts/*`, `docs/evaluation.md`, `experiments/PROTOCOL.md`).
- The 5070 Ti runs only M6's jobs (smoke tests, the unified re-run, rescoring
  weight search). Teammates train on their own machines.

## 2. Goals, non-goals, anti-goals

Goals:
- G1. Teammates can run `turas-check` on their own outputs and see pass or fail per contract rule before handing off.
- G2. One shared evaluator produces raw and Turkish-normalized WER/CER with S/D/I/N counts, plus the HF audio-course metrics, for every system.
- G3. One shared timing harness and inference interface, so the final re-run on the 5070 Ti gives comparable RTF for all nine systems.
- G4. Measured hardware evidence (VRAM, load time, rough speed) for each candidate model on the 5070 Ti, before teammates commit to checkpoints.
- G5. A split leakage audit that M6 runs on Member 1's release.

Non-goals:
- Writing Members 2–5's adapters, training code or rescorer (they own them; see TEAM_TASKS.md).
- Curating or verifying the dataset (Member 1).
- Choosing checkpoints (Members 2/3 decide; M6 supplies measurements).
- Producing any result, prediction or metric before real runs exist.

Anti-goals (failure even if it "works"):
- Any fabricated transcript, prediction, score or result, including test fixtures that look like dataset content (tests/README.md).
- Weakening a check so a teammate's bundle passes.
- Reading test references anywhere except the offline evaluator.
- Taking credit for teammates' work, or fixing their code without a reviewed PR.

## 3. Constraints

- Target machine: Windows 11, RTX 5070 Ti Laptop 12 GB (sm_120), driver 591.84, 31 GB RAM, Ryzen AI 9 HX 370, 720 GB free on C:. Pinned env: Python 3.13, torch 2.14.1+cu130, transformers 5.23.0, datasets 5.2.0, peft 0.21.3, jiwer 4.0.0, numpy 2.5.5 (from `nlp/project3-asr/requirements.txt`).
- Development machine: the GTX 1650 laptop (4 GB, no torch install yet). All Phase 0 code and tests must run on CPU without torch.
- CONTRIBUTING.md: member branch `feat/m6-integration-evaluation-qa`; logic in `src/turkish_asr_rdl/`; `scripts/` are thin wrappers; offline unit tests only (no model/data download, no GPU); M6 PRs need an independent non-author review.
- User standards: tests first, ≥80% coverage on new code, files 200–400 lines, functions <50 lines, no mutation, explicit errors with friendly messages, conventional commits, commit/push only when Member 6 says so.
- No new dependency where stdlib or a pinned package covers it. torch_geometric 2.8.1 is added only after its smoke test passes (P1).
- Course rule: "justify your choice of WER, CER, and additional metrics" from the HF audio course chapter 5; keep baseline results separate from the contribution.

## 4. Approaches considered

| | Approach | Trade-off |
|---|---|---|
| **A (chosen)** | Build the checker kit now (contract-only parts), smoke-test models on the 5070 Ti, then check, fix and re-run when teammates deliver | Most front-loaded work; fewest late fixes; teammates get the rules before coding |
| B | Build only normalizer and metrics now; write checks per teammate at the end | Less work now; format clashes surface at the deadline |
| C | A plus M6's own copies of B1–B4 inference | Strongest safety net; duplicates Members 2/3's owned work and roughly doubles M6's load |

Member 6 chose A: "if there is anything we can do before their part, yes it will save us time."

## 5. Six-day schedule (deadline Monday 2026-11-23, whole project)

Today is Tuesday 2026-11-17. M6's kit must be thin and finished by Wednesday night, because teammates start coding within days.

| Day | M6 work | Needs from team |
|---|---|---|
| Tue 17 (tonight) | Approve design and plan. PR1: environment in the repo, shared audio loader, `AsrAdapter`, `run_timed`, prediction/N-best validators, `AGENTS.md`. Push the branch; the 5070 Ti downloads models and runs smoke tests overnight. | Teammates confirm whether they have GPUs (see risks) |
| Wed 18 | PR1 review and merge. PR2: evaluator (normalizer, aligner, metrics, strict join) and the CLI. Send the smoke-test report (VRAM, speed per model) to Members 2–5. | M1: data subset and recording-grouped splits |
| Thu 19 | PR3: leakage audit, bootstrap, report tables. Audit Member 1's release and approve or return fixes. | M1: frozen release; M2/M3: first baseline predictions |
| Fri 20 | `turas-check` on incoming baselines; fix lists; overnight unified re-run of B1–B5 on the 5070 Ti. | M4/M5: fine-tuned checkpoints |
| Sat 21 | Check and re-run fine-tunes; tune the rescoring weights on validation once N-best and R1 exist. | M5: rescorer |
| Sun 22 | Final test evaluation, tables, RTF-vs-WER plot, error analysis; run the `submission-correctness` check (at least a day before the deadline); slides. | All: review claims |
| Mon 23 | Submit / present. | |

**Cuts forced by the deadline** (each reported as a limitation, never hidden):
- One training seed per system, labeled single-seed, instead of seeds 17/42/2026.
- Timing: batch 1, 3 repeats on a frozen 30-minute timing subset, instead of a 1-hour subset.
- Terminology P/R/F1: implemented only if Member 1 freezes a lexicon by Thursday; otherwise reported as not done.
- Contract amendments (normalization 0.2, nbest keys, data audio format, predictions order): one combined docs PR with `docs/decisions.md` entries, not separate contract PRs.
- Manifest and timing validators check required keys only.
- F2 (full fine-tune) and the R1 ablations run only if GPU time remains after the headline systems.

**Team-level risks M6 raises today** (outside M6's code, but they decide whether Monday works):
- Member 1's release is the critical path. Suggested size: train 20–30 h, validation about 1 h, test 1–2 h, split by recording group. Human listening checks on test only (about 240 clips per test hour, roughly 40 minutes per member if shared across six), with validation left as upstream text and labeled as such.
- If teammates have no GPU, the fine-tunes can't finish on their machines. The 5070 Ti would then need a training queue, which changes this design (it currently runs M6's jobs only).

## 6. Phases

### Phase 0: contract-only kit (now, CPU, this laptop)

Build order, one PR each (matches the schedule in section 5):
1. **PR1 (Tue):** P0.2 environment, P0.6 audio loader + `AsrAdapter` + `run_timed`, P0.5 prediction and N-best validators, P0.1 `AGENTS.md`. Once this is pushed to the M6 branch, Phase 1 can start without waiting for the merge.
2. **PR2 (Wed):** P0.3 contract amendments (one docs PR, run in parallel), P0.4 normalizer, aligner, metrics and strict join, and the P0.8 CLI subcommands that exist so far.
3. **PR3 (Thu):** P0.7 leakage audit, P0.4 bootstrap and report tables, and the remaining CLI subcommands.

**P0.1 `AGENTS.md` (repo root).** The master plan's agent rules, rewritten to the repo contracts:
1. Never write your own normalizer or WER/CER; import from `turkish_asr_rdl.evaluation`.
2. Never re-split data; read only `data/splits/<release>/{train,validation,test}.txt`.
3. Inference writes prediction JSONL per `docs/contracts/predictions.md` (no reference text) and N-best as a separate artifact per `docs/contracts/nbest.md`.
4. Wrap inference in the shared timing harness (P0.6); never time by hand.
5. Run `turas-check` on every output before handoff.
6. Known traps: in transformers 5.23.0, Whisper `generate(num_return_sequences>1)` returns copies of the 1-best; `asr_transcript` is never a reference; Python `lower()`/`casefold()` are wrong for Turkish.
Keep it short (one screen) and link the contracts for detail.

**P0.2 Environment in the repo.** Move the pinned environment from `nlp/project3-asr` into the repo so teammates and the 5070 Ti share it: `requirements/requirements.in`, `requirements/windows-cu130.txt` (the existing compiled pins), `scripts/check_env.py` (existing script and its 43 tests, moved to `tests/`), and a setup section in `docs/reproducibility.md` that includes the two Windows gotchas already found (FFmpeg *shared* build; `UV_PYTHON_INSTALL_DIR` outside AppData when running inside the Claude desktop app). Mac/Linux teammates compile their own lock with the header command.

**P0.3 Normalization contract 0.2.** Before coding the normalizer, amend `docs/contracts/normalization.md` (M6-owned, M1 review) to close the gaps the research found. Proposed rules (M6's call, marked as such in the PR):
- U+02BC (modifier letter apostrophe) is treated like U+0027/U+2019.
- `%` is kept like a symbol (it carries meaning in quantities, e.g. `%5`); other punctuation becomes a space as before.
- Unicode format characters (category Cf, e.g. U+200B, U+FEFF) are deleted in the normalized view.
- "Whitespace" means Python `str.isspace()`; documented as including U+001C–U+001F.
- Digit-adjacent apostrophes (`2026'da` → `2026 da`) and number grouping (`1.250,50` → `1 250 50`) stay as the contract says; documented as known token-count effects.
- Tie-break direction for alignment: backtrace from the end, preferring substitution, then deletion, then insertion; golden test `a b` → `b a` gives S=2.

**P0.4 Evaluator (`src/turkish_asr_rdl/evaluation/`).** Pure Python, no torch.
- `normalize`: `raw-v1` (identity) and `tr-normalized-v1` per contract 0.2; versioned and idempotent.
- `align`: own Levenshtein DP over token lists and code-point lists with the P0.3 tie-break; returns S, D, I, hits and N. `jiwer` is used only in tests to cross-check S+D+I totals, with a whitespace-split transform (jiwer splits on ASCII space only).
- `metrics`: per-utterance and corpus sums; WER, CER, WAcc = 1 − WER; empty references counted separately; corpus rate undefined when total N = 0.
- `join`: strict join of predictions to evaluator-only references by utterance ID; duplicate, missing, extra, audio-hash, release or split-hash mismatches fail before any metric. `error` rows produce a separate error-inclusive view (mapped to empty hypotheses, counted).
- `terminology`: exact canonical-term P/R/F1 (longest match, left to right, no overlap, canonical-ID tie-break). Built and tested now on a labeled toy lexicon; runs on the real lexicon when Member 1 freezes it.
- `bootstrap`: paired group bootstrap, 2,000 replicates, seed 2026, resampling leakage groups; the index matrix is hashed and the NumPy version recorded.
- `report`: Markdown/CSV tables with baselines and contributions in separate sections; HF-course names mapped explicitly (orthographic WER = raw-v1, normalised WER = tr-normalized-v1, RTFx = 1/RTF, WAcc). The course's `BasicTextNormalizer` is not used: it splits `İZMİR` into three tokens.

**P0.5 Validators (`src/turkish_asr_rdl/schemas/`).** Stdlib functions that return a list of rule violations (empty = pass):
- predictions JSONL: required fields and enums, finite numbers, forbidden keys (`reference_text`, `gold`, and similar), one row per frozen ID, stable order.
- N-best: `requested_n`/`actual_n`, unique candidate IDs, ranks 1..n, mandatory acoustic score with `score_convention_id`, null scores carry a reason, **distinct candidate texts** (catches the Whisper duplicate trap), no padding.
- timing sidecar and run manifest: required groups from `docs/contracts/artifacts.md` and `docs/evaluation.md`.
- split files: sorted, unique, LF, disjoint, union equals the release.
- inference export: only the allowlisted input fields.

**P0.6 Shared inference interface and timing harness (`src/turkish_asr_rdl/decoding/` shared parts, M6-owned).**
- A small `AsrAdapter` protocol that every model owner implements: load, transcribe a batch of audio, optionally return a scored N-best. torch is an optional import.
- `run_timed(...)`: excluded warm-up batches drawn from train/validation (at least 3), 3 timed repeats, batch size recorded. **Primary RTF is end-to-end** (audio read, decode and resample through result serialization), using `torch.cuda.synchronize()` plus `time.perf_counter()` at the boundaries. **Secondary "core-model" RTF** (the master plan's view) is labeled separately. It writes predictions, timing sidecars and a run manifest.
- Hardware records before and after each repeat: GPU name, driver, clocks, power, temperature and `clocks_event_reasons.*` from nvidia-smi. Thermal or hardware slowdown is flagged; `sw_power_cap` alone is normal on laptops and is only recorded.

**P0.7 Leakage audit.** Given Member 1's manifest and split files: recording-group overlap (`source_url` + `session_date`), time-interval overlap on (`source_url`, `start`, `end`), exact duplicate normalized text across splits, and a speaker-unknown statement. The output is an audit report that M6 signs.

**P0.8 `turas-check` CLI (`scripts/turas_check.py`, a thin wrapper).** Subcommands: `predictions`, `nbest`, `timing`, `manifest`, `splits`, `leakage`, `evaluate`, `report`. Exit 0 = pass; exit 1 = rule violations, one friendly line each; no tracebacks.

### Phase 1: 5070 Ti smoke tests (after P0.2 and P0.6 are pushed, before teammates deliver)

Needs model downloads but no teammate code. The smoke wrappers live on a separate branch, `feat/m6-smoke-tests`, which is never merged to `main`. They are harness checks, not the B1–B4 deliverables, and owners may copy them. Candidates are the research shortlist; if Members 2/3 later pin a different revision, M6 measures that revision. For each candidate:
- whisper-small and whisper-large-v3 (fp16; beam 10 through a path that gives distinct candidates);
- Qwen3-ASR-1.7B-hf and 0.6B-hf, in the main env, never the `qwen-asr` package;
- mpoyraz XLS-R-300m Turkish cv7 and cv8, with the processor built explicitly (AutoProcessor fails in the pinned env);
- a toy heterogeneous graph in torch_geometric 2.8.1 (forward and backward on CUDA).

Measure for each: load, a few clips from one upstream train shard (accuracy is not looked at), peak VRAM, load time, rough RTF and N-best distinctness. Also measure training-step memory for whisper-small full fine-tuning and whisper-large-v3 LoRA; whisper-large-v3 full fine-tuning should be infeasible on 12 GB, and the measurement records that. The output is a hardware feasibility report for Members 2–5 (TEAM_TASKS D05, D13, D14). This also exercises the timing harness on a real GPU.

### Phase 2: when teammates deliver (waits on Members 1–5)

1. Member 1's release: `turas-check splits` plus `leakage`; approve or return a fix list.
2. Each model bundle: `turas-check` passes, then the unified re-run on the 5070 Ti through the owner's adapter and `run_timed`.
3. Rescoring: grid-search the fusion weights on validation only, using Member 5's rescorer and Members 3/4's N-best; freeze before test.
4. Final test evaluation once per preregistered condition; bootstrap CIs; tables; RTF-vs-WER plot; error analysis; slides.

Fixes: a failing check becomes a fix list on the owner's branch, or an M6 PR into that branch with the owner's review. Never a force-push.

## 7. How code reaches the 5070 Ti

Code is written on this laptop on `feat/m6-integration-evaluation-qa`. Member 6 gave a standing OK (2026-11-17) to push the `feat/m6-*` branches; each PR into `main` is still asked separately. After a push, the 5070 Ti session clones or pulls the branch (gh is logged in as `afel6` there too) and runs the exact commands sent by message. Small reports (JSON or Markdown) come back by message. Large artifacts stay on the 5070 Ti outside git (for example `G:\turas-artifacts\`). The 5070 Ti session never designs or edits code.

## 8. Facts this design relies on (verified 2026-11-17)

- The WorldSpeech `tr_tr` upstream train and test splits share recordings, so TURAS re-splits by recording group. There is no speaker column. The test transcripts were kept only where whisper-large-v3-turbo agreed with them (CER < 0.3), which biases results toward Whisper and must be reported as a limitation. The sample showed no financial content.
- The mpoyraz CTC vocabulary has no digits, so B4 and F3 cannot emit numerals.
- Qwen3-ASR `-hf` runs natively in transformers ≥ 5.13.
- torch_geometric 2.8.1 is tested only up to torch 2.12.
- The CTC decoder libraries (torchaudio's decoder, flashlight-text, pyctcdecode) don't fit the pinned env; Member 3 writes a small prefix beam search, and the exact score comes from `CTCLoss`.

Full notes are in the session research summary; the key items are repeated here so this doc stands alone.

## 9. Decisions from the builder check (assistant's calls; Member 6 may override)

1. **Member 1 file formats.** The evaluator reads the curated manifest JSONL from `docs/contracts/data.md`, using only `utterance_id`, `audio_sha256`, `reference_text`, `leakage_group_id` and `split`. `split_manifest_sha256` is the SHA-256 of the split manifest file's bytes. "Stable input order" is the sorted split-file order. M6 proposes these in the predictions contract; Member 1 reviews.
2. **Leakage audit inputs.** Use `data.md` fields only, with no new fields. Blocking: a `leakage_group_id`, `duplicate_cluster_id` or `audio_sha256` appearing in two splits; overlapping `start_ms`/`end_ms` within one `source_recording_key` across splits (touching intervals, end == start, are not overlap). Report-only: identical normalized text across splits, with counts; texts of 10 or more tokens are flagged for M6's manual review. These thresholds are fixed now, before any data is seen.
3. **Manifest and N-best keys.** Validators enforce only keys the contracts name. For null scores, M6 proposes `score_details.null_reasons: {field: reason}` and a `dedup_map` in nbest contract 0.2 (Members 4/5 review).
4. **Distinct candidates.** Compare raw `text` byte for byte. A duplicate fails unless `dedup_map` records it. A list where every candidate equals rank 1 always fails when `actual_n` > 1. Candidates identical after `tr-normalized-v1` produce a warning, not a failure.
5. **Error rows.** The headline table is the all-ID, error-inclusive view: `error` rows are scored as empty hypotheses, and the error count is shown next to WER. M6 retries failing utterances once in Phase 2 and records it.
6. **Normalizer label.** It is built now against the contract 0.2 proposal and keeps the label `tr-normalized-v1`. It is frozen at the first real evaluation; any later change creates `v2` and a full re-evaluation.
7. **Adapter input.** `AsrAdapter.transcribe` receives float32 16 kHz mono arrays plus utterance IDs, produced by M6's shared audio loader inside the timed region. That gives one audio path for every system, so end-to-end RTF stays fair. Member 1's release stores audio as 16 kHz mono FLAC under an artifact root; M6 proposes this in the data contract.
8. **Official predictions.** The 5070 Ti re-run produces the official test predictions for all systems: full test set, batch 1, the precision each model card supports. The owner's own bundle serves as reproduction evidence; a gap above 0.5 absolute normalized WER is investigated.
9. **RTF condition.** The primary condition is batch 1. Timing repeats (3) run on a frozen 1-hour timing subset of test IDs, sampled by recording group with seed 2026 at release freeze. Batched throughput is an optional, labeled secondary view.
10. **R1 timing.** Time the producer run, then a separately timed rescoring pass over the saved N-best, and report the sum. Rescorer cold-load time is excluded and reported. Ablation variants (LM, shuffled graph, relation families) are scored offline from saved N-best without timing.
11. **Test references.** They are stored on the 5070 Ti in a separate protected folder that only `turas-check evaluate` reads, in its own process after inference has finished. Inference commands get only the inference export, and the runner refuses any config that points into the protected folder.
12. **Pinned environment.** Teammates' inference adapters must run in the pinned Windows env for the final re-run; their training env is free. Any exception (for example R1, if PyTorch Geometric fails its smoke test) is declared and recorded in the timing sidecar.
13. **Repository facts** (checked 2026-11-17): `Rxd3/project3-turkish-asr-rdl` redirects to `Rxd3/TURAS`, which is public; `main` is protected; collaborators are `Rxd3`, `afel6` (Member 6), `Devinci101` and `abdu3061`. M6 fixes the stale repo name in the README, D01 and `docs/repository_setup.md`. Audio, references and predictions never enter git (already in `.gitignore`).

14. **Delivery to teammates** (Member 6, 2026-11-17: a teammate will review "within days"). Phase 0 lands as small PRs into `main`, one per build-order step in section 6, each with an independent teammate review. Teammates get the kit by pulling `main`. Until a step is merged, `AGENTS.md` lists only what is already on `main`.

Open until Phase 2: how teammates deliver checkpoints to the 5070 Ti (D14). Proposal: private Hugging Face model repos with pinned revisions; decide before the first handoff.

## 10. Left to the builder

- Internal module split inside `evaluation/`, `schemas/`, `decoding/`, within the size limits.
- Exact CLI flag names and report column order.
- The test fixture strings (short, labeled synthetic, never dataset-like).
