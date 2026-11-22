"""Score baseline runs outside the frozen protocol with the frozen test scorer's shared calls.

The scoring core and the paired recording-group bootstrap follow
scripts/m45_final_evaluate.py; a run's manifest.json output hashes take the place of
the frozen completed.json. Every check runs and all scoring finishes in memory before
anything is written. Outputs are aggregates only: no reference or hypothesis text.
"""

import argparse
import errno
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import jiwer
import numpy as np

from turkish_asr_rdl.data.screening import Lexicon
from turkish_asr_rdl.evaluation.domain import evaluate_domain
from turkish_asr_rdl.evaluation.join import join
from turkish_asr_rdl.evaluation.metrics import UtteranceScore, corpus_rates, score_pair
from turkish_asr_rdl.evaluation.normalize import (
    RAW_V1,
    TR_NORMALIZED_V1,
    normalize,
    word_tokens,
)
from turkish_asr_rdl.schemas.nbest import validate_nbest
from turkish_asr_rdl.schemas.predictions import validate_predictions
from turkish_asr_rdl.training.acoustic import read_jsonl, sha256_file, write_json

PROG = "m6_score_baselines"
VIEWS = (RAW_V1, TR_NORMALIZED_V1)
CROSSCHECK_TOLERANCE = 1e-12
VERIFIED_STATUSES = ("verified", "adjudicated")
MANIFEST = "manifest.json"
RUN_FILES = ("predictions.jsonl", "nbest.jsonl")
SUMMARY = "summary.json"
SYSTEM_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
PROTOCOL_KEYS = (
    (("dataset_release",), str),
    (("split_manifest_sha256",), str),
    (("lexicon", "path"), str),
    (("lexicon", "sha256"), str),
    (("decoder", "requested_n"), int),
    (("bootstrap", "seed"), int),
    (("bootstrap", "replicates"), int),
)
TEST_PROTOCOL_KEYS = (
    (("inputs", "test_ids"), str),
    (("test_reference", "payload_sha256"), str),
    (("test_reference", "variant"), str),
)


@dataclass(frozen=True)
class Inputs:
    protocol: dict
    ids: list[str]
    refs: list[dict]
    lexicon: Lexicon
    groups: list[str]
    protocol_sha256: str
    reference_sha256: str


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # one line and exit 1 instead of usage text and exit 2
        raise ValueError(f"usage error: {message}")


def summarize(scores: list[UtteranceScore]) -> dict:
    if not scores:
        return {"status": "unavailable", "utterances": 0}
    result = corpus_rates(scores)
    return {**asdict(result), "reference_words": result.words.ref_len,
            "reference_characters": result.chars.ref_len}


def _summaries(scores: dict, wanted: set[str] | None = None) -> dict:
    return {view: summarize([s for s in values if wanted is None or s.utterance_id in wanted])
            for view, values in scores.items()}


def _protocol_value(protocol: object, *keys: str, kind: type = str):
    value = protocol
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    if not isinstance(value, kind):
        raise ValueError(f"protocol {'.'.join(keys)} is missing or not a {kind.__name__}")  # noqa: TRY004
    return value


def _read_jsonl(path: Path) -> list[dict]:
    try:
        return read_jsonl(path)
    except ValueError as exc:  # bad UTF-8 or JSON, a BOM, a blank line
        raise ValueError(f"cannot read {path}: {exc}") from exc


def load_inputs(protocol_path: Path, protocol_sha256: str, split: str, references: Path,
                references_sha256: str) -> Inputs:
    if sha256_file(protocol_path) != protocol_sha256:
        raise ValueError("protocol checksum mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    for keys, kind in PROTOCOL_KEYS + (TEST_PROTOCOL_KEYS if split == "test" else ()):
        _protocol_value(protocol, *keys, kind=kind)
    if sha256_file(references) != references_sha256:
        raise ValueError("reference checksum mismatch")
    if split == "test" and references_sha256 != protocol["test_reference"]["payload_sha256"]:
        raise ValueError("references sha256 is not the protocol test reference payload")
    if split == "test":
        ids_path = Path(protocol["inputs"]["test_ids"])
    else:
        ids_path = Path("data/splits") / protocol["dataset_release"] / "validation.txt"
    ids = ids_path.read_text(encoding="utf-8").splitlines()
    refs = _read_jsonl(references)
    if [r.get("utterance_id") for r in refs] != ids:
        raise ValueError(f"reference IDs differ from {ids_path} (identity mismatch)")
    lexicon_path = Path(protocol["lexicon"]["path"])
    if sha256_file(lexicon_path) != protocol["lexicon"]["sha256"]:
        raise ValueError("lexicon checksum mismatch")
    groups = sorted({r["leakage_group_id"] for r in refs if isinstance(r.get("leakage_group_id"), str)})
    return Inputs(protocol, ids, refs, Lexicon.load(lexicon_path), groups,
                  protocol_sha256, references_sha256)


def _check_manifest(run_dir: Path, protocol: dict, split: str) -> str:
    """Check the run manifest against its files and the protocol; return its sha256."""
    data = (run_dir / MANIFEST).read_bytes()
    try:
        manifest = json.loads(data.decode("utf-8"))
    except ValueError as exc:
        raise ValueError(f"cannot read {run_dir / MANIFEST}: {exc}") from exc
    outputs = manifest.get("outputs") if isinstance(manifest, dict) else None
    inputs = manifest.get("inputs") if isinstance(manifest, dict) else None
    if not isinstance(outputs, dict):
        raise ValueError(f"{MANIFEST} has no outputs map")  # noqa: TRY004
    if not isinstance(inputs, dict):
        raise ValueError(f"{MANIFEST} has no inputs map")  # noqa: TRY004
    for name in RUN_FILES:
        if name not in outputs:
            raise ValueError(f"{MANIFEST} lists no hash for {name}")
    for name, digest in outputs.items():
        if sha256_file(run_dir / name) != digest:
            raise ValueError(f"{name} changed after its manifest was written")
    expected = {"dataset_release": protocol["dataset_release"],
                "split_manifest_sha256": protocol["split_manifest_sha256"], "split": split}
    for field, value in expected.items():
        if inputs.get(field) != value:
            raise ValueError(f"manifest inputs.{field} is not {value}")
    return hashlib.sha256(data).hexdigest()


def _agrees(independent: float, ours: float | None) -> bool:
    return ours is not None and abs(independent - ours) <= CROSSCHECK_TOLERANCE


def _crosscheck(joined: tuple, metrics: dict) -> None:
    """Audit WER/CER with an independent edit-distance library, as the frozen scorer does."""
    for view in VIEWS:
        rtext = [normalize(r.reference, view) for r in joined]
        htext = [normalize(r.hypothesis, view) for r in joined]
        words = jiwer.process_words([" ".join(word_tokens(s)) for s in rtext],
                                    [" ".join(word_tokens(s)) for s in htext])
        chars = jiwer.ReduceToListOfListOfChars()
        cer = jiwer.process_characters(rtext, htext, reference_transform=chars,
                                       hypothesis_transform=chars)
        if not (_agrees(words.wer, metrics[view].get("wer")) and _agrees(cer.cer, metrics[view].get("cer"))):
            raise ValueError("independent jiwer WER/CER cross-check failed")


def _group_scores(scores: list[UtteranceScore], joined: tuple, groups: list[str]) -> list[UtteranceScore]:
    result = []
    for group in groups:
        rates = corpus_rates([s for s, r in zip(scores, joined) if r.leakage_group_id == group])
        result.append(UtteranceScore(group, TR_NORMALIZED_V1, "ok", rates.words, rates.chars))
    return result


def _read_run(run_dir: Path, ids: list[str]) -> tuple[list[dict], list[dict]]:
    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    nbest = _read_jsonl(run_dir / "nbest.jsonl")
    checks = (("predictions.jsonl", validate_predictions(predictions, ids)),
              ("nbest.jsonl", validate_nbest(nbest, ids)))
    for name, check in checks:
        if not check.ok:
            raise ValueError(f"{name} fails the contract ({len(check.errors)} problem(s)); "
                             "run turas-check on it for details")
    return predictions, nbest


def score_run(system_id: str, run_dir: Path, inputs: Inputs,
              split: str) -> tuple[dict, list[UtteranceScore], list[dict]]:
    """Score one run: its report, its per-group normalized scores and its prediction rows."""
    protocol = inputs.protocol
    manifest_sha256 = _check_manifest(run_dir, protocol, split)
    predictions, nbest = _read_run(run_dir, inputs.ids)
    joined = join(predictions, inputs.refs, split=split,
                  split_manifest_sha256=protocol["split_manifest_sha256"])
    domain = evaluate_domain(joined, inputs.lexicon)
    containing = set(domain.pop("term_containing_utterance_ids"))
    scores = {view: [score_pair(r.utterance_id, r.reference, r.hypothesis, view, r.status) for r in joined]
              for view in VIEWS}
    domain["term_containing_utterances"] = {"count": len(containing),
                                            "metrics": _summaries(scores, containing)}
    verified = {r["utterance_id"] for r in inputs.refs if r.get("verification_status") in VERIFIED_STATUSES}
    coverage = Counter(row["actual_n"] for row in nbest)
    report = {
        "system_id": system_id, "split": split, "protocol_sha256": inputs.protocol_sha256,
        "reference_variant": protocol["test_reference"]["variant"] if split == "test" else None,
        "reference_sha256": inputs.reference_sha256, "manifest_sha256": manifest_sha256,
        "predictions_sha256": sha256_file(run_dir / "predictions.jsonl"),
        "nbest_sha256": sha256_file(run_dir / "nbest.jsonl"),
        "metrics": _summaries(scores),
        "m1_reported_verified_subset": {"utterances": len(verified), "metrics": _summaries(scores, verified)},
        "domain": domain,
        "nbest_coverage": {"requested_n": protocol["decoder"]["requested_n"],
                           "actual_n_distribution": dict(sorted(coverage.items())),
                           "hypotheses": sum(row["actual_n"] for row in nbest)},
        "error_rows": sum(r["status"] == "error" for r in predictions),
    }
    _crosscheck(joined, report["metrics"])
    report["independent_jiwer_crosscheck"] = "passed both views"
    return report, _group_scores(scores[TR_NORMALIZED_V1], joined, inputs.groups), predictions


def bootstrap_samples(n_groups: int, seed: int, replicates: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_groups, size=(replicates, n_groups))


def paired_interval(system: str, control: str, a: list[UtteranceScore], b: list[UtteranceScore],
                    samples: np.ndarray) -> dict:
    diffs = [corpus_rates([a[i] for i in sample]).wer - corpus_rates([b[i] for i in sample]).wer
             for sample in samples]
    lo, hi = np.percentile(diffs, [2.5, 97.5]).tolist()
    return {"system": system, "control": control,
            "normalized_wer_difference_percentage_points": 100 * (corpus_rates(a).wer - corpus_rates(b).wer),
            "paired_recording_bootstrap_95_interval_percentage_points": [100 * lo, 100 * hi]}


def one_best_agreement(a: list[dict], b: list[dict]) -> dict:
    """Count rows whose hypothesis_text is identical; two error rows (no text) count as equal."""
    first = {row["utterance_id"]: row.get("hypothesis_text") for row in a}
    second = {row["utterance_id"]: row.get("hypothesis_text") for row in b}
    if first.keys() != second.keys():
        raise ValueError("agreement needs the same utterance IDs in both runs")
    equal = sum(first[uid] == second[uid] for uid in first)
    return {"utterances": len(first), "hypothesis_byte_equal": equal, "differ": len(first) - equal}


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog=PROG, description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--split", choices=("test", "validation"), required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--references-sha256", required=True)
    parser.add_argument("--run", action="append", required=True, metavar="SYSTEM_ID=DIR")
    parser.add_argument("--pair", action="append", metavar="SYSTEM=CONTROL")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _parse_runs(values: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for value in values:
        system, sep, directory = value.partition("=")
        if not sep or not directory or not SYSTEM_ID.fullmatch(system) or system.casefold() == "summary":
            raise ValueError(f"--run {value!r} must be SYSTEM_ID=DIR, with a SYSTEM_ID of letters, "
                             "digits, '.', '_' or '-' other than 'summary'")
        if system.casefold() in {name.casefold() for name in runs}:
            raise ValueError(f"--run {system} is given twice")
        runs = {**runs, system: Path(directory)}
    return runs


def _parse_pairs(values: list[str], runs: dict[str, Path]) -> list[tuple[str, str]]:
    pairs = []
    for value in values:
        system, sep, control = value.partition("=")
        if not sep:
            raise ValueError(f"--pair {value!r} must be SYSTEM=CONTROL")
        for name in (system, control):
            if name not in runs:
                raise ValueError(f"--pair names a system not given with --run: {name}")
        pairs.append((system, control))
    return pairs


def _check_output(output: Path) -> None:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"output parent folder does not exist: {output.parent}")


def _score_runs(runs: dict[str, Path], inputs: Inputs, split: str) -> dict[str, tuple]:
    scored = {}
    for system, run_dir in runs.items():
        try:
            scored[system] = score_run(system, run_dir, inputs, split)
        except OSError as exc:
            raise ValueError(f"{system}: {_os_message(exc)}") from exc
        except ValueError as exc:
            raise ValueError(f"{system}: {exc}") from exc
    return scored


def _summary(split: str, inputs: Inputs, scored: dict[str, tuple], pairs: list[tuple[str, str]]) -> dict:
    boot = inputs.protocol["bootstrap"]
    samples = bootstrap_samples(len(inputs.groups), boot["seed"], boot["replicates"]) if pairs else None
    intervals = [paired_interval(s, c, scored[s][1], scored[c][1], samples) for s, c in pairs]
    agreement = [{"system": s, "control": c, **one_best_agreement(scored[s][2], scored[c][2])}
                 for s, c in pairs]
    return {"status": "m6_baseline_scoring_complete", "split": split,
            "protocol_sha256": inputs.protocol_sha256, "reference_sha256": inputs.reference_sha256,
            "utterances": len(inputs.ids), "recording_groups": len(inputs.groups),
            "systems": [report for report, _, _ in scored.values()], "pairs": intervals,
            "agreement": agreement, "bootstrap": {**boot, "groups": len(inputs.groups)}}


def _write_outputs(output: Path, files: dict[str, dict]) -> dict[Path, str]:
    """Write every file into a new folder; on any failure remove the folder again."""
    output.mkdir()
    try:
        digests = {}
        for name, value in files.items():
            write_json(output / name, value)
            digests[output / name] = sha256_file(output / name)
        return digests
    except BaseException:
        shutil.rmtree(output)
        raise


def _run(args: argparse.Namespace) -> int:
    runs = _parse_runs(args.run)
    pairs = _parse_pairs(args.pair or [], runs)
    _check_output(args.output)
    inputs = load_inputs(args.protocol, args.protocol_sha256, args.split, args.references,
                         args.references_sha256)
    if pairs and any(not isinstance(r.get("leakage_group_id"), str) for r in inputs.refs):
        raise ValueError("pairs need leakage_group_id on every reference")
    scored = _score_runs(runs, inputs, args.split)
    files = {f"{system}.json": report for system, (report, _, _) in scored.items()}
    digests = _write_outputs(args.output, {**files, SUMMARY: _summary(args.split, inputs, scored, pairs)})
    for system, (report, _, _) in scored.items():
        print(json.dumps({"scored_system": system, "rows": len(inputs.ids),
                          "normalized_wer": report["metrics"][TR_NORMALIZED_V1].get("wer")}), flush=True)
    for path, digest in digests.items():
        print(json.dumps({"written": str(path), "sha256": digest}), flush=True)
    return 0


def _os_message(exc: OSError) -> str:
    name = exc.filename if exc.filename is not None else "a file"
    if exc.errno == errno.ENOENT:
        return f"file not found: {name}"
    if exc.errno == errno.EISDIR or (exc.errno == errno.EACCES and exc.filename is not None
                                     and Path(exc.filename).is_dir()):
        return f"is a directory, not a file: {name}"
    if exc.errno in (errno.EACCES, errno.EPERM):
        return f"permission denied: {name}"
    if exc.errno == errno.EEXIST:
        return f"already exists: {name}"
    if exc.errno == errno.ENOTDIR:
        return f"not a directory: {name}"
    return f"{exc.strerror or 'operating-system error'}: {name}"


def _refuse(message: str) -> int:
    encoding = getattr(sys.stderr, "encoding", None) or "utf-8"
    line = " ".join(f"{PROG}: {message}".splitlines())
    sys.stderr.write(line.encode(encoding, "backslashreplace").decode(encoding) + "\n")
    return 1


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(_parser().parse_args(argv))
    except OSError as exc:
        return _refuse(_os_message(exc))
    except ValueError as exc:
        return _refuse(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
