"""Score runs in tr-normalized-v1 and in tr-number-invariant-v0, where a number counts the same as digits or words.

tr-number-invariant-v0 is a report view outside the frozen protocol, named and versioned
apart from the mandatory views as docs/contracts/normalization.md asks. Each text goes through:

1. Numbers written with ASCII digits become Turkish words. This step reads the text before
   tr-normalized-v1, because that normalizer turns the thousands dot and the decimal comma
   into spaces (`1.250,50 -> 1 250 50`), after which `3,5` and `3-5` look the same.
   - An integer, plain (`2026`) or with thousands dots (`1.250.000`: a first group of one to
     three digits not starting with 0, then groups of exactly three), is read in groups of
     three up to trilyon: `105 -> yüz beş`, `1000 -> bin`, `21.000 -> yirmi bir bin`,
     `1000000 -> bir milyon`.
   - A decimal comma reads as virgül, and the digits after it as an integer:
     `3,5 -> üç virgül beş`, `3,25 -> üç virgül yirmi beş`.
   - A percent sign right before or after the number, spaces between allowed, reads as
     yüzde before it: `%5`, `5%`, `% 5 -> yüzde beş`.
   - Leading zeros read one by one: `0 -> sıfır`, `05 -> sıfır beş`.
   A percent sign directly followed by a digit opens the next number: `5 %80` and `5%80 -> beş yüzde seksen`.
   Each replacement is padded with spaces, so a suffix after the digits becomes its own
   token, with or without an apostrophe: `15'te -> on beş te`, `1'inci -> bir inci`,
   `1.ci -> bir ci`, `2026da -> iki bin yirmi altı da`.
2. tr-normalized-v1, unchanged (`normalize` from turkish_asr_rdl.evaluation.normalize).
3. A token made only of two or more number words is split into them, the way Turkish
   spelling writes numbers: `onbeş -> on beş`, `ikibinyirmialtı -> iki bin yirmi altı`.
   A word that only starts with a number word stays (`onlar`, `binlerce`, `yüzde`).
Text with no ASCII digit and no glued number words is exactly its tr-normalized-v1 text.

Not handled; these stay as steps 2 and 3 leave them:
- an integer, or a decimal part, of 10^15 or more leaves the whole number as written;
- a dot is never a decimal mark (`3.5 -> üç beş`, `14.30 -> on dört otuz`) and a comma is
  always one (`1,250` reads as bir virgül iki yüz elli); thousands split by spaces read as
  separate numbers;
- signs, fractions, dates, times, Roman numerals and non-ASCII digits are not read;
- ordinals and case suffixes are not matched to their spelled forms: `1'inci -> bir inci`
  while a spelled `birinci` stays one token, and `15'te -> on beş te` while `on beşte`
  stays two; a bare `1.` reads as a cardinal, since it cannot be told from a full stop;
- a glued number with a suffix (`onbeşinci`) is not split.

Scoring follows scripts/m6_score_baselines.py: the shared join, `score_pair`,
`corpus_rates` and the independent jiwer cross-check. The invariant view scores the
canonical texts under tr-normalized-v1, which leaves them unchanged. Each run's split
comes from its prediction rows, whose IDs must be exactly the reference IDs of that split.
An error row scores as an empty hypothesis and counts in error_rows. The output holds
aggregates and hashes only, never text.
"""

import argparse
import errno
import json
import re
import sys
from pathlib import Path
from typing import NoReturn

import jiwer

from turkish_asr_rdl.evaluation.align import EditCounts
from turkish_asr_rdl.evaluation.join import JoinedRow, join
from turkish_asr_rdl.evaluation.metrics import corpus_rates, score_pair
from turkish_asr_rdl.evaluation.normalize import TR_NORMALIZED_V1, normalize, word_tokens
from turkish_asr_rdl.schemas.predictions import validate_predictions
from turkish_asr_rdl.training.acoustic import read_jsonl, sha256_file, write_json

PROG = "m6_number_invariant_wer"
NUMBER_INVARIANT_V0 = "tr-number-invariant-v0"
VIEWS = (TR_NORMALIZED_V1, NUMBER_INVARIANT_V0)
CROSSCHECK_TOLERANCE = 1e-12

ZERO = "sıfır"
DECIMAL_COMMA = "virgül"
PERCENT = "yüzde"
ONES = ("", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz")
TENS = ("", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan")
HUNDRED = "yüz"
SCALES = ("", "bin", "milyon", "milyar", "trilyon")
THOUSAND_POWER = 1
MAX_SPELLED_DIGITS = 3 * len(SCALES)  # up to 999 trilyon
# The vocabulary is prefix-free, so a glued run of number words splits in only one way.
_WORD = "|".join(sorted({ZERO, HUNDRED, *ONES[1:], *TENS[1:], *SCALES[1:]}, key=len, reverse=True))
_NUMBER_WORD = re.compile(_WORD)
_GLUED_NUMBER_WORDS = re.compile(f"(?:{_WORD}){{2,}}")
_SPACES = r"[^\S\r\n]*"
_NUMBER = re.compile(
    rf"(?:(?P<pre>%{_SPACES})|(?<![0-9]))"
    r"(?P<int>[1-9][0-9]{0,2}(?:\.[0-9]{3})+(?![0-9])|[0-9]+)"
    r"(?:,(?P<frac>[0-9]+))?"
    rf"(?(pre)|(?P<post>{_SPACES}%(?![0-9]))?)"
)


def _below_thousand(number: int) -> list[str]:
    hundreds, rest = divmod(number, 100)
    lead = [] if hundreds == 0 else [HUNDRED] if hundreds == 1 else [ONES[hundreds], HUNDRED]
    return lead + [word for word in (TENS[rest // 10], ONES[rest % 10]) if word]


def _group_words(group: int, power: int) -> list[str]:
    lead = [] if group == 1 and power == THOUSAND_POWER else _below_thousand(group)
    return lead + ([SCALES[power]] if power else [])


def _integer_words(number: int) -> list[str]:
    groups = [(power, number // 1000**power % 1000) for power in reversed(range(len(SCALES)))]
    return [word for power, group in groups if group for word in _group_words(group, power)] or [ZERO]


def _digit_words(digits: str) -> list[str] | None:
    """Words for an ASCII digit string, or None when it is too long to spell."""
    significant = digits.lstrip("0")
    if len(significant) > MAX_SPELLED_DIGITS:
        return None
    zeros = [ZERO] * (len(digits) - len(significant))
    return zeros + (_integer_words(int(significant)) if significant else [])


def _spell(match: re.Match[str]) -> str:
    whole = _digit_words(match["int"].replace(".", ""))
    fraction = [] if match["frac"] is None else _digit_words(match["frac"])
    if whole is None or fraction is None:
        return match[0]
    decimal = [DECIMAL_COMMA, *fraction] if fraction else []
    percent = [PERCENT] if match["pre"] is not None or match["post"] is not None else []
    return f" {' '.join(percent + whole + decimal)} "


def _split_glued(token: str) -> list[str]:
    return _NUMBER_WORD.findall(token) if _GLUED_NUMBER_WORDS.fullmatch(token) else [token]


def number_invariant(text: str) -> str:
    """Return the tr-number-invariant-v0 text: digits spelled, then tr-normalized-v1, then glued numbers split."""
    normalized = normalize(_NUMBER.sub(_spell, text), TR_NORMALIZED_V1)
    return " ".join(word for token in word_tokens(normalized) for word in _split_glued(token))


_CANONICAL = {TR_NORMALIZED_V1: lambda text: text, NUMBER_INVARIANT_V0: number_invariant}


def _counts(counts: EditCounts) -> dict:
    return {"substitutions": counts.substitutions, "deletions": counts.deletions,
            "insertions": counts.insertions, "hits": counts.hits, "n": counts.ref_len}


def _agrees(independent: float, ours: float | None) -> bool:
    return ours is not None and abs(independent - ours) <= CROSSCHECK_TOLERANCE


def _crosscheck(refs: list[str], hyps: list[str], summary: dict) -> None:
    """Audit WER/CER with an independent edit-distance library, as the frozen scorer does."""
    words = jiwer.process_words([" ".join(word_tokens(s)) for s in refs],
                                [" ".join(word_tokens(s)) for s in hyps])
    chars = jiwer.ReduceToListOfListOfChars()
    cer = jiwer.process_characters(refs, hyps, reference_transform=chars, hypothesis_transform=chars)
    if not (_agrees(words.wer, summary["wer"]) and _agrees(cer.cer, summary["cer"])):
        raise ValueError("independent jiwer WER/CER cross-check failed")


def score_view(joined: tuple[JoinedRow, ...], view: str) -> dict:
    """Aggregate edit counts and rates of one view over the joined rows."""
    canonical = _CANONICAL[view]
    pairs = [(canonical(r.reference), canonical(r.hypothesis)) for r in joined]
    scores = [score_pair(r.utterance_id, ref, hyp, TR_NORMALIZED_V1, r.status)
              for r, (ref, hyp) in zip(joined, pairs)]
    rates = corpus_rates(scores)
    summary = {"words": _counts(rates.words), "characters": _counts(rates.chars),
               "wer": rates.wer, "cer": rates.cer}
    _crosscheck([normalize(ref, TR_NORMALIZED_V1) for ref, _ in pairs],
                [normalize(hyp, TR_NORMALIZED_V1) for _, hyp in pairs], summary)
    return summary


def _read_hashed(path: Path, expected_sha256: str | None = None) -> tuple[list[dict], str]:
    """Read JSONL rows and the sha256 of exactly the bytes read."""
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"references sha256 mismatch: {path}")
    try:
        rows = read_jsonl(path)
    except (ValueError, RecursionError) as exc:  # bad UTF-8 or JSON, a BOM, a blank line, deep nesting
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if sha256_file(path) != digest:
        raise ValueError(f"file changed while it was read: {path}")
    return rows, digest


def score_run(run_dir: Path, refs: list[dict]) -> dict:
    """Score one run's predictions.jsonl against the references of its split."""
    path = run_dir / "predictions.jsonl"
    predictions, digest = _read_hashed(path)
    check = validate_predictions(predictions)
    if not check.ok:
        raise ValueError(f"{path} fails the predictions contract ({len(check.errors)} problem(s)); "
                         f"first: {check.errors[0]}")
    if not predictions:
        raise ValueError(f"{path} has no rows")
    split = predictions[0]["split"]
    joined = join(predictions, refs, split=split)
    return {"split": split, "predictions_sha256": digest, "utterances": len(joined),
            "error_rows": sum(r.status == "error" for r in joined),
            "metrics": {view: score_view(joined, view) for view in VIEWS}}


def _score_named(system: str, run_dir: Path, refs: list[dict]) -> dict:
    try:
        return score_run(run_dir, refs)
    except OSError as exc:
        raise ValueError(f"--run {system}: {_os_message(exc)}") from exc
    except ValueError as exc:
        raise ValueError(f"--run {system}: {exc}") from exc


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:  # one line and exit 1 instead of usage text and exit 2
        raise ValueError(f"usage error: {message}")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog=PROG, description=__doc__.splitlines()[0])
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--references-sha256", required=True)
    parser.add_argument("--run", action="append", required=True, metavar="SYSTEM_ID=RUN_DIR")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _parse_runs(values: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for value in values:
        system, sep, directory = value.partition("=")
        if not sep or not system or not directory:
            raise ValueError(f"--run {value!r} must be SYSTEM_ID=RUN_DIR")
        if system in runs:
            raise ValueError(f"--run {system} is given twice")
        runs = {**runs, system: Path(directory)}
    return runs


def _check_output(output: Path) -> None:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"output parent folder does not exist: {output.parent}")


def _write(output: Path, summary: dict) -> None:
    try:
        write_json(output, summary)
    except BaseException:
        output.unlink(missing_ok=True)
        raise


def _run(args: argparse.Namespace) -> int:
    runs = _parse_runs(args.run)
    _check_output(args.output)
    refs, references_sha256 = _read_hashed(args.references, args.references_sha256)
    systems = {system: _score_named(system, run_dir, refs) for system, run_dir in runs.items()}
    summary = {"status": "m6_number_invariant_scoring_complete", "views": list(VIEWS),
               "script_sha256": sha256_file(Path(__file__)), "references_sha256": references_sha256,
               "systems": systems}
    _write(args.output, summary)
    print(json.dumps({"written": str(args.output), "sha256": sha256_file(args.output)}), flush=True)
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
