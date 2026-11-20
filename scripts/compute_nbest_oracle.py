"""Compute N-best oracle WER/CER and candidate diversity from TURAS JSONLs.

Writes aggregate-only metrics (never reference or hypothesis text) so the JSON
output can be committed safely under results/evaluation/.

Usage:
    python scripts/compute_nbest_oracle.py \
        --references path/to/test_references.jsonl \
        --nbest F1=path/to/f1.nbest.jsonl F2=path/to/f2.nbest.jsonl \
        --output results/evaluation/m6_nbest_oracle.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from turkish_asr_rdl.evaluation.metrics import UtteranceScore, corpus_rates, score_pair
from turkish_asr_rdl.evaluation.normalize import normalize


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{lineno}: expected JSON object")
        rows.append(obj)
    return rows


def _extract_reference_text(row: Mapping[str, Any]) -> str:
    for key in ("reference_text", "text", "reference", "transcript"):
        val = row.get(key)
        if isinstance(val, str):
            return val
    raise ValueError(
        f"reference row for {row.get('utterance_id')!r} missing text field"
    )


def _extract_candidate_texts(row: Mapping[str, Any]) -> list[str]:
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return [""]
    ordered = sorted(
        candidates,
        key=lambda c: int(c.get("rank", 10**9)) if isinstance(c, Mapping) else 10**9,
    )
    texts: list[str] = []
    for cand in ordered:
        if isinstance(cand, Mapping):
            text = cand.get("text")
            texts.append(text if isinstance(text, str) else "")
    return texts or [""]


def evaluate_nbest_oracle(
    references: Mapping[str, str],
    nbest_rows: Sequence[Mapping[str, Any]],
    view: str = "tr-normalized-v1",
) -> dict[str, Any]:
    """Compute 1-best vs N-best oracle WER/CER and candidate diversity."""
    by_uid: dict[str, Mapping[str, Any]] = {}
    for row in nbest_rows:
        uid = row.get("utterance_id")
        if not isinstance(uid, str) or not uid:
            raise ValueError("N-best row missing string utterance_id")
        if uid in by_uid:
            raise ValueError(f"duplicate utterance_id in N-best file: {uid}")
        by_uid[uid] = row

    ref_ids = set(references.keys())
    nbest_ids = set(by_uid.keys())
    if ref_ids != nbest_ids:
        missing = sorted(ref_ids - nbest_ids)[:5]
        extra = sorted(nbest_ids - ref_ids)[:5]
        raise ValueError(
            f"utterance_id mismatch (missing={missing}, extra={extra})"
        )

    one_best_scores: list[UtteranceScore] = []
    oracle_scores: list[UtteranceScore] = []
    oracle_beats_rank1 = 0
    total_candidates = 0
    total_unique_norm = 0

    for uid in sorted(ref_ids):
        ref_text = references[uid]
        row = by_uid[uid]
        status = str(row.get("status", "ok"))
        cand_texts = _extract_candidate_texts(row)
        total_candidates += len(cand_texts)
        total_unique_norm += len({normalize(t, view) for t in cand_texts})

        scored = [score_pair(uid, ref_text, text, view, status=status) for text in cand_texts]
        rank1 = scored[0]
        best = min(
            scored,
            key=lambda s: (s.words.errors, s.chars.errors),
        )
        one_best_scores.append(rank1)
        oracle_scores.append(best)
        if best.words.errors < rank1.words.errors:
            oracle_beats_rank1 += 1

    r1 = corpus_rates(one_best_scores)
    ro = corpus_rates(oracle_scores)
    n_utts = len(ref_ids)
    one_best_wer_pct = None if r1.wer is None else round(r1.wer * 100.0, 4)
    oracle_wer_pct = None if ro.wer is None else round(ro.wer * 100.0, 4)
    headroom_pp = (
        None
        if one_best_wer_pct is None or oracle_wer_pct is None
        else round((r1.wer - ro.wer) * 100.0, 4)  # type: ignore[operator]
    )

    return {
        "view": view,
        "utterances": n_utts,
        "one_best_wer_pct": one_best_wer_pct,
        "oracle_wer_pct": oracle_wer_pct,
        "headroom_pp": headroom_pp,
        "one_best_cer_pct": None if r1.cer is None else round(r1.cer * 100.0, 4),
        "oracle_cer_pct": None if ro.cer is None else round(ro.cer * 100.0, 4),
        "oracle_beats_rank1_count": oracle_beats_rank1,
        "oracle_beats_rank1_pct": round(oracle_beats_rank1 / n_utts * 100.0, 2),
        "mean_candidates": round(total_candidates / n_utts, 3),
        "mean_unique_normalized_candidates": round(total_unique_norm / n_utts, 3),
        "one_best_words": {
            "substitutions": r1.words.substitutions,
            "deletions": r1.words.deletions,
            "insertions": r1.words.insertions,
            "reference_words": r1.words.ref_len,
        },
        "oracle_words": {
            "substitutions": ro.words.substitutions,
            "deletions": ro.words.deletions,
            "insertions": ro.words.insertions,
            "reference_words": ro.words.ref_len,
        },
    }


def _parse_system_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"expected SYSTEM_ID=PATH, got {spec!r}")
    sid, raw_path = spec.split("=", 1)
    if not sid.strip() or not raw_path.strip():
        raise ValueError(f"invalid system spec {spec!r}")
    return sid.strip(), Path(raw_path.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute aggregate-only N-best oracle WER/CER diagnostics."
    )
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument(
        "--nbest",
        required=True,
        nargs="+",
        help="One or more SYSTEM_ID=PATH_TO_NBEST_JSONL entries.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--view", default="tr-normalized-v1")
    args = parser.parse_args(argv)

    ref_rows = _read_jsonl(args.references)
    references: dict[str, str] = {}
    for row in ref_rows:
        uid = row.get("utterance_id")
        if not isinstance(uid, str) or not uid:
            raise ValueError("reference row missing string utterance_id")
        if uid in references:
            raise ValueError(f"duplicate utterance_id in references: {uid}")
        references[uid] = _extract_reference_text(row)

    systems: dict[str, Any] = {}
    for spec in args.nbest:
        sid, path = _parse_system_spec(spec)
        nbest_rows = _read_jsonl(path)
        result = evaluate_nbest_oracle(references, nbest_rows, view=args.view)
        result["nbest_sha256"] = _sha256(path)
        systems[sid] = result

    payload = {
        "schema_version": "0.1",
        "view": args.view,
        "references_sha256": _sha256(args.references),
        "utterances": len(references),
        "systems": systems,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for sid, res in systems.items():
        print(
            f"{sid}: 1-best={res['one_best_wer_pct']}% | oracle={res['oracle_wer_pct']}% "
            f"| headroom={res['headroom_pp']} pp | beats_r1={res['oracle_beats_rank1_count']}/{res['utterances']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
