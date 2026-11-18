"""Listening-review records and status resolution (policy ``turas-verbatim-v0.1``).

A review kit exports one decision per utterance. ``validate_decision`` checks a
decision against the protected reference row and the verification guide;
``make_record`` turns it into a data-contract review record whose ID is a
content hash, so ingesting the same export twice adds nothing.
``resolve_status`` applies the guide's status mapping to all records of one
utterance:

- an ``adjudication`` record, if present, is final (latest one wins):
  accept/correct -> ``adjudicated``, reject -> ``rejected``;
- otherwise each primary reviewer's latest decision counts; if they disagree
  the utterance stays ``in_review`` until adjudicated;
- accept -> ``verified`` (``upstream_human``), correct -> ``verified``
  (``local_human``), reject -> ``rejected``.

Decisions are never inferred: an empty decision is skipped, not defaulted.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

POLICY_VERSION = "turas-verbatim-v0.1"
DECISIONS = ("accept", "correct", "reject")
ISSUE_CODES = ("boundary_cut", "overlap", "noise", "edited_minutes", "wrong_segment", "non_speech", "other")
ROLES = ("primary", "adjudication")
_REVIEWER = re.compile(r"^rev_[a-z0-9_]{1,32}$")


class ReviewError(ValueError):
    """A decision that the verification guide does not allow."""


def _utc(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ReviewError("review_utc missing")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReviewError(f"review_utc not ISO 8601: {value!r}") from exc
    return value


def validate_decision(d: dict, ref: dict | None) -> bool:
    """Raise ReviewError for an invalid decision; return False if undecided."""
    uid = d.get("utterance_id")
    if ref is None:
        raise ReviewError(f"{uid}: unknown utterance ID")
    if d.get("audio_sha256") != ref["audio_sha256"]:
        raise ReviewError(f"{uid}: audio hash differs from the release (stale kit?)")
    decision = d.get("decision")
    if decision is None:
        return False
    if decision not in DECISIONS:
        raise ReviewError(f"{uid}: decision {decision!r} is not one of {DECISIONS}")
    if d.get("annotation_policy_version") != POLICY_VERSION:
        raise ReviewError(f"{uid}: policy {d.get('annotation_policy_version')!r} != {POLICY_VERSION}")
    if d.get("reviewer_id") == "rev_unknown" or not _REVIEWER.match(str(d.get("reviewer_id", ""))):
        raise ReviewError(f"{uid}: reviewer_id must look like rev_m1 (got {d.get('reviewer_id')!r})")
    _utc(d.get("review_utc"))
    codes = d.get("issue_codes") or []
    bad = [c for c in codes if c not in ISSUE_CODES]
    if bad:
        raise ReviewError(f"{uid}: unknown issue codes {bad}")
    if decision == "accept" and d.get("text_changed"):
        raise ReviewError(f"{uid}: text was edited but the decision is accept")
    if decision == "correct":
        text = d.get("corrected_text")
        if not isinstance(text, str) or not text.strip():
            raise ReviewError(f"{uid}: correct needs a non-empty corrected_text")
        if text.strip() == ref.get("reference_text"):
            raise ReviewError(f"{uid}: correct but the text equals the reference (use accept)")
    if decision == "reject" and not codes:
        raise ReviewError(f"{uid}: reject needs at least one issue code")
    return True


def make_record(d: dict, kit: str | None, role: str) -> dict:
    if role not in ROLES:
        raise ReviewError(f"role {role!r} not in {ROLES}")
    codes = sorted(d.get("issue_codes") or [])
    body = {
        "utterance_id": d["utterance_id"], "reviewer_id": d["reviewer_id"], "role": role,
        "review_utc": d["review_utc"], "audio_sha256": d["audio_sha256"],
        "annotation_policy_version": d["annotation_policy_version"], "decision": d["decision"],
        "corrected_text": d["corrected_text"].strip() if d["decision"] == "correct" else None,
        "issue_codes": codes, "note": d.get("note", ""), "kit": kit,
    }
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {"verification_record_id": "rev_" + digest[:20], **body}


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _latest(records: Iterable[dict]) -> dict:
    return max(records, key=lambda r: (r["review_utc"], r["verification_record_id"]))


def resolve_status(records: list[dict]) -> dict:
    """Current status of one utterance from all its review records."""
    adj = [r for r in records if r["role"] == "adjudication"]
    if adj:
        r = _latest(adj)
        status = "rejected" if r["decision"] == "reject" else "adjudicated"
        return {"verification_status": status, "verification_record_id": r["verification_record_id"],
                "decision": r["decision"], "adjudicator_id": r["reviewer_id"],
                "reference_origin": "local_human" if r["decision"] == "correct" else "upstream_human"}
    by_reviewer = defaultdict(list)
    for r in records:
        by_reviewer[r["reviewer_id"]].append(r)
    current = [_latest(v) for _, v in sorted(by_reviewer.items())]
    outcomes = {(r["decision"], r["corrected_text"]) for r in current}
    if len(outcomes) > 1:
        return {"verification_status": "in_review", "verification_record_id": None, "decision": "conflict",
                "adjudicator_id": None, "reference_origin": None}
    r = _latest(current)
    return {"verification_status": "rejected" if r["decision"] == "reject" else "verified",
            "verification_record_id": r["verification_record_id"], "decision": r["decision"],
            "adjudicator_id": None,
            "reference_origin": "local_human" if r["decision"] == "correct" else "upstream_human"}


def ingest(refs: dict[str, dict], kits: list[dict], existing: list[dict], role: str) -> tuple[list[dict], Counter]:
    """Validate every decision first; return (new records, counters). All-or-nothing."""
    known = {r["verification_record_id"] for r in existing}
    new, stats = [], Counter()
    errors = []
    for kit in kits:
        for d in kit.get("reviews", []):
            try:
                if not validate_decision(d, refs.get(d.get("utterance_id"))):
                    stats["undecided"] += 1
                    continue
            except ReviewError as exc:
                errors.append(str(exc))
                continue
            rec = make_record(d, kit.get("kit"), role)
            if rec["verification_record_id"] in known:
                stats["already_ingested"] += 1
                continue
            known.add(rec["verification_record_id"])
            new.append(rec)
    if errors:
        raise ReviewError(f"{len(errors)} invalid decision(s), nothing ingested:\n" + "\n".join(errors))
    stats["new_records"] = len(new)
    return new, stats


def status_table(refs: dict[str, dict], records: list[dict]) -> dict[str, dict]:
    by_utt = defaultdict(list)
    for r in records:
        by_utt[r["utterance_id"]].append(r)
    return {u: {"utterance_id": u, "split": refs[u]["split"], **resolve_status(rs)}
            for u, rs in sorted(by_utt.items())}


def audit_markdown(release: str, refs: dict[str, dict], status: dict[str, dict],
                   records: list[dict], pilot_ids: set[str]) -> str:
    """Aggregate counts only: no reference or corrected text."""
    split_size = Counter(r["split"] for r in refs.values())
    lines = [f"# {release} listening-verification audit (aggregate, no text)", "",
             f"Policy: `{POLICY_VERSION}`. Generated by `scripts/m1_apply_reviews.py` from "
             f"{len(records)} review records ({len({r['reviewer_id'] for r in records})} reviewer IDs).", ""]

    def table(title: str, ids: set[str] | None) -> None:
        lines.extend([f"## {title}", "",
                      "| Split | In scope | Reviewed | accept | correct | reject | conflict | adjudicated |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for split in ("test", "validation", "train"):
            scope = {u for u, r in refs.items() if r["split"] == split and (ids is None or u in ids)}
            rows = [status[u] for u in scope if u in status]
            c = Counter(r["decision"] for r in rows)
            adj = sum(1 for r in rows if r["adjudicator_id"])
            lines.append(f"| {split} | {len(scope)} | {len(rows)} | {c['accept']} | {c['correct']} | "
                         f"{c['reject']} | {c['conflict']} | {adj} |")
        lines.append("")

    if pilot_ids:
        table("Pilot sample", pilot_ids)
    table("Whole release", None)
    issues = Counter(code for r in records for code in r["issue_codes"])
    lines += ["Issue codes over all records: " + (", ".join(f"`{k}` {v}" for k, v in issues.most_common()) or "none"), ""]
    frozen_ready = all(
        status.get(u, {}).get("verification_status") in ("verified", "adjudicated", "rejected")
        for u, r in refs.items() if r["split"] in ("validation", "test"))
    lines += [f"Validation/test fully reviewed (every row verified, adjudicated or rejected): "
              f"**{'yes' if frozen_ready else 'no'}** "
              f"(validation {split_size['validation']} rows, test {split_size['test']} rows).", ""]
    return "\n".join(lines)
