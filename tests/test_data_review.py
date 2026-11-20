"""Offline tests for listening-review ingestion (synthetic fixtures, no real text)."""

import pytest

from turkish_asr_rdl.data import review as RV

H = "a" * 64
REFS = {
    "utt_a": {"utterance_id": "utt_a", "split": "test", "audio_sha256": H, "reference_text": "bütçe açığı arttı"},
    "utt_b": {"utterance_id": "utt_b", "split": "validation", "audio_sha256": H, "reference_text": "faiz yüzde 3"},
    "utt_c": {"utterance_id": "utt_c", "split": "train", "audio_sha256": H, "reference_text": "kanun teklifi"},
}


def dec(uid, decision, reviewer="rev_m1", utc="2026-11-18T10:00:00Z", **kw):
    d = {"utterance_id": uid, "audio_sha256": H, "reviewer_id": reviewer, "review_utc": utc,
         "annotation_policy_version": RV.POLICY_VERSION, "decision": decision,
         "corrected_text": None, "text_changed": False, "issue_codes": [], "note": ""}
    d.update(kw)
    return d


def kit(*reviews):
    return {"kit": "pilot-v0.1", "reviews": list(reviews)}


def test_status_mapping_and_undecided_skipped():
    new, stats = RV.ingest(REFS, [kit(
        dec("utt_a", "accept"),
        dec("utt_b", "correct", corrected_text="  faiz yüzde 3 oldu ", text_changed=True),
        dec("utt_c", "reject", issue_codes=["overlap"]),
        dec("utt_c", None, reviewer="rev_m2"),
    )], [], "primary")
    assert stats["new_records"] == 3 and stats["undecided"] == 1
    st = RV.status_table(REFS, new)
    assert st["utt_a"]["verification_status"] == "verified"
    assert st["utt_a"]["reference_origin"] == "upstream_human"
    assert st["utt_b"]["verification_status"] == "verified"
    assert st["utt_b"]["reference_origin"] == "local_human"
    assert [r for r in new if r["utterance_id"] == "utt_b"][0]["corrected_text"] == "faiz yüzde 3 oldu"
    assert st["utt_c"]["verification_status"] == "rejected"


def test_reingest_is_noop():
    k = kit(dec("utt_a", "accept"))
    first, _ = RV.ingest(REFS, [k], [], "primary")
    again, stats = RV.ingest(REFS, [k], first, "primary")
    assert again == [] and stats["already_ingested"] == 1


def test_latest_decision_of_same_reviewer_wins():
    recs, _ = RV.ingest(REFS, [kit(
        dec("utt_a", "reject", utc="2026-11-18T10:00:00Z", issue_codes=["noise"]),
        dec("utt_a", "accept", utc="2026-11-18T11:00:00Z"),
    )], [], "primary")
    assert RV.status_table(REFS, recs)["utt_a"]["verification_status"] == "verified"


def test_disagreement_needs_adjudication():
    recs, _ = RV.ingest(REFS, [kit(
        dec("utt_a", "accept", reviewer="rev_m1"),
        dec("utt_a", "reject", reviewer="rev_m2", issue_codes=["boundary_cut"]),
    )], [], "primary")
    assert RV.status_table(REFS, recs)["utt_a"]["verification_status"] == "in_review"
    adj, _ = RV.ingest(REFS, [kit(dec("utt_a", "accept", reviewer="rev_m6", utc="2026-11-19T09:00:00Z"))],
                       recs, "adjudication")
    st = RV.status_table(REFS, recs + adj)["utt_a"]
    assert st["verification_status"] == "adjudicated" and st["adjudicator_id"] == "rev_m6"


@pytest.mark.parametrize("bad", [
    dec("utt_x", "accept"),                                          # unknown ID
    dec("utt_a", "accept", audio_sha256="b" * 64),                   # stale audio
    dec("utt_a", "Accept"),                                          # typo in decision
    dec("utt_a", "accept", annotation_policy_version="whatever"),
    dec("utt_a", "accept", reviewer="rev_unknown"),
    dec("utt_a", "accept", reviewer="Ali Veli"),
    dec("utt_a", "accept", utc=None),
    dec("utt_a", "accept", issue_codes=["made_up"]),
    dec("utt_a", "accept", text_changed=True),
    dec("utt_a", "correct", corrected_text=""),
    dec("utt_a", "correct", corrected_text="bütçe açığı arttı"),     # same as reference
    dec("utt_a", "reject"),                                          # no issue code
])
def test_invalid_decisions_are_refused_all_or_nothing(bad):
    with pytest.raises(RV.ReviewError):
        RV.ingest(REFS, [kit(dec("utt_b", "accept"), bad)], [], "primary")


def test_audit_has_counts_but_no_text():
    recs, _ = RV.ingest(REFS, [kit(
        dec("utt_a", "accept"),
        dec("utt_b", "correct", corrected_text="faiz yüzde 3 oldu", text_changed=True),
    )], [], "primary")
    md = RV.audit_markdown("rel", REFS, RV.status_table(REFS, recs), recs, {"utt_a", "utt_b"})
    assert "| test | 1 | 1 | 1 | 0 | 0 | 0 | 0 |" in md
    assert "faiz" not in md and "bütçe" not in md
    assert "fully reviewed" in md and "**yes**" in md


def test_equivalent_audio_hash_is_accepted_only_via_map():
    heard = "d" * 64
    d = dec("utt_a", "accept", audio_sha256=heard)
    with pytest.raises(RV.ReviewError):
        RV.ingest(REFS, [kit(d)], [], "primary")
    recs, stats = RV.ingest(REFS, [kit(d)], [], "primary", {heard: H})
    assert stats["heard_equivalent_audio"] == 1
    assert recs[0]["audio_sha256"] == H and recs[0]["reviewed_audio_sha256"] == heard
    with pytest.raises(RV.ReviewError):  # map to a different release hash does not help
        RV.ingest(REFS, [kit(d)], [], "primary", {heard: "e" * 64})


def test_apply_to_references_corrects_rejects_and_keeps_ids():
    recs, _ = RV.ingest(REFS, [kit(
        dec("utt_a", "accept"),
        dec("utt_b", "correct", corrected_text="faiz yüzde 3 oldu", text_changed=True),
        dec("utt_c", "reject", issue_codes=["noise"]),
    )], [], "primary")
    kept, rejected = RV.apply_to_references(REFS, recs)
    by = {r["utterance_id"]: r for r in kept}
    assert set(by) == {"utt_a", "utt_b"} and [r["utterance_id"] for r in rejected] == ["utt_c"]
    assert by["utt_a"]["reference_text"] == "bütçe açığı arttı" and by["utt_a"]["verification_status"] == "verified"
    assert by["utt_b"]["reference_text"] == "faiz yüzde 3 oldu" and by["utt_b"]["reference_origin"] == "local_human"
    assert by["utt_b"]["verification_record_id"].startswith("rev_")
    assert REFS["utt_b"]["reference_text"] == "faiz yüzde 3"  # input untouched
