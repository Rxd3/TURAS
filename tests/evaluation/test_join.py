"""SYNTHETIC: rows here are made-up fixtures, never dataset transcripts."""

import copy
import dataclasses

import pytest

from turkish_asr_rdl.evaluation.join import (
    REFERENCE_FIELDS,
    JoinedRow,
    JoinError,
    join,
)

MANIFEST = "m" * 64


def pred(uid, **overrides):
    row = {
        "utterance_id": uid,
        "audio_sha256": f"h-{uid}",
        "dataset_release": "rel-1",
        "split": "test",
        "split_manifest_sha256": MANIFEST,
        "status": "ok",
        "hypothesis_text": f"SYNTHETIC hyp {uid}",
    }
    return {**row, **overrides}


def ref(uid, **overrides):
    row = {
        "utterance_id": uid,
        "audio_sha256": f"h-{uid}",
        "reference_text": f"SYNTHETIC ref {uid}",
        "leakage_group_id": f"g-{uid}",
        "split": "test",
        "dataset_release": "rel-1",
    }
    return {**row, **overrides}


def problems_of(predictions, references, **kwargs):
    with pytest.raises(JoinError) as caught:
        join(predictions, references, split="test", **kwargs)
    return caught.value.problems


def test_reference_fields_constant():
    assert REFERENCE_FIELDS == (
        "utterance_id", "audio_sha256", "reference_text", "leakage_group_id", "split", "dataset_release",
    )


def test_join_matches_rows_ordered_by_id():
    rows = join([pred("u2"), pred("u1")], [ref("u1"), ref("u2")], split="test")
    assert rows == (
        JoinedRow("u1", "g-u1", "SYNTHETIC ref u1", "SYNTHETIC hyp u1", "ok"),
        JoinedRow("u2", "g-u2", "SYNTHETIC ref u2", "SYNTHETIC hyp u2", "ok"),
    )


def test_join_reports_missing_extra_and_hash_mismatch():
    with pytest.raises(JoinError) as e:
        join([pred("u1", audio_sha256="other"), pred("u3")], [ref("u1"), ref("u2")], split="test")
    assert any("missing" in p and "'u2'" in p for p in e.value.problems)
    assert any("extra" in p and "'u3'" in p for p in e.value.problems)
    assert any("audio_sha256" in p and "'u1'" in p for p in e.value.problems)


def test_join_error_rows_become_empty_hypothesis():
    predictions = [pred("u1", status="error", hypothesis_text=None), pred("u2", status="error")]
    rows = join(predictions, [ref("u1"), ref("u2")], split="test")
    assert [(r.hypothesis, r.status) for r in rows] == [("", "error"), ("", "error")]


def test_empty_status_keeps_empty_hypothesis():
    rows = join([pred("u1", status="empty", hypothesis_text="")], [ref("u1")], split="test")
    assert rows[0].hypothesis == ""
    assert rows[0].status == "empty"


def test_join_filters_to_requested_split():
    references = [ref("u1"), ref("u9", split="train")]
    rows = join([pred("u1")], references, split="test")
    assert [r.utterance_id for r in rows] == ["u1"]


def test_other_split_rows_need_only_id_and_split():
    references = [ref("u1"), {"utterance_id": "u9", "split": "train"}]
    assert len(join([pred("u1")], references, split="test")) == 1


def test_duplicate_ids_on_either_side():
    problems = problems_of([pred("u1"), pred("u1")], [ref("u1"), ref("u1")])
    assert any("duplicate" in p and "predictions" in p for p in problems)
    assert any("duplicate" in p and "references" in p for p in problems)


def test_reference_duplicate_across_splits_is_reported():
    problems = problems_of([pred("u1")], [ref("u1"), ref("u1", split="train")])
    assert any("duplicate" in p and "'u1'" in p for p in problems)


def test_dataset_release_mismatch():
    problems = problems_of([pred("u1", dataset_release="rel-2")], [ref("u1")])
    assert any("dataset_release" in p and "'u1'" in p for p in problems)


def test_split_manifest_mismatch_with_argument():
    problems = problems_of([pred("u1")], [ref("u1")], split_manifest_sha256="0" * 64)
    assert any("split_manifest_sha256" in p and "'u1'" in p for p in problems)


def test_split_manifest_match_and_unchecked_without_argument():
    assert join([pred("u1")], [ref("u1")], split="test", split_manifest_sha256=MANIFEST)
    assert join([pred("u1", split_manifest_sha256="x")], [ref("u1")], split="test")


def test_problem_lists_ten_ids_and_the_true_count():
    references = [ref(f"u{n:02d}") for n in range(12)]
    (problem,) = problems_of([], references)
    assert "12" in problem
    assert "'u09'" in problem
    assert "'u10'" not in problem and "'u11'" not in problem


def test_problems_never_show_reference_text():
    with pytest.raises(JoinError) as e:
        join([pred("u1", audio_sha256="x"), pred("u3")], [ref("u1"), ref("u2")], split="test")
    shown = " ".join(e.value.problems) + str(e.value)
    assert "SYNTHETIC ref" not in shown


def test_join_error_message_is_one_line():
    with pytest.raises(JoinError) as e:
        join([pred("u3")], [ref("u1")], split="test")
    assert "\n" not in str(e.value)
    assert isinstance(e.value, ValueError)
    assert len(e.value.problems) == 2


@pytest.mark.parametrize("bad_id", [None, 5, ["u1"]])
def test_non_string_prediction_id_is_a_problem_not_a_crash(bad_id):
    problems = problems_of([pred(bad_id)], [ref("u1")])
    assert any("predictions" in p and "utterance_id" in p and "row 1" in p for p in problems)


def test_missing_reference_id_and_split_are_problems():
    problems = problems_of([pred("u1")], [{"reference_text": "SYNTHETIC"}])
    assert any("utterance_id" in p for p in problems)
    assert any("split" in p for p in problems)


def test_row_that_is_not_an_object_is_a_problem():
    problems = problems_of([pred("u1"), ["not", "an", "object"]], [ref("u1")])
    assert any("not JSON objects" in p and "row 2" in p for p in problems)


@pytest.mark.parametrize("field", [f for f in REFERENCE_FIELDS if f not in ("utterance_id", "split")])
def test_reference_in_split_needs_every_reference_field(field):
    broken = {k: v for k, v in ref("u1").items() if k != field}
    problems = problems_of([pred("u1")], [broken])
    assert any(field in p for p in problems)


@pytest.mark.parametrize("hypothesis", [None, 3])
def test_non_error_row_needs_string_hypothesis(hypothesis):
    problems = problems_of([pred("u1", hypothesis_text=hypothesis)], [ref("u1")])
    assert any("hypothesis_text" in p for p in problems)


def test_missing_status_is_a_problem():
    row = {k: v for k, v in pred("u1").items() if k != "status"}
    assert any("status" in p for p in problems_of([row], [ref("u1")]))


def test_join_does_not_mutate_inputs():
    predictions = [pred("u1", status="error", hypothesis_text=None)]
    references = [ref("u1")]
    before = copy.deepcopy((predictions, references))
    join(predictions, references, split="test")
    assert (predictions, references) == before


def test_joined_row_is_frozen():
    row = JoinedRow("u1", "g", "r", "h", "ok")
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.status = "error"  # type: ignore[misc]
