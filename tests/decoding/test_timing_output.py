"""Timing harness: adapter results the output files cannot hold. SYNTHETIC tones and results only."""
import itertools
import types
from dataclasses import replace

import pytest

from turkish_asr_rdl.decoding.interface import Candidate, Hypothesis
from turkish_asr_rdl.decoding.timing import run_timed, write_run_outputs
from turkish_asr_rdl.schemas.jsonl import read_jsonl
from turkish_asr_rdl.schemas.nbest import validate_nbest
from turkish_asr_rdl.schemas.predictions import validate_predictions

WARM = ("warm_1", "warm_2", "warm_3")
TIMED = ("utt_a", "utt_b")


class FixedResultAdapter:
    """Returns the SYNTHETIC result the test gives, renamed for each utterance."""

    system_id = "sys_synthetic"
    model_artifact_sha256 = "b" * 64
    decoder_config_sha256 = "a" * 64

    def __init__(self, result, requested_n=1):
        self.result, self.requested_n = result, requested_n
        self.score_convention_id = "nbest-synthetic-v1" if requested_n else None
        self.config = {"beam": max(requested_n, 1)}

    def load(self):
        return None

    def transcribe(self, batch):
        return [replace(self.result, utterance_id=item.utterance_id) for item in batch]


class RaisingDetails(dict):
    """score_details whose lookup fails with a message that is not valid UTF-8 text."""

    def get(self, key, default=None):
        raise LookupError("SYNTHETIC \ud800")


@pytest.fixture
def run_and_write(tmp_path, write_tone, make_input, run_context):
    """Run the harness on SYNTHETIC 1 s tones with a fake clock, write the outputs; return (run, out dir)."""

    def _run(adapter):
        for uid in (*WARM, *TIMED):
            write_tone(f"audio/{uid}.flac", 16000, 1.0)
        timed, warm = [make_input(u) for u in TIMED], [make_input(u) for u in WARM]
        ticks = itertools.count(0.0, 0.5)
        result = run_timed(adapter, timed, warm, audio_root=tmp_path, run_id="run_synthetic", repeats=1,
                           clock=lambda: next(ticks), probe=lambda: {"ok": False, "error": "no gpu"},
                           sync=lambda: None)
        write_run_outputs(tmp_path / "out", result, timed, run_context, adapter, command="SYNTHETIC cmd")
        return result, tmp_path / "out"

    return _run


def with_candidate(**changes):
    candidate = replace(Candidate("c1", 1, "SYNTHETIC c1", (1,), -1.0, "kind"), **changes)
    return Hypothesis("utt_x", "ok", "SYNTHETIC c1", (candidate,))


UNWRITABLE = {
    "details_none": with_candidate(score_details=None),
    "details_list": with_candidate(score_details=["SYNTHETIC"]),
    "null_reasons_string": with_candidate(score_details={"null_reasons": "none"}),
    "keys_collide_as_json": with_candidate(score_details={1: "int key", "1": "str key"}),
    "null_reasons_keys_collide": with_candidate(score_details={"null_reasons": {None: "a", "null": "b"}}),
    "tuple_key": with_candidate(score_details={(1, 2): "tuple key"}),
    "object_value": with_candidate(score_details={"x": object()}),
    "infinite_score": with_candidate(acoustic_log_score=float("-inf")),
    "token_ids_none": with_candidate(token_ids=None),
    "lone_surrogate_in_candidate": with_candidate(text="SYNTHETIC \ud800"),
    "lone_surrogate_in_text": Hypothesis("utt_x", "ok", "SYNTHETIC \ud800"),
    "candidates_none": Hypothesis("utt_x", "ok", "SYNTHETIC", None),
    "dict_candidate": Hypothesis("utt_x", "ok", "SYNTHETIC", ({"candidate_id": "c1"},)),
    "object_error_message": Hypothesis("utt_x", "error", None, error_code="SYNTHETIC", error_message=object()),
}


@pytest.mark.parametrize("result", UNWRITABLE.values(), ids=UNWRITABLE.keys())
def test_unwritable_result_becomes_an_error_row_and_outputs_validate(run_and_write, result):
    run, out = run_and_write(FixedResultAdapter(result))
    assert {h.error_code for h in run.hypotheses} == {"adapter_output_invalid"}
    assert all(h.status == "error" and h.text is None and not h.candidates for h in run.hypotheses)
    assert validate_predictions(read_jsonl(out / "predictions.jsonl"), expected_ids=list(TIMED)).ok
    assert validate_nbest(read_jsonl(out / "nbest.jsonl")).ok


def test_error_message_that_is_not_valid_text_is_escaped(run_and_write):
    run, out = run_and_write(FixedResultAdapter(with_candidate(score_details=RaisingDetails())))
    assert {h.error_code for h in run.hypotheses} == {"adapter_output_invalid"}
    assert all("SYNTHETIC \\ud800" in h.error_message for h in run.hypotheses)
    assert validate_predictions(read_jsonl(out / "predictions.jsonl")).ok


def test_candidates_are_not_checked_when_no_nbest_is_written(run_and_write):
    run, out = run_and_write(FixedResultAdapter(with_candidate(score_details=None), requested_n=0))
    assert {h.status for h in run.hypotheses} == {"ok"}
    assert validate_predictions(read_jsonl(out / "predictions.jsonl")).ok
    assert not (out / "nbest.jsonl").exists()


def test_read_only_mapping_details_are_accepted(run_and_write):
    details = types.MappingProxyType({"length_penalty": 1.0})
    run, out = run_and_write(FixedResultAdapter(with_candidate(score_details=details)))
    assert {h.status for h in run.hypotheses} == {"ok"}
    assert validate_nbest(read_jsonl(out / "nbest.jsonl")).ok
