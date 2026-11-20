"""The inference interface every system implements for the shared timing harness."""
from __future__ import annotations

import reprlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from turkish_asr_rdl.schemas.jsonl import encode_jsonl
from turkish_asr_rdl.schemas.predictions import INFERENCE_INPUT_FIELDS, validate_inference_inputs


@dataclass(frozen=True)
class InferenceInput:
    """One allowlisted inference input row; it never carries reference text."""

    schema_version: str
    dataset_release: str
    split_manifest_sha256: str
    utterance_id: str
    audio_uri: str
    audio_sha256: str
    duration_s: float
    sample_rate_hz: int
    channels: int
    language: str

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> InferenceInput:
        """Build from a JSON row; raise ValueError listing every allowlist violation."""
        report = validate_inference_inputs([row])
        if not report.ok:
            raise ValueError("invalid inference input: " + "; ".join(report.errors))
        return cls(**{name: row[name] for name in INFERENCE_INPUT_FIELDS})


@dataclass(frozen=True)
class AudioItem:
    """Decoded float32 16 kHz mono samples for one utterance."""

    utterance_id: str
    samples: np.ndarray = field(compare=False)


@dataclass(frozen=True)
class Candidate:
    """One scored N-best candidate as produced by the model's own decoder."""

    candidate_id: str
    rank: int
    text: str
    token_ids: tuple[int, ...]
    acoustic_log_score: float
    acoustic_score_kind: str
    decoder_sequence_score: float | None = None
    score_details: Mapping[str, Any] = field(default_factory=dict)
    lm_log_score: float | None = None
    rdl_log_score: float | None = None
    combined_score: float | None = None


@dataclass(frozen=True)
class Hypothesis:
    """The system output for one utterance: status ok, empty or error."""

    utterance_id: str
    status: str
    text: str | None
    candidates: tuple[Candidate, ...] = ()
    error_code: str | None = None
    error_message: str | None = None
    selected_candidate_id: str | None = None


class AsrAdapter(Protocol):
    """What a model owner implements; requested_n 0 means the system has no N-best."""

    system_id: str
    model_artifact_sha256: str
    decoder_config_sha256: str
    requested_n: int
    score_convention_id: str | None
    config: Mapping[str, Any]

    def load(self) -> None:
        """Load weights and processors; timed once as the cold-load time."""

    def transcribe(self, batch: Sequence[AudioItem]) -> list[Hypothesis]:
        """Return one Hypothesis per AudioItem, in the same order."""


def check_adapter(adapter: AsrAdapter) -> None:
    """Raise a one-line ValueError when the adapter's N-best settings or config cannot be written."""
    requested_n = getattr(adapter, "requested_n", None)
    if not isinstance(requested_n, int) or isinstance(requested_n, bool) or requested_n < 0:
        raise ValueError(f"adapter.requested_n must be an integer of at least 0, got {reprlib.repr(requested_n)}")
    convention = getattr(adapter, "score_convention_id", None)
    if requested_n > 0 and not (isinstance(convention, str) and convention):
        raise ValueError("adapter.score_convention_id must be a non-empty string when requested_n > 0, "
                         f"got {reprlib.repr(convention)}")
    _check_config(getattr(adapter, "config", None))


def _check_config(config: Any) -> None:
    if not isinstance(config, Mapping):
        raise ValueError(f"adapter.config must be a mapping written as strict JSON, got {type(config).__name__}")
    try:
        encode_jsonl([config])
    except ValueError as exc:
        detail = " ".join(str(exc.__cause__ or exc).split())
        raise ValueError(f"adapter.config cannot be written as strict JSON: {detail}") from exc
