"""Shared parts of the 5070 Ti smoke adapters: hashing, GPU memory, candidate building, base adapters.

torch and transformers are imported inside the functions that use them, so this module
imports on a machine that has neither.
"""
from __future__ import annotations

import hashlib
import json
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from turkish_asr_rdl.decoding.interface import AudioItem, Candidate, Hypothesis
from turkish_asr_rdl.schemas.jsonl import sha256_file

DEVICE = "cuda"
GIB = 2**30
_MESSAGE_CHARS = 300
_NO_BEAM_SCORE = "greedy decoding has no beam score"


def one_line(exc: BaseException) -> str:
    """The exception as one short line: type name and collapsed message."""
    return " ".join(f"{type(exc).__name__}: {exc}".split())[:_MESSAGE_CHARS]


def sha256_json(value: Any) -> str:
    """SHA-256 of the canonical JSON form (sorted keys, no spaces)."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModelFiles:
    """A downloaded model snapshot: where it is and which exact weights it holds."""

    name: str
    repo_id: str
    revision: str
    path: Path
    artifact_sha256: str


def weights_sha256(folder: Path, pattern: str) -> str:
    """One hash over the name and SHA-256 of every weights file matching pattern."""
    files = sorted(Path(folder).glob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} weights file in {folder}")
    lines = "".join(f"{f.name} {sha256_file(f)}\n" for f in files)
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()



def cuda_torch() -> Any:
    """Import torch and require a CUDA device."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run the smoke tests on the GPU laptop")
    return torch


class CudaVram:
    """Peak GPU memory since reset(), with the free dedicated memory measured at reset(), in GiB."""

    def __init__(self) -> None:
        self._free_before: float | None = None

    def reset(self) -> None:
        torch = cuda_torch()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self._free_before = torch.cuda.mem_get_info()[0] / GIB

    def peak(self) -> dict[str, float | None]:
        torch = cuda_torch()
        return {
            "peak_vram_gib": torch.cuda.max_memory_allocated() / GIB,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
            "free_vram_before_gib": self._free_before,
            "total_vram_gib": torch.cuda.get_device_properties(0).total_memory / GIB,
        }


def environment() -> dict[str, Any]:
    """Library versions and the GPU name."""
    import torch
    import transformers

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
            "gpu": gpu}


def collapse_ctc(path: Sequence[int], blank: int) -> tuple[int, ...]:
    """CTC greedy rule: merge adjacent repeats, then drop blanks."""
    kept = [token for k, token in enumerate(path) if token != blank and (k == 0 or path[k - 1] != token)]
    return tuple(kept)


def cut_at_stop(tokens: Sequence[int], stop_ids: frozenset) -> tuple[int, ...]:
    """Generated tokens up to and including the first stop token; the padding after it is dropped."""
    end = next((k + 1 for k, token in enumerate(tokens) if token in stop_ids), len(tokens))
    return tuple(tokens[:end])


def make_hypothesis(utterance_id: str, candidates: tuple[Candidate, ...]) -> Hypothesis:
    """Rank 1 is the transcript; a blank rank 1 gives status "empty"."""
    text = candidates[0].text if candidates else ""
    if not text.strip():
        return Hypothesis(utterance_id, "empty", "", candidates)
    return Hypothesis(utterance_id, "ok", text, candidates)


def beam_hypothesis(utterance_id: str, texts: Sequence[str], token_rows: Sequence[Sequence[int]],
                    step_rows: Sequence[Sequence[float]], sequence_scores: Sequence[float | None], *,
                    kind: str, details: Mapping[str, Any], stop_ids: frozenset) -> Hypothesis:
    """Candidates in decoder rank order; the acoustic score sums step log-probs through the stop token."""
    candidates = []
    for rank, (text, tokens, steps, beam_score) in enumerate(
            zip(texts, token_rows, step_rows, sequence_scores), start=1):
        kept = cut_at_stop(tokens, stop_ids)
        reasons = {} if beam_score is not None else {"null_reasons": {"decoder_sequence_score": _NO_BEAM_SCORE}}
        candidates.append(Candidate(
            candidate_id=f"{utterance_id}-c{rank:02d}", rank=rank, text=text, token_ids=kept,
            acoustic_log_score=float(sum(steps[:len(kept)])), acoustic_score_kind=kind,
            decoder_sequence_score=None if beam_score is None else float(beam_score),
            score_details={**details, **reasons},
        ))
    return make_hypothesis(utterance_id, tuple(candidates))


class SmokeAdapter:
    """AsrAdapter attributes shared by every family; subclasses implement load() and decode(item)."""

    requested_n = 0
    score_convention_id: str | None = None

    def __init__(self, files: ModelFiles, settings: Mapping[str, Any]):
        self.files = files
        self.system_id = f"smoke-{files.name}"
        self.model_artifact_sha256 = files.artifact_sha256
        self.config = {"repo_id": files.repo_id, "revision": files.revision, **settings}
        self.decoder_config_sha256 = sha256_json(self.config)
        self.decode_path = "not run"
        self.first_traceback: str | None = None
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        raise NotImplementedError

    def decode(self, item: AudioItem) -> Hypothesis:
        raise NotImplementedError

    def transcribe(self, batch: Sequence[AudioItem]) -> list[Hypothesis]:
        """Decode each item; the first failure's full traceback is kept for the result file."""
        try:
            return [self.decode(item) for item in batch]
        except Exception:
            if self.first_traceback is None:
                self.first_traceback = traceback.format_exc()
            raise

    def extra_facts(self, items: Sequence[AudioItem]) -> dict[str, Any]:
        return {"decode_path": self.decode_path, "first_traceback": self.first_traceback}


class BeamAdapter(SmokeAdapter):
    """Tries beam search; after the first beam failure every item is decoded greedily and the failure kept."""

    def __init__(self, files: ModelFiles, settings: Mapping[str, Any]):
        super().__init__(files, settings)
        self.beam_error: str | None = None
        self.beam_traceback: str | None = None

    def beam(self, item: AudioItem) -> Hypothesis:
        raise NotImplementedError

    def greedy(self, item: AudioItem) -> Hypothesis:
        raise NotImplementedError

    def decode(self, item: AudioItem) -> Hypothesis:
        if self.beam_error is None:
            try:
                hypothesis = self.beam(item)
            except Exception as exc:  # recorded below; the greedy path still measures speed and memory
                self.beam_error, self.beam_traceback = one_line(exc), traceback.format_exc()
            else:
                self.decode_path = f"beam {self.config.get('num_beams')}"
                return hypothesis
        self.decode_path = "greedy (beam search failed, see beam_error)"
        return self.greedy(item)

    def extra_facts(self, items: Sequence[AudioItem]) -> dict[str, Any]:
        return {**super().extra_facts(items), "beam_error": self.beam_error, "beam_traceback": self.beam_traceback}
