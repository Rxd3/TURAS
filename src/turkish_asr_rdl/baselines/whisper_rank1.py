"""Reusable Whisper rank-1 comparison against single-result generation.

Compares rank 1 from the shared N-best decoding helper with Whisper's standard
generate path (num_return_sequences=1) at the same beam width and Turkish
transcription settings. Reference-free; performs verbatim comparison without
normalization or whitespace stripping.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


DEFAULT_WHISPER_PROMPT_TOKENS = [
    "<|startoftranscript|>",
    "<|tr|>",
    "<|transcribe|>",
    "<|notimestamps|>",
]


@dataclass
class TokenDiff:
    single_length: int
    nbest_rank1_length: int
    first_diff_index: int | None = None
    single_token_at_diff: int | None = None
    nbest_token_at_diff: int | None = None


@dataclass
class Rank1ComparisonResult:
    matched: bool
    status: str  # "match", "mismatch", "missing_rank1", "error"
    single_text: str | None = None
    nbest_rank1_text: str | None = None
    text_exact_match: bool = False
    tokens_exact_match: bool = False
    single_token_ids: list[int] = field(default_factory=list)
    nbest_rank1_token_ids: list[int] = field(default_factory=list)
    token_diff: TokenDiff | None = None
    details: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def strip_known_prompt(
    token_ids: list[int],
    prompt_token_ids: list[int] | None = None,
    eos_token_id: int | None = None,
) -> list[int]:
    """Strip known prompt prefix and truncate at EOS if present."""
    tokens = list(token_ids)
    if prompt_token_ids and len(tokens) >= len(prompt_token_ids):
        if tokens[:len(prompt_token_ids)] == prompt_token_ids:
            tokens = tokens[len(prompt_token_ids):]
    if eos_token_id is not None and eos_token_id in tokens:
        tokens = tokens[:tokens.index(eos_token_id) + 1]
    return tokens


def compare_rank1_outputs(
    single_result: dict[str, Any] | Any,
    nbest_candidates: list[dict[str, Any]] | dict[str, Any],
    *,
    prompt_token_ids: list[int] | None = None,
    eos_token_id: int | None = None,
) -> Rank1ComparisonResult:
    """Compare rank 1 from an N-best candidate list with single-result generation.

    Performs exact verbatim text comparison and separate token sequence comparison.
    Does not strip whitespace, normalize text, or assume the two paths match.
    """
    # 1. Check for decoding error in single-result
    if isinstance(single_result, dict):
        if single_result.get("status") == "error":
            err_msg = single_result.get("error") or "Single-result decoding reported an error"
            return Rank1ComparisonResult(
                matched=False,
                status="error",
                details=f"Single-result decode failed: {err_msg}",
                error=str(err_msg),
            )
        single_text = single_result.get("text")
        single_tokens_raw = single_result.get("token_ids", [])
    else:
        single_text = str(single_result)
        single_tokens_raw = []

    # 2. Check for decoding error or missing candidates in N-best
    if isinstance(nbest_candidates, dict):
        if nbest_candidates.get("status") == "error":
            err_msg = nbest_candidates.get("error") or "N-best decoding reported an error"
            return Rank1ComparisonResult(
                matched=False,
                status="error",
                details=f"N-best decode failed: {err_msg}",
                error=str(err_msg),
            )
        candidate_list = nbest_candidates.get("candidates", [])
    elif isinstance(nbest_candidates, list):
        candidate_list = nbest_candidates
    else:
        return Rank1ComparisonResult(
            matched=False,
            status="error",
            details=f"Unexpected N-best candidate container type: {type(nbest_candidates).__name__}",
            error="Invalid N-best container type",
        )

    if not candidate_list:
        return Rank1ComparisonResult(
            matched=False,
            status="missing_rank1",
            single_text=single_text,
            details="N-best candidate list is empty; no candidates returned",
        )

    # Find candidate with rank 1
    rank1_cand = None
    for cand in candidate_list:
        if cand.get("rank") == 1:
            rank1_cand = cand
            break

    if rank1_cand is None:
        return Rank1ComparisonResult(
            matched=False,
            status="missing_rank1",
            single_text=single_text,
            details="N-best output does not contain a candidate with rank == 1",
        )

    nbest_text = rank1_cand.get("text")
    nbest_tokens_raw = rank1_cand.get("token_ids", [])

    # Process tokens (strip known prompt prefix if present)
    single_tokens = strip_known_prompt(single_tokens_raw, prompt_token_ids, eos_token_id)
    nbest_tokens = strip_known_prompt(nbest_tokens_raw, prompt_token_ids, eos_token_id)

    # Verbatim text comparison (NO stripping of whitespace, NO normalization)
    text_match = (single_text == nbest_text)

    # Token comparison
    tokens_match = (single_tokens == nbest_tokens)

    # Compute token diff if mismatching
    token_diff = None
    if not tokens_match:
        first_diff = None
        single_val = None
        nbest_val = None
        min_len = min(len(single_tokens), len(nbest_tokens))
        for idx in range(min_len):
            if single_tokens[idx] != nbest_tokens[idx]:
                first_diff = idx
                single_val = single_tokens[idx]
                nbest_val = nbest_tokens[idx]
                break
        if first_diff is None and len(single_tokens) != len(nbest_tokens):
            first_diff = min_len
            single_val = single_tokens[min_len] if len(single_tokens) > min_len else None
            nbest_val = nbest_tokens[min_len] if len(nbest_tokens) > min_len else None

        token_diff = TokenDiff(
            single_length=len(single_tokens),
            nbest_rank1_length=len(nbest_tokens),
            first_diff_index=first_diff,
            single_token_at_diff=single_val,
            nbest_token_at_diff=nbest_val,
        )

    if text_match and tokens_match:
        status = "match"
        details = "Rank 1 exactly matches single-result generation in text and tokens."
        matched = True
    elif text_match and not tokens_match:
        status = "mismatch"
        details = f"Text matches verbatim ('{single_text}'), but token IDs differ."
        matched = False
    elif not text_match and tokens_match:
        status = "mismatch"
        details = f"Token IDs match, but decoded text differs ('{single_text}' vs '{nbest_text}')."
        matched = False
    else:
        status = "mismatch"
        details = f"Text and token mismatch. Single: '{single_text}' | N-best rank 1: '{nbest_text}'."
        matched = False

    return Rank1ComparisonResult(
        matched=matched,
        status=status,
        single_text=single_text,
        nbest_rank1_text=nbest_text,
        text_exact_match=text_match,
        tokens_exact_match=tokens_match,
        single_token_ids=single_tokens,
        nbest_rank1_token_ids=nbest_tokens,
        token_diff=token_diff,
        details=details,
    )


def run_whisper_single_generate(
    model: Any,
    processor: Any,
    input_features: Any,
    *,
    beam_width: int = 5,
    max_new_tokens: int = 440,
    prompt_tokens: list[str] | None = None,
) -> dict[str, Any]:
    """Execute Whisper standard generate with num_return_sequences=1."""
    import torch

    prompt_toks = prompt_tokens or DEFAULT_WHISPER_PROMPT_TOKENS
    prompt = processor.tokenizer.convert_tokens_to_ids(prompt_toks)
    eos_id = getattr(processor.tokenizer, "eos_token_id", None)

    config = copy.deepcopy(model.generation_config)
    config.forced_decoder_ids = None

    device = input_features.device
    decoder_input_ids = torch.tensor([prompt], device=device)

    try:
        with torch.inference_mode():
            output = model.generate(
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                generation_config=config,
                num_beams=beam_width,
                num_return_sequences=1,
                do_sample=False,
                length_penalty=1.0,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
            )
        seq = output.sequences[0].tolist()
        tokens = seq[len(prompt):]
        if eos_id is not None and eos_id in tokens:
            tokens = tokens[:tokens.index(eos_id) + 1]
        text = processor.tokenizer.decode(tokens, skip_special_tokens=True)
        return {
            "status": "ok",
            "text": text,
            "token_ids": tokens,
            "raw_sequence": seq,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "text": None,
            "token_ids": [],
        }


def run_whisper_rank1_comparison(
    model: Any,
    processor: Any,
    input_features: Any,
    *,
    nbest_fn: Callable[..., Any],
    beam_width: int = 5,
    max_new_tokens: int = 440,
    prompt_tokens: list[str] | None = None,
    language: str | None = None,
    task: str | None = None,
) -> Rank1ComparisonResult:
    """Run both decoding paths and compare rank 1 against single-result generation.

    Uses beam_width as requested_n for the proposed helper at b6afa13 because
    the helper ties requested_n and num_beams together.
    """
    if language is not None and language != "tr":
        raise ValueError(f"Whisper baseline decoding requires language='tr', got {language!r}")
    if task is not None and task != "transcribe":
        raise ValueError(f"Whisper baseline decoding requires task='transcribe', got {task!r}")
    prompt_toks = prompt_tokens or DEFAULT_WHISPER_PROMPT_TOKENS
    prompt_ids = processor.tokenizer.convert_tokens_to_ids(prompt_toks)
    eos_id = getattr(processor.tokenizer, "eos_token_id", None)

    # 1. Standard generate with num_return_sequences=1
    single_res = run_whisper_single_generate(
        model,
        processor,
        input_features,
        beam_width=beam_width,
        max_new_tokens=max_new_tokens,
        prompt_tokens=prompt_toks,
    )

    # 2. Shared N-best path (requested_n = beam_width)
    try:
        nbest_out = nbest_fn(
            model,
            processor,
            input_features,
            requested_n=beam_width,
            max_new_tokens=max_new_tokens,
            deduplicate=False,
        )
        if isinstance(nbest_out, tuple):
            candidates = nbest_out[0]
        else:
            candidates = nbest_out
    except Exception as exc:
        return Rank1ComparisonResult(
            matched=False,
            status="error",
            single_text=single_res.get("text"),
            details=f"N-best decode invocation failed: {type(exc).__name__}: {exc}",
            error=f"{type(exc).__name__}: {exc}",
        )

    # 3. Compare outputs
    return compare_rank1_outputs(
        single_res,
        candidates,
        prompt_token_ids=prompt_ids,
        eos_token_id=eos_id,
    )
