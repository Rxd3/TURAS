"""Genuine Whisper beam candidates with original-model teacher-forced scores."""
from __future__ import annotations

import copy
import math


def scored_token_ids(sequence, prompt_length: int, eos_id: int) -> list[int]:
    tokens = list(sequence[prompt_length:])
    if eos_id in tokens:
        tokens = tokens[:tokens.index(eos_id) + 1]
    if not tokens:
        raise ValueError("candidate has no generated tokens")
    return tokens


def teacher_forced_score(model, input_features, sequence: list[int], prompt_length: int):
    """Natural-log sum, generated text plus EOS, excluding the forced prefix."""
    import torch
    if not 1 <= prompt_length < len(sequence):
        raise ValueError("invalid prefix length")
    device = input_features.device
    ids = torch.tensor([sequence], device=device)
    with torch.inference_mode():
        logits = model(input_features=input_features, decoder_input_ids=ids[:, :-1], use_cache=False).logits
        log_probs = logits.float().log_softmax(-1)
        selected = log_probs.gather(-1, ids[:, 1:, None]).squeeze(-1)
        score = float(selected[:, prompt_length - 1:].sum())
    if not math.isfinite(score):
        raise ValueError("non-finite candidate score")
    return score


def whisper_nbest(model, processor, input_features, *, requested_n: int = 5, max_new_tokens: int = 440):
    """Return candidates and a removed-to-kept dedup map; never pad a short list."""
    import torch
    from transformers import GenerationMixin
    if requested_n < 1 or max_new_tokens < 1 or input_features.shape[0] != 1:
        raise ValueError("N-best extraction requires one utterance and positive beam/token limits")
    prompt_tokens = ["<|startoftranscript|>", "<|tr|>", "<|transcribe|>", "<|notimestamps|>"]
    prompt = processor.tokenizer.convert_tokens_to_ids(prompt_tokens)
    eos = processor.tokenizer.eos_token_id
    config = copy.deepcopy(model.generation_config)
    config.forced_decoder_ids = None
    # Bypass Whisper's long-form orchestration, which may collapse return sequences.
    # Rescore with the original logits; processed beam scores remain separate.
    with torch.inference_mode():
        output = GenerationMixin.generate(
            model, input_features=input_features,
            decoder_input_ids=torch.tensor([prompt], device=input_features.device),
            generation_config=config, num_beams=requested_n, num_return_sequences=requested_n,
            do_sample=False, length_penalty=1.0, max_new_tokens=max_new_tokens,
            return_dict_in_generate=True, output_scores=True,
        )
    candidates, dedup, kept = [], {}, {}
    for index, sequence in enumerate(output.sequences.tolist(), 1):
        tokens = scored_token_ids(sequence, len(prompt), eos)
        text = processor.tokenizer.decode(tokens, skip_special_tokens=True)
        candidate_id = f"beam-{index:04d}"
        if text in kept:
            dedup[candidate_id] = kept[text]
            continue
        kept[text] = candidate_id
        score = teacher_forced_score(model, input_features, prompt + tokens, len(prompt))
        beam_scores = getattr(output, "sequences_scores", None)
        candidates.append({
            "candidate_id": candidate_id, "rank": index, "text": text, "token_ids": tokens,
            "token_count": len(tokens), "acoustic_log_score": score,
            "acoustic_score_kind": "seq2seq_token_log_prob_sum",
            "decoder_sequence_score": float(beam_scores[index - 1]) if beam_scores is not None else None,
            "lm_log_score": None, "rdl_log_score": None, "combined_score": None,
            "score_details": {"log_base": "e", "prompt_excluded": prompt_tokens,
                              "includes_eos": eos in tokens, "terminated": eos in tokens,
                              "score_source": "teacher-forced original logits; no suppression/temperature transformation",
                              "nbest_algorithm": "beam", "num_beams": requested_n,
                              "decoder_sequence_score_formula": "framework beam score with length_penalty=1.0"},
        })
    return candidates, dedup
