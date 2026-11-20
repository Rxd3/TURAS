"""Prefix-beam CTC search and exact alignment-marginalized transcript scoring."""
from __future__ import annotations

import math

NEG_INF = float("-inf")


def logadd(*values: float) -> float:
    maximum = max(values)
    if maximum == NEG_INF:
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def validate_log_probs(frames, blank_id: int):
    frames = [list(map(float, frame)) for frame in frames]
    if not frames or not frames[0] or not 0 <= blank_id < len(frames[0]):
        raise ValueError("empty CTC frames or invalid blank ID")
    width = len(frames[0])
    for frame in frames:
        if len(frame) != width or any(math.isnan(x) or x == float("inf") or x > 1e-5 for x in frame):
            raise ValueError("CTC input must be consistent natural-log probabilities")
        if not math.isclose(sum(math.exp(x) for x in frame), 1.0, rel_tol=1e-4, abs_tol=1e-5):
            raise ValueError("CTC frame probabilities do not sum to one")
    return frames


def sequence_log_probability(frames, tokens, blank_id: int) -> float:
    """Forward dynamic program sums every valid CTC alignment, including blanks."""
    frames = validate_log_probs(frames, blank_id)
    tokens = tuple(tokens)
    if any(not isinstance(t, int) or t == blank_id or not 0 <= t < len(frames[0]) for t in tokens):
        raise ValueError("transcript tokens must be valid nonblank label IDs")
    extended = [blank_id]
    for token in tokens:
        extended.extend((token, blank_id))
    previous = [NEG_INF] * len(extended)
    previous[0] = 0.0
    for frame in frames:
        current = [NEG_INF] * len(extended)
        for state, token in enumerate(extended):
            incoming = [previous[state]]
            if state > 0:
                incoming.append(previous[state - 1])
            if state > 1 and token != blank_id and token != extended[state - 2]:
                incoming.append(previous[state - 2])
            current[state] = logadd(*incoming) + frame[token]
        previous = current
    return previous[0] if not tokens else logadd(previous[-1], previous[-2])


def prefix_beam_search(frames, *, blank_id: int, beam_size: int = 10, token_top_k: int | None = None):
    """Pruned candidate discovery; its beam mass is explicitly approximate."""
    frames = validate_log_probs(frames, blank_id)
    if beam_size < 1 or (token_top_k is not None and token_top_k < 1):
        raise ValueError("beam and token limits must be positive")
    beams = {(): (0.0, NEG_INF)}
    for frame in frames:
        labels = sorted(range(len(frame)), key=lambda i: (-frame[i], i))
        if token_top_k is not None:
            labels = labels[:token_top_k]
        labels = sorted(set(labels) | {blank_id})
        next_beams = {}
        def add(prefix, blank=NEG_INF, nonblank=NEG_INF):
            old_b, old_n = next_beams.get(prefix, (NEG_INF, NEG_INF))
            next_beams[prefix] = (logadd(old_b, blank), logadd(old_n, nonblank))
        for prefix, (pb, pn) in beams.items():
            total = logadd(pb, pn)
            add(prefix, blank=total + frame[blank_id])
            for token in labels:
                if token == blank_id:
                    continue
                if prefix and token == prefix[-1]:
                    add(prefix, nonblank=pn + frame[token])
                    add(prefix + (token,), nonblank=pb + frame[token])
                else:
                    add(prefix + (token,), nonblank=total + frame[token])
        ranked = sorted(next_beams, key=lambda p: (-logadd(*next_beams[p]), p))[:beam_size]
        beams = {p: next_beams[p] for p in ranked}
    return [(tokens, logadd(*scores)) for tokens, scores in sorted(
        beams.items(), key=lambda item: (-logadd(*item[1]), item[0]))]


def ctc_nbest(frames, tokenizer, *, blank_id: int, requested_n: int = 5,
              beam_size: int = 10, deduplicate: bool = False):
    if not 1 <= requested_n <= beam_size:
        raise ValueError("requested_n must be between one and beam_size")
    frames = validate_log_probs(frames, blank_id)
    beams = prefix_beam_search(frames, blank_id=blank_id, beam_size=beam_size)
    candidates, dedup, kept = [], {}, {}
    for rank, (tokens, beam_score) in enumerate(beams[:requested_n], 1):
        # Prefixes are already CTC-collapsed; group_tokens=False preserves repeats.
        text = tokenizer.decode(list(tokens), group_tokens=False, skip_special_tokens=True)
        cid = f"ctc-{rank:04d}"
        if deduplicate and text in kept:
            dedup[cid] = kept[text]
            continue
        score = sequence_log_probability(frames, tokens, blank_id)
        if not math.isfinite(score):
            continue  # impossible prefixes have zero probability, not artificial scores
        kept[text] = cid
        candidates.append({"candidate_id": cid, "rank": rank, "text": text,
                           "token_ids": list(tokens), "token_count": len(tokens),
                           "acoustic_log_score": score, "acoustic_score_kind": "ctc_sequence_log_probability",
                           "decoder_sequence_score": beam_score, "lm_log_score": None,
                           "rdl_log_score": None, "combined_score": None,
                           "score_details": {"log_base": "e", "blank_id": blank_id,
                                             "nbest_algorithm": "prefix_beam", "beam_size": beam_size,
                                             "sequence_score_exact": True, "beam_score_approximate": True}})
        if len(candidates) == requested_n:
            break
    return candidates, dedup
