"""Deterministic joint acoustic/optional-LM/relational candidate selection."""
from __future__ import annotations

import copy
import math


def rescore(candidates: list[dict], *, acoustic_weight: float = 1.0, lm_weight: float = 0.0,
            rdl_weight: float = 0.0, acoustic_exponent: float = 1.0, lm_exponent: float = 1.0):
    if not candidates:
        raise ValueError("cannot rescore an empty candidate list")
    params = (acoustic_weight, lm_weight, rdl_weight, acoustic_exponent, lm_exponent)
    if any(not math.isfinite(v) or v < 0 for v in params) or not any(params[:3]):
        raise ValueError("weights/exponents must be finite nonnegative numbers; enable at least one score")
    if len({c["candidate_id"] for c in candidates}) != len(candidates):
        raise ValueError("duplicate candidate ID")
    output = copy.deepcopy(candidates)
    for candidate in output:
        score = 0.0
        for key, weight, count_key, exponent in (
            ("acoustic_log_score", acoustic_weight, "token_count", acoustic_exponent),
            ("lm_log_score", lm_weight, "lm_token_count", lm_exponent),
            ("rdl_log_score", rdl_weight, None, 0.0),
        ):
            if not weight:
                continue
            value = candidate.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError("enabled component is unavailable/non-finite: " + key)
            divisor = 1.0
            if count_key:
                count = candidate.get(count_key)
                if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                    raise ValueError("missing/invalid token count: " + count_key)
                divisor = max(1, count) ** exponent
            score += weight * value / divisor
        if not math.isfinite(score):
            raise ValueError("combined score is non-finite")
        candidate["combined_score"] = score
        candidate.get("score_details", {}).get("null_reasons", {}).pop("combined_score", None)
    winner = min(output, key=lambda c: (-c["combined_score"], c["rank"], c["candidate_id"]))
    return winner["candidate_id"], output
