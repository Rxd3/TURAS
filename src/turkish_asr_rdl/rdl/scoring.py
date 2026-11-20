"""Frozen external-alias linking and learned hypothesis relation compatibility."""
from __future__ import annotations

import copy
import math
import re
from turkish_asr_rdl.evaluation.normalize import normalize, TR_NORMALIZED_V1


def words(text: str):
    return re.findall(r"\w+", normalize(text, TR_NORMALIZED_V1))


def linker_from_tables(entities: list[dict], aliases: list[dict], *, frozen_lexicon=None, lexicon_sha256=None):
    entity_topics = {r["entity_id"]: r["topic_id"] for r in entities if "topic_id" in r}
    # M1 entities.csv has no topic_id; entity/topic pairs come from the static
    # belongs_to relationships and are supplied by the training command below.
    if frozen_lexicon is not None:
        from turkish_asr_rdl.data.screening import Lexicon
        if not isinstance(lexicon_sha256, str) or not re.fullmatch(r'[a-f0-9]{64}', lexicon_sha256):
            raise ValueError('a frozen lexicon source checksum is required')
        Lexicon(frozen_lexicon)  # Validate the frozen syntax before serialization.
        expected = {(t['term_id'], alias) for t in frozen_lexicon['terms'] for alias in t['aliases']}
        actual = {(r['entity_id'], r['alias_text']) for r in aliases}
        if actual != expected or {r['entity_id'] for r in entities} != {t['term_id'] for t in frozen_lexicon['terms']}:
            raise ValueError('relational aliases/entities differ from the frozen lexicon')
        return {'policy': 'm1-frozen-lexicon-v1', 'lexicon': frozen_lexicon,
                'lexicon_source_sha256': lexicon_sha256, 'entity_topics': entity_topics,
                'normalization': 'frozen M1 screening_tokens on raw hypothesis; prefix/exclusion/longest-span rules preserved'}
    if any('*' in r['alias_text'] for r in aliases):
        raise ValueError('wildcard aliases require the frozen lexicon, including its exclusion rules')
    return {"aliases": [{"tokens": words(r["alias_text"]), "entity_id": r["entity_id"]} for r in aliases],
            "entity_topics": entity_topics, "normalization": "shared tr-normalized-v1 plus alphanumeric token matching"}


def link_entities(text: str, linker: dict):
    if linker.get('policy') == 'm1-frozen-lexicon-v1':
        from turkish_asr_rdl.data.screening import Lexicon, screening_tokens
        return sorted({m.term_id for m in Lexicon(linker['lexicon']).match(screening_tokens(text))})
    # Existing bundles retain their legacy exact-token policy for reproducibility.
    tokens = words(text)
    aliases = sorted(linker["aliases"], key=lambda r: (-len(r["tokens"]), r["entity_id"]))
    entities = set()
    position = 0
    while position < len(tokens):
        found = next((a for a in aliases if a["tokens"] and tokens[position:position + len(a["tokens"])] == a["tokens"]), None)
        if found:
            entities.add(found["entity_id"])
            position += len(found["tokens"])
        else:
            position += 1
    return sorted(entities)


def score_hypothesis(text: str, model, linker: dict, *, missing_probability: float = 0.1,
                     epsilon: float = 1e-8, quantity_extractor=None, states=None):
    """No reference arguments exist. Unknown quantities receive a declared floor.

    Co-mention/topic links use real encoded nodes. For an expresses query, an
    unseen utterance is represented by the mean encoded linked entity context.
    This pooling is an explicit approximation and needs downstream validation.
    """
    import torch
    from itertools import combinations
    if not 0 < epsilon < 0.5 or not epsilon <= missing_probability < 1:
        raise ValueError("invalid relation probability floor")
    if "rel_expresses" not in model.snapshot.relations:
        quantity_extractor = lambda text: []
    elif quantity_extractor is None:
        try:
            from turkish_asr_rdl.data.relational import extract_quantities
        except ImportError:
            raise ValueError("M1's approved quantity extractor is required; merge its reviewed implementation") from None
        quantity_extractor = extract_quantities
    snapshot = model.snapshot
    index = {t: {key: i for i, key in enumerate(ids)} for t, ids in snapshot.nodes.items()}
    entities = link_entities(text, linker)
    known = [e for e in entities if e in index["entity"]]
    queries = []
    for entity in known:
        topic = linker["entity_topics"].get(entity)
        if topic in index["topic"] and "rel_belongs_to" in snapshot.relations:
            queries.append(("entity", index["entity"][entity], "rel_belongs_to", "topic", index["topic"][topic]))
    if "rel_co_mentioned" in snapshot.relations:
        queries.extend(("entity", index["entity"][a], "rel_co_mentioned", "entity", index["entity"][b])
                       for a, b in combinations(known, 2))
    quantities = quantity_extractor(text)
    model.eval()
    with torch.inference_mode():
        states = model.encode() if states is None else states
        probabilities = model.score_edges(states, queries).sigmoid().tolist()
        unknown_quantities = 0
        for quantity in quantities:
            key = f"q__{quantity['quantifier_type']}__{quantity['unit']}__{quantity['numeric_value']}"
            if key in index["quantifier"] and known and "rel_expresses" in snapshot.relations:
                context = states["entity"][[index["entity"][e] for e in known]].mean(0)
                value = model.score_vectors(context, states["quantifier"][index["quantifier"][key]], "rel_expresses")
                probabilities.append(float(value.sigmoid()))
            else:
                probabilities.append(missing_probability)
                unknown_quantities += 1
    if probabilities:
        score = sum(math.log(max(epsilon, p)) for p in probabilities) / len(probabilities)
    else:
        score = math.log(missing_probability)
    return score, {"relation_count": len(probabilities), "linked_entities": len(known),
                   "unlinked_entities": len(entities) - len(known), "quantifier_count": len(quantities),
                   "unknown_or_contextless_quantifiers": unknown_quantities,
                   "missing_probability": missing_probability,
                   "utterance_query_representation": "mean encoded linked entities; not a learned test ID"}


def attach_scores(candidates: list[dict], model, linker: dict, *, states=None, **kwargs):
    import torch
    output = copy.deepcopy(candidates)
    model.eval()
    with torch.inference_mode():
        states = model.encode() if states is None else states
    for candidate in output:
        score, details = score_hypothesis(candidate["text"], model, linker, states=states, **kwargs)
        candidate["rdl_log_score"] = score
        candidate.setdefault("score_details", {})["rdl"] = details
        candidate["score_details"].get("null_reasons", {}).pop("rdl_log_score", None)
    return output
