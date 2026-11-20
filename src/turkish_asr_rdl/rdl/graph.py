"""Typed train-only CSV snapshots and group-held-out relation discovery."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
import random

from turkish_asr_rdl.training.acoustic import sha256_file


@dataclass
class GraphSnapshot:
    nodes: dict[str, list[str]]
    relations: dict[str, tuple[str, str, bool]]
    edges: list[tuple[str, int, str, str, int]]
    validation_edges: list[tuple[str, int, str, str, int]]
    train_groups: list[str]
    validation_groups: list[str]
    input_hashes: dict[str, str]
    overlapping_validation_relations: int = 0

    def to_dict(self):
        return asdict(self)


def read_table(path: Path, expected_hash: str) -> list[dict]:
    if sha256_file(path) != expected_hash:
        raise ValueError("relational CSV hash mismatch: " + path.name)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("missing or duplicate CSV header")
        rows = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError("CSV row width differs from header")
        return rows


def build_snapshot(static_dir: Path, train_dir: Path, hashes: dict[str, str], *,
                   approved_train_ids: set[str], dataset_release: str,
                   seed: int = 17, validation_fraction: float = 0.2,
                   exclude_relations: tuple[str, ...] = ()) -> GraphSnapshot:
    """Use only train annotations/external knowledge; internal dev uses train groups.

    M1 aggregates co-mentions across groups. Reconstruct these from per-utterance
    mentions before splitting, so the aggregate's first evidence ID cannot leak
    validation co-occurrences into the training graph.
    """
    if not 0 <= validation_fraction < 1 or not approved_train_ids:
        raise ValueError("invalid graph split or empty approved train IDs")
    tables = {}
    for scope, directory, names in (
        ("static", static_dir, ["topics", "entities", "entity_aliases", "relation_types", "provenance", "relationships_static"]),
        ("train", train_dir, ["utterances", "quantifiers", "relationships", "provenance"]),
    ):
        for name in names:
            key = f"{scope}/{name}.csv"
            if key not in hashes:
                raise ValueError("missing reviewed table hash: " + key)
            tables[key] = read_table(Path(directory) / (name + ".csv"), hashes[key])
    provenance = {}
    for scope in ("static", "train"):
        for row in tables[f"{scope}/provenance.csv"]:
            origin = "external_knowledge" if scope == "static" else "train_annotation"
            if row["origin_kind"] != origin or "train" not in row["allowed_split"].split("|"):
                raise ValueError("held-out or unauthorized provenance in graph inputs")
            if scope == "train" and row["allowed_split"] != "train":
                raise ValueError("training annotations must have train-only access")
            if row["provenance_id"] in provenance and provenance[row["provenance_id"]] != row:
                raise ValueError("conflicting provenance IDs")
            provenance[row["provenance_id"]] = row
    def allowed(prov_id, required_origin=None):
        if prov_id not in provenance:
            raise ValueError("missing provenance foreign key")
        if required_origin and provenance[prov_id]["origin_kind"] != required_origin:
            raise ValueError("unexpected provenance origin")
    utterances = {}
    for row in tables["train/utterances.csv"]:
        if row["split"] != "train" or row["dataset_release"] != dataset_release:
            raise ValueError("held-out utterance supplied to graph training")
        allowed(row["provenance_id"], "train_annotation")
        if row["utterance_id"] in utterances:
            raise ValueError("duplicate utterance ID")
        utterances[row["utterance_id"]] = row
    if set(utterances) != approved_train_ids:
        raise ValueError("relational utterances do not match approved frozen train IDs")
    groups = sorted({r["leakage_group_id"] for r in utterances.values()})
    random.Random(seed).shuffle(groups)
    if validation_fraction and len(groups) < 2:
        raise ValueError("need at least two training groups for relation validation")
    count = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction))) if validation_fraction else 0
    dev_groups = set(groups[:count])
    train_groups = set(groups[count:])
    train_utts = {u for u, row in utterances.items() if row["leakage_group_id"] in train_groups}
    relations = {}
    for row in tables["static/relation_types.csv"]:
        allowed(row["provenance_id"], "external_knowledge")
        if row["directed"] not in {"true", "false"} or row["relation_type_id"] in relations:
            raise ValueError("invalid or duplicate relation definition")
        relations[row["relation_type_id"]] = (row["head_type"], row["tail_type"], row["directed"] == "true")
    if set(exclude_relations) - set(relations):
        raise ValueError("unknown ablation relation type")
    nodes = {"entity": [], "topic": [], "quantifier": [], "utterance": sorted(train_utts)}
    for table, node_type, key in (("static/entities.csv", "entity", "entity_id"),
                                  ("static/topics.csv", "topic", "topic_id")):
        for row in tables[table]:
            allowed(row["provenance_id"], "external_knowledge")
            nodes[node_type].append(row[key])
        if len(set(nodes[node_type])) != len(nodes[node_type]):
            raise ValueError("duplicate static node ID")
        nodes[node_type].sort()
    alias_ids = set()
    for row in tables["static/entity_aliases.csv"]:
        allowed(row["provenance_id"], "external_knowledge")
        if row["alias_id"] in alias_ids or row["entity_id"] not in nodes["entity"] or not row["alias_text"]:
            raise ValueError("invalid alias ID/entity foreign key")
        alias_ids.add(row["alias_id"])
    quantifiers = {}
    for row in tables["train/quantifiers.csv"]:
        allowed(row["provenance_id"], "train_annotation")
        if row["quantifier_id"] in quantifiers:
            raise ValueError("duplicate quantifier ID")
        quantifiers[row["quantifier_id"]] = row
    mentions = {uid: set() for uid in utterances}
    raw_train, raw_dev = [], []
    relationship_ids = set()
    for row in tables["static/relationships_static.csv"] + tables["train/relationships.csv"]:
        if row["relationship_id"] in relationship_ids:
            raise ValueError("duplicate relationship ID")
        relationship_ids.add(row["relationship_id"])
        origin = provenance.get(row["evidence_provenance_id"], {}).get("origin_kind")
        allowed(row["evidence_provenance_id"])
        if "train" not in row["access_split"].split("|") or origin not in {"external_knowledge", "train_annotation"}:
            raise ValueError("held-out relationship supplied to graph training")
        if origin == "train_annotation" and row["access_split"] != "train":
            raise ValueError("annotation relation access must be train-only")
        rel = row["relation_type_id"]
        if rel not in relations or relations[rel][:2] != (row["head_type"], row["tail_type"]):
            raise ValueError("typed relation schema mismatch")
        uid = row["evidence_utterance_id"]
        if origin == "train_annotation" and uid not in utterances:
            raise ValueError("relationship evidence is outside frozen training IDs")
        if origin == "train_annotation" and row["head_type"] == "utterance" and row["head_id"] != uid:
            raise ValueError("utterance relationship head differs from its evidence ID")
        if rel == "rel_co_mentioned":
            continue  # unsafe cross-group aggregate; reconstructed below
        if rel == "rel_mentions":
            if row["head_id"] != uid or row["tail_id"] not in nodes["entity"]:
                raise ValueError("invalid mention foreign key")
            mentions[uid].add(row["tail_id"])
        edge = (row["head_type"], row["head_id"], rel, row["tail_type"], row["tail_id"])
        if origin == "external_knowledge" or uid in train_utts:
            raw_train.append(edge)
            if edge[3] == "quantifier":
                if edge[4] not in quantifiers:
                    raise ValueError("quantifier foreign key missing")
                nodes["quantifier"].append(edge[4])
    nodes["quantifier"] = sorted(set(nodes["quantifier"]))
    for uid, entities in mentions.items():
        destination = raw_train if uid in train_utts else raw_dev
        for a, b in combinations(sorted(entities), 2):
            destination.append(("entity", a, "rel_co_mentioned", "entity", b))
    # A relation already evidenced by a training group cannot be called an
    # unseen validation fact. Exclude it from the validation set rather than
    # hiding its direct edge while leaking it through training mention paths.
    dev_set = set(raw_dev)
    def canonical(edge):
        if not relations[edge[2]][2] and edge[0] == edge[3] and edge[1] > edge[4]:
            return (edge[3], edge[4], edge[2], edge[0], edge[1])
        return edge
    dev_set = {canonical(e) for e in dev_set}
    train_set = {canonical(e) for e in raw_train}
    overlap = len(dev_set & train_set)
    dev_set -= train_set
    train_set = {e for e in train_set if e[2] not in exclude_relations}
    dev_set = {e for e in dev_set if e[2] not in exclude_relations}
    relations = {r: spec for r, spec in relations.items() if r not in exclude_relations}
    index = {kind: {key: i for i, key in enumerate(keys)} for kind, keys in nodes.items()}
    def encode(edge):
        try:
            return edge[0], index[edge[0]][edge[1]], edge[2], edge[3], index[edge[3]][edge[4]]
        except KeyError:
            raise ValueError("node foreign key missing from train-only snapshot") from None
    return GraphSnapshot(nodes, relations, [encode(e) for e in sorted(train_set)],
                         [encode(e) for e in sorted(dev_set)], sorted(train_groups), sorted(dev_groups), dict(hashes), overlap)
