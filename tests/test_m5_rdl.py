"""Synthetic relational fixtures; they are never dataset or experiment evidence."""
import csv
import json
import math

import pytest

from turkish_asr_rdl.rdl.graph import build_snapshot
from turkish_asr_rdl.training.acoustic import sha256_file


REL_FIELDS = ["relationship_id", "head_type", "head_id", "relation_type_id", "tail_type", "tail_id",
              "evidence_provenance_id", "evidence_utterance_id", "annotation_status", "access_split"]


@pytest.fixture
def tables(tmp_path):
    static = tmp_path / "static"
    train = tmp_path / "train"
    static.mkdir()
    train.mkdir()
    hashes = {}
    def write(scope, name, header, rows):
        directory = static if scope == "static" else train
        path = directory / (name + ".csv")
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
        hashes[f"{scope}/{name}.csv"] = sha256_file(path)
    write("static", "provenance", ["provenance_id", "origin_kind", "allowed_split"], [["p-ext", "external_knowledge", "train|validation|test"]])
    write("train", "provenance", ["provenance_id", "origin_kind", "allowed_split"], [["p-train", "train_annotation", "train"]])
    write("static", "topics", ["topic_id", "canonical_label", "provenance_id"], [["topic", "fixture topic", "p-ext"]])
    write("static", "entities", ["entity_id", "canonical_label", "provenance_id"], [[e, e, "p-ext"] for e in "abcd"])
    write("static", "entity_aliases", ["alias_id", "entity_id", "alias_text", "provenance_id"], [[e, e, "fixture" + e, "p-ext"] for e in "abcd"])
    rels = [("rel_mentions", "utterance", "entity", "true"), ("rel_has_topic", "utterance", "topic", "true"),
            ("rel_expresses", "utterance", "quantifier", "true"), ("rel_belongs_to", "entity", "topic", "true"),
            ("rel_co_mentioned", "entity", "entity", "false")]
    write("static", "relation_types", ["relation_type_id", "head_type", "tail_type", "directed", "provenance_id"],
          [[*r, "p-ext"] for r in rels])
    write("static", "relationships_static", REL_FIELDS,
          [["s-" + e, "entity", e, "rel_belongs_to", "topic", "topic", "p-ext", "", "lexicon_defined", "train|validation|test"] for e in "abcd"])
    write("train", "utterances", ["utterance_id", "dataset_release", "split", "leakage_group_id", "provenance_id"],
          [["u" + str(i), "fixture-only", "train", "g" + str(i), "p-train"] for i in range(3)])
    write("train", "quantifiers", ["quantifier_id", "surface_form", "quantifier_type", "numeric_value", "unit", "provenance_id"],
          [["q" + str(i), "fixture quantity", "amount", str(i), "TRY", "p-train"] for i in range(3)])
    relationships = []
    for i, pair in enumerate(("ab", "bc", "cd")):
        uid = "u" + str(i)
        for e in pair:
            relationships.append([uid + e, "utterance", uid, "rel_mentions", "entity", e, "p-train", uid, "auto_extracted", "train"])
        relationships.extend([
            [uid + "t", "utterance", uid, "rel_has_topic", "topic", "topic", "p-train", uid, "auto_extracted", "train"],
            [uid + "q", "utterance", uid, "rel_expresses", "quantifier", "q" + str(i), "p-train", uid, "auto_extracted", "train"]])
    # Deliberately misleading aggregate: safe loader reconstructs from mentions.
    relationships.append(["aggregate", "entity", "a", "rel_co_mentioned", "entity", "d", "p-train", "u0", "auto_extracted:count=9999", "train"])
    write("train", "relationships", REL_FIELDS, relationships)
    return dict(static_dir=static, train_dir=train, hashes=hashes,
                approved_train_ids={"u0", "u1", "u2"}, dataset_release="fixture-only", seed=17, validation_fraction=.34)


def test_heldout_groups_and_cross_group_aggregate_never_enter_adjacency(tables):
    snapshot = build_snapshot(**tables)
    assert set(snapshot.train_groups).isdisjoint(snapshot.validation_groups)
    assert snapshot.validation_edges and set(snapshot.edges).isdisjoint(snapshot.validation_edges)
    assert len(snapshot.nodes["utterance"]) == 2
    assert len(snapshot.nodes["quantifier"]) == 2
    assert ("entity", 0, "rel_co_mentioned", "entity", 3) not in snapshot.edges + snapshot.validation_edges


def test_protected_provenance_rejected_even_when_file_hash_is_supplied(tables):
    path = tables["train_dir"] / "provenance.csv"
    path.write_text(path.read_text().replace("train_annotation", "test_annotation"))
    tables["hashes"]["train/provenance.csv"] = sha256_file(path)
    with pytest.raises(ValueError, match="unauthorized provenance"):
        build_snapshot(**tables)


def test_typed_negative_sampling_excludes_both_directions_of_known_edges(tables):
    pytest.importorskip("torch")
    from turkish_asr_rdl.rdl.model import canonical, sample_negatives
    snapshot = build_snapshot(**tables)
    negatives, saturated = sample_negatives(snapshot.edges, snapshot, seed=5)
    known = {canonical(e, snapshot) for e in snapshot.edges + snapshot.validation_edges}
    assert negatives and saturated > 0
    assert all(canonical(e, snapshot) not in known for e in negatives)
    for ht, h, r, tt, t in negatives:
        assert snapshot.relations[r][:2] == (ht, tt)


def test_relation_ablation_removes_both_learning_and_scoring_type(tables):
    tables["exclude_relations"] = ("rel_expresses",)
    snapshot = build_snapshot(**tables)
    assert "rel_expresses" not in snapshot.relations
    assert all(edge[2] != "rel_expresses" for edge in snapshot.edges)


def test_heterogeneous_model_learns_real_parameters_and_scores_without_gold(tables):
    torch = pytest.importorskip("torch")
    from turkish_asr_rdl.rdl.model import HeterogeneousRelationModel, train_relation_model, evaluate_relations
    from turkish_asr_rdl.rdl.scoring import score_hypothesis
    snapshot = build_snapshot(**tables)
    torch.manual_seed(17)
    model = HeterogeneousRelationModel(snapshot, hidden_dim=8, layers=2, dropout=0)
    before = model.decoder.detach().clone()
    history = train_relation_model(model, epochs=8, learning_rate=.03, seed=17)
    assert not torch.equal(before, model.decoder)
    assert history[-1]["loss"] < history[0]["loss"]
    evaluation = evaluate_relations(model)
    assert 0 <= evaluation["average_precision"] <= 1
    assert 0 < evaluation["mrr"] <= 1
    with pytest.raises(ValueError, match="relation query"):
        model.score_edges(model.encode(), [("entity", 0, "unknown_relation", "entity", 1)])
    linker = {"aliases": [{"tokens": ["fixturea"], "entity_id": "a"}, {"tokens": ["fixtureb"], "entity_id": "b"}],
              "entity_topics": {"a": "topic", "b": "topic"}}
    score, coverage = score_hypothesis("fixturea fixtureb", model, linker, quantity_extractor=lambda text: [])
    assert math.isfinite(score) and score <= 0 and coverage["relation_count"] == 3
    empty, coverage = score_hypothesis("unlinked fixture", model, linker, quantity_extractor=lambda text: [])
    assert empty == pytest.approx(math.log(.1)) and coverage["relation_count"] == 0


def test_safe_graph_bundle_roundtrip_and_tamper_rejection(tables, tmp_path):
    pytest.importorskip("torch")
    from turkish_asr_rdl.rdl.train import train, load_model
    ids = tmp_path / "train.txt"
    ids.write_text("u0\nu1\nu2\n")
    split = tmp_path / "split_manifest.json"
    split.write_text(json.dumps({"dataset_release": "fixture-only", "splits": {"train": {
        "sha256": sha256_file(ids), "utterances": 3}}}))
    config = {"protocol_reviewed": True, "dataset_release": "fixture-only", "split_manifest_sha256": sha256_file(split),
              "inputs": {"static_dir": str(tables["static_dir"]), "train_dir": str(tables["train_dir"]),
                         "train_ids": str(ids), "split_manifest": str(split), "tables_sha256": tables["hashes"]},
              "seed": 17, "validation_fraction": .34, "model": {"hidden_dim": 8, "layers": 1, "dropout": 0},
              "training": {"epochs": 2, "learning_rate": .01}}
    output = tmp_path / "graph"
    train(config, output)
    model, linker, manifest = load_model(output)
    assert manifest["dataset_release"] == "fixture-only" and len(linker["aliases"]) == 4
    assert model.snapshot.validation_groups
    (output / "linker.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_model(output)


def test_offline_rescoring_roundtrip_and_test_freeze_gate(tables, tmp_path):
    pytest.importorskip("torch")
    from turkish_asr_rdl.rdl.train import train
    from turkish_asr_rdl.rdl.rescore import run, preflight, reject_gold
    ids = tmp_path / "train.txt"
    ids.write_text("u0\nu1\nu2\n")
    split = tmp_path / "split_manifest.json"
    split.write_text(json.dumps({"dataset_release": "fixture-only", "splits": {"train": {
        "sha256": sha256_file(ids), "utterances": 3}}}))
    graph = tmp_path / "graph"
    train({"protocol_reviewed": True, "dataset_release": "fixture-only", "split_manifest_sha256": sha256_file(split),
           "inputs": {"static_dir": str(tables["static_dir"]), "train_dir": str(tables["train_dir"]),
                      "train_ids": str(ids), "split_manifest": str(split), "tables_sha256": tables["hashes"]},
           "seed": 17, "validation_fraction": .34, "model": {"hidden_dim": 8, "layers": 1, "dropout": 0},
           "training": {"epochs": 2, "learning_rate": .01}}, graph)
    expected_ids = tmp_path / "validation.txt"
    expected_ids.write_text("synthetic-validation\n")
    nbest = tmp_path / "fixture-nbest.jsonl"
    nbest.write_text(json.dumps({"schema_version": "0.1", "run_id": "fixture-only", "utterance_id": "synthetic-validation",
        "dataset_release": "fixture-only", "split": "validation", "split_manifest_sha256": sha256_file(split),
        "decoder_config_sha256": "a" * 64, "requested_n": 1, "actual_n": 1,
        "candidates": [{"candidate_id": "fixture-candidate", "rank": 1, "text": "fixturea fixtureb",
                        "acoustic_log_score": -3.0, "token_count": 2}]}) + "\n")
    config = {"protocol_reviewed": True, "dataset_release": "fixture-only", "split_manifest_sha256": sha256_file(split),
              "split": "validation", "fusion_frozen": False, "missing_probability": .1,
              "inputs": {"nbest": str(nbest), "nbest_sha256": sha256_file(nbest), "expected_ids": str(expected_ids),
                         "expected_ids_sha256": sha256_file(expected_ids), "graph_artifact": str(graph),
                         "graph_manifest_sha256": sha256_file(graph / "artifact-manifest.json")},
              "fusion": {"acoustic_weight": 1.0, "rdl_weight": .1}}
    output = tmp_path / "rescored.jsonl"
    result = run(config, output, quantity_extractor=lambda text: [])
    row = json.loads(output.read_text())
    assert result["status"] == "completed" and result["end_to_end_asr_rtf"] is None
    assert row["selected_candidate_id"] == "fixture-candidate"
    assert row["source_run_id"] == "fixture-only" and row["run_id"] != "fixture-only"
    assert row["candidates"][0]["rdl_log_score"] <= 0
    config["split"] = "test"
    assert any("frozen fusion" in b for b in preflight(config)["blockers"])
    with pytest.raises(ValueError, match="forbidden"):
        reject_gold({"nested": {"reference_text": "forbidden synthetic fixture field"}})
