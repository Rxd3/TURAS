"""Train/save a train-only heterogeneous graph; protected data is never fetched."""
from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from turkish_asr_rdl.training.acoustic import sha256_file, write_json
from turkish_asr_rdl.rdl.graph import GraphSnapshot, build_snapshot, read_table


def preflight(config):
    blockers = []
    for key in ("static_dir", "train_dir", "train_ids", "split_manifest"):
        value = config.get("inputs", {}).get(key)
        if not value or not Path(value).exists():
            blockers.append("missing input: " + key)
    if not config.get("inputs", {}).get("tables_sha256"):
        blockers.append("missing reviewed relational table hashes")
    if not config.get("protocol_reviewed"):
        blockers.append("graph provenance/split protocol awaits M1/M6 review")
    return {"status": "blocked" if blockers else "ready_for_input_validation", "blockers": blockers}


def train(config, output: Path):
    report = preflight(config)
    if report["blockers"]:
        raise ValueError("RDL training blocked: " + "; ".join(report["blockers"]))
    if output.exists():
        raise ValueError("output must be a new directory")
    inputs = config["inputs"]
    manifest_path = Path(inputs["split_manifest"])
    if sha256_file(manifest_path) != config["split_manifest_sha256"]:
        raise ValueError("split manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["dataset_release"] != config["dataset_release"]:
        raise ValueError("graph dataset release mismatch")
    ids_path = Path(inputs["train_ids"])
    if sha256_file(ids_path) != manifest["splits"]["train"]["sha256"]:
        raise ValueError("train ID hash mismatch")
    ids = ids_path.read_text(encoding="utf-8").splitlines()
    if len(ids) != len(set(ids)) or len(ids) != manifest["splits"]["train"]["utterances"]:
        raise ValueError("invalid frozen training ID inventory")
    snapshot = build_snapshot(Path(inputs["static_dir"]), Path(inputs["train_dir"]), inputs["tables_sha256"],
                              approved_train_ids=set(ids), dataset_release=config["dataset_release"],
                              seed=config["seed"], validation_fraction=config["validation_fraction"],
                              exclude_relations=tuple(config.get("exclude_relations", [])))
    import torch
    from safetensors.torch import save_file
    from turkish_asr_rdl.rdl.model import HeterogeneousRelationModel, evaluate_relations, train_relation_model
    from turkish_asr_rdl.rdl.scoring import linker_from_tables
    torch.manual_seed(config["seed"])
    model = HeterogeneousRelationModel(snapshot, **config["model"])
    history = train_relation_model(model, seed=config["seed"], **config["training"])
    evaluation = evaluate_relations(model)
    hashes = inputs["tables_sha256"]
    static = Path(inputs["static_dir"])
    aliases = read_table(static / "entity_aliases.csv", hashes["static/entity_aliases.csv"])
    entities = read_table(static / "entities.csv", hashes["static/entities.csv"])
    linker = linker_from_tables(entities, aliases)
    relations = read_table(static / "relationships_static.csv", hashes["static/relationships_static.csv"])
    linker["entity_topics"] = {r["head_id"]: r["tail_id"] for r in relations if r["relation_type_id"] == "rel_belongs_to"}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "snapshot.json", snapshot.to_dict())
    write_json(output / "resolved-config.json", config)
    write_json(output / "linker.json", linker)
    write_json(output / "training.json", {"history": history, "relation_evaluation": evaluation})
    import importlib.metadata
    import subprocess
    import sys
    code = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "diff", "HEAD"], text=True)
    import hashlib
    write_json(output / "environment.json", {"python": sys.version,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
        "code_commit": code, "dirty_diff_sha256": hashlib.sha256(dirty.encode()).hexdigest()})
    save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(output / "model.safetensors"))
    files = {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / "artifact-manifest.json", {
        "schema_version": "0.1", "run_id": "rdl-" + uuid.uuid4().hex,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "trained_internal_holdout_only" if config["validation_fraction"] else "trained_train_split",
        "dataset_release": config["dataset_release"],
        "split_manifest_sha256": config["split_manifest_sha256"], "files": files,
        "limitations": ["auto-extracted training labels, not human-verified relations",
                        "closed-world negatives may be true but unobserved",
                        "internal holdout is not corpus validation/test", "downstream ASR not evaluated"]})
    return evaluation


def load_model(directory: Path):
    """Verify all bundle members before loading safe weight tensors."""
    directory = Path(directory)
    manifest = json.loads((directory / "artifact-manifest.json").read_text(encoding="utf-8"))
    required = {"model.safetensors", "snapshot.json", "resolved-config.json", "linker.json", "training.json", "environment.json"}
    if not required.issubset(manifest.get("files", {})):
        raise ValueError("RDL artifact inventory lacks required members")
    for name, digest in manifest["files"].items():
        member = (directory / name).resolve()
        if not member.is_relative_to(directory.resolve()) or sha256_file(member) != digest:
            raise ValueError("RDL artifact path/hash mismatch")
    config = json.loads((directory / "resolved-config.json").read_text(encoding="utf-8"))
    snapshot = GraphSnapshot(**json.loads((directory / "snapshot.json").read_text(encoding="utf-8")))
    from safetensors.torch import load_file
    from turkish_asr_rdl.rdl.model import HeterogeneousRelationModel
    model = HeterogeneousRelationModel(snapshot, **config["model"])
    model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
    model.eval()
    linker = json.loads((directory / "linker.json").read_text(encoding="utf-8"))
    return model, linker, manifest


def main():
    parser = argparse.ArgumentParser(description="Train-only heterogeneous relation learning")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.preflight:
        print(json.dumps(preflight(config), indent=2))
        return
    if args.output is None:
        parser.error("--output is required for training")
    try:
        result = train(config, args.output)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
