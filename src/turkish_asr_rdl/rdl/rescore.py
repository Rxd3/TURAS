"""Offline learned-RDL rescoring of an existing, frozen N-best artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import uuid

from turkish_asr_rdl.decoding.rescoring import rescore
from turkish_asr_rdl.training.acoustic import read_jsonl, sha256_file, write_json


def reject_gold(value):
    if isinstance(value, dict):
        if any(k.lower() in {"reference_text", "gold", "target_text", "reference_tokens"} for k in value):
            raise ValueError("gold/reference fields are forbidden in N-best inputs")
        for nested in value.values():
            reject_gold(nested)
    elif isinstance(value, list):
        for nested in value:
            reject_gold(nested)


def preflight(config):
    blockers = []
    for name in ("nbest", "expected_ids", "graph_artifact"):
        value = config.get("inputs", {}).get(name)
        if not value or not Path(value).exists():
            blockers.append("missing input: " + name)
    for name in ("nbest_sha256", "expected_ids_sha256", "graph_manifest_sha256"):
        digest = config.get("inputs", {}).get(name)
        if not isinstance(digest, str) or len(digest) != 64:
            blockers.append("missing reviewed hash: " + name)
    if not config.get("protocol_reviewed"):
        blockers.append("rescoring protocol awaits M4/M6 review")
    if config.get("split") == "test":
        selection = config.get("validation_selection", {})
        if not config.get("fusion_frozen") or not selection.get("path") or not selection.get("sha256"):
            blockers.append("test rescoring requires frozen fusion and hashed validation selection evidence")
    elif config.get("split") != "validation":
        blockers.append("offline selection uses validation; final reporting uses test")
    return {"status": "blocked" if blockers else "ready_for_input_validation", "blockers": blockers}


def run(config, output: Path, *, quantity_extractor=None):
    report = preflight(config)
    if report["blockers"]:
        raise ValueError("rescoring blocked: " + "; ".join(report["blockers"]))
    if output.exists() or output.with_suffix(output.suffix + ".timing.json").exists():
        raise ValueError("output already exists")
    inputs = config["inputs"]
    for name, digest_key in (("nbest", "nbest_sha256"), ("expected_ids", "expected_ids_sha256")):
        if sha256_file(Path(inputs[name])) != inputs[digest_key]:
            raise ValueError("rescoring input hash mismatch: " + name)
    graph = Path(inputs["graph_artifact"])
    graph_hash = sha256_file(graph / "artifact-manifest.json")
    if graph_hash != inputs["graph_manifest_sha256"]:
        raise ValueError("graph manifest hash mismatch")
    if config["split"] == "test":
        selection = config["validation_selection"]
        if sha256_file(Path(selection["path"])) != selection["sha256"]:
            raise ValueError("validation selection evidence hash mismatch")
    from turkish_asr_rdl.rdl.train import load_model
    from turkish_asr_rdl.rdl.scoring import attach_scores
    cold = time.perf_counter()
    model, linker, graph_manifest = load_model(graph)
    cold_seconds = time.perf_counter() - cold
    if (graph_manifest["dataset_release"] != config["dataset_release"] or
            graph_manifest["split_manifest_sha256"] != config["split_manifest_sha256"]):
        raise ValueError("graph artifact release/split mismatch")
    started = time.perf_counter()
    rows = read_jsonl(Path(inputs["nbest"]))
    ids = Path(inputs["expected_ids"]).read_text(encoding="utf-8").splitlines()
    if len(ids) != len(set(ids)) or not ids or [r.get("utterance_id") for r in rows] != ids:
        raise ValueError("N-best IDs must match the frozen ordered input set exactly")
    for row in rows:
        reject_gold(row)
        if (row.get("schema_version") != "0.1" or row.get("dataset_release") != config["dataset_release"] or
                row.get("split") != config["split"] or row.get("split_manifest_sha256") != config["split_manifest_sha256"]):
            raise ValueError("N-best schema/release/split mismatch")
        if row.get("actual_n") != len(row.get("candidates", [])) or row["actual_n"] > row.get("requested_n", 0):
            raise ValueError("N-best counts are inconsistent")
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()
    run_id = "rdl-" + uuid.uuid4().hex
    import torch
    with torch.inference_mode():
        encoded = model.encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents replacing an existing experimental artifact.
    # An exception leaves a partial file, never silently presented as completed.
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            updated = attach_scores(row["candidates"], model, linker, states=encoded,
                                    missing_probability=config["missing_probability"], quantity_extractor=quantity_extractor)
            selected, updated = rescore(updated, **config["fusion"])
            result = {**row, "source_run_id": row["run_id"], "run_id": run_id,
                      "source_decoder_config_sha256": row["decoder_config_sha256"],
                      "decoder_config_sha256": config_hash, "rdl_artifact_sha256": graph_hash,
                      "candidates": updated, "selected_candidate_id": selected}
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
    seconds = time.perf_counter() - started
    sidecar = {"status": "completed", "run_id": run_id, "utterance_count": len(rows),
               "seconds": seconds, "cold_graph_load_seconds": cold_seconds,
               "scope": "offline N-best read, CPU RDL/fusion and JSONL write; excludes acoustic generation/audio decoding",
               "end_to_end_asr_rtf": None, "source_nbest_sha256": inputs["nbest_sha256"],
               "output_sha256": sha256_file(output), "config_sha256": config_hash}
    write_json(output.with_suffix(output.suffix + ".timing.json"), sidecar)
    return sidecar


def main():
    parser = argparse.ArgumentParser(description="Reference-free offline RDL N-best rescoring")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.preflight:
        print(json.dumps(preflight(config), indent=2))
        return
    if args.output is None:
        parser.error("--output is required")
    try:
        result = run(config, args.output)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
