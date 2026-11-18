"""Compose acoustic and learned relational inference inside M6's timed adapter."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from turkish_asr_rdl.decoding.export import adapter_result_fields
from turkish_asr_rdl.decoding.interface import Hypothesis
from turkish_asr_rdl.decoding.rescoring import rescore
from turkish_asr_rdl.training.acoustic import sha256_file


class RelationalAsrAdapter:
    """The shared run_timed harness measures both models and CPU/GPU transfers."""
    def __init__(self, acoustic_adapter, graph_directory: Path, graph_manifest_sha256: str, *,
                 dataset_release: str, split_manifest_sha256: str, fusion: dict,
                 missing_probability: float = 0.1):
        if fusion.get("lm_weight", 0) != 0:
            raise ValueError("this combined adapter has no external LM; use LM-disabled fusion")
        self.acoustic = acoustic_adapter
        self.graph_directory = Path(graph_directory)
        self.graph_manifest_sha256 = graph_manifest_sha256
        self.dataset_release = dataset_release
        self.split_manifest_sha256 = split_manifest_sha256
        base_manifest = getattr(acoustic_adapter, "manifest", {})
        if base_manifest.get("training_mode") in {"full", "lora", "ctc_finetune"}:
            if (base_manifest.get("dataset_release") != dataset_release or
                    base_manifest.get("split_manifest_sha256") != split_manifest_sha256):
                raise ValueError("adapted acoustic checkpoint uses a different release or split")
        self.fusion = dict(fusion)
        self.missing_probability = missing_probability
        self.system_id = acoustic_adapter.system_id + "+rdl"
        self.model_artifact_sha256 = acoustic_adapter.model_artifact_sha256
        self.requested_n = acoustic_adapter.requested_n
        self.score_convention_id = acoustic_adapter.score_convention_id + "+rdl-v1"
        self.config = {**acoustic_adapter.config, "rdl_graph_manifest_sha256": graph_manifest_sha256,
                       "fusion": self.fusion, "missing_probability": missing_probability,
                       "dataset_release": dataset_release, "split_manifest_sha256": split_manifest_sha256}
        self.decoder_config_sha256 = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()

    def load(self):
        import torch
        from turkish_asr_rdl.rdl.train import load_model
        if sha256_file(self.graph_directory / "artifact-manifest.json") != self.graph_manifest_sha256:
            raise ValueError("graph manifest hash mismatch")
        self.graph_model, self.linker, manifest = load_model(self.graph_directory)
        if (manifest["dataset_release"] != self.dataset_release or
                manifest["split_manifest_sha256"] != self.split_manifest_sha256):
            raise ValueError("acoustic/RDL release or split mismatch")
        self.acoustic.load()
        self.graph_model.eval()
        with torch.inference_mode():
            self.graph_states = self.graph_model.encode()

    def transcribe(self, batch):
        from turkish_asr_rdl.rdl.scoring import attach_scores
        result = []
        for hyp in self.acoustic.transcribe(batch):
            if hyp.status == "error":
                result.append(hyp)
                continue
            try:
                # Let the shared exporter choose canonical raw-text candidates.
                # Preserve the acoustic raw list for the final shared export.
                canonical = adapter_result_fields(hyp, self.requested_n)["candidates"]
                scored = attach_scores(canonical, self.graph_model, self.linker, states=self.graph_states,
                                       missing_probability=self.missing_probability)
                selected_id, scored = rescore(scored, **self.fusion)
                by_text = {c["text"]: c for c in scored}
                updated = []
                for candidate in hyp.candidates:
                    source = by_text[candidate.text]
                    one = {**source, "candidate_id": candidate.candidate_id, "rank": candidate.rank,
                           "acoustic_log_score": candidate.acoustic_log_score,
                           "token_count": len(candidate.token_ids)}
                    _, combined = rescore([one], **self.fusion)
                    details = {**candidate.score_details, "rdl": source["score_details"]["rdl"]}
                    reasons = {k: v for k, v in details.get("null_reasons", {}).items()
                               if k not in {"rdl_log_score", "combined_score"}}
                    details["null_reasons"] = reasons
                    updated.append(replace(candidate, rdl_log_score=source["rdl_log_score"],
                                           combined_score=combined[0]["combined_score"], score_details=details))
                selected = next(c for c in updated if c.candidate_id == selected_id)
                result.append(replace(hyp, text=selected.text, status="ok" if selected.text else "empty",
                                      candidates=tuple(updated), selected_candidate_id=selected_id))
            except Exception as exc:
                result.append(Hypothesis(hyp.utterance_id, "error", None,
                                         error_code=type(exc).__name__, error_message="relational inference failed"))
        return result
