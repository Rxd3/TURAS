"""Verified adapted checkpoint loading and the optional M6 AsrAdapter bridge."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from turkish_asr_rdl.training.acoustic import sha256_file


def verify_checkpoint(directory: Path, expected_manifest_sha256: str):
    directory = Path(directory).resolve()
    path = directory / "checkpoint-manifest.json"
    if sha256_file(path) != expected_manifest_sha256:
        raise ValueError("checkpoint manifest hash mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("files"):
        raise ValueError("checkpoint inventory is empty")
    for name, digest in manifest["files"].items():
        member = (directory / name).resolve()
        if not member.is_relative_to(directory) or sha256_file(member) != digest:
            raise ValueError("checkpoint member path/hash mismatch")
    if manifest.get("training_mode") not in {"lora", "full", "ctc_finetune", "ctc_vocab_extension"}:
        raise ValueError("unsupported checkpoint training mode")
    return manifest


class AdaptedAsrAdapter:
    """Duck-compatible with M6's harness; no M6 module is copied or rewritten.

    load()/nbest() work standalone. transcribe() uses the reviewed M6 dataclasses
    when that branch is integrated. No gold/reference argument is accepted.
    """
    def __init__(self, directory: Path, manifest_sha256: str, *, system_id: str,
                 device: str = "cpu", precision: str = "fp32", requested_n: int = 5,
                 beam_size: int = 10, max_new_tokens: int = 440):
        if requested_n < 1 or beam_size < requested_n or max_new_tokens < 1 or precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("invalid decoding limits or precision")
        self.directory = Path(directory)
        self.model_artifact_sha256 = manifest_sha256
        self.system_id = system_id
        self.requested_n = requested_n
        self.device = device
        self.precision = precision
        self.manifest = verify_checkpoint(self.directory, manifest_sha256)
        self.family = "ctc" if self.manifest["training_mode"] in {"ctc_finetune", "ctc_vocab_extension"} else "whisper"
        self._configure_decoder(device, precision, requested_n, beam_size, max_new_tokens)

    def _configure_decoder(self, device, precision, requested_n, beam_size, max_new_tokens):
        self.device, self.precision, self.requested_n = device, precision, requested_n
        self.score_convention_id = ("ctc-exact-alignment-marginalized-v1" if self.family == "ctc"
                                    else "whisper-original-logits-generated-plus-eos-v1")
        self.config = {"family": self.family, "device": device, "precision": precision,
                       "requested_n": requested_n, "beam_size": beam_size, "language": "tr", "task": "transcribe",
                       "max_new_tokens": max_new_tokens, "score_convention_id": self.score_convention_id}
        self.decoder_config_sha256 = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()

    def load(self):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor, Wav2Vec2ForCTC, Wav2Vec2Processor
        verify_checkpoint(self.directory, self.model_artifact_sha256)
        if self.device == "cpu" and self.precision != "fp32":
            raise ValueError("use fp32 for CPU decoding")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        if self.family == "whisper":
            self.processor = WhisperProcessor.from_pretrained(self.directory)
            if self.manifest["training_mode"] == "lora":
                from huggingface_hub import hf_hub_download
                from peft import PeftModel
                parent = self.manifest["parent"]
                weight = Path(hf_hub_download(parent["id"], parent["weight_file"], revision=parent["revision"]))
                if sha256_file(weight) != parent["weight_sha256"]:
                    raise ValueError("LoRA parent checksum mismatch")
                base = WhisperForConditionalGeneration.from_pretrained(parent["id"], revision=parent["revision"], use_safetensors=True)
                self.model = PeftModel.from_pretrained(base, self.directory).merge_and_unload()
            else:
                self.model = WhisperForConditionalGeneration.from_pretrained(self.directory, use_safetensors=True)
        else:
            self.processor = Wav2Vec2Processor.from_pretrained(self.directory)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.directory, use_safetensors=True)
            if self.model.config.vocab_size != len(self.processor.tokenizer) or self.model.config.pad_token_id != self.processor.tokenizer.pad_token_id:
                raise ValueError("adapted CTC head/tokenizer mismatch")
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[self.precision]
        self.model.to(device=self.device, dtype=dtype).eval()

    def nbest(self, samples, *, deduplicate: bool = False):
        """One 16 kHz mono utterance; caller owns audio hashing and RTF timing."""
        import torch
        import numpy as np
        if not isinstance(samples, np.ndarray) or samples.ndim != 1 or not 0 < len(samples) <= 480000 or not np.isfinite(samples).all():
            raise ValueError("expected finite 16 kHz mono samples of at most 30 seconds")
        with torch.inference_mode():
            if self.family == "whisper":
                from turkish_asr_rdl.decoding.whisper import whisper_nbest
                features = self.processor.feature_extractor(samples, sampling_rate=16000, return_tensors="pt").input_features
                return whisper_nbest(self.model, self.processor, features.to(self.device, dtype=self.model.dtype),
                                      requested_n=self.requested_n, max_new_tokens=self.config["max_new_tokens"],
                                      deduplicate=deduplicate)
            from turkish_asr_rdl.decoding.ctc import ctc_nbest
            batch = self.processor.feature_extractor(samples, sampling_rate=16000, return_attention_mask=True, return_tensors="pt")
            batch = {key: value.to(self.device, dtype=self.model.dtype) if key == "input_values" else value.to(self.device)
                     for key, value in batch.items()}
            frames = self.model(**batch).logits[0].float().log_softmax(-1).cpu().tolist()
            return ctc_nbest(frames, self.processor.tokenizer, blank_id=self.model.config.pad_token_id,
                              requested_n=self.requested_n, beam_size=self.config["beam_size"], deduplicate=deduplicate)

    def transcribe(self, batch):
        try:
            from turkish_asr_rdl.decoding.interface import Candidate, Hypothesis
        except ImportError:
            raise ValueError("M6's reviewed decoding.interface must be integrated before harness use") from None
        output = []
        for item in batch:
            try:
                rows, _ = self.nbest(item.samples, deduplicate=False)
                if not rows:
                    raise ValueError("decoder returned no valid candidates")
                candidates = tuple(Candidate(
                    candidate_id=row["candidate_id"], rank=row["rank"], text=row["text"], token_ids=tuple(row["token_ids"]),
                    acoustic_log_score=row["acoustic_log_score"], acoustic_score_kind=row["acoustic_score_kind"],
                    decoder_sequence_score=row["decoder_sequence_score"],
                    score_details=row["score_details"],
                ) for row in rows)
                text = candidates[0].text
                output.append(Hypothesis(item.utterance_id, "ok" if text else "empty", text, candidates))
            except Exception as exc:
                output.append(Hypothesis(item.utterance_id, "error", None,
                                         error_code=type(exc).__name__, error_message="adapted decoding failed"))
        return output


class PretrainedAsrAdapter(AdaptedAsrAdapter):
    """Unmodified pinned parent using exactly the adapted system's decoding path."""
    def __init__(self, parent: dict, *, family: str, system_id: str,
                 device="cuda", precision="bf16", requested_n=5, beam_size=10, max_new_tokens=440):
        import re
        if (family not in {"whisper", "ctc"} or not re.fullmatch(r"[a-f0-9]{40}", parent.get("revision", ""))
                or not re.fullmatch(r"[a-f0-9]{64}", parent.get("weight_sha256", ""))
                or requested_n < 1 or beam_size < requested_n or precision not in {"fp32", "bf16", "fp16"}):
            raise ValueError("invalid pinned pretrained adapter configuration")
        self.parent, self.family, self.system_id = dict(parent), family, system_id
        self.model_artifact_sha256 = parent["weight_sha256"]
        self._configure_decoder(device, precision, requested_n, beam_size, max_new_tokens)

    def load(self):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import WhisperForConditionalGeneration, WhisperProcessor, Wav2Vec2ForCTC, Wav2Vec2Processor
        parent = self.parent
        weight = Path(hf_hub_download(parent["id"], parent["weight_file"], revision=parent["revision"], local_files_only=True))
        if sha256_file(weight) != parent["weight_sha256"]:
            raise ValueError("pretrained parent checksum mismatch")
        model_cls, processor_cls = ((WhisperForConditionalGeneration, WhisperProcessor) if self.family == "whisper"
                                    else (Wav2Vec2ForCTC, Wav2Vec2Processor))
        self.processor = processor_cls.from_pretrained(parent["id"], revision=parent["revision"], local_files_only=True)
        self.model = model_cls.from_pretrained(parent["id"], revision=parent["revision"], local_files_only=True)
        if self.family == "ctc" and (self.model.config.vocab_size != len(self.processor.tokenizer)
                                    or self.model.config.pad_token_id != self.processor.tokenizer.pad_token_id):
            raise ValueError("pretrained CTC head/tokenizer mismatch")
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[self.precision]
        self.model.to(device=self.device, dtype=dtype).eval()
