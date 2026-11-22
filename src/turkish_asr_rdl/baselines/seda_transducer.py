"""Pinned Turkish Seda Zipformer2 RNN-T, CPU int8, greedy, without external LM."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np

from turkish_asr_rdl.decoding.interface import Hypothesis


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


class SedaTransducerAdapter:
    system_id = "baseline-seda-zipformer2-rnnt"
    requested_n = 0
    score_convention_id = None

    def __init__(self, directory):
        self.directory = Path(directory)
        self.inventory = self.directory / "verified-inventory.json"
        self.model_artifact_sha256 = file_sha(self.inventory)
        self.config = {"model": {"id": "atasoglu/seda-v0.1", "revision": "5cbbda1b34bde5e3004e79b137dc3871587ea282",
                      "architecture": "streaming Zipformer2 RNN-T", "precision": "int8 encoder/joiner; fp32 decoder",
                      "device": "cpu", "threads": 1},
                       "decoding": {"algorithm": "greedy_search", "external_lm": False, "hotwords": False,
                          "endpointing": False, "sample_rate_hz": 16000, "feature_dim": 80,
                          "end_padding_samples": 4800, "formatting": "raw decoder text", "requested_n": 0}}
        self.decoder_config_sha256 = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()

    def load(self):
        import sherpa_onnx
        if file_sha(self.inventory) != self.model_artifact_sha256:
            raise ValueError("Transducer inventory changed")
        inventory = json.loads(self.inventory.read_text())
        if inventory["revision"] != self.config["model"]["revision"]:
            raise ValueError("Transducer revision mismatch")
        for name in ["encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx", "tokens.txt"]:
            if file_sha(self.directory / name) != inventory["files"][name]["sha256"]:
                raise ValueError(f"Transducer file hash mismatch: {name}")
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(self.directory / "tokens.txt"), encoder=str(self.directory / "encoder.int8.onnx"),
            decoder=str(self.directory / "decoder.onnx"), joiner=str(self.directory / "joiner.int8.onnx"),
            num_threads=1, sample_rate=16000, feature_dim=80, decoding_method="greedy_search",
            enable_endpoint_detection=False, provider="cpu")

    def transcribe(self, batch):
        results = []
        for item in batch:
            stream = self.recognizer.create_stream()
            stream.accept_waveform(16000, np.ascontiguousarray(item.samples, dtype=np.float32))
            stream.accept_waveform(16000, np.zeros(4800, dtype=np.float32))
            stream.input_finished()
            while self.recognizer.is_ready(stream):
                self.recognizer.decode_stream(stream)
            text = self.recognizer.get_result(stream)
            if not isinstance(text, str):
                raise ValueError("Expected sherpa-onnx raw result text")
            results.append(Hypothesis(item.utterance_id, "ok" if text else "empty", text))
        return results
