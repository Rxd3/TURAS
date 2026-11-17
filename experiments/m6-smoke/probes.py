"""Training-memory and PyTorch Geometric probes for the 5070 Ti smoke tests.

torch, transformers, peft and torch_geometric are imported inside the probes, so this
module imports on a machine that has none of them.
"""
from __future__ import annotations

import math
import time
import traceback
from collections.abc import Sequence
from typing import Any

import numpy as np

import adapters
import smoke_common
from turkish_asr_rdl.decoding.audio import TARGET_SAMPLE_RATE_HZ

# probe name -> (base model, batch size, "full" or "lora")
TRAINING_PROBES = {
    "train-whisper-small-full": ("whisper-small", 8, "full"),
    "train-whisper-large-v3-lora": ("whisper-large-v3", 2, "lora"),
    "train-whisper-large-v3-full": ("whisper-large-v3", 1, "full"),
}
TRAIN_STEPS = 3
LEARNING_RATE = 1e-5
MAX_LABEL_TOKENS = 448
LORA_SETTINGS = {"r": 32, "lora_alpha": 64, "lora_dropout": 0.0, "target_modules": ["q_proj", "v_proj"],
                 "bias": "none"}
PYG_VERSION = "2.8.1"
PYG_SEED = 2026
PYG_HIDDEN = 32
PYG_NODES = {"word": (12, 16), "concept": (6, 8)}  # node type -> (count, feature size)
PYG_RELATIONS = ("mentions", "defines")
PYG_EDGES_PER_RELATION = 30


def train_probe(name: str, clips: Sequence[tuple[np.ndarray, str]], vram: Any) -> dict[str, Any]:
    """Three bf16 AdamW steps with gradient checkpointing; an out-of-memory error is a recorded outcome."""
    base, batch_size, mode = TRAINING_PROBES[name]
    if len(clips) < batch_size:
        raise ValueError(f"{name} needs {batch_size} clips, got {len(clips)}; prepare more with --clips")
    torch = smoke_common.cuda_torch()
    files = adapters.fetch_model(base)
    vram.reset()
    try:
        result = _train(torch, files, mode, clips[:batch_size])
    except torch.cuda.OutOfMemoryError as exc:
        result = {"outcome": "out_of_memory", "oom_message": smoke_common.one_line(exc)}
    memory = vram.peak()
    fits = result["outcome"] == "completed" and memory["peak_vram_gib"] < memory["total_vram_gib"]
    return {
        "base_model": {"repo_id": files.repo_id, "revision": files.revision},
        "mode": mode, "batch_size": batch_size, "steps": TRAIN_STEPS, "learning_rate": LEARNING_RATE,
        "precision": "bfloat16 weights, gradients and AdamW states", "gradient_checkpointing": "non-reentrant",
        "lora": LORA_SETTINGS if mode == "lora" else None,
        **result, **memory, "fits_in_dedicated_vram": fits,
    }


def _load_for_training(torch: Any, files: smoke_common.ModelFiles, mode: str) -> tuple[Any, Any, int]:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(files.path)
    processor.tokenizer.set_prefix_tokens(language="turkish", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(files.path, dtype=torch.bfloat16)
    start_id = model.config.decoder_start_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if mode == "lora":
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(**LORA_SETTINGS))
    return model.to(smoke_common.DEVICE).train(), processor, start_id


def _training_batch(torch: Any, processor: Any, clips: Sequence[tuple[np.ndarray, str]],
                    start_id: int) -> tuple[Any, Any]:
    """Log-mel features and label IDs from the train-split transcripts; padding is ignored by the loss."""
    features = processor.feature_extractor([audio for audio, _ in clips], sampling_rate=TARGET_SAMPLE_RATE_HZ,
                                           return_tensors="pt").input_features
    tokens = processor.tokenizer([text for _, text in clips], padding=True, truncation=True,
                                 max_length=MAX_LABEL_TOKENS, return_tensors="pt")
    labels = tokens.input_ids.masked_fill(tokens.attention_mask.ne(1), -100)
    if bool((labels[:, 0] == start_id).all()):
        labels = labels[:, 1:]  # the model prepends the start token itself
    return features.to(smoke_common.DEVICE, torch.bfloat16), labels.to(smoke_common.DEVICE)


def _train(torch: Any, files: smoke_common.ModelFiles, mode: str,
           clips: Sequence[tuple[np.ndarray, str]]) -> dict[str, Any]:
    model, processor, start_id = _load_for_training(torch, files, mode)
    features, labels = _training_batch(torch, processor, clips, start_id)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE)
    losses = []
    start = time.perf_counter()
    for _ in range(TRAIN_STEPS):
        loss = model(input_features=features, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    return {
        "outcome": "completed",
        "seconds_per_step": (time.perf_counter() - start) / TRAIN_STEPS,
        "losses_finite": all(math.isfinite(value) for value in losses),
        "trainable_params": sum(p.numel() for p in params),
        "total_params": sum(p.numel() for p in model.parameters()),
    }


def pyg_probe(vram: Any) -> dict[str, Any]:
    """Two HeteroConv(SAGEConv) layers forward and backward on CUDA, in fp32 and in bf16."""
    torch = smoke_common.cuda_torch()
    import torch_geometric

    vram.reset()
    checks = {label: _pyg_check(torch, dtype) for label, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16))}
    return {
        "torch_geometric": torch_geometric.__version__, "expected_version": PYG_VERSION,
        "outcome": "completed" if all(check["ok"] for check in checks.values()) else "failed",
        "graph": {"nodes": PYG_NODES, "relations": list(PYG_RELATIONS), "reverse_edges": True,
                  "edges_per_relation": PYG_EDGES_PER_RELATION, "layers": 2, "hidden": PYG_HIDDEN},
        **checks, **vram.peak(),
    }


def _pyg_check(torch: Any, dtype: Any) -> dict[str, Any]:
    try:
        loss, grads_finite = _hetero_pass(torch, dtype)
    except Exception as exc:  # each precision is its own check; its failure is the recorded finding
        return {"ok": False, "error": smoke_common.one_line(exc), "traceback": traceback.format_exc()}
    finite = math.isfinite(loss)
    return {"ok": finite and grads_finite, "loss_finite": finite, "grads_finite": grads_finite}


def _toy_graph(torch: Any) -> Any:
    from torch_geometric.data import HeteroData

    generator = torch.Generator().manual_seed(PYG_SEED)
    data = HeteroData()
    for node, (count, size) in PYG_NODES.items():
        data[node].x = torch.randn(count, size, generator=generator)
    words, concepts = PYG_NODES["word"][0], PYG_NODES["concept"][0]
    for relation in PYG_RELATIONS:
        src = torch.randint(0, words, (PYG_EDGES_PER_RELATION,), generator=generator)
        dst = torch.randint(0, concepts, (PYG_EDGES_PER_RELATION,), generator=generator)
        data["word", relation, "concept"].edge_index = torch.stack([src, dst])
        data["concept", f"rev_{relation}", "word"].edge_index = torch.stack([dst, src])
    return data.to(smoke_common.DEVICE)


def _hetero_pass(torch: Any, dtype: Any) -> tuple[float, bool]:
    from torch_geometric.nn import HeteroConv, SAGEConv

    data = _toy_graph(torch)
    sizes = {node: size for node, (_, size) in PYG_NODES.items()}
    first = HeteroConv({et: SAGEConv((sizes[et[0]], sizes[et[2]]), PYG_HIDDEN) for et in data.edge_types}, aggr="sum")
    second = HeteroConv({et: SAGEConv((PYG_HIDDEN, PYG_HIDDEN), PYG_HIDDEN) for et in data.edge_types}, aggr="sum")
    layers = torch.nn.ModuleList([first, second]).to(smoke_common.DEVICE, dtype)
    hidden = {node: x.to(dtype) for node, x in data.x_dict.items()}
    hidden = {node: x.relu() for node, x in layers[0](hidden, data.edge_index_dict).items()}
    hidden = layers[1](hidden, data.edge_index_dict)
    loss = sum(x.float().pow(2).mean() for x in hidden.values())
    loss.backward()
    torch.cuda.synchronize()
    grads = [p.grad for p in layers.parameters() if p.grad is not None]
    grads_finite = bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
    return float(loss.detach()), grads_finite
