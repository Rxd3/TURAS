"""Synthetic optimizer-resource probes: no corpus, pretrained weights or ASR results."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

from turkish_asr_rdl.training.acoustic import sha256_file, write_json


def probe(config, *, target_length, measured_steps=2):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import WhisperConfig, WhisperForConditionalGeneration, Wav2Vec2Config, Wav2Vec2ForCTC
    from turkish_asr_rdl.training.whisper import configure_whisper
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("this probe requires working CUDA and bf16")
    if target_length < 1 or measured_steps < 1:
        raise ValueError("positive target length/step count required")
    torch.manual_seed(config["training"]["seed"])
    torch.cuda.empty_cache()
    free_before, total = torch.cuda.mem_get_info()
    record = {"kind": "synthetic_training_resource_probe", "mode": config["mode"],
              "parent": config["model"], "target_length": target_length, "batch_size": 1,
              "accumulation": 1, "warmup_optimizer_steps": 1, "measured_optimizer_steps": measured_steps,
              "precision": "bf16 autocast, fp32 parameters", "pretrained_weights_used": False,
              "corpus_audio_or_transcripts_used": False, "free_before_bytes": free_before,
              "total_vram_bytes": total, "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0)}
    model = optimizer = features = labels = loss = None
    try:
        parent = config["model"]
        source = Path(hf_hub_download(parent["id"], "config.json", revision=parent["revision"]))
        record["architecture_config_sha256"] = sha256_file(source)
        if config["mode"] in {"lora", "full"}:
            architecture = WhisperConfig.from_json_file(source)
            if target_length >= architecture.max_target_positions:
                raise ValueError("target length exceeds architecture context")
            model = configure_whisper(WhisperForConditionalGeneration(architecture), config["mode"], config.get("lora", {}))
            features = torch.randn(1, architecture.num_mel_bins, architecture.max_source_positions * 2, device="cuda")
            labels = torch.randint(100, min(1000, architecture.vocab_size), (1, target_length), device="cuda")
            batch = {"input_features": features, "labels": labels}
        else:
            architecture = Wav2Vec2Config.from_json_file(source)
            model = Wav2Vec2ForCTC(architecture)
            model.freeze_feature_encoder()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.config.ctc_zero_infinity = False
            features = torch.randn(1, 30 * 16000, device="cuda")
            labels = torch.randint(4, architecture.vocab_size, (1, target_length), device="cuda")
            batch = {"input_values": features, "labels": labels}
        labels[0, -1] = architecture.eos_token_id if config["mode"] in {"lora", "full"} else 4
        model.to("cuda").train()
        parameters = [p for p in model.parameters() if p.requires_grad]
        record["trainable_parameters"] = sum(p.numel() for p in parameters)
        record["total_parameters"] = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=config["training"]["learning_rate"])
        elapsed = []
        for index in range(measured_steps + 1):
            if index == 1:
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError("synthetic loss is not finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            if index:
                elapsed.append(time.perf_counter() - started)
        record.update(status="passed_synthetic_probe", optimizer_step_seconds=elapsed,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except torch.OutOfMemoryError:
        record.update(status="out_of_memory", peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    finally:
        model = optimizer = features = labels = loss = None
        if "batch" in locals():
            batch.clear()
        if "parameters" in locals():
            parameters.clear()
        gc.collect()
        torch.cuda.empty_cache()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-lengths", type=int, nargs="+", default=[128, 440])
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    records = []
    for length in args.target_lengths:
        record = probe(config, target_length=length, measured_steps=args.steps)
        records.append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, {"status": "resource_probe_only", "records": records,
                                "limitations": ["random weights and tensor targets; not corpus training",
                                                "no recognition accuracy, inference RTF or epoch-duration claim",
                                                "probe accumulation=1; training configuration uses accumulation=16"]})
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
