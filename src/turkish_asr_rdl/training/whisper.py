"""Whisper LoRA/full training against an immutable multilingual parent."""
from __future__ import annotations

import re

from turkish_asr_rdl.training.acoustic import load_audio, sha256_file, training_main


def configure_whisper(model, mode: str, lora: dict):
    if mode == "lora":
        from peft import LoraConfig, get_peft_model
        # Generic PEFT wrapper preserves input_features; a text-only seq2seq
        # wrapper can incorrectly forward input_ids to Whisper.
        model = get_peft_model(model, LoraConfig(
            r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
            target_modules=["q_proj", "v_proj"], bias="none",
        ))
    elif mode == "full":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    else:
        raise ValueError("Whisper mode must be lora or full")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def build_model(config: dict):
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    parent = config["model"]
    if not re.fullmatch(r"[a-f0-9]{40}", parent["revision"]) or parent["id"].endswith(".en"):
        raise ValueError("an immutable multilingual Whisper checkpoint is required")
    weight = Path(hf_hub_download(parent["id"], parent["weight_file"], revision=parent["revision"]))
    if sha256_file(weight) != parent["weight_sha256"]:
        raise ValueError("parent weight checksum mismatch")
    processor = WhisperProcessor.from_pretrained(parent["id"], revision=parent["revision"], language="turkish", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(parent["id"], revision=parent["revision"], use_safetensors=True)
    return configure_whisper(model, config["mode"], config.get("lora", {})), processor


def build_collator(processor):
    import torch

    def collate(rows):
        audio = [load_audio(row) for row in rows]
        features = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
        labels = [processor.tokenizer(row["reference_text"]).input_ids for row in rows]
        if any(len(ids) > 448 for ids in labels):
            raise ValueError("Whisper target exceeds context; approved resegmentation is required")
        padded = processor.tokenizer.pad({"input_ids": labels}, return_tensors="pt")
        target = padded.input_ids.masked_fill(padded.attention_mask.ne(1), -100)
        start = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if torch.all(target[:, 0] == start):
            target = target[:, 1:]
        return {"input_features": features, "labels": target}

    return collate


def main():
    training_main(build_model, build_collator)


if __name__ == "__main__":
    main()
