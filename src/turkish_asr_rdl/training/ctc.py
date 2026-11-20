"""Pinned XLS-R adaptation with strict or explicit train-only vocabulary extension."""
from __future__ import annotations

import re

from turkish_asr_rdl.training.acoustic import load_audio, sha256_file, training_main
from turkish_asr_rdl.evaluation.normalize import normalize, TR_NORMALIZED_V1


def label_text(text: str) -> str:
    """Training-label spelling view only; evaluation references remain unchanged."""
    return normalize(text, TR_NORMALIZED_V1)


def audit_targets(rows, tokenizer):
    """Describe OOVs without deleting targets or inferring spoken numerals."""
    from collections import Counter
    counts = Counter()
    affected = []
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("CTC target audits accept training references only")
        text = label_text(row["reference_text"])
        if tokenizer.unk_token_id in tokenizer(text).input_ids:
            affected.append(row["utterance_id"])
            vocab = tokenizer.get_vocab()
            counts.update(char for char in text if char != " " and char not in vocab)
    return {"rows": len(rows), "affected_utterance_ids": affected,
            "unsupported_characters": dict(sorted(counts.items())),
            "policy": "strict OOV; no invented numeral readings or silently removed characters"}


def extend_vocabulary(model, tokenizer, characters):
    """Append audited character IDs and output rows, preserving the entire old head."""
    import torch
    from torch import nn
    old_vocab = tokenizer.get_vocab().copy()
    old_size = len(tokenizer)
    if model.lm_head.out_features != old_size or set(old_vocab.values()) != set(range(old_size)):
        raise ValueError("parent vocabulary/head must be contiguous and compatible")
    characters = sorted(characters)
    if not characters or len(set(characters)) != len(characters) or any(len(c) != 1 or c.isspace() or c in old_vocab for c in characters):
        raise ValueError("extension requires unique, new, single nonspace characters")
    if tokenizer.add_tokens(characters, special_tokens=False) != len(characters):
        raise ValueError("not every audited character was added")
    if any(tokenizer.get_vocab()[token] != index for token, index in old_vocab.items()):
        raise ValueError("extension changed a pretrained token ID")
    old = model.lm_head
    new = nn.Linear(old.in_features, len(tokenizer), bias=old.bias is not None,
                    device=old.weight.device, dtype=old.weight.dtype)
    nn.init.normal_(new.weight, mean=0.0, std=model.config.initializer_range)
    if new.bias is not None:
        nn.init.zeros_(new.bias)
    with torch.no_grad():
        new.weight[:old_size].copy_(old.weight)
        if old.bias is not None:
            new.bias[:old_size].copy_(old.bias)
    model.lm_head = new
    model.config.vocab_size = len(tokenizer)
    return {"old_vocab_size": old_size, "new_vocab_size": len(tokenizer),
            "added_token_ids": {c: tokenizer.convert_tokens_to_ids(c) for c in characters},
            "old_output_rows_preserved": True, "blank_id": model.config.pad_token_id,
            "new_row_initialization": "normal(0, parent initializer_range); zero bias"}


def build_model(config: dict):
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from turkish_asr_rdl.training.acoustic import read_jsonl
    parent = config["model"]
    if not re.fullmatch(r"[a-f0-9]{40}", parent["revision"]) or config["mode"] not in {"ctc_finetune", "ctc_vocab_extension"}:
        raise ValueError("an immutable CTC ASR parent and explicit CTC mode are required")
    processor = Wav2Vec2Processor.from_pretrained(parent["id"], revision=parent["revision"])
    rows = read_jsonl(Path(config["inputs"]["references"]))
    audit = audit_targets(rows, processor.tokenizer)
    extension = config["mode"] == "ctc_vocab_extension"
    if extension and sorted(config.get("vocabulary_extension", {}).get("characters", [])) != sorted(audit["unsupported_characters"]):
        raise ValueError("configured vocabulary extension differs from the train-only target audit")
    if audit["affected_utterance_ids"] and not extension:
        raise ValueError("CTC target audit failed before weight download: " + str(audit["unsupported_characters"]))
    weight = Path(hf_hub_download(parent["id"], parent["weight_file"], revision=parent["revision"]))
    if sha256_file(weight) != parent["weight_sha256"]:
        raise ValueError("CTC parent weight checksum mismatch")
    model = Wav2Vec2ForCTC.from_pretrained(parent["id"], revision=parent["revision"])
    if model.config.vocab_size != len(processor.tokenizer) or model.config.pad_token_id != processor.tokenizer.pad_token_id:
        raise ValueError("CTC head/vocabulary/blank mismatch")
    if extension:
        details = extend_vocabulary(model, processor.tokenizer, audit["unsupported_characters"])
        if audit_targets(rows, processor.tokenizer)["affected_utterance_ids"]:
            raise ValueError("extension still leaves unsupported targets")
        config["vocabulary_extension"]["resolved"] = details
    # Frozen convolutional frontend is an explicit adaptation choice, not a new head.
    model.freeze_feature_encoder()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.ctc_zero_infinity = False
    return model, processor


def build_collator(processor):
    def collate(rows):
        batch = processor.feature_extractor([load_audio(row) for row in rows], sampling_rate=16000,
                                            padding=True, return_attention_mask=True, return_tensors="pt")
        labels = [processor.tokenizer(label_text(row["reference_text"])).input_ids for row in rows]
        if any(not ids or processor.tokenizer.unk_token_id in ids or processor.tokenizer.pad_token_id in ids for ids in labels):
            raise ValueError("CTC labels contain empty/OOV/blank targets; follow the recorded strict target policy")
        padded = processor.tokenizer.pad({"input_ids": labels}, padding=True, return_tensors="pt")
        batch["labels"] = padded.input_ids.masked_fill(padded.attention_mask.ne(1), -100)
        return batch
    return collate


def main():
    training_main(build_model, build_collator)


if __name__ == "__main__":
    main()
