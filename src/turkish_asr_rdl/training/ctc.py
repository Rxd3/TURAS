"""Matched XLS-R CTC fine-tuning; never replace or resize the pretrained head."""
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


def build_model(config: dict):
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from turkish_asr_rdl.training.acoustic import read_jsonl
    parent = config["model"]
    if not re.fullmatch(r"[a-f0-9]{40}", parent["revision"]) or config["mode"] != "ctc_finetune":
        raise ValueError("an immutable CTC ASR parent and ctc_finetune mode are required")
    processor = Wav2Vec2Processor.from_pretrained(parent["id"], revision=parent["revision"])
    audit = audit_targets(read_jsonl(Path(config["inputs"]["references"])), processor.tokenizer)
    if audit["affected_utterance_ids"]:
        raise ValueError("CTC target audit failed before weight download: " + str(audit["unsupported_characters"]))
    weight = Path(hf_hub_download(parent["id"], parent["weight_file"], revision=parent["revision"]))
    if sha256_file(weight) != parent["weight_sha256"]:
        raise ValueError("CTC parent weight checksum mismatch")
    model = Wav2Vec2ForCTC.from_pretrained(parent["id"], revision=parent["revision"])
    if model.config.vocab_size != len(processor.tokenizer) or model.config.pad_token_id != processor.tokenizer.pad_token_id:
        raise ValueError("CTC head/vocabulary/blank mismatch")
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
