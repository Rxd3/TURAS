"""Matched XLS-R CTC fine-tuning; never replace or resize the pretrained head."""
from __future__ import annotations

import re
import unicodedata

from turkish_asr_rdl.training.acoustic import load_audio, sha256_file, training_main


def label_text(text: str) -> str:
    """Training-label spelling view only; evaluation references remain unchanged."""
    text = unicodedata.normalize("NFC", text).replace("I", "ı").replace("İ", "i").lower()
    text = "".join(" " if unicodedata.category(c).startswith("P") and c != "'" else c for c in text)
    return " ".join(text.split())


def build_model(config: dict):
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    parent = config["model"]
    if not re.fullmatch(r"[a-f0-9]{40}", parent["revision"]) or config["mode"] != "ctc_finetune":
        raise ValueError("an immutable CTC ASR parent and ctc_finetune mode are required")
    weight = Path(hf_hub_download(parent["id"], parent["weight_file"], revision=parent["revision"]))
    if sha256_file(weight) != parent["weight_sha256"]:
        raise ValueError("CTC parent weight checksum mismatch")
    processor = Wav2Vec2Processor.from_pretrained(parent["id"], revision=parent["revision"])
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
            raise ValueError("CTC labels contain empty/OOV/blank targets; M3/M6 must approve a vocabulary/number policy")
        padded = processor.tokenizer.pad({"input_ids": labels}, padding=True, return_tensors="pt")
        batch["labels"] = padded.input_ids.masked_fill(padded.attention_mask.ne(1), -100)
        return batch
    return collate


def main():
    training_main(build_model, build_collator)


if __name__ == "__main__":
    main()
