"""Smoke-test adapters for the 5070 Ti: Whisper, Qwen3-ASR and XLS-R behind the shared AsrAdapter interface.

torch, transformers and huggingface_hub are imported inside the functions that use them,
so this module imports on a machine that has none of them.
"""
from __future__ import annotations

import copy
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from smoke_common import (
    DEVICE,
    BeamAdapter,
    ModelFiles,
    SmokeAdapter,
    beam_hypothesis,
    collapse_ctc,
    cuda_torch,
    make_hypothesis,
    one_line,
    weights_sha256,
)
from turkish_asr_rdl.decoding.audio import TARGET_SAMPLE_RATE_HZ
from turkish_asr_rdl.decoding.interface import AudioItem, Candidate, Hypothesis

# name -> (Hugging Face repo, pinned commit, weights file pattern)
MODEL_REPOS = {
    "whisper-small": ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d", "*.safetensors"),
    "whisper-large-v3": ("openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1", "*.safetensors"),
    "qwen3-asr-1.7b": ("Qwen/Qwen3-ASR-1.7B-hf", "bcd2b5b7f32b480ab5790554cfa8347f246a14f3", "*.safetensors"),
    "qwen3-asr-0.6b": ("Qwen/Qwen3-ASR-0.6B-hf", "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c", "*.safetensors"),
    "xlsr-tr-cv7": ("mpoyraz/wav2vec2-xls-r-300m-cv7-turkish", "708639f50559d7970f462e13ec64d3f059ca89f6", "*.bin"),
    "xlsr-tr-cv8": ("mpoyraz/wav2vec2-xls-r-300m-cv8-turkish", "bdd0bb878d8bf59509611631e334eb073ba57773", "*.bin"),
}
SKIPPED_FILES = ("*.msgpack", "*.h5", "*fp32*", "language_model/*", "eval.py")
WHISPER_PROMPT_TOKENS = ("<|startoftranscript|>", "<|tr|>", "<|transcribe|>", "<|notimestamps|>")
MAX_NEW_TOKENS = 440


def fetch_model(name: str) -> ModelFiles:
    """Download the pinned snapshot (outside any timing) and hash its weights."""
    from huggingface_hub import snapshot_download

    repo_id, revision, weights = MODEL_REPOS[name]
    ignore = SKIPPED_FILES if weights == "*.bin" else SKIPPED_FILES + ("*.bin",)
    path = Path(snapshot_download(repo_id, revision=revision, ignore_patterns=list(ignore)))
    return ModelFiles(name, repo_id, revision, path, weights_sha256(path, weights))


def transformers_classes(names: Sequence[str]) -> dict[str, Any]:
    """Import the named transformers classes now, so the timed load() does not import them itself."""
    import transformers

    return {name: getattr(transformers, name) for name in names}


def _score_rows(model: Any, out: Any, beams: int) -> list[list[float]]:
    """Per-step log-probs of the chosen tokens; greedy scores are raw logits, so they are normalized."""
    steps = model.compute_transition_scores(out.sequences, out.scores, getattr(out, "beam_indices", None),
                                            normalize_logits=beams == 1)
    return steps.float().tolist()


def _beam_scores(out: Any, beams: int, count: int) -> list[float | None]:
    return out.sequences_scores.float().tolist() if beams > 1 else [None] * count


class WhisperAdapter(BeamAdapter):
    """N-best through core GenerationMixin.generate with a fixed Turkish transcribe prompt."""

    requested_n = 10
    score_convention_id = "smoke-whisper-token-logprob-sum-v1"
    KIND = "seq2seq_token_log_prob_sum"
    TRANSFORMERS_CLASSES = ("WhisperForConditionalGeneration", "WhisperProcessor")

    def __init__(self, files: ModelFiles, classes: Mapping[str, Any]):
        super().__init__(files, classes, {
            "family": "whisper", "precision": "float16", "language": "tr", "task": "transcribe",
            "timestamps": False, "num_beams": 10, "num_return_sequences": 10, "max_new_tokens": MAX_NEW_TOKENS,
            "length_penalty": 1.0, "generate": "transformers.GenerationMixin.generate",
        })

    def load(self) -> None:
        torch = cuda_torch()
        self._processor = self._classes["WhisperProcessor"].from_pretrained(self.files.path)
        model = self._classes["WhisperForConditionalGeneration"].from_pretrained(self.files.path, dtype=torch.float16)
        self._model = model.to(DEVICE).eval()
        tokenizer = self._processor.tokenizer
        self._prompt = tuple(tokenizer.convert_tokens_to_ids(list(WHISPER_PROMPT_TOKENS)))
        self._stop = frozenset({tokenizer.eos_token_id})
        self._generation_config = copy.deepcopy(model.generation_config)
        if getattr(self._generation_config, "forced_decoder_ids", None) is not None:
            self._generation_config.forced_decoder_ids = None  # the prompt is passed as decoder_input_ids

    def _features(self, samples: Any) -> Any:
        import torch

        features = self._processor.feature_extractor(samples, sampling_rate=TARGET_SAMPLE_RATE_HZ,
                                                     return_tensors="pt").input_features
        return features.to(DEVICE, torch.float16)

    def _generate(self, item: AudioItem, beams: int) -> Hypothesis:
        import torch
        import transformers

        prompt = torch.tensor([self._prompt], device=DEVICE)
        with torch.inference_mode():
            out = transformers.GenerationMixin.generate(
                self._model, input_features=self._features(item.samples), decoder_input_ids=prompt,
                generation_config=self._generation_config, num_beams=beams, num_return_sequences=beams,
                do_sample=False, output_scores=True, return_dict_in_generate=True, max_new_tokens=MAX_NEW_TOKENS)
            steps = _score_rows(self._model, out, beams)
        generated = out.sequences[:, len(self._prompt):]
        texts = self._processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
        details = {
            "score_source": "transformers compute_transition_scores, summed over generated tokens through end-of-text",
            "log_base": "e", "includes_eos": True, "prompt_excluded": list(WHISPER_PROMPT_TOKENS),
            "temperature": 1.0, "nbest_algorithm": "beam search" if beams > 1 else "greedy", "num_beams": beams,
            "decoder_sequence_score_formula": "transformers beam score: sum of log-probs / length ** 1.0",
            "library": f"transformers {transformers.__version__}",
        }
        return beam_hypothesis(item.utterance_id, texts, generated.tolist(), steps,
                               _beam_scores(out, beams, len(texts)), kind=self.KIND, details=details,
                               stop_ids=self._stop)

    def beam(self, item: AudioItem) -> Hypothesis:
        return self._generate(item, self.config["num_beams"])

    def greedy(self, item: AudioItem) -> Hypothesis:
        return self._generate(item, 1)

    def extra_facts(self, items: Sequence[AudioItem]) -> dict[str, Any]:
        return {**super().extra_facts(items), "whisper_generate_trap": self._trap(items)}

    def _trap(self, items: Sequence[AudioItem]) -> dict[str, Any]:
        """Duplicates from WhisperForConditionalGeneration.generate with the same beam settings."""
        try:
            counts = [self._trap_one(item) for item in items]
        except Exception as exc:  # the trap is a side measurement; its failure is recorded, not raised
            return {"error": one_line(exc), "traceback": traceback.format_exc()}
        return {
            "utterances": len(counts),
            "returned_per_utterance": sorted({returned for returned, _ in counts}),
            "duplicate_texts": sum(duplicates for _, duplicates in counts),
            "utterances_with_duplicates": sum(1 for _, duplicates in counts if duplicates),
        }

    def _trap_one(self, item: AudioItem) -> tuple[int, int]:
        import torch

        with torch.inference_mode():
            out = self._model.generate(
                input_features=self._features(item.samples), language="tr", task="transcribe", num_beams=10,
                num_return_sequences=10, do_sample=False, output_scores=True, return_dict_in_generate=True,
                max_new_tokens=MAX_NEW_TOKENS)
        sequences = out["sequences"] if isinstance(out, Mapping) else out
        texts = self._processor.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        return len(texts), len(texts) - len(set(texts))


class QwenAsrAdapter(BeamAdapter):
    """Qwen3-ASR -hf through transformers' native classes, language forced to Turkish."""

    requested_n = 5
    score_convention_id = "smoke-qwen3-asr-joint-token-logprob-sum-v1"
    KIND = "speech_llm_joint_token_log_prob_sum"
    TRANSFORMERS_CLASSES = ("AutoModelForMultimodalLM", "AutoProcessor")

    def __init__(self, files: ModelFiles, classes: Mapping[str, Any]):
        super().__init__(files, classes, {
            "family": "qwen3-asr", "precision": "bfloat16", "language": "tr", "num_beams": 5,
            "num_return_sequences": 5, "max_new_tokens": MAX_NEW_TOKENS, "model_class": "AutoModelForMultimodalLM",
            "input": "AutoProcessor.apply_transcription_request",
        })

    def load(self) -> None:
        torch = cuda_torch()
        self._processor = self._classes["AutoProcessor"].from_pretrained(self.files.path)
        model = self._classes["AutoModelForMultimodalLM"].from_pretrained(self.files.path, dtype=torch.bfloat16)
        self._model = model.to(DEVICE).eval()
        eos = self._model.generation_config.eos_token_id
        self._stop = frozenset(eos if isinstance(eos, (list, tuple)) else [eos])

    def _generate(self, item: AudioItem, beams: int) -> Hypothesis:
        import torch
        import transformers

        inputs = self._processor.apply_transcription_request(audio=[item.samples], language="tr")
        inputs = inputs.to(self._model.device, self._model.dtype)
        prompt_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = self._model.generate(**inputs, num_beams=beams, num_return_sequences=beams, do_sample=False,
                                       output_scores=True, return_dict_in_generate=True,
                                       max_new_tokens=MAX_NEW_TOKENS)
            steps = _score_rows(self._model, out, beams)
        generated = out.sequences[:, prompt_length:]
        texts = self._processor.decode(generated, return_format="transcription_only")
        details = {
            "score_source": "transformers compute_transition_scores, summed over generated tokens through end-of-turn",
            "joint": "speech-LLM score; no separate acoustic-only component exists",
            "log_base": "e", "includes_eos": True, "prompt_excluded": "chat prompt with forced 'language Turkish'",
            "nbest_algorithm": "beam search" if beams > 1 else "greedy", "num_beams": beams,
            "library": f"transformers {transformers.__version__}",
        }
        return beam_hypothesis(item.utterance_id, list(texts), generated.tolist(), steps,
                               _beam_scores(out, beams, len(texts)), kind=self.KIND, details=details,
                               stop_ids=self._stop)

    def beam(self, item: AudioItem) -> Hypothesis:
        return self._generate(item, self.config["num_beams"])

    def greedy(self, item: AudioItem) -> Hypothesis:
        return self._generate(item, 1)


class XlsrAdapter(SmokeAdapter):
    """Greedy CTC with the exact sequence log-probability of the decoded label sequence."""

    requested_n = 1
    score_convention_id = "smoke-ctc-exact-sequence-logprob-v1"
    KIND = "ctc_sequence_log_prob_exact"
    TRANSFORMERS_CLASSES = ("Wav2Vec2CTCTokenizer", "Wav2Vec2FeatureExtractor", "Wav2Vec2ForCTC")

    def __init__(self, files: ModelFiles, classes: Mapping[str, Any]):
        super().__init__(files, classes, {
            "family": "wav2vec2-ctc", "precision": "float32", "decoding": "greedy",
            "processor": "Wav2Vec2CTCTokenizer and Wav2Vec2FeatureExtractor, no language model",
        })

    def load(self) -> None:
        torch = cuda_torch()
        self._tokenizer = self._classes["Wav2Vec2CTCTokenizer"].from_pretrained(self.files.path)
        self._processor = self._classes["Wav2Vec2FeatureExtractor"].from_pretrained(self.files.path)
        model = self._classes["Wav2Vec2ForCTC"].from_pretrained(self.files.path, dtype=torch.float32)
        self._model = model.to(DEVICE).eval()
        self._blank = self._tokenizer.pad_token_id

    def decode(self, item: AudioItem) -> Hypothesis:
        import torch

        values = self._processor(item.samples, sampling_rate=TARGET_SAMPLE_RATE_HZ, return_tensors="pt").input_values
        with torch.inference_mode():
            logits = self._model(values.to(DEVICE)).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)[0].cpu()
            labels = collapse_ctc(log_probs.argmax(dim=-1).tolist(), self._blank)
            loss = torch.nn.functional.ctc_loss(
                log_probs.unsqueeze(1), torch.tensor([labels], dtype=torch.long),
                torch.tensor([log_probs.shape[0]]), torch.tensor([len(labels)]),
                blank=self._blank, reduction="none", zero_infinity=False)
        self.decode_path = "greedy"
        candidate = Candidate(
            candidate_id=f"{item.utterance_id}-c01", rank=1,
            text=self._tokenizer.decode(list(labels), group_tokens=False), token_ids=labels,
            acoustic_log_score=-float(loss[0]), acoustic_score_kind=self.KIND,
            score_details={
                "score_source": "-torch.nn.functional.ctc_loss(reduction='none') on the greedy label sequence",
                "exact": True, "log_base": "e", "blank_id": self._blank,
                "decoding": "argmax path, repeats merged, blanks removed",
                "null_reasons": {"decoder_sequence_score": "greedy CTC decoding has no framework sequence score"},
            },
        )
        return make_hypothesis(item.utterance_id, (candidate,))


def build_adapter(name: str) -> SmokeAdapter:
    """Download the model files and import its transformers classes; return the adapter, not loaded yet.

    Both happen before run_timed, so its timed cold load leaves them out. It still covers all of
    from_pretrained (config, tokenizer and weight files) and the move to the GPU, and may include
    importing the concrete modules that an Auto class or a processor resolves there (for example
    AutoProcessor, AutoModelForMultimodalLM, or WhisperProcessor's tokenizer and feature extractor).
    """
    files = fetch_model(name)
    if name.startswith("whisper"):
        family = WhisperAdapter
    elif name.startswith("qwen3-asr"):
        family = QwenAsrAdapter
    else:
        family = XlsrAdapter
    return family(files, transformers_classes(family.TRANSFORMERS_CLASSES))
