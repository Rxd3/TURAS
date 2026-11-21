# Turkish ASR support audit for HuBERT and transducers

Reviewed 2026-11-21 for the course baseline-family coverage noted in the
[owner decisions](owner_decisions.md#things-approval-cannot-replace). This is
a review of model cards, architecture and published training provenance.
The revisions in the source links identify the checked files. No weights
were downloaded and no candidate was run on TURAS.

## HuBERT

| Candidate | Published evidence and decision |
| --- | --- |
| `facebook/hubert-base-ls960` | Meta's release is a feature extractor with no trained transcription head in the published checkpoint. Exclude it as an out-of-the-box ASR baseline. [Pinned card](https://huggingface.co/facebook/hubert-base-ls960/blob/dba3bb02fda4248b6e082697eee756de8fe8aa8a/README.md) |
| `facebook/hubert-large-ls960-ft` | The CTC model is trained for English LibriSpeech. Its released vocabulary lacks Turkish characters such as `Ç`, `Ğ`, `İ`, `Ö`, `Ş` and `Ü`. Exclude it from the Turkish ASR comparison. [Pinned card](https://huggingface.co/facebook/hubert-large-ls960-ft/blob/ece5fabbf034c1073acae96d5401b25be96709d8/README.md) [Pinned vocabulary](https://huggingface.co/facebook/hubert-large-ls960-ft/blob/ece5fabbf034c1073acae96d5401b25be96709d8/vocab.json) |
| `utter-project/mHuBERT-147` | The multilingual encoder's reported pretraining data include Turkish MediaSpeech. The release provides representations and has no trained Turkish transcription head. It is a candidate encoder for a separate experiment using a Turkish CTC head trained on the frozen train split. The card lists CC BY-NC-SA 4.0. [Pinned card](https://huggingface.co/utter-project/mHuBERT-147/blob/7ad3fc0bc5106c58c9c13526abccad527150d135/README.md) [Publisher paper](https://arxiv.org/abs/2406.06371) |

None of the checked HuBERT checkpoints supplies a Turkish-ready transcription
head. A Turkish head trained for a future comparison would need its own
checkpoint, vocabulary, data provenance and validation selection.

## Transducers

| Candidate | Published evidence and decision |
| --- | --- |
| `nvidia/parakeet-tdt-0.6b-v3` | NVIDIA's Token-and-Duration Transducer supports 25 named languages. The supported-language list omits Turkish. Exclude this revision from Turkish ASR. [Pinned card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/blob/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/README.md) |
| `nvidia/nemotron-3.5-asr-streaming-0.6b` | The published cache-aware FastConformer RNN-T card explicitly lists Turkish (`tr-TR`) as transcription-ready and reports Turkish FLEURS results. This is a Turkish-capable candidate. The card names proprietary and public training sources without a complete clip-level manifest. Its overlap with TURAS is unresolved. The card lists OpenMDW 1.1. [Pinned card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/README.md) |
| `atasoglu/seda-v0.1` | A Turkish Zipformer2 RNN-T with an ONNX encoder, decoder, joiner and 500-token vocabulary. The publisher lists Common Voice 27 Turkish and ISSAI Turkish Speech Corpus as its acoustic training sources and lists Apache-2.0 for the acoustic files. WorldSpeech does not appear in that list. Prefer this candidate for a follow-up when those source manifests can be checked against the TURAS test recordings. Its optional language model and formatting can affect results and require explicit configuration. [Pinned card](https://huggingface.co/atasoglu/seda-v0.1/blob/5cbbda1b34bde5e3004e79b137dc3871587ea282/README.md) |
| `duxx/turkish-stt-zipformer` | A Turkish Zipformer RNN-Transducer with released FP32 and INT8 ONNX weights. It explicitly trained on Turkish WorldSpeech, the upstream source of TURAS. Exclude from the primary comparison unless its training manifest demonstrates that TURAS test recordings were held out. The card lists CC BY-NC 4.0. [Pinned card](https://huggingface.co/duxx/turkish-stt-zipformer/blob/cebec199e0a9a1bdcd195b9124922b4a33e98bc8/README.md) |

## Benchmark status

A later transducer run should verify the pinned checkpoint and tokenizer,
document training-data overlap where it can be checked, and freeze the language
prompt or language-selection behavior, decoding, precision, optional language
model and text-emission policy using validation only. Evaluate on the frozen
636 IDs with the shared scorer. Report known or unresolved overlap and measure
runtime using the shared protocol. [Dataset handoff](m1_deadline_handoff.md)
[Evaluation policy](evaluation.md)

This audit documents the supported and excluded candidates. A matched transducer
run and the separate GMM-HMM baseline remain outstanding.
