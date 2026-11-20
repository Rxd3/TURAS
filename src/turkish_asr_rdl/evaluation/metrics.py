"""Per-utterance scores, corpus WER/CER/WAcc (sum over sum) and RTF/RTFx."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from turkish_asr_rdl.evaluation.align import EditCounts, align_counts
from turkish_asr_rdl.evaluation.normalize import normalize, word_tokens


@dataclass(frozen=True)
class UtteranceScore:
    utterance_id: str
    view: str
    status: str
    words: EditCounts
    chars: EditCounts


@dataclass(frozen=True)
class CorpusRates:
    view: str
    utterances: int
    zero_reference_utterances: int
    error_rows: int
    words: EditCounts
    chars: EditCounts
    wer: float | None
    cer: float | None
    wacc: float | None


def score_pair(
    utterance_id: str, reference: str, hypothesis: str, view: str, status: str = "ok"
) -> UtteranceScore:
    """Align one pair in `view`: word tokens for WER, code points for CER."""
    ref = normalize(reference, view)
    hyp = normalize(hypothesis, view)
    return UtteranceScore(
        utterance_id=utterance_id,
        view=view,
        status=status,
        words=align_counts(word_tokens(ref), word_tokens(hyp)),
        chars=align_counts(ref, hyp),
    )


def _rate(counts: EditCounts) -> float | None:
    return counts.errors / counts.ref_len if counts.ref_len else None


def corpus_rates(scores: Sequence[UtteranceScore]) -> CorpusRates:
    """Sum S/D/I/N over utterances of one view; a rate is None when its N is 0.

    An utterance whose reference has no words in this view counts as a
    zero-reference utterance: its insertions add to the errors, not to N.
    """
    if not scores:
        raise ValueError("no utterance scores to aggregate")
    views = sorted({score.view for score in scores})
    if len(views) != 1:
        raise ValueError(f"scores must share one normalization view, got {views}")
    words = sum((score.words for score in scores), EditCounts())
    chars = sum((score.chars for score in scores), EditCounts())
    wer = _rate(words)
    return CorpusRates(
        view=views[0],
        utterances=len(scores),
        zero_reference_utterances=sum(1 for s in scores if s.words.ref_len == 0),
        error_rows=sum(1 for s in scores if s.status == "error"),
        words=words,
        chars=chars,
        wer=wer,
        cer=_rate(chars),
        wacc=None if wer is None else 1 - wer,
    )


def _check_durations(processing_s: float, audio_s: float) -> None:
    if not (math.isfinite(audio_s) and audio_s > 0):
        raise ValueError(f"audio_s must be a finite number above 0, got {audio_s!r}")
    if not (math.isfinite(processing_s) and processing_s >= 0):
        raise ValueError(f"processing_s must be a finite number of 0 or more, got {processing_s!r}")


def rtf(processing_s: float, audio_s: float) -> float:
    """Real-time factor: processing seconds per audio second (below 1 is faster than real time)."""
    _check_durations(processing_s, audio_s)
    return processing_s / audio_s


def rtfx(processing_s: float, audio_s: float) -> float:
    """Inverse real-time factor (1 / RTF): audio seconds per processing second."""
    _check_durations(processing_s, audio_s)
    if processing_s == 0:
        raise ValueError("processing_s must be above 0 for RTFx")
    return audio_s / processing_s
