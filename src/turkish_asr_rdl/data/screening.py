"""Text screening for TURAS curation: transcript issue codes and lexicon matching.

``screening_lower`` exists only to match lexicon aliases while choosing and
labelling segments. It is NOT the evaluation normalizer: WER/CER and terminology
scores use ``turkish_asr_rdl.evaluation`` (M6) on the frozen references.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_UPPER = "A-ZÇĞİÖŞÜÂÎÛ"

# Non-spoken material that the TBMM stenographic minutes insert into
# WorldSpeech `human_transcript`. Each pattern yields one issue code.
ISSUE_PATTERNS: dict[str, re.Pattern[str]] = {
    # "BAŞKAN - ", "ÖZGÜR ÖZEL (Manisa) - ", "AK PARTİ GRUP BAŞKANVEKİLİ ... – "
    "speaker_header": re.compile(
        rf"(?:^|\s)[{_UPPER}]{{2,}}(?:\s+[{_UPPER}]{{2,}})*\s*(?:\([^)]*\))?\s*[-–—]\s"
    ),
    # "(Devamla)", "(CHP sıralarından alkışlar)", "(İstanbul)", "(1)"
    "parenthetical": re.compile(r"\([^)]*\)"),
    "bracketed": re.compile(r"\[[^\]]*\]"),
    # Speech cut by the segment boundary or by the stenographer: "... Ersever..."
    "trailing_ellipsis": re.compile(r"(?:\.\.\.|…)\s*$"),
}


def transcript_issues(text: str) -> list[str]:
    """Return sorted issue codes for a raw upstream transcript (empty list = clean)."""
    if not text or not text.strip():
        return ["empty"]
    return sorted(code for code, pat in ISSUE_PATTERNS.items() if pat.search(text))


def screening_lower(text: str) -> str:
    """NFC, Turkish dotted/dotless I mapping, then lowercase (screening only)."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("I", "ı").replace("İ", "i")
    return unicodedata.normalize("NFC", text.lower())


_TOKEN = re.compile(r"[^\W_]+")


def screening_tokens(text: str) -> list[str]:
    """Letter/digit runs of the screening-lowercased text.

    Apostrophes and hyphens split tokens, so ``Merkez Bankası'nın`` gives
    ``merkez``, ``bankası``, ``nın`` and the alias ``merkez bankas*`` matches.
    """
    return _TOKEN.findall(screening_lower(text))


@dataclass(frozen=True)
class _Alias:
    term_id: str
    tokens: tuple[str, ...]
    last_prefix: bool
    exclude: tuple[str, ...]

    def matches_at(self, toks: list[str], i: int) -> bool:
        n = len(self.tokens)
        if i + n > len(toks):
            return False
        for k in range(n - 1):
            if toks[i + k] != self.tokens[k]:
                return False
        last, want = toks[i + n - 1], self.tokens[-1]
        if not self.last_prefix:
            return last == want
        if not last.startswith(want):
            return False
        return not any(last.startswith(x) for x in self.exclude)


@dataclass(frozen=True)
class TermMatch:
    term_id: str
    start: int
    end: int  # exclusive token index


class Lexicon:
    """Loaded ``turas-domain-lexicon`` JSON with longest-span matching."""

    def __init__(self, data: dict):
        self.version: str = data["lexicon_version"]
        self.topics: dict[str, dict] = {t["topic_id"]: t for t in data["topics"]}
        self.terms: dict[str, dict] = {}
        aliases: list[_Alias] = []
        for term in data["terms"]:
            tid = term["term_id"]
            if tid in self.terms:
                raise ValueError(f"duplicate term_id {tid}")
            if term["topic_id"] not in self.topics:
                raise ValueError(f"{tid}: unknown topic {term['topic_id']}")
            self.terms[tid] = term
            exclude = tuple(term.get("exclude", ()))
            for raw in term["aliases"]:
                toks = raw.split()
                prefix = toks[-1].endswith("*")
                if prefix:
                    toks[-1] = toks[-1][:-1]
                if any(not t or t != screening_lower(t) for t in toks):
                    raise ValueError(f"{tid}: alias {raw!r} is not screening-lowercase")
                aliases.append(_Alias(tid, tuple(toks), prefix, exclude))
        # Multi-token aliases (and exact single tokens) are indexed by their
        # exact first token; single-token prefix aliases by their stem.
        self._by_first: dict[str, list[_Alias]] = {}
        self._by_stem: dict[str, list[_Alias]] = {}
        for a in aliases:
            if len(a.tokens) == 1 and a.last_prefix:
                self._by_stem.setdefault(a.tokens[0], []).append(a)
            else:
                self._by_first.setdefault(a.tokens[0], []).append(a)
        units = data.get("quantity_units", {})
        self.scale_words = frozenset(units.get("scale", ()))
        self.currency_terms = frozenset(units.get("currency_terms", ()))

    @classmethod
    def load(cls, path: str | Path) -> "Lexicon":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def topic_of(self, term_id: str) -> str:
        return self.terms[term_id]["topic_id"]

    def _candidates(self, toks: list[str], i: int) -> Iterable[_Alias]:
        yield from self._by_first.get(toks[i], ())
        tok = toks[i]
        for n in range(1, len(tok) + 1):
            yield from self._by_stem.get(tok[:n], ())

    def match(self, toks: list[str]) -> list[TermMatch]:
        """Longest token spans, left to right, without overlap.

        Equal-length candidates at one position are broken by term_id.
        """
        out: list[TermMatch] = []
        i = 0
        while i < len(toks):
            best: _Alias | None = None
            for a in self._candidates(toks, i):
                if not a.matches_at(toks, i):
                    continue
                if best is None or (len(a.tokens), best.term_id) > (len(best.tokens), a.term_id):
                    best = a
            if best is None:
                i += 1
                continue
            out.append(TermMatch(best.term_id, i, i + len(best.tokens)))
            i += len(best.tokens)
        return out


_NUMBER = re.compile(r"^\d+$")


_PERCENT_SIGN = re.compile(r"%\s?\d|\d\s?%")


def count_quantities(
    text: str, toks: list[str], matches: Iterable[TermMatch], lexicon: Lexicon
) -> dict[str, int]:
    """Count amount and percentage expressions in one transcript.

    amount: a digit token, optionally followed by more digit/scale tokens, then
    a currency term (``5 milyar lira``, ``100 dolar``). percentage: ``yüzde``
    followed by any token, or a ``%`` sign next to a digit in the raw text.
    Counts are curation statistics, not gold quantity annotations.
    """
    currency_starts = {m.start for m in matches if m.term_id in lexicon.currency_terms}
    amount = 0
    percentage = len(_PERCENT_SIGN.findall(text))
    for i, t in enumerate(toks):
        if _NUMBER.match(t):
            j = i + 1
            while j < len(toks) and (toks[j] in lexicon.scale_words or _NUMBER.match(toks[j])):
                j += 1
            if j in currency_starts:
                amount += 1
        elif t == "yüzde" and i + 1 < len(toks):
            percentage += 1
    return {"amount": amount, "percentage": percentage}
