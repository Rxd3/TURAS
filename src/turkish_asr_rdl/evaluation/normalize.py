"""Versioned text views for evaluation: `raw-v1` and `tr-normalized-v1`.

`tr-normalized-v1` follows the normalization contract 0.2 rules, in this order:
NFC; Turkish-aware lowercase; delete format characters (Cf); delete an
apostrophe (U+0027, U+2019, U+02BC) between two letters, otherwise make it a
space; every other punctuation character except `%` becomes a space; collapse
`str.isspace()` runs to one ASCII space and strip both ends; NFC again, so a
pair left next to each other by a deletion is composed and normalizing twice
gives the same text as normalizing once.
"""

import unicodedata

RAW_V1 = "raw-v1"
TR_NORMALIZED_V1 = "tr-normalized-v1"

_APOSTROPHES = frozenset("'\u2019\u02bc")
_KEPT_PUNCTUATION = frozenset("%")


def _is_letter(char: str) -> bool:
    return unicodedata.category(char).startswith("L")


def _between_letters(text: str, index: int) -> bool:
    if index == 0 or index == len(text) - 1:
        return False
    return _is_letter(text[index - 1]) and _is_letter(text[index + 1])


def _map_char(text: str, index: int) -> str:
    char = text[index]
    if char in _APOSTROPHES:
        return "" if _between_letters(text, index) else " "
    if char not in _KEPT_PUNCTUATION and unicodedata.category(char).startswith("P"):
        return " "
    return char


def _tr_normalized_v1(text: str) -> str:
    composed = unicodedata.normalize("NFC", text)
    lowered = composed.replace("I", "ı").replace("İ", "i").lower()
    visible = "".join(c for c in lowered if unicodedata.category(c) != "Cf")
    spaced = "".join(_map_char(visible, i) for i in range(len(visible)))
    return unicodedata.normalize("NFC", " ".join(spaced.split()))


def normalize(text: str, version: str) -> str:
    """Return the `version` view of `text`; raises ValueError for an unknown version."""
    if version == RAW_V1:
        return text
    if version == TR_NORMALIZED_V1:
        return _tr_normalized_v1(text)
    raise ValueError(
        f"unknown normalization version {version!r}; use {RAW_V1!r} or {TR_NORMALIZED_V1!r}"
    )


def word_tokens(text: str) -> list[str]:
    """Split on runs of Unicode whitespace (`str.isspace()`)."""
    return text.split()
