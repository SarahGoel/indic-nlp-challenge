"""Conservative language identification for Hindi and Sanskrit text.

This module uses explicit lexical evidence and Unicode script inspection.
It is not a trained statistical language detector. Since Hindi and Sanskrit
share Devanagari script and vocabulary, uncertain cases are reported as
unknown instead of being assigned a language without sufficient evidence.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

LanguageCode = Literal["hin", "san", "mixed", "unknown"]

# These are deliberately small, conservative lexical indicators.
# Expand them only after reviewing representative manuscript examples.
_HINDI_MARKERS = frozenset(
    {
        "और", "लेकिन", "क्योंकि", "इसलिए", "है", "हैं", "था", "थे",
        "थी", "को", "से", "में", "पर", "यह", "वह", "जो", "कि",
        "करता", "करते", "किया", "होगा", "होगी", "रहेगा", "रहेगी",
    }
)

_SANSKRIT_MARKERS = frozenset(
    {
        "नमः", "अथ", "इति", "तथा", "एव", "च", "वा", "यः", "यत्",
        "भवति", "भवन्ति", "उच्यते", "सर्वम्", "फलम्", "योगः",
        "ग्रहाः", "राशिः", "लग्नम्",
    }
)

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
# Match a Unicode letter followed by letters or combining marks.
# Devanagari vowel signs and virama must remain attached to their word.
_WORD_RE = re.compile(
    r"[^\W\d_](?:[^\W\d_]|[\u0300-\u036F]|[\u0900-\u0903]|"
    r"[\u093A-\u094F]|[\u0951-\u0957]|[\u0962-\u0963])*",
    flags=re.UNICODE,
)

@dataclass(frozen=True, slots=True)
class LanguageIdentification:
    """Result of a conservative language-identification operation."""

    language: LanguageCode
    confidence: float
    hindi_markers: tuple[str, ...]
    sanskrit_markers: tuple[str, ...]
    reason: str


def validate_language_code(language: str) -> bool:
    """Return whether a value is a supported corpus language label."""
    return language in {"hin", "san", "mixed", "unknown"}


def identify_language(text: str) -> LanguageIdentification:
    """Estimate whether a text segment is Hindi, Sanskrit, mixed, or unknown.

    The result is based on a small set of explicit lexical markers. The
    confidence is a heuristic score, not a calibrated probability.

    Args:
        text: Text segment to inspect. The input is never modified.

    Returns:
        A structured result with the label, heuristic confidence, matched
        markers, and explanation.

    Raises:
        TypeError: If text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text.strip():
        return LanguageIdentification(
            language="unknown",
            confidence=0.0,
            hindi_markers=(),
            sanskrit_markers=(),
            reason="Input is empty or contains only whitespace.",
        )

    normalized = unicodedata.normalize("NFC", text)
    if not _DEVANAGARI_RE.search(normalized):
        return LanguageIdentification(
            language="unknown",
            confidence=0.0,
            hindi_markers=(),
            sanskrit_markers=(),
            reason="No Devanagari characters were detected.",
        )

    words = tuple(word for word in _WORD_RE.findall(normalized))
    word_set = set(words)

    hindi_hits = tuple(sorted(word_set & _HINDI_MARKERS))
    sanskrit_hits = tuple(sorted(word_set & _SANSKRIT_MARKERS))

    if hindi_hits and sanskrit_hits:
        return LanguageIdentification(
            language="mixed",
            confidence=0.6,
            hindi_markers=hindi_hits,
            sanskrit_markers=sanskrit_hits,
            reason=(
                "Markers from both lexical lists were found. "
                "Manual review may be needed to distinguish mixed text "
                "from shared vocabulary or quoted material."
            ),
        )

    if hindi_hits:
        confidence = min(0.85, 0.55 + 0.1 * len(hindi_hits))
        return LanguageIdentification(
            language="hin",
            confidence=confidence,
            hindi_markers=hindi_hits,
            sanskrit_markers=(),
            reason="One or more configured Hindi markers were found.",
        )

    if sanskrit_hits:
        confidence = min(0.85, 0.55 + 0.1 * len(sanskrit_hits))
        return LanguageIdentification(
            language="san",
            confidence=confidence,
            hindi_markers=(),
            sanskrit_markers=sanskrit_hits,
            reason="One or more configured Sanskrit markers were found.",
        )

    return LanguageIdentification(
        language="unknown",
        confidence=0.0,
        hindi_markers=(),
        sanskrit_markers=(),
        reason=(
            "No configured language markers were found. "
            "Devanagari script alone cannot distinguish Hindi from Sanskrit."
        ),
    )
