"""Conservative Unicode normalization for Indic manuscript text."""

from __future__ import annotations

import re
import unicodedata


_HORIZONTAL_WHITESPACE = re.compile(r"[ \t\f\v]+")

# Danda recovery patterns
_SINGLE_DANDA = re.compile(r"(?<!\|)\|(?!\|)")  # Single | not followed by another |
_DOUBLE_DANDA = re.compile(r"\|\|")  # Double ||


def _recover_danda(text: str) -> str:
    """Recover Devanagari danda punctuation from ASCII bar substitutions.

    Maps:
    - Single | (pipe) to Devanagari danda (।)
    - Double || to Devanagari double danda (॥)

    Handles edge cases:
    - Only converts standalone bars, not bars within other contexts
    - Preserves spacing around punctuation
    - Conservative approach to avoid false positives

    Args:
        text: Text containing ASCII bar substitutions for danda.

    Returns:
        Text with danda punctuation restored to Devanagari.
    """
    # Replace double danda first (more specific pattern)
    text = _DOUBLE_DANDA.sub("॥", text)

    # Replace single danda (excluding those already converted to double)
    text = _SINGLE_DANDA.sub("।", text)

    return text


def normalize_text(
    text: str,
    *,
    normalize_whitespace: bool = True,
    remove_nul: bool = True,
    recover_danda: bool = False,
) -> str:
    """Normalize Unicode text without rewriting legitimate manuscript content.

    Processing order:
    1. Apply Unicode NFC normalization.
    2. Optionally remove NUL characters.
    3. Optionally recover danda punctuation from ASCII bars.
    4. Optionally normalize horizontal whitespace and line endings.

    Indic characters, matras, viramas, nukta characters, punctuation,
    numerals, and internal blank lines are otherwise preserved.

    Args:
        text: Original OCR text or a human-verified transcription.
        normalize_whitespace: Normalize horizontal whitespace and line endings.
        remove_nul: Remove U+0000 NUL characters.
        recover_danda: Recover Devanagari danda (।, ॥) from ASCII bars (|, ||).

    Returns:
        Normalized text.

    Raises:
        TypeError: If text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFC", text)

    if remove_nul:
        normalized = normalized.replace("\x00", "")

    if recover_danda:
        normalized = _recover_danda(normalized)

    if normalize_whitespace:
        normalized = normalized.replace("\r\n", "\n")
        normalized = normalized.replace("\r", "\n")

        lines = [
            _HORIZONTAL_WHITESPACE.sub(" ", line).strip(" ")
            for line in normalized.split("\n")
        ]

        normalized = "\n".join(lines).strip("\n")

    return normalized