"""Conservative Unicode normalization for Indic manuscript text."""

from __future__ import annotations

import re
import unicodedata


_HORIZONTAL_WHITESPACE = re.compile(r"[ \t\f\v]+")


def normalize_text(
    text: str,
    *,
    normalize_whitespace: bool = True,
    remove_nul: bool = True,
) -> str:
    """Normalize Unicode text without rewriting legitimate manuscript content.

    Processing order:
    1. Apply Unicode NFC normalization.
    2. Optionally remove NUL characters.
    3. Optionally normalize horizontal whitespace and line endings.

    Indic characters, matras, viramas, nukta characters, punctuation,
    numerals, and internal blank lines are otherwise preserved.

    Args:
        text: Original OCR text or a human-verified transcription.
        normalize_whitespace: Normalize horizontal whitespace and line endings.
        remove_nul: Remove U+0000 NUL characters.

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

    if normalize_whitespace:
        normalized = normalized.replace("\r\n", "\n")
        normalized = normalized.replace("\r", "\n")

        lines = [
            _HORIZONTAL_WHITESPACE.sub(" ", line).strip(" ")
            for line in normalized.split("\n")
        ]

        normalized = "\n".join(lines).strip("\n")

    return normalized