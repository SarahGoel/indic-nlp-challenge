
"""Transliteration utilities for Devanagari and supported Indic scripts.

This module performs script/representation conversion, not translation.
The input text is never modified in place.
"""

from __future__ import annotations

import unicodedata

from indicnlp.transliterate.unicode_transliterate import (
    UnicodeIndicTransliterator,
)
from loguru import logger


# Independent Devanagari vowels.
_INDEPENDENT_VOWELS: dict[str, str] = {
    "अ": "a", "आ": "ā", "इ": "i", "ई": "ī",
    "उ": "u", "ऊ": "ū", "ऋ": "ṛ", "ॠ": "ṝ",
    "ऌ": "ḷ", "ॡ": "ḹ", "ए": "e", "ऐ": "ai",
    "ओ": "o", "औ": "au",
}

# Dependent vowel signs. An absent sign after a consonant means inherent a.
_VOWEL_SIGNS: dict[str, str] = {
    "ा": "ā", "ि": "i", "ी": "ī", "ु": "u",
    "ू": "ū", "ृ": "ṛ", "ॄ": "ṝ", "ॢ": "ḷ",
    "ॣ": "ḹ", "े": "e", "ै": "ai", "ो": "o",
    "ौ": "au",
}

# Devanagari consonants in their standard IAST representation.
_CONSONANTS: dict[str, str] = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ṅ",
    "च": "c", "छ": "ch", "ज": "j", "झ": "jh", "ञ": "ñ",
    "ट": "ṭ", "ठ": "ṭh", "ड": "ḍ", "ढ": "ḍh", "ण": "ṇ",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "ś", "ष": "ṣ", "स": "s", "ह": "h",
    "ळ": "ḷ",
}

# Consonants formed with the Devanagari nukta.
_NUKTA_CONSONANTS: dict[str, str] = {
    "क़": "q", "ख़": "ḵẖ", "ग़": "ġ", "ज़": "z",
    "ड़": "ṛ", "ढ़": "ṛh", "फ़": "f", "य़": "ẏ",
}

_SPECIALS: dict[str, str] = {
    "ं": "ṃ",       # anusvara
    "ः": "ḥ",       # visarga
    "ँ": "m̐",      # chandrabindu
    "ऽ": "’",       # avagraha
    "ॐ": "oṃ",
    "।": "।",
    "॥": "॥",
}

_DEVANAGARI_DIGITS = str.maketrans(
    "०१२३४५६७८९", "0123456789"
)

# IndicNLP's Unicode transliterator uses ISO 15919-style script codes.
# Hindi and Sanskrit both use Devanagari as their source script.
_INDIC_SCRIPT_CODES: dict[str, str] = {
    "hin": "hi",
    "san": "hi",
    "ben": "bn",
    "guj": "gu",
    "mar": "mr",
    "tam": "ta",
    "tel": "te",
    "kan": "kn",
    "mal": "ml",
    "pan": "pa",
    "asm": "as",
    "ory": "or",
}

_CONSONANT_KEYS = sorted(
    set(_CONSONANTS) | set(_NUKTA_CONSONANTS),
    key=len,
    reverse=True,
)


def _transliterate_iast(text: str) -> str:
    """Convert Devanagari text to IAST using orthographic mappings."""
    output: list[str] = []
    i = 0

    while i < len(text):
        # Check two-character nukta consonants before individual characters.
        matched = next(
            (key for key in _CONSONANT_KEYS if text.startswith(key, i)),
            None,
        )

        if matched is not None:
            output.append(
                _NUKTA_CONSONANTS.get(
                    matched, _CONSONANTS.get(matched, "")
                )
            )
            i += len(matched)

            # A consonant may be followed by a vowel sign or virama.
            if i < len(text) and text[i] in _VOWEL_SIGNS:
                output.append(_VOWEL_SIGNS[text[i]])
                i += 1
            elif i < len(text) and text[i] == "्":
                # Virama suppresses the inherent vowel.
                i += 1
            else:
                # Devanagari consonants normally carry inherent a.
                output.append("a")
            continue

        char = text[i]

        if char in _INDEPENDENT_VOWELS:
            output.append(_INDEPENDENT_VOWELS[char])
        elif char in _SPECIALS:
            output.append(_SPECIALS[char])
        elif char in _VOWEL_SIGNS:
            # An isolated vowel sign is unusual; preserve it rather than
            # silently discarding source information.
            output.append(char)
        elif char == "्":
            # A standalone virama is preserved for auditability.
            output.append(char)
        elif "०" <= char <= "९":
            output.append(char.translate(_DEVANAGARI_DIGITS))
        else:
            # Preserve spaces, punctuation, Latin text, and unknown symbols.
            output.append(char)

        i += 1

    return "".join(output)


def transliterate_devanagari_to_iast(text: str) -> str:
    """Convert Devanagari text into IAST.

    Args:
        text: Hindi or Sanskrit text written in Devanagari.

    Returns:
        IAST text, with whitespace and unrelated punctuation preserved.

    Raises:
        TypeError: If text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFC", text)
    if not normalized:
        return ""

    result = _transliterate_iast(normalized)
    logger.debug("Converted Devanagari text to IAST")
    return result


def transliterate_devanagari_to_script(
    text: str,
    target_language: str,
) -> str:
    """Convert Devanagari text to a supported Indic script.

    Args:
        text: Text written in Devanagari.
        target_language: ISO-style project code such as 'ben', 'guj',
            'tam', or 'urd'.

    Returns:
        Text transliterated into the target script.

    Raises:
        TypeError: If text or target_language is not a string.
        ValueError: If the target is not supported by this module.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(target_language, str):
        raise TypeError("target_language must be a string")

    target = target_language.strip().lower()
    normalized = unicodedata.normalize("NFC", text)

    if target not in _INDIC_SCRIPT_CODES:
        raise ValueError(
            f"Unsupported target language: {target_language!r}. "
            f"Supported codes: {', '.join(_INDIC_SCRIPT_CODES)}"
        )

    # Hindi and Sanskrit both use Devanagari; no script conversion is needed.
    if target in {"hin", "san"}:
        return normalized

    if not normalized:
        return ""

    try:
        result = UnicodeIndicTransliterator.transliterate(
            normalized, "hi", _INDIC_SCRIPT_CODES[target]
        )
    except Exception:
        logger.exception(
            "Indic script transliteration failed for target {}",
            target,
        )
        raise

    logger.debug("Converted Devanagari text to target script {}", target)
    return result


def transliterate_text(
    text: str,
    target: str,
) -> str:
    """Transliterate Devanagari text to IAST or a supported Indic script.

    Args:
        text: Source text in Devanagari.
        target: 'iast' or a supported project language code.

    Returns:
        The transliterated text.

    Raises:
        TypeError: If either argument is not a string.
        ValueError: If the target is unsupported.
    """
    if not isinstance(target, str):
        raise TypeError("target must be a string")

    normalized_target = target.strip().lower()

    if normalized_target == "iast":
        return transliterate_devanagari_to_iast(text)

    return transliterate_devanagari_to_script(text, normalized_target)
