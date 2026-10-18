
"""Tests for Devanagari transliteration."""

import pytest

from indicnlp_pipeline.transliteration.script_transliterate import (
    transliterate_devanagari_to_iast,
    transliterate_devanagari_to_script,
    transliterate_text,
)


def test_sanskrit_iast_transliteration() -> None:
    assert transliterate_devanagari_to_iast(
        "धर्मः सत्यं च"
    ) == "dharmaḥ satyaṃ ca"


def test_conjuncts_and_virama() -> None:
    assert transliterate_devanagari_to_iast("क्षेत्र") == "kṣetra"
    assert transliterate_devanagari_to_iast("विद्या") == "vidyā"


def test_independent_and_dependent_vowels() -> None:
    assert transliterate_devanagari_to_iast(
        "अ आ इ ई उ ऊ ऋ ए ऐ ओ औ"
    ) == "a ā i ī u ū ṛ e ai o au"
    assert transliterate_devanagari_to_iast("की") == "kī"


def test_nukta_consonants() -> None:
    assert transliterate_devanagari_to_iast("ज़् फ़्") == "z f"


def test_punctuation_numerals_and_whitespace() -> None:
    assert transliterate_devanagari_to_iast(
        "रामः। (१४३७) ॥"
    ) == "rāmaḥ। (1437) ॥"


def test_empty_text() -> None:
    assert transliterate_devanagari_to_iast("") == ""
    assert transliterate_devanagari_to_script("", "ben") == ""


def test_non_string_text_is_rejected() -> None:
    with pytest.raises(TypeError):
        transliterate_devanagari_to_iast(None)  # type: ignore[arg-type]


def test_bengali_script_conversion() -> None:
    result = transliterate_devanagari_to_script("धर्मः", "ben")
    assert result == "ধর্মঃ"


def test_hindi_and_sanskrit_keep_devanagari() -> None:
    text = "धर्मः सत्यं च"
    assert transliterate_devanagari_to_script(text, "hin") == text
    assert transliterate_devanagari_to_script(text, "san") == text


def test_unsupported_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported target language"):
        transliterate_devanagari_to_script("नमस्ते", "urd")


def test_dispatch_to_iast() -> None:
    assert transliterate_text("धर्मः", "iast") == "dharmaḥ"


def test_dispatch_to_bengali() -> None:
    assert transliterate_text("धर्मः", "ben") == "ধর্মঃ"


def test_target_is_case_insensitive() -> None:
    assert transliterate_text("धर्मः", "IAST") == "dharmaḥ"


def test_non_string_target_is_rejected() -> None:
    with pytest.raises(TypeError):
        transliterate_text("धर्मः", None)  # type: ignore[arg-type]
