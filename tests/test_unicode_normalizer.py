"""Tests for conservative Unicode normalization."""

import unicodedata

import pytest

from indicnlp_pipeline.normalization.unicode_normalizer import normalize_text


def test_nfc_normalization():
    text = "e\u0301"
    assert normalize_text(text, normalize_whitespace=False) == "\u00e9"


def test_devanagari_combining_sequence():
    text = "\u0915\u093c"
    result = normalize_text(text, normalize_whitespace=False)

    assert result == unicodedata.normalize("NFC", text)


def test_matras_are_preserved():
    text = "\u0915\u093f \u0915\u0940 \u0915\u0941 \u0915\u0942"
    assert normalize_text(text) == text


def test_virama_and_conjuncts_are_preserved():
    text = "\u0915\u094d\u0937 \u0924\u094d\u0930 \u091c\u094d\u091e"
    assert normalize_text(text) == text


def test_danda_is_preserved():
    text = "\u0930\u093e\u092e\u0964"
    assert normalize_text(text) == text


def test_double_danda_is_preserved():
    text = "\u0936\u094d\u0932\u094b\u0915\u0965"
    assert normalize_text(text) == text


def test_devanagari_digits_are_preserved():
    text = "\u0966\u0967\u0968\u0969\u096a\u096b\u096c\u096d\u096e\u096f"
    assert normalize_text(text) == text


def test_whitespace_is_normalized_without_merging_lines():
    text = "  \u0930\u093e\u092e   \u0938\u0940\u0924\u093e  \r\n"
    text += "\t\u0939\u0930\u093f  \u0913\u092e\t"

    expected = "\u0930\u093e\u092e \u0938\u0940\u0924\u093e\n"
    expected += "\u0939\u0930\u093f \u0913\u092e"

    assert normalize_text(text) == expected


def test_blank_lines_are_preserved():
    text = "\u0930\u093e\u092e\n\n\u0938\u0940\u0924\u093e"
    assert normalize_text(text) == text


def test_nul_ocr_noise_is_removed_by_default():
    text = "\u0930\u093e\x00\u092e"
    assert normalize_text(text) == "\u0930\u093e\u092e"


def test_other_potential_ocr_noise_is_not_silently_deleted():
    text = "\ufffd\u0930\u093e\u092e"
    assert normalize_text(text) == text


def test_nul_removal_can_be_disabled():
    text = "\u0930\u093e\x00\u092e"
    assert normalize_text(text, remove_nul=False) == text


def test_whitespace_cleanup_can_be_disabled():
    text = "  \u0930\u093e\u092e  \r\n"
    expected = unicodedata.normalize("NFC", text)

    assert normalize_text(text, normalize_whitespace=False) == expected


def test_normalization_is_idempotent():
    text = "  \u0915\u094d\u0937   \u0930\u093e\u092e\u0964  "
    once = normalize_text(text)
    twice = normalize_text(once)

    assert twice == once


def test_non_string_input_raises_type_error():
    with pytest.raises(TypeError):
        normalize_text(None)  # type: ignore[arg-type]