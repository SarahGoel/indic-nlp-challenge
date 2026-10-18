"""Tests for conservative Hindi/Sanskrit language identification."""

import pytest

from indicnlp_pipeline.normalization.language_identifier import (
    identify_language,
    validate_language_code,
)


def test_identifies_hindi_from_configured_markers() -> None:
    result = identify_language("यह अच्छा है और वह भी अच्छा है।")
    assert result.language == "hin"
    assert "है" in result.hindi_markers
    assert result.confidence > 0.0


def test_identifies_sanskrit_from_configured_markers() -> None:
    result = identify_language("अथ योगः इति।")
    assert result.language == "san"
    assert result.confidence > 0.0


def test_marks_text_with_both_marker_sets_as_mixed() -> None:
    result = identify_language("यह है तथा योगः")
    assert result.language == "mixed"
    assert result.hindi_markers
    assert result.sanskrit_markers


@pytest.mark.parametrize("text", ["", "   ", "12345", "Hello world"])
def test_marks_empty_or_non_devanagari_text_unknown(text: str) -> None:
    assert identify_language(text).language == "unknown"


def test_ambiguous_devanagari_text_is_unknown() -> None:
    result = identify_language("राम")
    assert result.language == "unknown"
    assert "cannot distinguish" in result.reason


def test_does_not_modify_original_text() -> None:
    original = "यह है।"
    identify_language(original)
    assert original == "यह है।"


def test_rejects_non_string_input() -> None:
    with pytest.raises(TypeError):
        identify_language(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("hin", True),
        ("san", True),
        ("mixed", True),
        ("unknown", True),
        ("eng", False),
        ("", False),
    ],
)
def test_validate_language_code(language: str, expected: bool) -> None:
    assert validate_language_code(language) is expected
