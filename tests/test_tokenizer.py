"""Tests for Unicode-aware Hindi and Sanskrit tokenization."""

import pytest

from indicnlp_pipeline.segmentation.sentence_segmenter import segment_text
from indicnlp_pipeline.segmentation.tokenizer import (
    tokenize_segment,
    tokenize_text,
    tokens_to_dicts,
)


def test_hindi_words_and_danda_are_tokenized() -> None:
    tokens = tokenize_text("राम सीता।", "page-1", "page-1::seg-0001")

    assert [token.text for token in tokens] == ["राम", "सीता", "।"]
    assert [token.token_type for token in tokens] == [
        "word",
        "word",
        "punctuation",
    ]


def test_sanskrit_conjuncts_and_virama_are_preserved() -> None:
    text = "धर्मः क्षेत्रम्"

    tokens = tokenize_text(text, "page-2", "page-2::seg-0001")

    assert [token.text for token in tokens] == ["धर्मः", "क्षेत्रम्"]
    assert "".join(tokens[0].text) == "धर्मः"
    assert "क्ष" in tokens[1].text


def test_devanagari_matras_and_nukta_are_preserved() -> None:
    text = "किताब ज़रूरी"

    tokens = tokenize_text(text, "page-3", "page-3::seg-0001")

    assert [token.text for token in tokens] == ["किताब", "ज़रूरी"]


def test_double_danda_is_a_separate_token() -> None:
    tokens = tokenize_text("धर्मो रक्षति॥", "page-4", "seg-1")

    assert [token.text for token in tokens] == ["धर्मो", "रक्षति", "॥"]
    assert tokens[-1].token_type == "punctuation"


def test_numbered_entry_keeps_brackets_and_digits() -> None:
    tokens = tokenize_text("(१४३७)", "page-5", "seg-1")

    assert [token.text for token in tokens] == ["(", "१४३७", ")"]
    assert tokens[1].token_type == "number"


def test_ascii_numbers_are_preserved() -> None:
    tokens = tokenize_text("अध्याय 1437", "page-6", "seg-1")

    assert [token.text for token in tokens] == ["अध्याय", "1437"]
    assert tokens[-1].token_type == "number"


def test_punctuation_is_not_discarded() -> None:
    tokens = tokenize_text("रामः, सीता।", "page-7", "seg-1")

    assert [token.text for token in tokens] == ["रामः", ",", "सीता", "।"]


def test_whitespace_separates_tokens() -> None:
    tokens = tokenize_text("एक\tदो\nतीन", "page-8", "seg-1")

    assert [token.text for token in tokens] == ["एक", "दो", "तीन"]


def test_empty_or_whitespace_input_returns_no_tokens() -> None:
    assert tokenize_text("", "page-9", "seg-1") == []
    assert tokenize_text(" \n\t ", "page-9", "seg-1") == []


def test_source_offsets_point_to_original_text() -> None:
    text = "रामः, सीता।"

    tokens = tokenize_text(text, "page-10", "seg-1")

    for token in tokens:
        assert text[token.start_char:token.end_char] == token.text


def test_token_indices_are_zero_based_and_ordered() -> None:
    tokens = tokenize_text("एक दो तीन", "page-11", "seg-1")

    assert [token.token_index for token in tokens] == [0, 1, 2]


def test_segment_metadata_is_preserved() -> None:
    segments = segment_text("रामः सीताम्।", "BRIGU_V5_P0001")

    tokens = tokenize_segment(segments[0])

    assert all(token.page_id == "BRIGU_V5_P0001" for token in tokens)
    assert all(token.segment_id == segments[0].segment_id for token in tokens)


def test_serialization_returns_token_dictionaries() -> None:
    tokens = tokenize_text("रामः", "page-12", "seg-1")

    records = tokens_to_dicts(tokens)

    assert records[0]["text"] == "रामः"
    assert records[0]["token_index"] == 0
    assert records[0]["start_char"] == 0
    assert records[0]["end_char"] == len("रामः")


def test_non_string_text_is_rejected() -> None:
    with pytest.raises(TypeError):
        tokenize_text(None, "page-13", "seg-1")  # type: ignore[arg-type]


def test_blank_page_id_is_rejected() -> None:
    with pytest.raises(ValueError):
        tokenize_text("पाठ", " ", "seg-1")


def test_blank_segment_id_is_rejected() -> None:
    with pytest.raises(ValueError):
        tokenize_text("पाठ", "page-14", " ")


def test_invalid_segment_type_is_rejected() -> None:
    with pytest.raises(TypeError):
        tokenize_segment(None)  # type: ignore[arg-type]


def test_tokenization_is_deterministic() -> None:
    first = tokenize_text("धर्मः रक्षति॥", "page-15", "seg-1")
    second = tokenize_text("धर्मः रक्षति॥", "page-15", "seg-1")

    assert first == second


def test_original_text_is_not_modified() -> None:
    text = "  धर्मः   रक्षति॥  "
    original = text

    tokenize_text(text, "page-16", "seg-1")

    assert text == original
